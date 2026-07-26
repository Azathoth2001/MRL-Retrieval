"""LoRA 注入：给冻结 OpenCLIP 的 vision/text 塔挂低秩增量。单一职责：参数高效微调的接线。

**为什么手写而不用 peft**：open_clip 的注意力是 `nn.MultiheadAttention`，q/k/v 融合在一个
`in_proj_weight` [3d, d] 的裸 Parameter 里，不是独立的 `nn.Linear`。peft 只能抓到 `out_proj`
和 MLP 的 Linear，**挂不到 q/k/v** —— 而那恰恰是 LoRA 最该挂的位置，漏掉会低估 LoRA、
可能得出"解冻没用"的错误结论。手写还顺便省掉一个依赖。

三类位点（`lora.targets` 控制）：
  in_proj  —— MHA 融合的 q/k/v（一个 [3d, r] 的 B 同时覆盖三者）
  out_proj —— MHA 输出投影（普通 Linear）
  mlp      —— 每个 block 的 c_fc / c_proj

初始化：A ~ kaiming，**B = 0**，所以注入瞬间 ΔW = 0、模型输出与冻结骨干**逐位相同**（有断言验证）。
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET_KINDS = ("in_proj", "out_proj", "mlp")


class LoRALinear(nn.Module):
    """包一层冻结 Linear，加上 B@A 的低秩增量。base 保持原 dtype，增量算完再 cast 回去。"""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scale = r, alpha / r
        # 建在 base 权重所在设备上：inject_lora 是在模型已搬到 GPU **之后**调的，
        # 若不指定 device，新 Parameter 会落在 CPU，前向立刻报跨设备错。
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32, device=dev))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=dev))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # 包装层必须把这几个属性透出去：open_clip 会用 `mlp.c_fc.weight.dtype` 探测精度
    # （`ResidualAttentionBlock.get_weight_dtype`），拿不到就 AttributeError。
    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        out = self.base(x)
        delta = self.drop(x.to(self.A.dtype)) @ self.A.t() @ self.B.t()
        return out + self.scale * delta.to(out.dtype)


class LoRAMultiheadAttention(nn.Module):
    """给 MHA 的 in_proj（融合 q/k/v）和 out_proj 挂 LoRA。

    **为什么这两处都得用"注入权重"而不能包成 LoRALinear**：`nn.MultiheadAttention.forward`
    是把权重**张量**交给 `F.multi_head_attention_forward` 的（`self.in_proj_weight`、
    `self.out_proj.weight`），并不会去调用 `out_proj(x)`。所以把 out_proj 换成一个包装模块会直接
    报 `'LoRALinear' object has no attribute 'weight'`。

    做法：把基权重从 `_parameters` 摘出来存成 buffer，forward 时把 `base + scale*B@A` 作为
    **普通属性**赋回去 —— MHA 只是读这个属性，前向正常，梯度沿 B/A 回传。
    in_proj 用一个 [3d, r] 的 B 同时覆盖 q/k/v 三段。
    """

    def __init__(self, mha: nn.MultiheadAttention, r: int, alpha: float, dropout: float = 0.0,
                 adapt_in=True, adapt_out=True):
        super().__init__()
        self.mha = mha
        for p in self.mha.parameters():
            p.requires_grad_(False)
        d = mha.embed_dim
        self.r, self.scale = r, alpha / r
        self.adapt_in, self.adapt_out = adapt_in, adapt_out
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if adapt_in:
            self._demote(mha, "in_proj_weight", "in_proj_weight_base")
            dev = mha.in_proj_weight_base.device
            self.A_in = nn.Parameter(torch.empty(r, d, dtype=torch.float32, device=dev))
            self.B_in = nn.Parameter(torch.zeros(3 * d, r, dtype=torch.float32, device=dev))
            nn.init.kaiming_uniform_(self.A_in, a=math.sqrt(5))
        if adapt_out:
            self._demote(mha.out_proj, "weight", "weight_base")
            dev = mha.out_proj.weight_base.device
            self.A_out = nn.Parameter(torch.empty(r, d, dtype=torch.float32, device=dev))
            self.B_out = nn.Parameter(torch.zeros(d, r, dtype=torch.float32, device=dev))
            nn.init.kaiming_uniform_(self.A_out, a=math.sqrt(5))

    @staticmethod
    def _demote(module, param_name, buffer_name):
        """把可训练 Parameter 降级成 buffer，腾出同名属性给 forward 里算出来的张量。"""
        w = getattr(module, param_name).detach().clone()
        del module._parameters[param_name]
        module.register_buffer(buffer_name, w)

    def forward(self, *args, **kwargs):
        if self.adapt_in:
            base = self.mha.in_proj_weight_base
            d = self.scale * (self.B_in @ self.drop(self.A_in))
            self.mha.in_proj_weight = base + d.to(base.dtype)
        if self.adapt_out:
            base = self.mha.out_proj.weight_base
            d = self.scale * (self.B_out @ self.drop(self.A_out))
            self.mha.out_proj.weight = base + d.to(base.dtype)
        return self.mha(*args, **kwargs)


def _tower_of(name):
    return "visual" if name.startswith("visual") else "text"


def inject_lora(model, r=8, alpha=16, dropout=0.0,
                targets=TARGET_KINDS, towers=("visual", "text")):
    """就地给 model 挂 LoRA，返回 {统计信息}。骨干全部冻结，只有 LoRA 参数可训。"""
    bad = set(targets) - set(TARGET_KINDS)
    if bad:
        raise ValueError(f"未知 lora.targets {sorted(bad)}，可选 {TARGET_KINDS}")

    # 注入前先记总量：in_proj 的基权重会被降级成 buffer（24×2304×768 ≈ 42.5M），
    # 注入后再数 model.parameters() 会漏掉这批，占比算出来是错的。
    n_total = sum(p.numel() for p in model.parameters())
    model.requires_grad_(False)                        # 先全冻，再逐个挂
    n_sites = {k: 0 for k in TARGET_KINDS}

    blocks = []
    if "visual" in towers:
        blocks += [(f"visual.transformer.resblocks.{i}", b)
                   for i, b in enumerate(model.visual.transformer.resblocks)]
    if "text" in towers:
        blocks += [(f"transformer.resblocks.{i}", b)
                   for i, b in enumerate(model.transformer.resblocks)]

    adapt_in, adapt_out = "in_proj" in targets, "out_proj" in targets
    for name, blk in blocks:
        if adapt_in or adapt_out:
            blk.attn = LoRAMultiheadAttention(blk.attn, r, alpha, dropout,
                                             adapt_in=adapt_in, adapt_out=adapt_out)
            n_sites["in_proj"] += int(adapt_in)
            n_sites["out_proj"] += int(adapt_out)
        if "mlp" in targets:
            blk.mlp.c_fc = LoRALinear(blk.mlp.c_fc, r, alpha, dropout)
            blk.mlp.c_proj = LoRALinear(blk.mlp.c_proj, r, alpha, dropout)
            n_sites["mlp"] += 2

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"sites": {k: v for k, v in n_sites.items() if v},
            "n_trainable": n_train, "n_total": n_total,
            "pct": 100.0 * n_train / n_total, "r": r, "alpha": alpha,
            "targets": list(targets), "towers": list(towers)}


LORA_SUFFIXES = (".A", ".B", ".A_in", ".B_in", ".A_out", ".B_out")


def lora_state_dict(model):
    """只导出 LoRA 参数（A/B），checkpoint 才不会有几百 MB 的冻结骨干。"""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k.endswith(LORA_SUFFIXES)}


def load_lora_state_dict(model, sd):
    """载入 LoRA 权重。模型必须已经 inject_lora 过（形状要对得上）。"""
    known = set(dict(model.named_parameters())) | set(dict(model.named_buffers()))
    unexpected = [k for k in sd if k not in known]
    if unexpected:
        raise RuntimeError(f"checkpoint 里有对不上的键（模型是否已 inject_lora？）：{unexpected[:5]}")
    model.load_state_dict(sd, strict=False)
