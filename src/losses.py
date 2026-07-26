"""损失：嵌套 InfoNCE（温度 fixed/learnable/per_dim，权重 equal/inverse/linear）。单一职责：损失数学。"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# τ 的合法区间。防温度塌缩（τ→0 时 logits 爆炸）与失效（τ 过大时对比信号消失）。
TAU_MIN, TAU_MAX = 0.01, 1.0


def dim_weights(dims, mode="equal"):
    """各档权重 c_m，归一化到 **sum = len(dims)**（保证不同 mode 间总梯度尺度可比）。

    mode: equal（c_m=1，中性默认）| inverse（∝1/m，压小维度）| linear（∝m，压大维度）。
    """
    dims = list(dims)
    if mode == "equal":
        raw = [1.0] * len(dims)
    elif mode == "inverse":
        raw = [1.0 / m for m in dims]
    elif mode == "linear":
        raw = [float(m) for m in dims]
    else:
        raise ValueError(f"weight.mode 只能是 equal/inverse/linear，收到 {mode!r}")
    scale = len(dims) / sum(raw)
    return [r * scale for r in raw]


class Temperature(nn.Module):
    """温度 τ。以 log τ 参数化（保证 τ>0、优化更稳），forward(i) 给出第 i 档的 τ。

    mode: fixed（常数、不进优化器）| learnable（共享一个可学 τ）| per_dim（每档一个 τ_m，校准各档相似度尺度）。
    τ 在 exp 内、非线性 —— 改它等于改目标本身，与求和外的线性权重 c_m 不可互相替代。
    """

    def __init__(self, mode="fixed", init=0.07, n_dims=1):
        super().__init__()
        if mode not in ("fixed", "learnable", "per_dim"):
            raise ValueError(f"temperature.mode 只能是 fixed/learnable/per_dim，收到 {mode!r}")
        if not TAU_MIN <= init <= TAU_MAX:
            raise ValueError(f"temperature.init={init} 超出 [{TAU_MIN}, {TAU_MAX}]")
        self.mode = mode
        log_t = torch.full((n_dims if mode == "per_dim" else 1,), math.log(init))
        if mode == "fixed":
            self.register_buffer("log_t", log_t)      # buffer：随模型搬设备但不被训练
        else:
            self.log_t = nn.Parameter(log_t)

    def forward(self, i=0):
        idx = i if self.mode == "per_dim" else 0
        return self.log_t[idx].clamp(math.log(TAU_MIN), math.log(TAU_MAX)).exp()

    def values(self):
        """当前各档 τ（float 列表），日志用。"""
        with torch.no_grad():
            return [float(self(i)) for i in range(self.log_t.numel())]


def matryoshka_infonce(e_img, e_txt, dims, temperature, weights):
    """嵌套对称 InfoNCE。

    对每个 m：**先取前缀 [:m] 再各自 L2 归一化**（切勿先归一化整段再切——最常见的坑，
    那样低维前缀的模长信息会被高维分量污染），算双向交叉熵，再按 c_m 加权求和。

    e_img / e_txt : [B, D] adapter 输出（未归一化）
    dims          : 嵌套档位，需满足 max(dims) <= D
    temperature   : Temperature 模块（按档取 τ），或一个标量 float
    weights       : 与 dims 等长的 c_m 列表
    返回 (total_loss, {m: 该档 loss（已 detach）})
    """
    if e_img.shape != e_txt.shape:
        raise ValueError(f"图文 embedding 形状不一致：{tuple(e_img.shape)} vs {tuple(e_txt.shape)}")
    if len(weights) != len(dims):
        raise ValueError(f"weights({len(weights)}) 与 dims({len(dims)}) 长度不一致")
    if max(dims) > e_img.shape[1]:
        raise ValueError(f"dims 最大 {max(dims)} 超过 embedding 维度 {e_img.shape[1]}")

    target = torch.arange(e_img.shape[0], device=e_img.device)
    total, per_dim = e_img.new_zeros(()), {}
    for i, m in enumerate(dims):
        zi = F.normalize(e_img[:, :m], dim=-1)          # 先切 -> 再归一化
        zt = F.normalize(e_txt[:, :m], dim=-1)
        tau = temperature(i) if callable(temperature) else temperature
        logits = zi @ zt.t() / tau                      # [B, B]，行=图 列=文
        loss_m = 0.5 * (F.cross_entropy(logits, target) +
                        F.cross_entropy(logits.t(), target))
        total = total + weights[i] * loss_m
        per_dim[m] = loss_m.detach()
    return total, per_dim
