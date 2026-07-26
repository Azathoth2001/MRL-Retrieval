"""训练：在缓存特征上训 adapter（nested 与否、温度、权重全读 config）。单一职责：训练循环。

`loss.nested` 决定档位集合：true -> cfg.mrl.dims（④ MRL）；false -> 只有 full_dim（③ 消融）。
两者共用这一份代码、同架构同预算，唯一差别就是嵌套，才能把"nesting 的贡献"隔离出来。

**每 epoch 每图只采 1 条 caption**（5 条里轮换）：若一个 batch 里同时出现同一张图的两条 caption，
InfoNCE 会把其中一条当负例（假负例）。batch=1024 时每步约 14 对，虽不致命但白送噪声，采样避开更干净。
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .adapter import build_adapter
from .data import train_holdout_indices
from .features import load_features
from .losses import Temperature, dim_weights, matryoshka_infonce
from .metrics import recall_at_k


def _pair_table(img_ids, txt_ids):
    """[n_img, n_caps] 的 caption 下标表 —— 第 i 行是第 i 张图那几条 caption 在 txt 里的位置。"""
    pos = {iid: k for k, iid in enumerate(img_ids)}
    per_img = defaultdict(list)
    for k, tid in enumerate(txt_ids):
        base = tid.rsplit("_cap", 1)[0]
        if base not in pos:
            raise KeyError(f"txt_id {tid!r} 找不到对应图 {base!r}")
        per_img[pos[base]].append(k)
    counts = {len(v) for v in per_img.values()}
    if len(per_img) != len(img_ids) or len(counts) != 1:
        raise ValueError(f"图文对齐异常：{len(per_img)}/{len(img_ids)} 张图有 caption，每图条数 {counts}")
    return np.array([per_img[i] for i in range(len(img_ids))], dtype=np.int64)


def _l2_prefix(x, m):
    """先切前缀再归一化（顺序不能反）。x: [N, D] torch tensor -> numpy [N, m]。"""
    z = torch.nn.functional.normalize(x[:, :m], dim=-1)
    return z.detach().cpu().numpy()


@torch.no_grad()
def _val_score(adapter, val_img, val_txt, val_img_ids, val_txt_ids, dims):
    """val 上各档 R@1 的均值（i2t 与 t2i 各半）—— 选 best checkpoint 的依据。

    这里的 dims 一律用**评测档位** cfg.mrl.dims，不是训练档位：③ 与 ④ 的差别必须只有
    "损失是否嵌套"这一条，若 ③ 只按 512 维选、④ 按各档均值选，就多了个模型选择的混杂因素。
    """
    adapter.eval()
    ei, et = adapter(val_img), adapter(val_txt)
    per_dim = {}
    for m in dims:
        rec = recall_at_k(_l2_prefix(ei, m), _l2_prefix(et, m), val_img_ids, val_txt_ids, ks=(1,))
        per_dim[m] = 0.5 * (rec["i2t"][1] + rec["t2i"][1])
    adapter.train()
    return float(np.mean(list(per_dim.values()))), per_dim


def train(cfg):
    """训 adapter 并存 outputs/checkpoints/adapter_{experiment.name}.pt（存 val 最优那一版）。"""
    torch.manual_seed(cfg.train.seed)
    rng = np.random.default_rng(cfg.train.seed)
    device = cfg.model.device

    tr_img, tr_img_ids = load_features(cfg, "train", "image")
    tr_txt, tr_txt_ids = load_features(cfg, "train", "text")
    va_img, va_img_ids = load_features(cfg, "val", "image")
    va_txt, va_txt_ids = load_features(cfg, "val", "text")
    pairs = _pair_table(tr_img_ids, tr_txt_ids)                  # [n_img, 5]

    if getattr(cfg.data, "use_holdout", False):
        # 留出一批 train 图不参与训练，供"同域干净干扰项"实验用（见 data.train_holdout_indices）。
        # pairs 里存的是 caption 在完整 tr_txt 中的下标，所以 tr_txt 保持整份、只筛图这一侧。
        keep, held = train_holdout_indices(len(tr_img_ids), cfg.data.holdout_train)
        tr_img, pairs = tr_img[keep], pairs[keep]
        tr_img_ids = [tr_img_ids[i] for i in keep]
        print(f"  [holdout] 训练用 {len(keep)} 图，留出 {len(held)} 图不参与训练")

    to = lambda a: torch.from_numpy(a).to(device)                # 特征全放显存：29k+145k×512 fp32 ≈ 356MB
    tr_img_t, tr_txt_t, va_img_t, va_txt_t = to(tr_img), to(tr_txt), to(va_img), to(va_txt)

    eval_dims = list(cfg.mrl.dims)                               # 评测档位（③④ 共用，用于选 best）
    dims = eval_dims if cfg.loss.nested else [cfg.mrl.full_dim]  # 训练档位（③ 只有满维）
    weights = dim_weights(dims, cfg.loss.weight.mode)
    adapter = build_adapter(cfg).to(device).train()
    temperature = Temperature(cfg.loss.temperature.mode, cfg.loss.temperature.init,
                              n_dims=len(dims)).to(device)

    params = list(adapter.parameters()) + [p for p in temperature.parameters()]
    n_param = sum(p.numel() for p in adapter.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    n_img, bs = len(tr_img_ids), cfg.train.batch_size
    steps_per_epoch = max(1, n_img // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.train.epochs * steps_per_epoch)

    print(f"[train] experiment={cfg.experiment.name}  nested={cfg.loss.nested}  dims={dims}")
    print(f"  权重({cfg.loss.weight.mode}) = {[round(w, 4) for w in weights]}")
    print(f"  τ({cfg.loss.temperature.mode}) 初始 = {[round(t, 4) for t in temperature.values()]}")
    print(f"  adapter {n_param/1e6:.2f}M 参数；{n_img} 图/epoch，batch={bs} -> {steps_per_epoch} step/epoch")

    best, best_state, best_epoch = -1.0, None, -1
    for epoch in range(1, cfg.train.epochs + 1):
        order = rng.permutation(n_img)
        cap_choice = rng.integers(0, pairs.shape[1], size=n_img)     # 每图本轮用哪条 caption
        running = defaultdict(float)
        for step in range(steps_per_epoch):
            idx = order[step * bs:(step + 1) * bs]
            if len(idx) < 2:                                          # InfoNCE 至少要 2 个样本
                continue
            t_idx = pairs[idx, cap_choice[idx]]
            e_img = adapter(tr_img_t[torch.from_numpy(idx).to(device)])
            e_txt = adapter(tr_txt_t[torch.from_numpy(t_idx).to(device)])

            loss, per_dim = matryoshka_infonce(e_img, e_txt, dims, temperature, weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running["total"] += float(loss)
            for m, v in per_dim.items():
                running[m] += float(v)

        score, per_dim_r1 = _val_score(adapter, va_img_t, va_txt_t, va_img_ids, va_txt_ids, eval_dims)
        tag = ""
        if score > best:
            best, best_epoch = score, epoch
            best_state = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
            tag = "  <- best"
        losses = "  ".join(f"m{m}={running[m]/steps_per_epoch:.3f}" for m in dims)
        print(f"  ep{epoch:>3}  loss={running['total']/steps_per_epoch:.4f}  [{losses}]"
              f"  val R@1均值={score:.2f}{tag}")

    ck_dir = Path(cfg.paths.output_root) / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    out = ck_dir / f"{cfg.experiment.name}.pt"      # experiment.name 已含 adapter_ 前缀
    torch.save({
        "state_dict": best_state,
        "dim": cfg.mrl.full_dim,
        "hidden": cfg.adapter.hidden,
        "experiment": cfg.experiment.name,
        "nested": cfg.loss.nested,
        "dims": dims,                 # 训练档位
        "eval_dims": eval_dims,       # 选 best 用的档位（③④ 一致）
        "weight_mode": cfg.loss.weight.mode,
        "weights": weights,
        "temperature_mode": cfg.loss.temperature.mode,
        "temperature": temperature.values(),
        "epochs": cfg.train.epochs,
        "lr": cfg.train.lr,
        "use_holdout": bool(getattr(cfg.data, "use_holdout", False)),
        "holdout_train": cfg.data.holdout_train if getattr(cfg.data, "use_holdout", False) else 0,
        "n_train_images": n_img,
        "best_val_r1": best,
        "best_epoch": best_epoch,
    }, out)
    print(f"  best ep{best_epoch}（val R@1均值={best:.2f}）-> {out}")
    return out
