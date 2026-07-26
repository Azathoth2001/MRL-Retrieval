"""Step 6（Phase 2）：LoRA 微调骨干 + 同一个嵌套损失。

与 Phase 1 的关键区别：LoRA 每步都改骨干 → **嵌入随之变化、无法复用缓存特征**，
必须每步整套前向过 ViT + 文本编码器。这才是真正吃算力的阶段。

精度策略：骨干 **fp32 权重 + autocast(fp16) + GradScaler**，而不是 fp16 主权重 ——
后者配上 LoRA 的小梯度容易下溢。实测 8GB 显存下 batch 384 峰值 6.66GB，故默认 256 留余量。

**必须同时报 Flickr30k 与 COCO**（见 RESULTS.md §6、§7）：冻结 adapter 已经测出"域内赚
0.3~1.0 个点、域外亏 4.3~6.0 个点"，所以只报域内涨幅是没有意义的。

    python scripts/06_train_lora.py                     # 训练 + 双数据集评测
    python scripts/06_train_lora.py --eval-only         # 只用已有 checkpoint 评测
"""
import argparse
import json
import sys
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.adapter import build_adapter
from src.config import load_config
from src.data import build_extract_loader, build_pair_loader, get_coco_splits, get_splits
from src.features import load_features
from src.losses import Temperature, dim_weights, matryoshka_infonce
from src.metrics import recall_at_k
from src.lora import inject_lora, load_lora_state_dict, lora_state_dict
from src.model import load_clip

EVAL_SETS = [("flickr30k", "test"), ("coco", "coco_test")]


def build_model(cfg):
    """fp32 骨干 + LoRA + adapter。cfg 的 model.precision 必须已是 fp32（见模块 docstring）。"""
    model, preprocess, tokenizer = load_clip(cfg)
    info = inject_lora(model, r=cfg.lora.r, alpha=cfg.lora.alpha, dropout=cfg.lora.dropout,
                       targets=cfg.lora.targets, towers=cfg.lora.towers)
    if cfg.lora.grad_checkpointing:
        model.set_grad_checkpointing(True)

    adapter = build_adapter(cfg).to(cfg.model.device)
    init = getattr(cfg.lora, "init_adapter", None)
    if init:
        p = pathlib.Path(cfg.paths.output_root) / "checkpoints" / f"{init}.pt"
        if not p.exists():
            raise FileNotFoundError(f"热启动用的 adapter 不存在：{p}（或把 lora.init_adapter 设为 null）")
        adapter.load_state_dict(torch.load(p, map_location="cpu", weights_only=True)["state_dict"])
        print(f"  adapter 从 {p.name} 热启动")
    return model, adapter, preprocess, tokenizer, info


def records_of(cfg, split):
    """按 split 名取 records（`coco_*` 走 COCO，其余走 Flickr30k），与 features 的分派一致。"""
    if split.startswith("coco_"):
        s = split[len("coco_"):]
        return get_coco_splits(cfg, [s])[s]
    return get_splits(cfg)[split]


@torch.no_grad()
def encode_split(cfg, model, adapter, split, preprocess, tokenizer):
    """用当前（已微调的）模型编码一个 split，返回 adapter 后的图/文 embedding + ids。"""
    was_training = model.training
    model.eval()
    adapter.eval()
    dev = cfg.model.device
    records = records_of(cfg, split)
    out = {}
    for modality in ("image", "text"):
        loader, ids = build_extract_loader(cfg, records, modality,
                                          preprocess=preprocess, tokenizer=tokenizer)
        chunks = []
        for batch in loader:
            batch = batch.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                e = model.encode_image(batch) if modality == "image" else model.encode_text(batch)
                e = adapter(e.float())
            chunks.append(e.float().cpu().numpy())
        out[modality] = (np.concatenate(chunks, 0), ids)
    if was_training:
        model.train()
        adapter.train()
    return out


def recall_by_dim(emb, dims):
    """对每个档位先切前缀再归一化，算双向 R@1。emb = {'image': (E,ids), 'text': (E,ids)}。"""
    (img, img_ids), (txt, txt_ids) = emb["image"], emb["text"]
    rows = []
    for m in dims:
        l2 = lambda x: x[:, :m] / np.maximum(np.linalg.norm(x[:, :m], axis=1, keepdims=True), 1e-12)
        r = recall_at_k(l2(img), l2(txt), img_ids, txt_ids, ks=(1, 5, 10))
        rows.append({"dim": m, **{f"{d}_r{k}": r[d][k] for d in ("i2t", "t2i") for k in (1, 5, 10)}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--name", default="lora_mrl")
    ap.add_argument("--smoke", type=int, default=0,
                    help="冒烟测试：只用前 N 张训练图跑 1 epoch，验证整条链路")
    ap.add_argument("--set", action="append", default=[], metavar="k=v")
    args = ap.parse_args()

    import yaml
    ov = {"model.precision": "fp32"}     # LoRA 训练固定 fp32 权重 + autocast，不用 fp16 主权重
    for item in args.set:
        k, v = item.split("=", 1)
        ov[k] = yaml.safe_load(v)
    cfg = load_config(overrides=ov)
    if args.smoke:
        cfg.lora.epochs = 1
    dev = cfg.model.device
    dims = list(cfg.mrl.dims) if cfg.loss.nested else [cfg.mrl.full_dim]
    ck_dir = pathlib.Path(cfg.paths.output_root) / "checkpoints"
    res_dir = pathlib.Path(cfg.paths.output_root) / "results"
    ck_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    ck_path = ck_dir / f"{args.name}.pt"

    torch.manual_seed(cfg.train.seed)
    rng = np.random.default_rng(cfg.train.seed)

    model, adapter, preprocess, tokenizer, info = build_model(cfg)
    print(f"[LoRA] r={cfg.lora.r} alpha={cfg.lora.alpha} 位点={info['sites']} "
          f"towers={info['towers']}；可训练 {info['n_trainable']/1e6:.2f}M / "
          f"{info['n_total']/1e6:.1f}M = {info['pct']:.2f}%")

    if args.eval_only:
        if not ck_path.exists():
            raise FileNotFoundError(f"缺 {ck_path}")
        ck = torch.load(ck_path, map_location=dev, weights_only=True)
        load_lora_state_dict(model, ck["lora"])
        adapter.load_state_dict(ck["adapter"])
    else:
        records = get_splits(cfg)["train"]
        tr_img_cached, _ = load_features(cfg, "train", "image")   # 蒸馏目标：冻结 CLIP 的原始嵌入
        tr_txt_cached, tr_txt_ids = load_features(cfg, "train", "text")
        from src.train import _pair_table
        pairs = _pair_table([r["image_id"] for r in records], tr_txt_ids)
        if args.smoke:      # 只留前 N 张（pairs 与 records 同序，一起截即可）
            records, pairs = records[:args.smoke], pairs[:args.smoke]
            print(f"  [smoke] 只用 {len(records)} 张训练图、1 epoch")

        weights = dim_weights(dims, cfg.loss.weight.mode)
        temperature = Temperature(cfg.loss.temperature.mode, cfg.loss.temperature.init,
                                  n_dims=len(dims)).to(dev)
        lora_params = [p for p in model.parameters() if p.requires_grad]
        # `--set lora.towers=[]` 时一个 LoRA 参数都没有 → 这就是**只训 adapter 的对照组**：
        # 同 epoch、同 batch、同 LR，唯一差别是骨干有没有解冻，从而把"解冻的净贡献"隔离出来。
        # 空参数组会让 AdamW 报错，所以为空就不加这一组。
        groups = [{"params": list(adapter.parameters()) + list(temperature.parameters()),
                   "lr": cfg.lora.adapter_lr}]
        if lora_params:
            groups.insert(0, {"params": lora_params, "lr": cfg.lora.lr})
        else:
            print("  [对照组] 没有 LoRA 参数（towers 为空）→ 只训 adapter")
        opt = torch.optim.AdamW(groups, weight_decay=cfg.lora.weight_decay)
        scaler = torch.amp.GradScaler("cuda")
        bs = cfg.lora.batch_size
        steps = len(records) // bs
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.lora.epochs * steps)
        print(f"  可训练：LoRA {sum(p.numel() for p in lora_params)/1e6:.2f}M + "
              f"adapter {sum(p.numel() for p in adapter.parameters())/1e6:.2f}M")
        print(f"  {len(records)} 图/epoch，batch={bs} -> {steps} step/epoch，"
              f"{cfg.lora.epochs} epoch；蒸馏权重={cfg.lora.distill}")

        img_tgt = torch.from_numpy(tr_img_cached).to(dev)
        txt_tgt = torch.from_numpy(tr_txt_cached).to(dev)
        best, best_state, best_ep = -1.0, None, -1
        for ep in range(1, cfg.lora.epochs + 1):
            cap_choice = rng.integers(0, pairs.shape[1], size=len(records))
            loader = build_pair_loader(cfg, records, cap_choice, preprocess, tokenizer, bs)
            model.train()
            adapter.train()
            run, run_d, n = 0.0, 0.0, 0
            t0 = time.perf_counter()
            for img, tok, idx in tqdm(loader, desc=f"ep{ep}", unit="step"):
                img, tok = img.to(dev, non_blocking=True), tok.to(dev, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    ei = model.encode_image(img)
                    et = model.encode_text(tok)
                zi, zt = adapter(ei.float()), adapter(et.float())
                loss, _ = matryoshka_infonce(zi, zt, dims, temperature, weights)
                if cfg.lora.distill > 0:      # 抗遗忘：拉住骨干输出别偏离原始 CLIP 太远
                    ii = idx.to(dev)
                    ti = torch.from_numpy(pairs[idx.numpy(), cap_choice[idx.numpy()]]).to(dev)
                    d = (2 - F.cosine_similarity(ei.float(), img_tgt[ii], dim=-1).mean()
                         - F.cosine_similarity(et.float(), txt_tgt[ti], dim=-1).mean())
                    loss = loss + cfg.lora.distill * d
                    run_d += float(d)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                run += float(loss)
                n += 1

            emb = encode_split(cfg, model, adapter, "val", preprocess, tokenizer)
            rows = recall_by_dim(emb, dims)
            score = float(np.mean([0.5 * (r["i2t_r1"] + r["t2i_r1"]) for r in rows]))
            tag = ""
            if score > best:
                best, best_ep = score, ep
                best_state = ({k: v.clone() for k, v in lora_state_dict(model).items()},
                              {k: v.detach().clone() for k, v in adapter.state_dict().items()})
                tag = "  <- best"
            print(f"  ep{ep}  loss={run/max(n,1):.4f}"
                  + (f"  distill={run_d/max(n,1):.4f}" if cfg.lora.distill > 0 else "")
                  + f"  val R@1均值={score:.2f}  ({time.perf_counter()-t0:.0f}s){tag}")

        torch.save({"lora": best_state[0], "adapter": best_state[1],
                    "dim": cfg.mrl.full_dim, "hidden": cfg.adapter.hidden,
                    "dims": dims, "lora_cfg": {k: getattr(cfg.lora, k) for k in
                                               ("r", "alpha", "targets", "towers", "lr",
                                                "epochs", "batch_size", "distill")},
                    "temperature": temperature.values(),
                    "best_val_r1": best, "best_epoch": best_ep}, ck_path)
        print(f"  best ep{best_ep}（val R@1均值={best:.2f}）-> {ck_path}")
        load_lora_state_dict(model, best_state[0])
        adapter.load_state_dict(best_state[1])

    print("\n=== 双数据集评测（域内 + 域外，缺一不可）===")
    all_rows = []
    for label, split in EVAL_SETS:
        emb = encode_split(cfg, model, adapter, split, preprocess, tokenizer)
        rows = recall_by_dim(emb, list(cfg.mrl.dims))
        print(f"\n[{label}] {split}: {len(emb['image'][1])} 图 / {len(emb['text'][1])} caption")
        for r in rows:
            print(f"  m={r['dim']:>4}  i2t R@1={r['i2t_r1']:6.2f}  t2i R@1={r['t2i_r1']:6.2f}")
        all_rows += [{"dataset": label, "condition": args.name, **r} for r in rows]

    import pandas as pd
    out = res_dir / f"{args.name}.csv"
    pd.DataFrame(all_rows).to_csv(out, index=False, encoding="utf-8")
    print(f"\n结果 -> {out}")


if __name__ == "__main__":
    main()
