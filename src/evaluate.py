"""评测编排：给定一个 reducer，跑遍各维度 -> 质量(metrics)+工程(retrieval) -> 汇总结果表。单一职责：编排评测。

四组（truncate / pca / adapter_plain / adapter_mrl）走的是**同一段代码**，
唯一差别是传进来的 reducer —— 这样"MRL 赢了"不可能是评测路径不同带来的假象。

质量在 1000 图 test 上算（与文献可比）；延迟/存储在 31k 全量图库上测（1000 张太小，各维度都亚毫秒）。
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

from .features import load_features, load_gallery
from .metrics import recall_at_k
from .retrieval import bench


def _results_dir(cfg) -> Path:
    d = Path(cfg.paths.output_root) / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def evaluate_condition(cfg, reducer, name, split="test", with_bench=True):
    """对 cfg.mrl.dims 每个 m 评一遍，汇总成表并存 outputs/results/{name}.csv。

    行 = 维度；列 = 双向 R@1/5/10 + 相对满维的保持率 + 各精度的延迟/存储。
    """
    img, img_ids = load_features(cfg, split, "image")
    txt, txt_ids = load_features(cfg, split, "text")
    gallery, _ = load_gallery(cfg) if with_bench else (None, None)

    dims = list(cfg.mrl.dims)
    print(f"[{name}] {split}: {len(img_ids)} 图 / {len(txt_ids)} caption"
          + (f"；图库 {len(gallery)} 图" if with_bench else "")
          + f"；维度 {dims}")

    rows = []
    for m in dims:
        row = {"condition": name, "dim": m}
        rec = recall_at_k(reducer.transform(img, m), reducer.transform(txt, m),
                          img_ids, txt_ids, ks=(1, 5, 10))
        for d in ("i2t", "t2i"):
            for k in (1, 5, 10):
                row[f"{d}_r{k}"] = rec[d][k]

        if with_bench:
            # query = caption（text→image 检索，图库是 31k 张图），与 t2i 方向一致
            gal_m, q_m = reducer.transform(gallery, m), reducer.transform(txt, m)
            for prec in cfg.eval.precisions:
                b = bench(gal_m, q_m, m, precision=prec, index=cfg.eval.faiss_index)
                row[f"lat_ms_{prec}"] = b["lat_ms"]
                row[f"bytes_{prec}"] = b["bytes_per_vec"]
                row[f"index_mb_{prec}"] = b["index_mb"]
            row["n_gallery"] = b["n_gallery"]

        rows.append(row)
        print(f"  m={m:>4}  i2t R@1={row['i2t_r1']:6.2f}  t2i R@1={row['t2i_r1']:6.2f}"
              + (f"  lat={row[f'lat_ms_{cfg.eval.precisions[0]}']:6.3f}ms" if with_bench else ""))

    df = pd.DataFrame(rows)
    full = max(dims)                                  # 相对满维的保持率 —— 头牌指标
    for col in ("i2t_r1", "t2i_r1"):
        base = df.loc[df["dim"] == full, col].iloc[0]
        df[f"{col}_ret"] = 100.0 * df[col] / base if base else float("nan")

    out = _results_dir(cfg) / f"{name}.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"  -> {out}")
    return df


def load_condition_results(cfg, names):
    """读回若干组的结果 CSV 并纵向拼起来（05 汇总用）。缺的组跳过并提示。"""
    frames = []
    for n in names:
        p = _results_dir(cfg) / f"{n}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
        else:
            print(f"  (跳过 {n}：{p.name} 不存在)")
    if not frames:
        raise FileNotFoundError("一组结果都没有，先跑 03 / 04")
    return pd.concat(frames, ignore_index=True)
