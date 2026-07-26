"""Step 13：蒸馏正则扫描的权衡图 —— 横轴域外(COCO)、纵轴域内(Flickr30k)，每个 λ 一个点。

正则扫描该看的不是"哪个 λ 的某一项最高"，而是**能不能把域外买回来而不牺牲域内** ——
所以画成权衡平面：右上角为佳，一条向右上移动的轨迹说明蒸馏是"免费"的，
向左上/右下弯则说明存在实打实的取舍。
"""
import sys
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.config import load_config

SWEEP = [(0.0, "lora_mrl.csv"), (0.5, "lora_distill05.csv"),
         (2.0, "lora_distill2.csv"), (8.0, "lora_distill8.csv")]
# 参照点：(图例, 颜色, 标记, Flickr csv, COCO csv)
REFS = [("frozen CLIP", "tab:gray", "X", "baseline.csv", "coco_baseline.csv"),
        ("Phase1 adapter", "tab:blue", "P", "adapter_mrl.csv", "coco_adapter_mrl.csv")]


def val(res_dir, name, dataset, dim, metric):
    df = pd.read_csv(res_dir / name)
    if "dataset" in df.columns:
        df = df[df["dataset"] == dataset]
    row = df[df["dim"] == dim]
    return float(row[metric].iloc[0]) if len(row) else None


def main():
    cfg = load_config()
    res_dir = pathlib.Path(cfg.paths.output_root) / "results"
    fig_dir = pathlib.Path(cfg.paths.output_root) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    have = [(lam, f) for lam, f in SWEEP if (res_dir / f).exists()]
    missing = [f for lam, f in SWEEP if not (res_dir / f).exists()]
    if missing:
        print(f"（缺 {missing}，先跑完对应的 06_train_lora）")
    if not have:
        raise FileNotFoundError("一个扫描结果都没有")

    print(f"{'λ':>6}" + "".join(f"{f'm={m} {d}':>16}" for m in (32, 512)
                                 for d in ("Flickr t2i", "COCO t2i")))
    rows = []
    for lam, f in have:
        r = {"lambda": lam}
        for m in (32, 512):
            r[f"fk_{m}"] = val(res_dir, f, "flickr30k", m, "t2i_r1")
            r[f"co_{m}"] = val(res_dir, f, "coco", m, "t2i_r1")
        rows.append(r)
        print(f"{lam:>6.1f}" + "".join(f"{r[f'{p}_{m}']:>16.2f}"
                                      for m in (32, 512) for p in ("fk", "co")))
    sw = pd.DataFrame(rows).sort_values("lambda")
    sw.to_csv(res_dir / "distill_sweep.csv", index=False, encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, m in zip(axes, (32, 512)):
        ax.plot(sw[f"co_{m}"], sw[f"fk_{m}"], "-o", color="tab:red", ms=7, zorder=3,
                label="LoRA + distill (sweep)")
        for _, r in sw.iterrows():
            ax.annotate(f"λ={r['lambda']:g}", (r[f"co_{m}"], r[f"fk_{m}"]),
                        textcoords="offset points", xytext=(7, -3), fontsize=8)
        for label, color, mk, f_csv, c_csv in REFS:
            if not (res_dir / f_csv).exists():
                continue
            x = val(res_dir, c_csv, "coco", m, "t2i_r1")
            y = val(res_dir, f_csv, "flickr30k", m, "t2i_r1")
            if x is not None and y is not None:
                ax.plot(x, y, mk, color=color, ms=12, label=label, zorder=4)
        ax.set(xlabel="out-of-domain: COCO t2i R@1 (%)",
               ylabel="in-domain: Flickr30k t2i R@1 (%)",
               title=f"m = {m}   (upper-right is better)")   # 标签用英文：默认字体没中文
        ax.grid(alpha=.3)
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Distillation sweep — can we buy back out-of-domain without losing in-domain?",
                 fontsize=12)
    fig.tight_layout()
    out = fig_dir / "distill_sweep.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n结果 -> {res_dir / 'distill_sweep.csv'}\n图 -> {out}")


if __name__ == "__main__":
    main()
