"""Step 12：Phase 2 汇总图 —— 域内(Flickr30k) vs 域外(COCO) 并排，看解冻骨干的收益与遗忘代价。

四条线：① 冻结 CLIP（原生截断）/ Phase1 adapter（冻结骨干，batch1024×20ep）/
对照组（同 LoRA 预算但骨干仍冻结）/ LoRA。
关键在于**两块面板必须一起看**：只看左边会把"伤了通用能力"藏起来。
"""
import sys
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.config import load_config

# (图例, 颜色, 线型, 标记, Flickr30k 结果 csv, COCO 结果 csv)
SERIES = [
    ("① frozen CLIP (truncate)", "tab:gray",  "--", "o", "baseline.csv",          "coco_baseline.csv"),
    ("Phase1 adapter (frozen)",  "tab:blue",  "-.", "s", "adapter_mrl.csv",       "coco_adapter_mrl.csv"),
    ("control (same budget)",    "tab:green", ":",  "^", "ctrl_adapter_only.csv", None),
    ("LoRA (unfrozen)",          "tab:red",   "-",  "D", "lora_mrl.csv",          None),
]


def load(res_dir, name, dataset):
    """两种格式：普通 evaluate 的单数据集 csv，和 06 输出的含 dataset 列的双数据集 csv。"""
    df = pd.read_csv(res_dir / name)
    if "dataset" in df.columns:
        df = df[df["dataset"] == dataset]
    return df.sort_values("dim")


def main():
    cfg = load_config()
    res_dir = pathlib.Path(cfg.paths.output_root) / "results"
    fig_dir = pathlib.Path(cfg.paths.output_root) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for col, (dataset, title) in enumerate((("flickr30k", "in-domain: Flickr30k 1k test"),
                                            ("coco", "out-of-domain: MS-COCO 5k test"))):
        for row, metric in enumerate(("i2t_r1", "t2i_r1")):
            ax = axes[row][col]
            for label, color, ls, mk, f_csv, c_csv in SERIES:
                name = f_csv if dataset == "flickr30k" else (c_csv or f_csv)
                if not (res_dir / name).exists():
                    continue
                d = load(res_dir, name, dataset)
                if d.empty:
                    continue
                ax.plot(d["dim"], d[metric], ls, color=color, marker=mk, ms=4, label=label)
            ax.set_xscale("log", base=2)
            ax.grid(alpha=.3)
            if row == 0:
                ax.set_title(title)
            if row == 1:
                ax.set_xlabel("embedding dim (prefix length)")
            if col == 0:
                ax.set_ylabel(f"{'i2t' if metric == 'i2t_r1' else 't2i'} R@1 (%)")
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Phase 2 — LoRA: in-domain gain vs. out-of-domain cost", fontsize=12)
    fig.tight_layout()
    out = fig_dir / "phase2_lora.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print("=== 满维(512) 对照，含相对冻结 CLIP 的差值 ===")
    print(f"{'条件':<26}{'Flickr i2t':>12}{'Flickr t2i':>12}{'COCO i2t':>12}{'COCO t2i':>12}")
    ref = {}
    for label, _, _, _, f_csv, c_csv in SERIES:
        if not (res_dir / f_csv).exists():
            continue
        f = load(res_dir, f_csv, "flickr30k")
        c = load(res_dir, c_csv or f_csv, "coco")
        vals = [f[f.dim == 512]["i2t_r1"].iloc[0], f[f.dim == 512]["t2i_r1"].iloc[0],
                c[c.dim == 512]["i2t_r1"].iloc[0], c[c.dim == 512]["t2i_r1"].iloc[0]]
        if not ref:
            ref = dict(zip("abcd", vals))
        delta = [v - r for v, r in zip(vals, ref.values())]
        print(f"{label:<26}" + "".join(f"{v:>7.2f}({d:+.2f})" for v, d in zip(vals, delta)))
    print(f"\n图 -> {out}")


if __name__ == "__main__":
    main()
