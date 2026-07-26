"""Step 5：四组统一评测 + 出对比表/曲线（R@m + 延迟/存储）。

已有 CSV 的组直接复用；有 checkpoint 但没评过的 adapter 组现场评。
图里用英文标签 —— matplotlib 默认字体没有中文，用中文会变成豆腐块。
"""
import sys
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.evaluate import evaluate_condition, load_condition_results
from src.reducers import AdapterReducer

# (结果名, checkpoint 名或 None, 图例, 颜色, 线型)
CONDITIONS = [
    ("baseline",      None,             "① CLIP truncate",  "tab:gray",   "--"),
    ("pca",           None,             "② PCA",            "tab:orange", "-."),
    ("adapter_plain", "adapter_plain",  "③ Adapter (plain)", "tab:green",  ":"),
    ("adapter_mrl",   "adapter_mrl",    "④ Adapter (MRL)",  "tab:red",    "-"),
]


def ensure_evaluated(cfg):
    """adapter 组：有 ckpt 但没结果 CSV 的，现场评一遍。"""
    res_dir = pathlib.Path(cfg.paths.output_root) / "results"
    ck_dir = pathlib.Path(cfg.paths.output_root) / "checkpoints"
    for name, ck, *_ in CONDITIONS:
        if ck is None or (res_dir / f"{name}.csv").exists():
            continue
        ck_path = ck_dir / f"{ck}.pt"
        if ck_path.exists():
            print(f"[{name}] 有 checkpoint 未评测，现在评：")
            evaluate_condition(cfg, AdapterReducer(ck_path, device=cfg.model.device), name)


def plot_recall(df, present, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, col, title in zip(axes, ("i2t_r1", "t2i_r1"),
                              ("image -> text  R@1", "text -> image  R@1")):
        for name, _, label, color, ls in present:
            d = df[df["condition"] == name].sort_values("dim")
            ax.plot(d["dim"], d[col], ls, color=color, marker="o", ms=4, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("embedding dim (prefix length)")
        ax.set_ylabel("R@1 (%)")
        ax.set_title(title)
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Flickr30k 1k test — recall vs. retrieval dimension", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_cost(df, present, prec, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, _, label, color, ls in present:
        d = df[df["condition"] == name].sort_values("dim")
        axes[0].plot(d["dim"], d[f"lat_ms_{prec}"], ls, color=color, marker="o", ms=4, label=label)
        axes[1].plot(d[f"bytes_{prec}"], d["t2i_r1"], ls, color=color, marker="o", ms=4, label=label)
    axes[0].set(xscale="log", xlabel="embedding dim", ylabel=f"latency (ms/query, {prec})",
                title=f"faiss single-query latency ({prec})")
    axes[0].set_xscale("log", base=2)
    axes[1].set(xlabel=f"bytes per vector ({prec})", ylabel="t2i R@1 (%)",
                title="quality vs. storage cost")
    for ax in axes:
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    cfg = load_config()
    ensure_evaluated(cfg)
    df = load_condition_results(cfg, [c[0] for c in CONDITIONS])
    present = [c for c in CONDITIONS if c[0] in set(df["condition"])]
    prec = cfg.eval.precisions[0]

    res_dir = pathlib.Path(cfg.paths.output_root) / "results"
    fig_dir = pathlib.Path(cfg.paths.output_root) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(res_dir / "summary.csv", index=False, encoding="utf-8")

    show = ["condition", "dim", "i2t_r1", "t2i_r1", "i2t_r1_ret", "t2i_r1_ret",
            f"lat_ms_{prec}", f"bytes_{prec}"]
    print("\n=== 汇总（ret = 相对满维的保持率 %）===")
    print(df[[c for c in show if c in df.columns]].to_string(index=False,
          float_format=lambda v: f"{v:.2f}"))

    plot_recall(df, present, fig_dir / "r1_vs_dim.png")
    if f"lat_ms_{prec}" in df.columns:
        plot_cost(df, present, prec, fig_dir / "quality_vs_cost.png")
    print(f"\n结果 -> {res_dir / 'summary.csv'}\n图 -> {fig_dir}")


if __name__ == "__main__":
    main()
