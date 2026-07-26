"""Step 7（可选消融）：温度 τ × 权重 c_m 的 3×3 网格。

τ ∈ {fixed, learnable, per_dim} × c ∈ {equal, inverse, linear}，共 9 组，同架构同预算同 seed。
跑全网格而非单因子扫描 —— 二者可能有交互（例如 per_dim τ 也许只在 inverse 权重下才有用）。

消融只看质量，**关掉 faiss 测量**：延迟只取决于维度、与降维方法无关（Step 5 已验证四组曲线重合），
每组再测一遍是纯浪费。已有 checkpoint / 结果 CSV 的组自动跳过，可断点续跑。
"""
import itertools
import sys
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.evaluate import evaluate_condition
from src.reducers import AdapterReducer
from src.train import train

TAU_MODES = ["fixed", "learnable", "per_dim"]
WEIGHT_MODES = ["equal", "inverse", "linear"]
TAU_STYLE = {"fixed": ("tab:blue", "-"), "learnable": ("tab:red", "--"), "per_dim": ("tab:green", ":")}


def run_cell(tau, weight, force=False):
    """训 + 评一格。返回 (结果 DataFrame, checkpoint 元信息)。"""
    name = f"abl_{tau}_{weight}"
    cfg = load_config(overrides={
        "loss.temperature.mode": tau,
        "loss.weight.mode": weight,
        "loss.nested": True,
        "experiment.name": name,
    })
    ck = pathlib.Path(cfg.paths.output_root) / "checkpoints" / f"{name}.pt"
    csv = pathlib.Path(cfg.paths.output_root) / "results" / f"{name}.csv"

    if force or not ck.exists():
        train(cfg)
    else:
        print(f"[{name}] checkpoint 已存在，跳过训练")

    meta = {k: v for k, v in torch.load(ck, map_location="cpu", weights_only=True).items()
            if k != "state_dict"}
    if force or not csv.exists():
        df = evaluate_condition(cfg, AdapterReducer(ck, device=cfg.model.device), name,
                                with_bench=False)
    else:
        print(f"[{name}] 结果已存在，跳过评测")
        df = pd.read_csv(csv)
    df = df.assign(tau_mode=tau, weight_mode=weight)
    return df, meta


def main():
    cfg = load_config()
    dims = list(cfg.mrl.dims)
    frames, metas = [], {}

    for tau, weight in itertools.product(TAU_MODES, WEIGHT_MODES):
        print(f"\n{'='*70}\nτ={tau}  c={weight}\n{'='*70}")
        df, meta = run_cell(tau, weight)
        frames.append(df)
        metas[(tau, weight)] = meta

    all_df = pd.concat(frames, ignore_index=True)
    res_dir = pathlib.Path(cfg.paths.output_root) / "results"
    fig_dir = pathlib.Path(cfg.paths.output_root) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(res_dir / "ablation_loss.csv", index=False, encoding="utf-8")

    for col, label in (("t2i_r1", "t2i R@1"), ("i2t_r1", "i2t R@1")):
        piv = all_df.pivot_table(index=["tau_mode", "weight_mode"], columns="dim", values=col)
        piv["mean"] = piv.mean(axis=1)
        print(f"\n=== {label}（%）按维度 ===")
        print(piv.to_string(float_format=lambda v: f"{v:.2f}"))

    print("\n=== 学到的 τ（fixed 组不训、恒为 0.07）===")
    for (tau, weight), meta in metas.items():
        vals = [round(v, 4) for v in meta["temperature"]]
        print(f"  τ={tau:<10} c={weight:<8} -> {vals}   best ep{meta['best_epoch']:>3} "
              f"val={meta['best_val_r1']:.2f}")

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True, sharey="row")
    for j, weight in enumerate(WEIGHT_MODES):
        for i, (col, label) in enumerate((("i2t_r1", "i2t R@1"), ("t2i_r1", "t2i R@1"))):
            ax = axes[i][j]
            for tau in TAU_MODES:
                d = all_df[(all_df.tau_mode == tau) & (all_df.weight_mode == weight)].sort_values("dim")
                color, ls = TAU_STYLE[tau]
                ax.plot(d["dim"], d[col], ls, color=color, marker="o", ms=4, label=f"tau={tau}")
            ax.set_xscale("log", base=2)
            ax.grid(alpha=.3)
            if i == 0:
                ax.set_title(f"weight = {weight}")
            if i == 1:
                ax.set_xlabel("embedding dim")
            if j == 0:
                ax.set_ylabel(f"{label} (%)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Loss ablation — temperature x dimension-weight (Flickr30k 1k test)", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "ablation_loss.png", dpi=150)
    plt.close(fig)
    print(f"\n结果 -> {res_dir / 'ablation_loss.csv'}\n图 -> {fig_dir / 'ablation_loss.png'}")


if __name__ == "__main__":
    main()
