"""Step 11：真正的跨域评测 —— 在 **COCO 自己的检索任务** 上评四组。

与"跨域级联"（`08_cascade.py --gallery cross_domain`）的区别（容易混）：
  · 跨域级联：任务仍是 Flickr30k 检索，COCO 图只当**干扰项**填大干草堆。
  · 本脚本：query = COCO test 的 caption，正确答案 = COCO test 的图，
    即换了一个**完整的检索任务**，考的是 Flickr30k 上学到的嵌套结构换域后还稳不稳。

PCA 与两个 adapter 都是在 Flickr30k train 上拟合/训练的，**一点 COCO 数据都没见过** —— 这正是要考的。
参照线：冻结 CLIP 在 COCO 5k test 上的官方数字 i2t 59.44 / t2i 42.31（已逐位复现）。
不测延迟（延迟只取决于维度、与用哪种降维和哪个数据集都无关，已在四组曲线重合上验证过）。
"""
import sys
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.evaluate import evaluate_condition
from src.features import load_features
from src.reducers import AdapterReducer, PCAReducer, TruncateReducer

SPLIT = "coco_test"
CONDITIONS = [
    ("baseline",      None,            "① CLIP truncate",   "tab:gray",   "--", "o"),
    ("pca",           None,            "② PCA",             "tab:orange", "-.", "s"),
    ("adapter_plain", "adapter_plain", "③ Adapter (plain)", "tab:green",  ":",  "^"),
    ("adapter_mrl",   "adapter_mrl",   "④ Adapter (MRL)",   "tab:red",    "-",  "D"),
]


def main():
    cfg = load_config()
    ck_dir = pathlib.Path(cfg.paths.output_root) / "checkpoints"
    res_dir = pathlib.Path(cfg.paths.output_root) / "results"
    fig_dir = pathlib.Path(cfg.paths.output_root) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    tr_img, _ = load_features(cfg, "train", "image")
    tr_txt, _ = load_features(cfg, "train", "text")

    frames, present = [], []
    for name, ck, label, color, ls, mk in CONDITIONS:
        if name == "baseline":
            red = TruncateReducer()
        elif name == "pca":
            red = PCAReducer(n_components=cfg.mrl.full_dim).fit(tr_img, tr_txt)
        else:
            p = ck_dir / f"{ck}.pt"
            if not p.exists():
                print(f"  (跳过 {name}：{p.name} 不存在)")
                continue
            red = AdapterReducer(p, device=cfg.model.device)
        df = evaluate_condition(cfg, red, f"coco_{name}", split=SPLIT, with_bench=False)
        frames.append(df.assign(condition=name))
        present.append((name, label, color, ls, mk))

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(res_dir / "cross_domain_coco.csv", index=False, encoding="utf-8")

    for col, lab in (("i2t_r1", "i2t R@1"), ("t2i_r1", "t2i R@1")):
        piv = all_df.pivot_table(index="dim", columns="condition", values=col)
        print(f"\n=== COCO 5k test {lab}（%）===")
        print(piv.to_string(float_format=lambda v: f"{v:.2f}"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, col, title in zip(axes, ("i2t_r1", "t2i_r1"),
                              ("image -> text  R@1", "text -> image  R@1")):
        for name, label, color, ls, mk in present:
            d = all_df[all_df.condition == name].sort_values("dim")
            ax.plot(d["dim"], d[col], ls, color=color, marker=mk, ms=4, label=label)
        ax.set_xscale("log", base=2)
        ax.set(xlabel="embedding dim (prefix length)", ylabel="R@1 (%)", title=title)
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Cross-domain transfer: trained on Flickr30k, evaluated on MS-COCO 5k test",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "cross_domain_coco.png", dpi=150)
    plt.close(fig)
    print(f"\n结果 -> {res_dir / 'cross_domain_coco.csv'}\n图 -> {fig_dir / 'cross_domain_coco.png'}")


if __name__ == "__main__":
    main()
