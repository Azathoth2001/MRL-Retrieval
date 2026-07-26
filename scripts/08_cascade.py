"""Step 8：级联检索（funnel）—— 低维粗筛 top-K + 满维重排，量端到端延迟与召回。

核心设计：**四组都跑**。漏斗只在前缀本身是良好表示时才立得住 —— 粗筛漏掉的正确答案，
后面重排再准也救不回来。所以"级联能省多少"直接由前缀质量决定，这是 MRL 的下游变现。

延迟只取决于 (粗筛维度, K)、与用哪种降维无关，故每个组合只计时一次并复用，
避开跨运行约 20% 的测量噪声。

图库三选一（`--gallery`）：
  full31k      —— train+val+test 共 31,014 张。**含 adapter 训练过的图**，跨组绝对 R@1 不可信，
                  只能看每组自己的保持率（这是最初那版）。
  same_domain  —— test 1,000 + Flickr30k train 留出的 6,000 张。干净、同域，但图库偏小。
                  **必须配 `_ho` adapter**（训练时排除了留出集），脚本自动挑，配错就没意义了。
  cross_domain —— test 1,000 + COCO restval 30,504 张。干净、图库大，但存在域偏移。
                  COCO 图现有 adapter 从没见过，所以用常规 checkpoint。
"""
import argparse
import sys
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.cascade import (build_flat_index, cascade_recall, gt_indices,
                         time_cascade, time_flat)
from src.config import load_config
from src.features import load_clean_gallery, load_features, load_gallery
from src.reducers import AdapterReducer, PCAReducer, TruncateReducer

SHORTLIST_DIMS = [32, 64, 128]
K_GRID = [10, 50, 100, 500, 1000]
TIMING_PASSES = 3          # 扫几轮网格取逐格最小值
# 同域干净图库的干扰项是"留出的 train 图"，只有用 *_ho checkpoint（训练时排除了它们）才干净
CKPT_SUFFIX = {"full31k": "", "same_domain": "_ho", "cross_domain": ""}
STYLE = {
    "baseline":      ("① CLIP truncate",   "tab:gray",   "--", "o"),
    "pca":           ("② PCA",             "tab:orange", "-.", "s"),
    "adapter_plain": ("③ Adapter (plain)", "tab:green",  ":",  "^"),
    "adapter_mrl":   ("④ Adapter (MRL)",   "tab:red",    "-",  "D"),
}


def get_gallery(cfg, kind):
    if kind == "full31k":
        return load_gallery(cfg)
    return load_clean_gallery(cfg, kind)


def build_reducers(cfg, suffix=""):
    """四组 reducer。缺 checkpoint 的 adapter 组自动跳过。

    PCA 一律只在 train 上拟合。注意同域干净实验下 PCA 仍看过全部 train 图（含留出集），
    所以 ② 对留出干扰项并非严格"没见过" —— 但 PCA 是无监督、零训练的线性投影，
    不存在对比学习那种"把见过的图磨得更自信"的机制，影响远小于 ③④，此处不另做处理。
    """
    out = {"baseline": TruncateReducer()}

    tr_img, _ = load_features(cfg, "train", "image")
    tr_txt, _ = load_features(cfg, "train", "text")
    out["pca"] = PCAReducer(n_components=cfg.mrl.full_dim).fit(tr_img, tr_txt)

    ck_dir = pathlib.Path(cfg.paths.output_root) / "checkpoints"
    for name in ("adapter_plain", "adapter_mrl"):
        p = ck_dir / f"{name}{suffix}.pt"
        if p.exists():
            out[name] = AdapterReducer(p, device=cfg.model.device)
        else:
            print(f"  (跳过 {name}：{p.name} 不存在"
                  + ("，同域干净实验需先训 *_ho：--set data.use_holdout=true)" if suffix else ")"))
    return out


def pareto(d):
    """(延迟↓, R@1↑) 上的帕雷托前沿。三个 m_s 分支在延迟轴上互相交错，
    把所有点按延迟连成一条线会画出来回折的假曲线 —— 只有前沿才是真正可达的操作包络。"""
    d = d.sort_values("lat_ms")
    keep, best = [], -1.0
    for _, r in d.iterrows():
        if r["r1"] > best:
            keep.append(r)
            best = r["r1"]
    return pd.DataFrame(keep)


def make_figure(df, out, tag):
    n_gal = int(df["n_gallery"].dropna().iloc[0]) if "n_gallery" in df else 0
    tag = f"{tag}, {n_gal} gallery" if n_gal else tag
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for name, (label, color, ls, mk) in STYLE.items():
        d = df[(df.condition == name) & (df["mode"] == "cascade") & (df.shortlist_dim == 64)]
        if d.empty:
            continue
        ax.plot(d["K"], d["ceiling"], ls, color=color, marker=mk, ms=5, label=label)
    ax.set(xscale="log", xlabel="shortlist size K", ylabel="stage-1 recall@K (%)",
           title=f"Funnel ceiling: is the answer in the shortlist?\n"
                 f"(shortlist dim = 64 — {tag})")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    for name, (label, color, ls, mk) in STYLE.items():
        d = df[(df.condition == name) & (df["mode"] == "cascade")]
        if d.empty:
            continue
        ax.scatter(d["lat_ms"], d["r1"], s=12, color=color, alpha=.25, marker=mk)
        f = pareto(d)
        ax.plot(f["lat_ms"], f["r1"], ls, color=color, marker=mk, ms=5, label=label)
        e = df[(df.condition == name) & (df["mode"] == "exact")]
        ax.plot(e["lat_ms"], e["r1"], marker="*", ms=16, color=color, ls="none",
                markeredgecolor="k", markeredgewidth=.5)
    ax.set(xscale="log", xlabel="end-to-end latency (ms/query, log)", ylabel="t2i R@1 (%)",
           title=f"Cascade Pareto frontier (dots = all K x shortlist-dim)\n"
                 f"star = exact full-dim search — {tag}")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", default="full31k", choices=list(CKPT_SUFFIX))
    ap.add_argument("--plot-only", action="store_true",
                    help="只重画图（召回是确定性的，计时很贵）")
    args = ap.parse_args()

    cfg = load_config()
    full = cfg.mrl.full_dim
    res_dir = pathlib.Path(cfg.paths.output_root) / "results"
    fig_dir = pathlib.Path(cfg.paths.output_root) / "figures"
    csv_path = res_dir / f"cascade_{args.gallery}.csv"
    png_path = fig_dir / f"cascade_{args.gallery}.png"

    if args.plot_only:
        make_figure(pd.read_csv(csv_path), png_path, args.gallery)
        print(f"图 -> {png_path}")
        return

    gallery, gallery_ids = get_gallery(cfg, args.gallery)
    txt, txt_ids = load_features(cfg, "test", "text")
    gt = gt_indices(gallery_ids, txt_ids)
    suffix = CKPT_SUFFIX[args.gallery]
    print(f"[级联/{args.gallery}] 图库 {len(gallery_ids)} 图，query {len(txt_ids)} 条 test caption，方向 t2i")
    print(f"       粗筛维度 {SHORTLIST_DIMS} × K {K_GRID}；重排用满维 {full}"
          + (f"；adapter checkpoint 后缀 '{suffix}'" if suffix else ""))
    if args.gallery == "full31k":
        print("       ⚠️ 图库含 adapter 训练过的图 —— 跨组绝对 R@1 不可信，只看各组自己的保持率")
    print()

    reducers = build_reducers(cfg, suffix)
    rows = []

    # ---- 参照：满维精确检索（延迟与组无关，只测一次）----
    lat_exact = None
    for name, red in reducers.items():
        g_full, q_full = red.transform(gallery, full), red.transform(txt, full)
        idx_full = build_flat_index(g_full)
        if lat_exact is None:
            lat_exact = min(time_flat(idx_full, q_full, n_queries=300, repeats=2, warmup=50)
                            for _ in range(TIMING_PASSES))
        # 满维精确检索：粗筛与重排都是满维，取 K=10 即等价于直接 top-10（重排只是同分重排序）
        rec, _ = cascade_recall(idx_full, g_full, q_full, q_full, gt, K=10)
        rows.append({"condition": name, "mode": "exact", "shortlist_dim": full, "K": 0,
                     "lat_ms": lat_exact, "ceiling": float("nan"),   # 参照组无"粗筛上限"可言
                     **{f"r{k}": v for k, v in rec.items()}})
        print(f"  [{name}] 满维精确  R@1={rec[1]:6.2f}  R@10={rec[10]:6.2f}  lat={lat_exact:.3f}ms")

    # ---- 延迟：只取决于 (m_s, K)，与用哪种降维无关，所以每格测一次给各组复用 ----
    # 多轮扫过整个网格、逐格取最小值：单次调用内重复取 min 躲不开持续性的系统干扰
    # （实测过一次，某个 m_s 整组被抬高到 2 倍，跨 m_s 的缩放关系直接乱掉）。
    print("\n[计时] 多轮扫网格取逐格最小值 ...")
    ref = reducers["adapter_mrl"] if "adapter_mrl" in reducers else next(iter(reducers.values()))
    g_full_ref, q_full_ref = ref.transform(gallery, full), ref.transform(txt, full)
    idx_lo_ref = {m: build_flat_index(ref.transform(gallery, m)) for m in SHORTLIST_DIMS}
    q_lo_ref = {m: ref.transform(txt, m) for m in SHORTLIST_DIMS}

    lat_cache = {}
    for p in range(TIMING_PASSES):
        for m_s in SHORTLIST_DIMS:
            for K in K_GRID:
                t = time_cascade(idx_lo_ref[m_s], g_full_ref, q_lo_ref[m_s], q_full_ref, K,
                                 n_queries=300, repeats=2, warmup=50)
                key = (m_s, K)
                lat_cache[key] = t if key not in lat_cache else min(lat_cache[key], t)
        print(f"  pass {p+1}/{TIMING_PASSES} 完成")

    # ---- 召回：确定性，各组分别算 ----
    print()
    for m_s in SHORTLIST_DIMS:
        for name, red in reducers.items():
            g_lo, q_lo = red.transform(gallery, m_s), red.transform(txt, m_s)
            g_full, q_full = red.transform(gallery, full), red.transform(txt, full)
            idx_lo = build_flat_index(g_lo)
            for K in K_GRID:
                rec, ceiling = cascade_recall(idx_lo, g_full, q_lo, q_full, gt, K)
                rows.append({"condition": name, "mode": "cascade", "shortlist_dim": m_s, "K": K,
                             "lat_ms": lat_cache[(m_s, K)], "ceiling": ceiling,
                             **{f"r{k}": v for k, v in rec.items()}})
        for K in K_GRID:
            lat = lat_cache[(m_s, K)]
            best = max(r["r1"] for r in rows if r["mode"] == "cascade"
                       and r["shortlist_dim"] == m_s and r["K"] == K)
            print(f"  m_s={m_s:>3} K={K:>4}  lat={lat:6.3f}ms ({lat_exact/lat:5.2f}× 加速)  "
                  f"四组最好 R@1={best:.2f}")

    df = pd.DataFrame(rows).assign(gallery=args.gallery, n_gallery=len(gallery_ids))
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df["r1_ret"] = df.apply(
        lambda r: 100.0 * r["r1"] / df[(df.condition == r.condition)
                                       & (df["mode"] == "exact")]["r1"].iloc[0], axis=1)
    df["speedup"] = lat_exact / df["lat_ms"]
    df.to_csv(csv_path, index=False, encoding="utf-8")

    print("\n=== 粗筛召回上限（正确答案落在 top-K 内的比例，%）—— 漏斗的天花板 ===")
    ceil_piv = df[df["mode"] == "cascade"].pivot_table(
        index=["shortlist_dim", "K"], columns="condition", values="ceiling")
    print(ceil_piv.to_string(float_format=lambda v: f"{v:.2f}"))

    print("\n=== 级联 R@1（%）===")
    r1_piv = df[df["mode"] == "cascade"].pivot_table(
        index=["shortlist_dim", "K"], columns="condition", values="r1")
    print(r1_piv.to_string(float_format=lambda v: f"{v:.2f}"))

    print("\n=== 相对该组满维精确检索的保持率（%）===")
    ret_piv = df[df["mode"] == "cascade"].pivot_table(
        index=["shortlist_dim", "K"], columns="condition", values="r1_ret")
    print(ret_piv.to_string(float_format=lambda v: f"{v:.2f}"))

    make_figure(df, png_path, args.gallery)
    print(f"\n结果 -> {csv_path}\n图 -> {png_path}")


if __name__ == "__main__":
    main()
