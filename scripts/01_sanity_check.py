"""Step 1：sanity check —— 冻结 CLIP 零样本跑检索基准，核对官方 R@K。
对不上先修 preprocess/protocol，别往下走。

    python scripts/01_sanity_check.py                  # Flickr30k 1000图 test
    python scripts/01_sanity_check.py --dataset coco   # MS-COCO 5000图 test（需先跑 09）

顺带把该 split 的特征缓存下来（走 02 的同一条批处理路径）。
"""
import argparse
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.features import extract_and_cache, load_features
from src.metrics import recall_at_k

# openclip 官方 retrieval results（ViT-B-16 / laion2b_s34b_b88k）
# docs/openclip_retrieval_results.csv 里 "image retr."=t2i，"text retr."=i2t（反直觉，别搞反）
BENCHMARKS = {
    "flickr30k": {
        "split": "test",
        "desc": "Flickr30k 1000图 test",
        "targets": {"i2t": {1: 86.30, 5: 97.90, 10: 99.40},
                    "t2i": {1: 69.84, 5: 90.38, 10: 94.56}},
    },
    "coco": {
        "split": "coco_test",
        "desc": "MS-COCO 5000图 test",
        "targets": {"i2t": {1: 59.44, 5: 81.78, 10: 88.62},
                    "t2i": {1: 42.31, 5: 67.70, 10: 77.06}},
    },
}
TOL = 2.0   # 百分点。fp16 + caption 取法等实现细节差异的容忍带

FAIL_HINTS = [
    "preprocess 是否用 open_clip 随模型返回的那个 transform",
    "split 是否为 Karpathy test（1000 图 / 5000 caption）",
    "归一化顺序：取前缀 [:m] 之后才归一化",
    "i2t 是否按「top-K 命中任一 caption」算",
]


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="flickr30k", choices=list(BENCHMARKS))
    args = ap.parse_args()
    bench = BENCHMARKS[args.dataset]
    split, targets = bench["split"], bench["targets"]

    cfg = load_config()
    print(f"[Step 1] sanity check  {cfg.model.name} / {cfg.model.pretrained}  "
          f"precision={cfg.model.precision}")
    print(f"         基准：{bench['desc']}")

    extract_and_cache(cfg, splits=(split,))
    img_emb, img_ids = load_features(cfg, split, "image")
    txt_emb, txt_ids = load_features(cfg, split, "text")
    print(f"{split}: {len(img_ids)} 图 / {len(txt_ids)} caption  dim={img_emb.shape[1]}")

    res = recall_at_k(l2(img_emb), l2(txt_emb), img_ids, txt_ids, ks=(1, 5, 10))

    print(f"\n{'方向':<6}{'K':>4}{'实测':>9}{'官方':>9}{'差':>8}")
    worst = 0.0
    for d in ("i2t", "t2i"):
        for k in (1, 5, 10):
            got, want = res[d][k], targets[d][k]
            diff = got - want
            worst = max(worst, abs(diff))
            print(f"{d:<6}{k:>4}{got:>8.2f}%{want:>8.2f}%{diff:>+8.2f}")

    print()
    if worst <= TOL:
        print(f"[PASS] 最大偏差 {worst:.2f} 个百分点（阈值 {TOL}）。地基可用，可跑 Step 2。")
    else:
        print(f"[FAIL] 最大偏差 {worst:.2f} 个百分点（阈值 {TOL}）。先查：")
        for h in FAIL_HINTS:
            print(f"  - {h}")
        sys.exit(1)


if __name__ == "__main__":
    main()
