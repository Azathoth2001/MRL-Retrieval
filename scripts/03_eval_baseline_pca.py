"""Step 3：baseline（原生截断）+ PCA 两组，零训练出曲线。"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.evaluate import evaluate_condition
from src.features import load_features
from src.reducers import PCAReducer, TruncateReducer


def main():
    cfg = load_config()

    evaluate_condition(cfg, TruncateReducer(), "baseline")            # ① 原生截断

    # ② PCA：只在 train 上拟合（图文合一、共用一个投影），绝不碰 test —— 防泄漏
    tr_img, _ = load_features(cfg, "train", "image")
    tr_txt, _ = load_features(cfg, "train", "text")
    pca = PCAReducer(n_components=cfg.mrl.full_dim).fit(tr_img, tr_txt)
    evaluate_condition(cfg, pca, "pca")


if __name__ == "__main__":
    main()
