"""Step 10：抽 COCO 特征。restval 只抽图（纯干扰池，用不到 caption），test 图文都抽（跨域基准）。"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.features import extract_and_cache

if __name__ == "__main__":
    cfg = load_config()
    extract_and_cache(cfg, splits=("coco_restval",), modalities=("image",))
    extract_and_cache(cfg, splits=("coco_test",), modalities=("image", "text"))
