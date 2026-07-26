"""Step 2：抽取并缓存冻结 CLIP 的 512 维特征（★地基★）。"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.features import extract_and_cache

if __name__ == "__main__":
    extract_and_cache(load_config())
