"""Step 0b：一次性取 Flickr30k 到本地 cfg.paths.data_root（Karpathy 标注 CSV + 31,014 张图）。

取自 nlphuji/flickr30k：flickr_annotations_30k.csv(13MB, 含 split 列) + flickr30k-images.zip(4.4GB)。
直下 local_dir 而非 hub blob，避免 Windows 无符号链接下占双份盘。跑完可删 {data_root}/_staging。
"""
import os
import shutil
import sys
import pathlib
import zipfile

# 必须在 import huggingface_hub 之前设：hf_xet 后端会把整个 4.4GB zip 缓冲进内存
# （实测磁盘文件恒 0 字节、RSS 单调涨），16GB 机器上不可取。关掉走普通 HTTP 流式+断点续传。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.data import ANNOTATION_CSV, IMAGE_SUBDIR

REPO = "nlphuji/flickr30k"


def main():
    from huggingface_hub import hf_hub_download

    cfg = load_config()
    root = pathlib.Path(cfg.paths.data_root)
    staging, img_dir = root / "_staging", root / IMAGE_SUBDIR
    root.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] 标注 CSV ...", flush=True)
    csv_src = hf_hub_download(REPO, ANNOTATION_CSV, repo_type="dataset", local_dir=str(staging))
    shutil.copy2(csv_src, root / ANNOTATION_CSV)
    print("  ->", root / ANNOTATION_CSV, flush=True)

    print("[2/3] 图像 zip 4.4GB（断点续传）...", flush=True)
    zip_path = hf_hub_download(REPO, "flickr30k-images.zip", repo_type="dataset",
                              local_dir=str(staging))
    print(f"  -> {zip_path}  {pathlib.Path(zip_path).stat().st_size / 1e9:.2f} GB", flush=True)

    print("[3/3] 解压铺平到 images/ ...", flush=True)
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = pathlib.Path(info.filename).name
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            # zip 是 macOS 打的，带一套 __MACOSX/._xxx.jpg 资源叉副本（与真图数量 1:1）。
            # 按文件名铺平会把它们也写成"图片"，必须跳过。
            if name.startswith("._") or info.filename.startswith("__MACOSX"):
                continue
            target = img_dir / name
            if not (target.exists() and target.stat().st_size == info.file_size):
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=1 << 20)
            n += 1
            if n % 5000 == 0:
                print(f"  {n} 张", flush=True)
    print(f"完成：{n} 张 -> {img_dir}")
    print(f"可删暂存：{staging}")


if __name__ == "__main__":
    main()
