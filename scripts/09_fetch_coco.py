"""Step 9：取 MS-COCO（Karpathy split）到 cfg.paths.coco_root。用途有两个：

1. **干净的跨域干扰池** —— restval 的 30,504 张，adapter 从没见过，用来修
   "图库里含训练过的图" 这个混杂因素（见 RESULTS.md §5 的测量注意）。
2. **跨域基准** —— test 的 5,000 张是标准 MS-COCO 检索基准，官方数字 i2t 59.44 / t2i 42.31，
   既是第二个 sanity check，也用来看 Flickr30k 上训的 adapter 换域后是涨是跌。

只需下 **val2014.zip 一个文件**：实测 Karpathy 的 validation+test+restval 全部落在 val2014
（5000+5000+30504 = 40504 = val2014 全量），train 全部落在 train2014。
所以 train2014 天然就是**训练池**（留给 Phase 2 LoRA），与干扰池永不相交。

下载用 8 路并行分段：单连接实测只有 1.88 MB/s（延迟瓶颈），并行能快 3~5 倍。分段可断点续传。
"""
import os
import sys
import pathlib
import zipfile
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.config import load_config

ZIP_URL = "http://images.cocodataset.org/zips/val2014.zip"
ANNOTATION_CSV = "karpathy_coco.csv"
IMAGE_SUBDIR = "images"
N_PARTS = 8
CHUNK = 1 << 20
# Karpathy COCO 官方规模，用于下载后自检
SPLIT_SIZES = {"train": 82783, "validation": 5000, "test": 5000, "restval": 30504}


def save_metadata(root):
    """把 Karpathy COCO 划分落成一份 CSV（filename / split / captions），之后只读磁盘。"""
    import json
    import pandas as pd
    from datasets import load_dataset

    ds = load_dataset("yerevann/coco-karpathy")
    rows = []
    for split in ds:
        if len(ds[split]) != SPLIT_SIZES.get(split):
            raise ValueError(f"split {split}: {len(ds[split])} 行，期望 {SPLIT_SIZES.get(split)}")
        for r in ds[split]:
            rows.append({"filename": r["filename"], "filepath": r["filepath"],
                         "split": split, "cocoid": r["cocoid"],
                         "raw": json.dumps(r["sentences"], ensure_ascii=False)})
    df = pd.DataFrame(rows)
    out = root / ANNOTATION_CSV
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"  标注 -> {out}  ({len(df)} 行)")
    return df


def _total_size(url):
    import requests
    r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
    r.raise_for_status()
    return int(r.headers["Content-Range"].split("/")[-1])


def _fetch_part(args):
    """下一段字节区间到 .part 文件；已完整的段直接跳过（断点续传）。"""
    import requests
    url, start, end, path, i = args
    want = end - start + 1
    if path.exists() and path.stat().st_size == want:
        return i, want, True
    got = path.stat().st_size if path.exists() else 0
    if got >= want:
        path.unlink(missing_ok=True)
        got = 0
    with requests.get(url, headers={"Range": f"bytes={start + got}-{end}"},
                      stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(path, "ab") as f:
            for chunk in r.iter_content(CHUNK):
                f.write(chunk)
    return i, path.stat().st_size, False


def download_parallel(url, dest, staging, n_parts=N_PARTS):
    """8 路并行分段下载再拼接。"""
    if dest.exists():
        print(f"  zip 已存在，跳过下载：{dest} ({dest.stat().st_size / 1e9:.2f} GB)")
        return dest
    total = _total_size(url)
    print(f"  总大小 {total / 1e9:.2f} GB，分 {n_parts} 段并行下载 ...", flush=True)
    step = total // n_parts
    jobs = []
    for i in range(n_parts):
        s = i * step
        e = total - 1 if i == n_parts - 1 else (i + 1) * step - 1
        jobs.append((url, s, e, staging / f"part{i:02d}", i))

    with ThreadPoolExecutor(max_workers=n_parts) as ex:
        for i, size, cached in ex.map(_fetch_part, jobs):
            print(f"    段{i} 完成 {size / 1e6:8.1f} MB{'（已缓存）' if cached else ''}", flush=True)

    print("  拼接 ...", flush=True)
    tmp = dest.with_suffix(".tmp")
    with open(tmp, "wb") as out:
        for i in range(n_parts):
            p = staging / f"part{i:02d}"
            with open(p, "rb") as f:
                while True:
                    b = f.read(CHUNK * 8)
                    if not b:
                        break
                    out.write(b)
    if tmp.stat().st_size != total:
        raise RuntimeError(f"拼接后 {tmp.stat().st_size} != 期望 {total}")
    tmp.replace(dest)
    for i in range(n_parts):
        (staging / f"part{i:02d}").unlink(missing_ok=True)
    return dest


def extract(zip_path, img_dir):
    """解压铺平到 images/。跳过 macOS 资源叉（Flickr30k 那个 zip 踩过）。"""
    import shutil
    img_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = pathlib.Path(info.filename).name
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            if name.startswith("._") or info.filename.startswith("__MACOSX"):
                continue
            target = img_dir / name
            if not (target.exists() and target.stat().st_size == info.file_size):
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=CHUNK)
            n += 1
            if n % 5000 == 0:
                print(f"    {n} 张", flush=True)
    return n


def main():
    cfg = load_config()
    root = pathlib.Path(cfg.paths.coco_root)
    staging, img_dir = root / "_staging", root / IMAGE_SUBDIR
    root.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)

    print("[1/3] Karpathy COCO 标注 ...")
    df = save_metadata(root)

    print("[2/3] val2014.zip（含 test 5k + val 5k + restval 30.5k）...")
    zip_path = download_parallel(ZIP_URL, staging / "val2014.zip", staging)

    print("[3/3] 解压 ...")
    n = extract(zip_path, img_dir)
    print(f"  解出 {n} 张 -> {img_dir}")

    on_disk = {p.name for p in img_dir.iterdir() if p.is_file()}
    print("\n各 split 的图像到位情况：")
    for split in ("test", "validation", "restval", "train"):
        want = df[df.split == split]["filename"]
        have = sum(1 for f in want if f in on_disk)
        note = "（在 train2014，本次不下，留给 Phase 2 LoRA）" if split == "train" else ""
        print(f"  {split:12s} {have:>6}/{len(want):<6} {note}")
    print(f"\n干扰池 = restval（{SPLIT_SIZES['restval']} 张，永不参与训练）")
    print(f"训练池 = train（{SPLIT_SIZES['train']} 张，在 train2014.zip，需要时再下）")
    print(f"可删暂存：{staging}")


if __name__ == "__main__":
    main()
