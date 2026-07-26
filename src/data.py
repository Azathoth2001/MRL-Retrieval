"""数据：Flickr30k(Karpathy split) 本地读取 + 图/文分离 Dataset。单一职责：数据 I/O。

数据来源（方案A，全本地、不走网络）：
  {data_root}/flickr_annotations_30k.csv   Karpathy 标注：raw(5条caption) / split / filename
  {data_root}/images/{filename}            31,014 张 jpg
CSV 与图像均由 scripts/00_fetch_data.py 从 nlphuji/flickr30k 一次性取到本地。

约定：records **只带图像路径、不持有 PIL 对象**（16GB 内存下不可全量解码进内存）；
图、文两条流长度不同（1图↔5caption），故分成两个 Dataset，各自 shuffle=False，
顺序即 dataset.ids 的顺序 —— 特征与 id 靠这个顺序对齐。
"""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

ANNOTATION_CSV = "flickr_annotations_30k.csv"
IMAGE_SUBDIR = "images"
CAPS_PER_IMAGE = 5
# Karpathy split 官方规模，用于加载后自检（对不上说明标注文件被换过）
SPLIT_SIZES = {"train": 29000, "val": 1014, "test": 1000}

# 留出划分**专用**的固定 seed。刻意不用 cfg.train.seed —— 留出集是"数据集的属性"，
# 不是某次训练的属性；若跟着训练 seed 变，换个 seed 重训就会让"这些图从未被训练过"的保证失效。
HOLDOUT_SEED = 12345


def _parse_captions(raw):
    """CSV 的 raw 列是 JSON 数组字符串（caption 内含双引号已转义）。"""
    caps = json.loads(raw)
    if not isinstance(caps, list):
        raise ValueError(f"caption 字段不是列表：{raw[:120]!r}")
    return [str(c) for c in caps]


@lru_cache(maxsize=4)
def _read_annotations(root_str: str):
    """读标注 CSV -> {split: records}。lru_cache 避免被重复调用时反复解析 13MB CSV。"""
    root = Path(root_str)
    csv_path, img_dir = root / ANNOTATION_CSV, root / IMAGE_SUBDIR
    if not csv_path.exists() or not img_dir.is_dir():
        raise FileNotFoundError(
            f"缺少 Flickr30k 本地数据：\n  {csv_path}\n  {img_dir}\n"
            f"先跑：python scripts/00_fetch_data.py"
        )

    df = pd.read_csv(csv_path)
    for col in ("raw", "split", "filename"):
        if col not in df.columns:
            raise ValueError(f"标注 CSV 缺列 {col!r}，实际列：{list(df.columns)}")

    on_disk = {p.name for p in img_dir.iterdir() if p.is_file()}
    missing = [f for f in df["filename"] if f not in on_disk]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} 张图不在 {img_dir}（如 {missing[:3]}）。"
            f"图像解压不全，重跑 scripts/00_fetch_data.py"
        )

    splits: dict[str, list[dict]] = {}
    for row in df.itertuples(index=False):
        caps = _parse_captions(row.raw)
        if len(caps) != CAPS_PER_IMAGE:
            raise ValueError(f"{row.filename} 有 {len(caps)} 条 caption，期望 {CAPS_PER_IMAGE}")
        splits.setdefault(row.split, []).append({
            "image_id": Path(row.filename).stem,     # 如 "1000092795"
            "image_path": str(img_dir / row.filename),
            "captions": caps,
            "split": row.split,
        })

    for name, want in SPLIT_SIZES.items():
        got = len(splits.get(name, []))
        if got != want:
            raise ValueError(f"split {name}：{got} 张，期望 {want}（Karpathy split 对不上）")
    return splits


def get_splits(cfg):
    """返回 {'train'/'val'/'test': records}。record: image_id / image_path / captions(5条) / split。"""
    splits = _read_annotations(str(Path(cfg.paths.data_root)))
    return {k: list(v) for k, v in splits.items()}


COCO_ANNOTATION_CSV = "karpathy_coco.csv"
# Karpathy COCO 官方规模。train 全在 train2014，validation/test/restval 全在 val2014（实测）。
COCO_SPLIT_SIZES = {"train": 82783, "validation": 5000, "test": 5000, "restval": 30504}


@lru_cache(maxsize=2)
def _read_coco_annotations(root_str: str):
    """读 Karpathy COCO 标注 CSV -> {split: records}，格式与 Flickr30k 完全一致。

    只返回**图像已在本地**的 split：本项目默认只下 val2014（= validation+test+restval），
    train2014（训练池，留给 Phase 2 LoRA）不下也不该被当成干扰项。
    """
    root = Path(root_str)
    csv_path, img_dir = root / COCO_ANNOTATION_CSV, root / IMAGE_SUBDIR
    if not csv_path.exists() or not img_dir.is_dir():
        raise FileNotFoundError(
            f"缺少 COCO 本地数据：\n  {csv_path}\n  {img_dir}\n先跑：python scripts/09_fetch_coco.py")

    df = pd.read_csv(csv_path)
    on_disk = {p.name for p in img_dir.iterdir() if p.is_file()}
    splits: dict[str, list[dict]] = {}
    for row in df.itertuples(index=False):
        if row.filename not in on_disk:
            continue
        caps = _parse_captions(row.raw)[:CAPS_PER_IMAGE]   # COCO 有些图 6~7 条，按标准协议截到 5
        if len(caps) < CAPS_PER_IMAGE:
            continue                                        # 极少数不足 5 条的直接排除，保持协议一致
        splits.setdefault(row.split, []).append({
            "image_id": f"coco{row.cocoid}",
            "image_path": str(img_dir / row.filename),
            "captions": caps,
            "split": row.split,
        })
    return splits


def get_coco_splits(cfg, splits=("test", "restval")):
    """返回 COCO 的 {split: records}。缺图的 split 会报错并提示该下什么。"""
    all_splits = _read_coco_annotations(str(Path(cfg.paths.coco_root)))
    out = {}
    for s in splits:
        if s not in all_splits:
            raise FileNotFoundError(
                f"COCO split {s!r} 的图像不在本地（已有：{sorted(all_splits)}）。"
                f"train 在 train2014.zip 里，本项目默认不下。")
        out[s] = list(all_splits[s])
    return out


def train_holdout_indices(n_train, n_holdout):
    """确定性地把 train 切成 (训练用下标, 留出下标)，两者都已排序。

    留出的这批图**不参与任何训练**，用作**同域**干净干扰项 —— 修 §九quater 那个
    "31k 图库里含 adapter 训练过的图，于是 top-1 优势消失" 的混杂因素。
    同一 (n_train, n_holdout) 永远给出同一切分（见 HOLDOUT_SEED）。
    """
    import numpy as np

    if not 0 <= n_holdout < n_train:
        raise ValueError(f"holdout_train={n_holdout} 必须在 [0, {n_train}) 内")
    perm = np.random.default_rng(HOLDOUT_SEED).permutation(n_train)
    return np.sort(perm[n_holdout:]), np.sort(perm[:n_holdout])


class ImageDataset(Dataset):
    """1 item = 1 张图（已 preprocess 成 tensor）。self.ids 与遍历顺序一一对应。"""

    def __init__(self, records, preprocess):
        self.records = records
        self.preprocess = preprocess
        self.ids = [r["image_id"] for r in records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        with Image.open(self.records[idx]["image_path"]) as im:
            return self.preprocess(im.convert("RGB"))


class TextDataset(Dataset):
    """1 item = 1 条 caption（已 tokenize 成 [77]）。5 条/图摊平；id 格式 "{image_id}_cap{j}"。"""

    def __init__(self, records, tokenizer):
        self.tokenizer = tokenizer
        self.texts, self.ids = [], []
        for r in records:
            for j, cap in enumerate(r["captions"]):
                self.texts.append(cap)
                self.ids.append(f"{r['image_id']}_cap{j}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.tokenizer(self.texts[idx])[0]      # [1,77] -> [77]，交给 default collate


class PairDataset(Dataset):
    """1 item = (预处理图, tokenized caption, 记录下标)。LoRA 端到端训练用。

    `cap_choice[i]` 指定第 i 张图本轮用第几条 caption —— **每图每轮只采 1 条**：
    若一个 batch 里出现同图的两条 caption，InfoNCE 会把其中一条当负例（假负例）。
    由调用方每个 epoch 重新掷一次并重建 loader，这样 worker 进程拿到的一定是当轮的选择。
    """

    def __init__(self, records, cap_choice, preprocess, tokenizer):
        if len(cap_choice) != len(records):
            raise ValueError(f"cap_choice {len(cap_choice)} != records {len(records)}")
        self.records, self.cap_choice = records, cap_choice
        self.preprocess, self.tokenizer = preprocess, tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        with Image.open(rec["image_path"]) as im:
            img = self.preprocess(im.convert("RGB"))
        cap = rec["captions"][int(self.cap_choice[idx]) % len(rec["captions"])]
        return img, self.tokenizer(cap)[0], idx


def build_pair_loader(cfg, records, cap_choice, preprocess, tokenizer, batch_size,
                      shuffle=True, num_workers=None):
    """配对 DataLoader（图 + 当轮选中的那条 caption + 下标）。下标用于取蒸馏目标。"""
    ds = PairDataset(records, cap_choice, preprocess, tokenizer)
    nw = cfg.data.num_workers if num_workers is None else num_workers
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=nw,
                      pin_memory=True, drop_last=True)   # drop_last：避免最后一个小 batch 削弱对比信号


def build_extract_loader(cfg, records, modality, preprocess=None, tokenizer=None):
    """抽特征用 DataLoader。modality in {'image','text'}。
    返回 (loader, ids)；**shuffle=False**，loader 产出顺序即 ids 顺序，特征据此对齐。"""
    if modality == "image":
        if preprocess is None:
            raise ValueError("modality='image' 需要 preprocess")
        ds = ImageDataset(records, preprocess)
    elif modality == "text":
        if tokenizer is None:
            raise ValueError("modality='text' 需要 tokenizer")
        ds = TextDataset(records, tokenizer)
    else:
        raise ValueError(f"modality 只能是 image/text，收到 {modality!r}")

    nw = cfg.data.num_workers if modality == "image" else 0   # tokenize 很快，起 worker 反而亏
    loader = DataLoader(
        ds,
        batch_size=cfg.data.batch_size_extract,
        num_workers=nw,
        shuffle=False,
        pin_memory=True,
        persistent_workers=bool(nw),
    )
    return loader, ds.ids
