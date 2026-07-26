"""特征：抽取并缓存冻结 CLIP 的 512 维 embedding（.npy）及加载接口。★地基★。单一职责：特征缓存。

约定（方案 §8.3）：存**原始未归一化 fp32**；图/文分开存 + id 对齐表；已存在则跳过。
.npy 为唯一真源。之后 PCA / adapter 训练 / 四组评测全在缓存上跑。
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .data import build_extract_loader, get_coco_splits, get_splits
from .model import encode

MODALITIES = ("image", "text")
COCO_PREFIX = "coco_"


def _resolve_splits(cfg, splits):
    """按 split 名分派数据集：`coco_*` -> MS-COCO，其余 -> Flickr30k。

    这样 `load_features(cfg, "coco_restval", "image")` 与 Flickr30k 的调用方式完全一致，
    上层（evaluate / cascade）不必知道特征来自哪个数据集。
    """
    flickr = [s for s in splits if not s.startswith(COCO_PREFIX)]
    coco = [s[len(COCO_PREFIX):] for s in splits if s.startswith(COCO_PREFIX)]
    out = {}
    if flickr:
        out.update({k: v for k, v in get_splits(cfg).items() if k in flickr})
    if coco:
        out.update({COCO_PREFIX + k: v for k, v in get_coco_splits(cfg, coco).items()})
    return out


def _feature_dir(cfg) -> Path:
    d = Path(cfg.paths.output_root) / "features"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _paths(cfg, split, modality):
    d = _feature_dir(cfg)
    return d / f"{split}_{modality}.npy", d / f"{split}_{modality}_ids.json"


def _image_input_dtype(model):
    """图像 tensor 首先撞上的那层权重的 dtype —— preprocess 产出 fp32，不匹配就 RuntimeError。

    坑：open_clip 的 precision='fp16' 产出的是**混合精度**模型，不是整体 half。实测 3.3.0：
      positional_embedding fp32 / logit_scale fp32 / visual.conv1.weight **fp16**
    所以 next(model.parameters()).dtype（拿到的是 positional_embedding）会误报 fp32。
    这里取 visual 里第一个 Conv/Linear 的权重 dtype —— ViT 是 conv1，timm 骨干是 patch_embed.proj。
    """
    for m in model.visual.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)):
            w = getattr(m, "weight", None)
            if w is not None:
                return w.dtype
    return next(model.visual.parameters()).dtype


@torch.no_grad()
def _encode_loader(loader, ids, model, modality, device, img_dtype, desc):
    """跑一遍 loader，拼出 [N, dim] fp32 未归一化特征。顺序 == ids 顺序（loader shuffle=False）。"""
    out = []
    for batch in tqdm(loader, desc=desc, unit="batch"):
        batch = batch.to(device, non_blocking=True)
        if modality == "image":
            emb = encode(model, images=batch.to(img_dtype))
        else:
            emb = encode(model, texts=batch)          # token 是整型，交给 embedding 层，不能转
        out.append(emb.float().cpu().numpy())
    emb = np.concatenate(out, axis=0).astype(np.float32)
    if len(emb) != len(ids):
        raise RuntimeError(f"{desc}: 特征 {len(emb)} 条 != id {len(ids)} 条，对齐已坏")
    return emb


def extract_and_cache(cfg, splits=("train", "val", "test"), modalities=MODALITIES,
                      overwrite=False):
    """对指定 split 的图与文跑一遍冻结 CLIP，缓存 512 维 fp32 到 outputs/features/。

    `modalities` 可只要 image —— 纯干扰池（如 coco_restval）用不到 caption，
    抽 15 万条文本纯属浪费。
    """
    from .model import load_clip

    todo = [(s, m) for s in splits for m in modalities
            if overwrite or not _paths(cfg, s, m)[0].exists()]
    if not todo:
        print("特征已全部缓存，跳过。删 outputs/features/ 或 overwrite=True 可重抽。")
        return {}

    print(f"待抽取：{[f'{s}/{m}' for s, m in todo]}")
    all_splits = _resolve_splits(cfg, sorted({s for s, _ in todo}))
    model, preprocess, tokenizer = load_clip(cfg)
    device = cfg.model.device
    img_dtype = _image_input_dtype(model)
    print(f"图像输入 dtype = {img_dtype}（precision={cfg.model.precision}）")

    done = {}
    for split, modality in todo:
        records = all_splits[split]
        loader, ids = build_extract_loader(cfg, records, modality,
                                          preprocess=preprocess, tokenizer=tokenizer)
        emb = _encode_loader(loader, ids, model, modality, device, img_dtype,
                             desc=f"{split}/{modality}")
        emb_path, ids_path = _paths(cfg, split, modality)
        np.save(emb_path, emb)
        ids_path.write_text(json.dumps(ids), encoding="utf-8")
        print(f"  {emb_path.name}  {emb.shape}  {emb.nbytes / 1e6:.1f} MB")
        done[(split, modality)] = emb.shape
    return done


def load_features(cfg, split, modality):
    """加载缓存。modality in {'image','text'}。返回 (emb[N,512] float32 未归一化, ids[N])。"""
    if modality not in MODALITIES:
        raise ValueError(f"modality 只能是 image/text，收到 {modality!r}")
    emb_path, ids_path = _paths(cfg, split, modality)
    if not emb_path.exists():
        raise FileNotFoundError(f"缺特征缓存 {emb_path}，先跑：python scripts/02_extract_features.py")
    emb = np.load(emb_path).astype(np.float32)
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    return emb, ids


def load_gallery(cfg, splits=("train", "val", "test")):
    """拼出 ~31k 全量图库（延迟/存储用，方案 §六）。返回 (emb[N,512] fp32, ids[N])。

    ⚠️ 含 29,000 张 adapter 训练过的 train 图 —— 做**召回**的跨组比较时不能用这个图库，
    见方案 §九quater 的混杂因素说明；干净图库用 `load_clean_gallery`。
    """
    embs, ids = [], []
    for s in splits:
        e, i = load_features(cfg, s, "image")
        embs.append(e)
        ids.extend(i)
    return np.concatenate(embs, axis=0), ids


def load_clean_gallery(cfg, kind="same_domain"):
    """**干净**图库：不含任何 adapter 训练过的图。返回 (emb[N,512] fp32, ids[N])。

    kind:
      same_domain  —— test 1,000 + Flickr30k train 的留出集（同域，规模小但分布一致）
      cross_domain —— test 1,000 + COCO restval 30,504（规模大，但存在分布偏移：
                      COCO 偏物体中心、Flickr30k 偏人物动作，异域干扰项可能更容易被排除）
    两种都报才知道结论稳不稳。
    """
    from .data import train_holdout_indices

    te, te_ids = load_features(cfg, "test", "image")
    if kind == "same_domain":
        tr, tr_ids = load_features(cfg, "train", "image")
        _, held = train_holdout_indices(len(tr_ids), cfg.data.holdout_train)
        return (np.concatenate([te, tr[held]], axis=0),
                list(te_ids) + [tr_ids[i] for i in held])
    if kind == "cross_domain":
        co, co_ids = load_features(cfg, "coco_restval", "image")
        return np.concatenate([te, co], axis=0), list(te_ids) + list(co_ids)
    raise ValueError(f"kind 只能是 same_domain/cross_domain，收到 {kind!r}")
