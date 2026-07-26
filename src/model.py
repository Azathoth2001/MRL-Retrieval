"""模型：加载冻结 OpenCLIP，暴露 preprocess/tokenizer/编码/logit_scale。单一职责：模型+其预处理。"""
from __future__ import annotations
import torch
import open_clip


def load_clip(cfg):
    """返回 (model, preprocess, tokenizer)。model 需 eval() + requires_grad_(False)（冻结），搬到 cfg.model.device。"""
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.model.name,
        pretrained=cfg.model.pretrained,
        device=cfg.model.device,
        precision=cfg.model.precision,
    )
    tokenizer = open_clip.get_tokenizer(cfg.model.name)
    model.eval()
    model.requires_grad_(False)
    return model, preprocess, tokenizer


def encode(model, images=None, texts=None):
    """用冻结 model 编码图/文，返回**未归一化**的 512 维 embedding（归一化留到用时现算）。"""
    with torch.no_grad():
        if images is not None:
            img_emb = model.encode_image(images)
            return img_emb
        if texts is not None:
            txt_emb = model.encode_text(texts)
            return txt_emb
        return None
