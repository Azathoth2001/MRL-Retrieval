"""质量指标：R@1/5/10 双向（i2t + t2i）。单一职责：召回计算。

协议（与文献一致，1000 图 test）：
  t2i —— 每条 caption 有 1 张正确图，命中即算。
  i2t —— 每张图有 5 条正确 caption，top-K 里命中任一即算。
分块 argpartition，不一次性物化整个相似度矩阵。
"""
from __future__ import annotations
import numpy as np

_NORM_TOL = 1e-2


def _check_normalized(emb, name):
    """输入必须已 L2 归一化（点积才等于余弦）。顺手挡住"先归一化整段再切前缀"那个坑。"""
    norms = np.linalg.norm(emb[:: max(1, len(emb) // 512)], axis=1)
    if np.abs(norms - 1.0).max() > _NORM_TOL:
        raise ValueError(
            f"{name} 未 L2 归一化（norm 范围 {norms.min():.4f}~{norms.max():.4f}）。"
            f"注意：取前缀 [:m] **之后**才归一化。"
        )


def _owner_index(img_ids, txt_ids):
    """txt_id("{image_id}_cap{j}") -> 所属图在 img_ids 中的下标；返回 [n_txt] int64。"""
    pos = {iid: k for k, iid in enumerate(img_ids)}
    owner = np.empty(len(txt_ids), dtype=np.int64)
    for k, tid in enumerate(txt_ids):
        base = tid.rsplit("_cap", 1)[0]
        if base not in pos:
            raise KeyError(f"txt_id {tid!r} 的所属图 {base!r} 不在 img_ids 中，对齐已坏")
        owner[k] = pos[base]
    return owner


def _topk_desc(query, gallery, k, chunk=2048):
    """分块取 query@gallery.T 每行 top-k 的下标（按分数降序）。返回 [n_query, k] int64。"""
    k = min(k, gallery.shape[0])
    out = np.empty((query.shape[0], k), dtype=np.int64)
    for s in range(0, query.shape[0], chunk):
        sim = query[s:s + chunk] @ gallery.T
        part = np.argpartition(-sim, k - 1, axis=1)[:, :k]      # 先粗筛 top-k
        rows = np.arange(part.shape[0])[:, None]
        out[s:s + part.shape[0]] = part[rows, np.argsort(-sim[rows, part], axis=1)]
    return out


def recall_at_k(img_emb, txt_emb, img_ids, txt_ids, ks=(1, 5, 10)):
    """双向 R@K。img_emb[n_img,m] / txt_emb[n_txt,m] 需**已降到 m 维并 L2 归一化**。
    返回 {'i2t': {1:..,5:..,10:..}, 't2i': {...}}，单位 %。"""
    if len(img_emb) != len(img_ids) or len(txt_emb) != len(txt_ids):
        raise ValueError("emb 与 ids 长度不一致，对齐已坏")
    _check_normalized(img_emb, "img_emb")
    _check_normalized(txt_emb, "txt_emb")

    owner = _owner_index(img_ids, txt_ids)
    kmax = max(ks)

    # i2t：每图取 top-kmax caption，命中 = 该 caption 属于本图
    top_txt = _topk_desc(img_emb, txt_emb, kmax)
    hit_i2t = owner[top_txt] == np.arange(len(img_ids))[:, None]

    # t2i：每 caption 取 top-kmax 图，命中 = 正好是它所属那张
    top_img = _topk_desc(txt_emb, img_emb, kmax)
    hit_t2i = top_img == owner[:, None]

    return {
        "i2t": {k: float(100.0 * hit_i2t[:, :k].any(axis=1).mean()) for k in ks},
        "t2i": {k: float(100.0 * hit_t2i[:, :k].any(axis=1).mean()) for k in ks},
    }
