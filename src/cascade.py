"""级联检索（funnel）：低维前缀粗筛 top-K → 满维在候选内重排。单一职责：两阶段检索与其测量。

MRL 的性质在这里才真正变现：前面的曲线给的是"质量 ↔ 成本"的**权衡**（省成本要让出质量），
级联把它变成"**用低维的成本拿到接近满维的质量**"—— 只有前缀本身是良好表示时，
粗筛才能把正确答案稳定地留在 top-K 里，所以这是 MRL 的直接下游应用（原论文思路）。

口径说明（重要）：
  · 图库 = 全部 31,014 张图（train+val+test），query = 1,000 图 test 的 5,000 条 caption，方向 t2i。
  · 因此这里的 R@K **与 §九bis 的 1000 图 test 数字不可比**（多了 30,014 个干扰项，必然低得多）。
    本节的意义在于**同一图库内**"级联 vs 满维精确检索"的相对比较，两臂受同样的干扰。
  · 干扰项含 train 图（adapter 见过），这是个温和的混杂因素，但对被比较的两臂完全一致。
"""
from __future__ import annotations
import time

import faiss
import numpy as np


def build_flat_index(emb):
    """归一化向量上的精确内积索引（= 余弦）。

    这里**不动 faiss 线程数** —— 线程数只在计时函数里局部设为 1 再还原：
    延迟要单线程才干净，但召回计算是纯吞吐、该吃满所有核。
    """
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(np.ascontiguousarray(emb, dtype=np.float32))
    return index


def _rerank(gallery_full, cand, q_full):
    """在候选内用满维重排。cand: [B, K] 候选下标；返回 [B, K] 分数。

    `gallery_full[cand]` 这一步的随机访存是真实系统里必然付的代价（要把满维向量取回来），
    所以计时时**必须包含它**，不能只算点积。
    """
    return np.einsum("bkd,bd->bk", gallery_full[cand], q_full)


def cascade_recall(index_lo, gallery_full, q_lo, q_full, gt, K, ks=(1, 5, 10),
                   chunk_bytes=200_000_000):
    """级联的召回 + 粗筛召回上限（stage-1 是否把正确答案留在了 top-K 里）。

    gt: [n_query] 每条 query 的正确图在图库中的下标。
    分块以免 gallery_full[cand] 物化出 [B,K,D] 的巨大中间量。
    返回 (recall_dict, stage1_ceiling)。
    """
    n_q, dim_full = len(q_lo), gallery_full.shape[1]
    per_row = max(1, K * dim_full * 4)
    chunk = max(1, min(n_q, chunk_bytes // per_row))
    kmax = max(ks)

    hits = {k: 0 for k in ks}
    ceiling = 0
    for s in range(0, n_q, chunk):
        e = min(s + chunk, n_q)
        _, cand = index_lo.search(np.ascontiguousarray(q_lo[s:e]), K)   # [b, K]
        gt_b = gt[s:e][:, None]
        ceiling += int((cand == gt_b).any(axis=1).sum())               # 粗筛有没有捞到

        scores = _rerank(gallery_full, cand, q_full[s:e])              # [b, K]
        top = np.argsort(-scores, axis=1)[:, :kmax]
        reranked = np.take_along_axis(cand, top, axis=1)               # [b, kmax]
        for k in ks:
            hits[k] += int((reranked[:, :k] == gt_b).any(axis=1).sum())

    return ({k: 100.0 * v / n_q for k, v in hits.items()},
            100.0 * ceiling / n_q)


def time_cascade(index_lo, gallery_full, q_lo, q_full, K, k=10,
                 n_queries=200, repeats=3, warmup=20, threads=1, seed=0):
    """单条 query 的端到端延迟（ms）：stage-1 faiss 检索 + 取满维向量 + 重排。"""
    prev = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(threads)
    try:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(q_lo), size=min(n_queries, len(q_lo)), replace=False)
        lo = np.ascontiguousarray(q_lo[pick])
        full = np.ascontiguousarray(q_full[pick])

        def one(i):
            _, cand = index_lo.search(lo[i:i + 1], K)
            sc = _rerank(gallery_full, cand, full[i:i + 1])
            np.argsort(-sc, axis=1)[:, :k]

        for i in range(min(warmup, len(lo))):
            one(i)
        best = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            for i in range(len(lo)):
                one(i)
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
    finally:
        faiss.omp_set_num_threads(prev)
    return 1000.0 * best / len(lo)


def time_flat(index, queries, k=10, n_queries=200, repeats=3, warmup=20, threads=1, seed=0):
    """单阶段精确检索的单条 query 延迟（ms）—— 级联的对照。"""
    prev = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(threads)
    try:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(queries), size=min(n_queries, len(queries)), replace=False)
        q = np.ascontiguousarray(queries[pick])
        for i in range(min(warmup, len(q))):
            index.search(q[i:i + 1], k)
        best = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            for i in range(len(q)):
                index.search(q[i:i + 1], k)
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
    finally:
        faiss.omp_set_num_threads(prev)
    return 1000.0 * best / len(q)


def gt_indices(gallery_ids, txt_ids):
    """每条 caption 的正确图在图库中的下标。txt_id 形如 "{image_id}_cap{j}"。"""
    pos = {gid: i for i, gid in enumerate(gallery_ids)}
    out = np.empty(len(txt_ids), dtype=np.int64)
    for i, tid in enumerate(txt_ids):
        base = tid.rsplit("_cap", 1)[0]
        if base not in pos:
            raise KeyError(f"caption {tid!r} 的正确图 {base!r} 不在图库中")
        out[i] = pos[base]
    return out
