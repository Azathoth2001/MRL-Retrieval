"""工程指标：faiss 索引 + 延迟/存储/索引大小。单一职责：检索工程测量。

延迟按**单条 query**测（这才是线上感知到的检索延迟），线程数固定为 1：
31k 库上多线程会把维度差异淹没在调度噪声里，单线程的维度-延迟关系才干净可读。
fp16/int8 用 faiss 的 ScalarQuantizer 真实量化存储，不是"转一圈再转回来"的假降精度。
"""
from __future__ import annotations
import time

import faiss
import numpy as np

BYTES_PER_DIM = {"fp32": 4.0, "fp16": 2.0, "int8": 1.0}
_QUANT = {"fp16": "QT_fp16", "int8": "QT_8bit"}


def _build_index(emb, precision, index_type, nlist=None):
    """按 精度×索引类型 建 faiss 索引并 add 完数据。返回 (index, 说明字符串)。"""
    n, d = emb.shape
    metric = faiss.METRIC_INNER_PRODUCT              # 向量已归一化，内积=余弦
    if precision not in BYTES_PER_DIM:
        raise ValueError(f"precision 只能是 {list(BYTES_PER_DIM)}，收到 {precision!r}")

    if index_type == "flat":
        if precision == "fp32":
            index, desc = faiss.IndexFlatIP(d), "FlatIP"
        else:
            qt = getattr(faiss.ScalarQuantizer, _QUANT[precision])
            index, desc = faiss.IndexScalarQuantizer(d, qt, metric), f"SQ({precision})"
    elif index_type == "ivf":
        nlist = nlist or max(1, int(4 * np.sqrt(n)))
        coarse = faiss.IndexFlatIP(d)
        if precision == "fp32":
            index, desc = faiss.IndexIVFFlat(coarse, d, nlist, metric), f"IVF{nlist},Flat"
        else:
            qt = getattr(faiss.ScalarQuantizer, _QUANT[precision])
            index = faiss.IndexIVFScalarQuantizer(coarse, d, nlist, qt, metric)
            desc = f"IVF{nlist},SQ({precision})"
        index.nprobe = max(1, nlist // 16)
    else:
        raise ValueError(f"index 只能是 flat/ivf，收到 {index_type!r}")

    if not index.is_trained:
        index.train(emb)
    index.add(emb)
    return index, desc


def bench(gallery_emb, query_emb, m, precision="fp32", index="flat",
          k=10, n_queries=200, repeats=3, threads=1, seed=0):
    """在 gallery 上建索引测：单条 query 延迟(ms)/QPS、单向量字节数、索引大小(MB)。

    gallery_emb / query_emb 需已降到 m 维并 L2 归一化（reducer.transform 的输出）。
    返回 dict，字段见末尾。
    """
    gallery = np.ascontiguousarray(gallery_emb, dtype=np.float32)
    queries = np.ascontiguousarray(query_emb, dtype=np.float32)
    if gallery.shape[1] != m or queries.shape[1] != m:
        raise ValueError(f"期望 {m} 维，实际 gallery={gallery.shape[1]} query={queries.shape[1]}")

    prev_threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(threads)
    try:
        index_obj, desc = _build_index(gallery, precision, index)
        index_bytes = int(faiss.serialize_index(index_obj).nbytes)

        rng = np.random.default_rng(seed)
        pick = rng.choice(len(queries), size=min(n_queries, len(queries)), replace=False)
        probe = np.ascontiguousarray(queries[pick])

        for i in range(min(20, len(probe))):          # 预热：让 BLAS/量化查表进缓存
            index_obj.search(probe[i:i + 1], k)

        best = None                                   # 取多轮最小值，抑制系统抖动
        for _ in range(repeats):
            t0 = time.perf_counter()
            for i in range(len(probe)):
                index_obj.search(probe[i:i + 1], k)
            elapsed = time.perf_counter() - t0
            best = elapsed if best is None else min(best, elapsed)
        lat_ms = 1000.0 * best / len(probe)

        t0 = time.perf_counter()                      # 批量吞吐（另一种口径，供参考）
        index_obj.search(probe, k)
        batch_qps = len(probe) / max(time.perf_counter() - t0, 1e-9)
    finally:
        faiss.omp_set_num_threads(prev_threads)

    return {
        "dim": m,
        "precision": precision,
        "index": desc,
        "n_gallery": len(gallery),
        "lat_ms": lat_ms,                                    # 单条 query 延迟
        "qps": 1000.0 / lat_ms,
        "batch_qps": batch_qps,
        "bytes_per_vec": m * BYTES_PER_DIM[precision],       # 理论单向量字节
        "index_mb": index_bytes / 1e6,                       # 实测序列化后索引大小
        "index_bytes_per_vec": index_bytes / len(gallery),
    }
