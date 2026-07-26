"""降维统一接口：把 512 维特征映射到 m 维检索向量。三法同一接口，供 evaluate 统一调用。单一职责：降维。

关键约定：`transform(emb, m)` 返回的向量**已 L2 归一化**，且归一化一律发生在**截到 m 维之后**。
`_project` 的结果按输入数组做单条备忘 —— evaluate 会对同一份 emb 连续问 5 个 m，
不缓存的话 PCA 投影 / adapter 前向要白跑 5 遍。
"""
from __future__ import annotations
from pathlib import Path

import numpy as np


def l2_normalize(x):
    """按行 L2 归一化。零向量兜底成 1，避免 0/0。"""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


class Reducer:
    """基类：fit(train...) 拟合（可空）；transform(emb, m) -> 归一化后的 m 维向量。

    子类只需实现 `_project(emb) -> [N, D]`（全维表示），截断与归一化由基类统一处理。
    """

    name = "reducer"

    _MEMO_CAP = 8

    def __init__(self):
        # id(emb) -> (输入数组的引用, 投影结果)。同时存住引用：既能用 `is` 确认没撞 id，
        # 又能防对象被回收后 id 复用。evaluate 会在 test图/test文/31k图库 三者间轮换，
        # 所以必须是多条备忘而非单条，否则每次都失效。
        self._memo = {}

    def fit(self, train_img, train_txt):
        return self

    def _project(self, emb):
        return emb

    def transform(self, emb, m):
        hit = self._memo.get(id(emb))
        if hit is None or hit[0] is not emb:
            if len(self._memo) >= self._MEMO_CAP:
                self._memo.clear()
            hit = (emb, np.ascontiguousarray(self._project(emb), dtype=np.float32))
            self._memo[id(emb)] = hit
        proj = hit[1]
        if m > proj.shape[1]:
            raise ValueError(f"{self.name}: 要 {m} 维，但表示只有 {proj.shape[1]} 维")
        return l2_normalize(proj[:, :m])


class TruncateReducer(Reducer):
    """① 原生截断：直接取前 m 维再 L2 归一化（无需 fit）。这是"不做任何处理"的对照基线。"""

    name = "truncate"


class PCAReducer(Reducer):
    """② PCA：train 上（图文合一）拟合**一个**投影，apply 到两侧；只在 train 拟合、防泄漏。

    一次拟合到满维，取前 m 个主成分即得 m 维 —— 天然"嵌套"，与 MRL 前缀正好可比：
    差别只在 PCA 按方差排序、MRL 按训练目标把判别信息压到前缀。
    """

    name = "pca"

    def __init__(self, n_components=None):
        super().__init__()
        self.n_components = n_components
        self.pca = None

    def fit(self, train_img, train_txt):
        from sklearn.decomposition import PCA

        X = np.concatenate([np.asarray(train_img, dtype=np.float32),
                            np.asarray(train_txt, dtype=np.float32)], axis=0)
        k = self.n_components or min(X.shape[1], X.shape[0])
        try:    # n_samples >> n_features 时走协方差特征分解最快
            self.pca = PCA(n_components=k, svd_solver="covariance_eigh").fit(X)
        except (TypeError, ValueError):
            self.pca = PCA(n_components=k).fit(X)
        ev = self.pca.explained_variance_ratio_
        print(f"  PCA fit on {X.shape[0]} 条（图{len(train_img)}+文{len(train_txt)}）-> {k} 维；"
              f"累计解释方差 32维={ev[:32].sum():.3f} 128维={ev[:128].sum():.3f} 全量={ev.sum():.3f}")
        self._memo.clear()
        return self

    def _project(self, emb):
        if self.pca is None:
            raise RuntimeError("PCAReducer 未 fit，先在 train 特征上调用 .fit()")
        return self.pca.transform(np.asarray(emb, dtype=np.float32))


class AdapterReducer(Reducer):
    """③④ Adapter：加载训练好的 adapter，前向后取前 m 维再归一化。

    ③ adapter_plain（只在满维算 InfoNCE）与 ④ adapter_mrl（嵌套 InfoNCE）用的是同一个类，
    差别只在 checkpoint —— 保证两组评测路径完全一致。
    """

    def __init__(self, ckpt_path, device="cpu"):
        super().__init__()
        import torch
        from .adapter import MRLAdapter

        self.ckpt_path = Path(ckpt_path)
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"缺 adapter checkpoint {self.ckpt_path}，先跑 scripts/04_train_adapter.py")
        ck = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
        self.name = ck.get("experiment", self.ckpt_path.stem)
        self.device = device
        self.adapter = MRLAdapter(dim=ck["dim"], hidden=ck["hidden"])
        self.adapter.load_state_dict(ck["state_dict"])
        self.adapter.eval().requires_grad_(False).to(device)
        self.meta = {k: v for k, v in ck.items() if k != "state_dict"}

    def _project(self, emb):
        import torch

        with torch.no_grad():
            x = torch.from_numpy(np.asarray(emb, dtype=np.float32)).to(self.device)
            return self.adapter(x).cpu().numpy()
