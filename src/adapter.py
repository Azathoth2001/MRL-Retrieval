"""Adapter：512->1024->512 残差 MLP + LayerNorm（hidden 仅内部宽度，输出=嵌入=512）。单一职责：模块定义。（已实现）"""
from __future__ import annotations
import torch.nn as nn
import torch.nn.functional as F


class MRLAdapter(nn.Module):
    """图文共享。fc1(512->hidden) -> GELU -> fc2(hidden->512) -> +残差 -> LayerNorm(512)。"""

    def __init__(self, dim: int = 512, hidden: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.fc2(F.gelu(self.fc1(x)))
        return self.norm(x + h)          # 残差直连，初始接近恒等


def build_adapter(cfg):
    """按 cfg.adapter 构造 MRLAdapter。"""
    return MRLAdapter(dim=cfg.mrl.full_dim, hidden=cfg.adapter.hidden)
