"""配置读取：configs/*.yaml -> 可点访问对象；支持 CLI 覆盖。单一职责：只管配置。（已实现，可直接用）"""
from __future__ import annotations
import argparse
from pathlib import Path
from types import SimpleNamespace
import yaml


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def load_config(path="configs/default.yaml", overrides=None):
    """读取 yaml -> SimpleNamespace（可 cfg.model.name 点访问）。overrides={'a.b': val}。"""
    with open(Path(path), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for dotted, val in (overrides or {}).items():
        d, keys = cfg, dotted.split(".")
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = val
    return _to_ns(cfg)


def parse_cli():
    """命令行：--config 指定配置；--set a.b=value 覆盖（可多次）。"""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", action="append", default=[], help="覆盖，如 --set train.epochs=5")
    args = p.parse_args()
    ov = {}
    for item in args.set:
        k, v = item.split("=", 1)
        ov[k] = yaml.safe_load(v)   # 自动转类型
    return load_config(args.config, ov)
