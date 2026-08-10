from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(path.resolve())
    cfg["_config_dir"] = str(path.resolve().parent)
    return cfg


def resolve_path(cfg: Dict[str, Any], value: str | None) -> str | None:
    if not value:
        return value
    p = Path(value)
    if p.is_absolute():
        return str(p)
    if p.exists():
        return str(p.resolve())
    base = Path(cfg.get("_config_dir", ".")).parent
    return str((base / p).resolve())


def env_value(name: str | None, default: str = "") -> str:
    if not name:
        return default
    return os.environ.get(name, default)


def deep_get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur
