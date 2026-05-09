from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return cfg


def resolve_path(value: str | Path, base: str | Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if base is None:
        return path
    return Path(base) / path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
