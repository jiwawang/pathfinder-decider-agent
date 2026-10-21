"""Config loading: one YAML, resolved relative to the project root."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | os.PathLike | None = None) -> dict:
    cfg_path = Path(path) if path else PROJECT_ROOT / "configs" / "default.yaml"
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def resolve(p: str | os.PathLike) -> Path:
    """Resolve a config path relative to the project root unless absolute."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def slide_id_from(case_id: str, slide_name: str) -> str:
    """Stable, filesystem-safe slide id. Source names may contain spaces."""
    safe = slide_name.replace(" ", "-")
    return f"{case_id}_{safe}"
