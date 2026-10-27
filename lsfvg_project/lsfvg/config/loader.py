"""Load + validate the YAML config; save a snapshot with every run."""
from __future__ import annotations

import copy
import dataclasses as dc
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import yaml

from .schema import AppConfig, ConfigError, _build, validate


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def config_from_dict(data: dict[str, Any]) -> AppConfig:
    """Build and validate an :class:`AppConfig` from a plain dict."""
    errors: list[str] = []
    cfg = _build(AppConfig, data or {}, "", errors)
    if not errors:
        errors.extend(validate(cfg))
    if errors:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))
    return cfg


def load_config(path: str | Path, overrides: Optional[dict[str, Any]] = None) -> AppConfig:
    """Load YAML from ``path`` (optionally deep-merged with ``overrides``) and validate it."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if overrides:
        data = _deep_merge(data, overrides)
    return config_from_dict(data)


def config_snapshot(cfg: AppConfig) -> dict[str, Any]:
    return dc.asdict(cfg)


def config_hash(cfg: AppConfig) -> str:
    """Stable short hash of the full config (stored with every run)."""
    blob = json.dumps(config_snapshot(cfg), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def save_snapshot(cfg: AppConfig, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "config_snapshot.yaml"
    with open(p, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config_snapshot(cfg), fh, sort_keys=False)
    return p
