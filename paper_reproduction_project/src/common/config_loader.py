"""Small YAML/JSON config loader with safe defaults."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union


def load_config(path: Optional[Union[str, Path]], defaults: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    config = dict(defaults or {})
    if path is None:
        return config
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix.lower() == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
    else:
        loaded = _load_yaml(path)
    return deep_update(config, loaded)


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        return dict(value or {})
    except ImportError:
        return _load_simple_yaml(path)


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line or ":" not in line or line.startswith(" "):
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = _coerce_scalar(value.strip())
    return data


def _coerce_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("'\"")

