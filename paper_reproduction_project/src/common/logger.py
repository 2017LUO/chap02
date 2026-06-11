"""JSONL and artifact logging utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Union


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Union[str, Path], data: Any) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def append_jsonl(path: Union[str, Path], row: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Union[str, Path], rows: Iterable[dict[str, Any]]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


class JSONLLogger:
    """Append-only JSONL logger."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self.path.write_text("", encoding="utf-8")

    def log(self, row: dict[str, Any]) -> None:
        append_jsonl(self.path, row)

