"""Persistent feedback memory for risk-aware action correction."""

from __future__ import annotations

from typing import Optional, Union

import json
from pathlib import Path

from src.common.logger import ensure_dir

from .feedback_schema import FeedbackRecord


class FeedbackMemory:
    def __init__(self, records: Optional[list[FeedbackRecord]] = None) -> None:
        self.records = list(records or [])

    def add(self, record: FeedbackRecord) -> None:
        self.records.append(record)

    def violation_rate(self, scene: str, skill_id: str) -> float:
        rows = [row for row in self.records if row.scene == scene and row.skill_id == skill_id]
        if not rows:
            return 0.0
        return sum(1 for row in rows if row.violation) / len(rows)

    def average_risk(self, scene: str, skill_id: str) -> float:
        rows = [row for row in self.records if row.scene == scene and row.skill_id == skill_id]
        if not rows:
            return 0.0
        return sum(row.risk_score for row in rows) / len(rows)

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        ensure_dir(path.parent)
        path.write_text(json.dumps([row.to_dict() for row in self.records], ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FeedbackMemory":
        path = Path(path)
        if not path.exists():
            return cls()
        rows = json.loads(path.read_text(encoding="utf-8"))
        return cls([FeedbackRecord.from_dict(row) for row in rows])

