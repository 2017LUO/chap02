"""Retrieve similar feedback records."""

from __future__ import annotations

from .feedback_memory import FeedbackMemory
from .feedback_schema import FeedbackRecord


def retrieve_feedback(memory: FeedbackMemory, scene: str, skill_id: str, limit: int = 5) -> list[FeedbackRecord]:
    rows = [row for row in memory.records if row.scene == scene and row.skill_id == skill_id]
    rows.sort(key=lambda row: (not row.violation, -row.risk_score))
    return rows[:limit]

