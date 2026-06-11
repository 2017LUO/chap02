"""Structured execution feedback records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FeedbackRecord:
    scene: str
    skill_id: str
    risk_score: float
    reward: float
    success: bool
    violation: bool
    state_signature: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "skill_id": self.skill_id,
            "risk_score": self.risk_score,
            "reward": self.reward,
            "success": self.success,
            "violation": self.violation,
            "state_signature": self.state_signature,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "FeedbackRecord":
        return cls(
            scene=str(row["scene"]),
            skill_id=str(row["skill_id"]),
            risk_score=float(row.get("risk_score", 0.0)),
            reward=float(row.get("reward", 0.0)),
            success=bool(row.get("success", False)),
            violation=bool(row.get("violation", False)),
            state_signature=str(row.get("state_signature", "")),
            metadata=dict(row.get("metadata") or {}),
        )

