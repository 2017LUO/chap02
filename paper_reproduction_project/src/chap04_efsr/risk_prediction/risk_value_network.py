"""Interpretable risk value function for official environments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

from src.chap03_lacm.confidence.candidate_selector import risk_score_for_action
from src.chap04_efsr.execution_feedback import FeedbackMemory
from src.common.logger import ensure_dir


class RiskValueFunction:
    def __init__(self, memory: Optional[FeedbackMemory] = None, feedback_weight: float = 0.35) -> None:
        self.memory = memory or FeedbackMemory()
        self.feedback_weight = feedback_weight

    def predict(self, scene: str, obs: dict[str, Any], skill_id: str) -> float:
        safety = dict(obs.get("safety_state") or {})
        rule_risk = risk_score_for_action(scene, safety, skill_id)
        rows = [row for row in self.memory.records if row.scene == scene and row.skill_id == skill_id]
        if not rows:
            return rule_risk
        feedback_risk = self.memory.average_risk(scene, skill_id)
        violation_boost = self.memory.violation_rate(scene, skill_id)
        risk = (1.0 - self.feedback_weight) * rule_risk + self.feedback_weight * max(feedback_risk, violation_boost)
        return max(0.0, min(1.0, risk))

    def update(self, scene: str, skill_id: str, obs: dict[str, Any], reward: float, success: bool) -> None:
        from src.chap04_efsr.execution_feedback import collect_feedback

        self.memory.add(collect_feedback(scene, skill_id, obs, reward, success))

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        ensure_dir(path.parent)
        path.write_text(
            json.dumps(
                {
                    "feedback_weight": self.feedback_weight,
                    "records": [row.to_dict() for row in self.memory.records],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RiskValueFunction":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        memory = FeedbackMemory()
        from src.chap04_efsr.execution_feedback.feedback_schema import FeedbackRecord

        for row in data.get("records", []):
            memory.add(FeedbackRecord.from_dict(row))
        return cls(memory=memory, feedback_weight=float(data.get("feedback_weight", 0.35)))
