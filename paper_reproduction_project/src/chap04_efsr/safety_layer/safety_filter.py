"""Risk-based behavior correction layer."""

from __future__ import annotations

from typing import Any, Optional

from src.chap04_efsr.risk_prediction import RiskValueFunction
from src.common.data_structures import ActionBehavior

from .action_candidates import candidate_corrections


class SafetyFilter:
    def __init__(self, risk_value: Optional[RiskValueFunction] = None, risk_threshold: float = 0.55) -> None:
        self.risk_value = risk_value or RiskValueFunction()
        self.risk_threshold = risk_threshold

    def filter(
        self,
        scene: str,
        obs: dict[str, Any],
        raw_behavior: ActionBehavior,
    ) -> tuple[ActionBehavior, dict[str, Any]]:
        raw_risk = self.risk_value.predict(scene, obs, raw_behavior.skill_id)
        if raw_risk <= self.risk_threshold:
            return raw_behavior, {"corrected": False, "raw_risk": raw_risk, "chosen_risk": raw_risk}
        candidates = candidate_corrections(scene, raw_behavior)
        scored = [(candidate, self.risk_value.predict(scene, obs, candidate.skill_id)) for candidate in candidates]
        chosen, chosen_risk = min(scored, key=lambda item: (_bias_penalty(raw_behavior, item[0]) + item[1], item[1]))
        return chosen, {
            "corrected": chosen.skill_id != raw_behavior.skill_id,
            "raw_risk": raw_risk,
            "chosen_risk": chosen_risk,
            "candidate_risks": {candidate.skill_id: risk for candidate, risk in scored},
        }


def _bias_penalty(raw_behavior: ActionBehavior, candidate: ActionBehavior) -> float:
    return 0.0 if raw_behavior.skill_id == candidate.skill_id else 0.08

