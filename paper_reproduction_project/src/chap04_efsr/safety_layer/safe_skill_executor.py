"""Execute high-level behaviors through the safety layer."""

from __future__ import annotations

from typing import Any, Optional

from src.chap04_efsr.risk_prediction import RiskValueFunction
from src.common.data_structures import ActionBehavior

from .safety_filter import SafetyFilter


class SafeSkillExecutor:
    def __init__(self, safety_filter: Optional[SafetyFilter] = None) -> None:
        self.safety_filter = safety_filter or SafetyFilter()

    @property
    def risk_value(self) -> RiskValueFunction:
        return self.safety_filter.risk_value

    def execute(
        self,
        env: Any,
        scene: str,
        obs: dict[str, Any],
        raw_behavior: ActionBehavior,
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        chosen, safety_info = self.safety_filter.filter(scene, obs, raw_behavior)
        next_obs, reward, done, info = env.apply_skill(chosen.skill_id, chosen.to_skill_args())
        success = _success(scene, next_obs, done)
        self.risk_value.update(scene, chosen.skill_id, next_obs, reward, success)
        info = {
            **info,
            "raw_skill_id": raw_behavior.skill_id,
            "executed_skill_id": chosen.skill_id,
            "safety_info": safety_info,
        }
        return next_obs, reward, done, info


def _success(scene: str, obs: dict[str, Any], done: bool) -> bool:
    task = dict(obs.get("task_state") or {})
    if bool(task.get("task_success")):
        return True
    if scene == "crafter":
        inventory = dict(obs.get("inventory") or {})
        achievements = dict(obs.get("achievements") or {})
        return bool(inventory.get("stone_pickaxe", 0) or achievements.get("make_stone_pickaxe"))
    if scene == "highway":
        safety = dict(obs.get("safety_state") or {})
        return not bool(safety.get("collision", False)) and bool(done)
    return bool(done)

