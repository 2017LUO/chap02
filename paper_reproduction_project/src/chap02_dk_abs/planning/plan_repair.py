"""Plan repair rules for runtime mismatch or invalid planner output."""

from __future__ import annotations

from typing import Any, Optional

from src.common.data_structures import ActionBehavior

from .plan_generator import generate_action_behavior_plan
from .plan_validator import validate_plan


def repair_plan(
    scene: str,
    plan: list[ActionBehavior],
    task_state: dict[str, Any],
    domain_knowledge: dict[str, Any],
    config: Optional[dict[str, Any]] = None,
) -> list[ActionBehavior]:
    ok, _ = validate_plan(scene, plan, max_plan_length=(config or {}).get("max_plan_length"))
    if ok:
        return plan
    return generate_action_behavior_plan(scene, task_state, domain_knowledge, config)

