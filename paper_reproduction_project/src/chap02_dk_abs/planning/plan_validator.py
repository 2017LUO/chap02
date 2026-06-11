"""Validate high-level behavior plans against scene skills."""

from __future__ import annotations

from typing import Optional

from src.common.data_structures import ActionBehavior
from src.common.skill_registry import skill_ids_for_scene


def validate_plan(scene: str, plan: list[ActionBehavior], max_plan_length: Optional[int] = None) -> tuple[bool, list[str]]:
    errors: list[str] = []
    valid_skills = set(skill_ids_for_scene(scene))
    if not plan:
        errors.append("empty_plan")
    if max_plan_length is not None and len(plan) > max_plan_length:
        errors.append("plan_too_long")
    for index, item in enumerate(plan):
        if item.skill_id not in valid_skills:
            errors.append(f"unknown_skill[{index}]={item.skill_id}")
        if item.max_steps is not None and int(item.max_steps) <= 0:
            errors.append(f"non_positive_max_steps[{index}]")
    return not errors, errors

