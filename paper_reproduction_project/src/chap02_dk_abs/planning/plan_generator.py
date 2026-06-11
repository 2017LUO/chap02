"""Domain-knowledge constrained high-level behavior planner."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

from src.common.data_structures import ActionBehavior
from src.common.skill_registry import CRAFTER_STONE_PICKAXE_PLAN, HIGHWAY_CRUISE_SKILLS, UGV_PARKING_SKILLS, UGV_PATROL_SKILLS


def generate_action_behavior_plan(
    scene: str,
    task_state: dict[str, Any],
    domain_knowledge: dict[str, Any],
    config: Optional[dict[str, Any]] = None,
) -> list[ActionBehavior]:
    config = dict(config or {})
    planner_mode = str(config.get("mode", config.get("planner_mode", "llm_simulated"))).lower()
    if planner_mode not in {"rule", "rule_based", "deterministic"}:
        try:
            from .llm_planner import generate_llm_action_behavior_plan

            return generate_llm_action_behavior_plan(scene, task_state, domain_knowledge, config).plan
        except Exception:
            if not bool(config.get("fallback_to_rule", True)):
                raise
            return generate_rule_action_behavior_plan(scene, task_state, domain_knowledge, config)
    return generate_rule_action_behavior_plan(scene, task_state, domain_knowledge, config)


def generate_rule_action_behavior_plan(
    scene: str,
    task_state: dict[str, Any],
    domain_knowledge: dict[str, Any],
    config: Optional[dict[str, Any]] = None,
) -> list[ActionBehavior]:
    config = dict(config or {})
    scene = scene.strip().lower()
    if scene == "crafter":
        return _crafter_plan(config)
    if scene == "highway":
        return _highway_plan(task_state, config)
    if scene == "ugv_parking":
        return _ugv_sequence_plan(task_state, config, UGV_PARKING_SKILLS)
    if scene == "ugv_patrol":
        return _ugv_sequence_plan(task_state, config, UGV_PATROL_SKILLS)
    raise RuntimeError(f"No official planner is implemented for scene: {scene}")


def _crafter_plan(config: dict[str, Any]) -> list[ActionBehavior]:
    wood_target = int(config.get("wood_target_count", 4))
    stone_target = int(config.get("stone_target_count", 1))
    max_skill_steps = int(config.get("max_skill_steps", 350))
    plan = [deepcopy(item) for item in CRAFTER_STONE_PICKAXE_PLAN]
    plan[0].args["target_count"] = wood_target
    plan[0].max_steps = max_skill_steps
    plan[3].args["target_count"] = stone_target
    plan[3].max_steps = max_skill_steps
    for item in plan:
        item.source = "dk_abs_rule_planner"
    return plan


def _highway_plan(task_state: dict[str, Any], config: dict[str, Any]) -> list[ActionBehavior]:
    horizon = int(config.get("planning_horizon", 60))
    target_speed = float(task_state.get("target_speed", config.get("target_speed", 30.0)))
    speed = float(task_state.get("speed", 0.0) or 0.0)
    first_skill = "speed_up" if speed < target_speed * 0.85 else "safe_follow"
    template = [deepcopy(item) for item in HIGHWAY_CRUISE_SKILLS]
    first = next(item for item in template if item.skill_id == first_skill)
    follow = next(item for item in template if item.skill_id == "safe_follow")
    keep = next(item for item in template if item.skill_id == "keep_lane")
    plan: list[ActionBehavior] = []
    for index in range(horizon):
        item = deepcopy(first if index == 0 else follow if index % 3 else keep)
        item.source = "dk_abs_rule_planner"
        item.max_steps = 1
        plan.append(item)
    return plan


def _ugv_sequence_plan(task_state: dict[str, Any], config: dict[str, Any], template: list[ActionBehavior]) -> list[ActionBehavior]:
    phase = str(task_state.get("phase", "") or "")
    if phase == "task_complete":
        return []
    sequence = [deepcopy(item) for item in template if item.skill_id != "safe_stop"]
    start_index = 0
    for index, item in enumerate(sequence):
        if phase and (phase == item.expected_phase or phase == item.skill_id):
            start_index = index
            break
    remaining = sequence[start_index:]
    for item in remaining:
        item.source = "dk_abs_rule_planner"
        if item.max_steps is None:
            item.max_steps = int(config.get("max_skill_steps", 120))
    horizon = config.get("planning_horizon")
    if horizon is not None and int(horizon) > len(remaining) and remaining:
        padded = list(remaining)
        filler = deepcopy(remaining[-1])
        filler.source = "dk_abs_rule_planner"
        while len(padded) < int(horizon):
            padded.append(deepcopy(filler))
        return padded
    return remaining
