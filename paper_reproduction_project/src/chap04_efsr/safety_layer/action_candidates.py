"""Candidate action corrections for safety filtering."""

from __future__ import annotations

from src.common.data_structures import ActionBehavior


def candidate_corrections(scene: str, raw_behavior: ActionBehavior) -> list[ActionBehavior]:
    if scene == "highway":
        order = ["safe_follow", "slow_down", "keep_lane"]
        if raw_behavior.skill_id in {"lane_left", "lane_right"}:
            order = ["keep_lane", "safe_follow", "slow_down"]
        return [ActionBehavior(skill_id, source="efsr_candidate", max_steps=1) for skill_id in order]
    if scene == "crafter":
        if raw_behavior.skill_id == "collect_stone":
            return [ActionBehavior("collect_wood", source="efsr_candidate", max_steps=80), raw_behavior]
        return [raw_behavior]
    if scene == "ugv_parking":
        if raw_behavior.skill_id in {"pass_intersection", "parking_control", "target_ball_clear"}:
            return [
                ActionBehavior("safe_stop", source="efsr_candidate", max_steps=20),
                ActionBehavior("obstacle_avoidance", source="efsr_candidate", max_steps=60),
                raw_behavior,
            ]
        return [ActionBehavior("safe_stop", source="efsr_candidate", max_steps=20), raw_behavior]
    if scene == "ugv_patrol":
        if raw_behavior.skill_id in {"pass_intersection", "route_switch", "parking_control"}:
            return [
                ActionBehavior("safe_stop", source="efsr_candidate", max_steps=20),
                ActionBehavior("checkpoint_approach", source="efsr_candidate", max_steps=60),
                raw_behavior,
            ]
        return [ActionBehavior("safe_stop", source="efsr_candidate", max_steps=20), raw_behavior]
    return [raw_behavior]
