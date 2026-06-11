"""Task relevance filter for high-level skills."""

from __future__ import annotations

from src.common.skill_registry import skill_ids_for_scene


def filter_relevant_skills(scene: str, task_state: dict, descriptors: dict[str, str]) -> list[str]:
    if scene == "crafter":
        phase = str(task_state.get("phase", "collect_wood"))
        ordered = ["collect_wood", "place_table", "make_wood_pickaxe", "collect_stone", "make_stone_pickaxe"]
        if phase in ordered:
            return ordered[ordered.index(phase) :]
        return ordered
    if scene == "highway":
        safety = task_state.get("safety_state") if isinstance(task_state.get("safety_state"), dict) else {}
        if safety.get("front_vehicle_dist") is not None:
            return ["safe_follow", "keep_lane", "slow_down", "speed_up"]
        return ["safe_follow", "keep_lane", "speed_up", "lane_left", "lane_right"]
    return [skill for skill in skill_ids_for_scene(scene) if skill in descriptors]

