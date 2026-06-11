"""Skill descriptors for structured planning prompts and validators."""

from __future__ import annotations

from typing import Any

from src.common.skill_registry import skill_ids_for_scene


def extract_skill_descriptors(scene: str, domain_knowledge: dict[str, Any]) -> dict[str, str]:
    policies = domain_knowledge.get("skill_policies")
    if isinstance(policies, dict) and policies:
        return {str(key): str(value) for key, value in policies.items()}
    return {skill_id: _default_description(scene, skill_id) for skill_id in skill_ids_for_scene(scene)}


def _default_description(scene: str, skill_id: str) -> str:
    if scene == "crafter":
        return {
            "collect_wood": "collect enough wood for table and pickaxe crafting",
            "place_table": "place the crafting table in a free adjacent cell",
            "make_wood_pickaxe": "craft a wood pickaxe near a table",
            "collect_stone": "collect stone using the wood pickaxe",
            "make_stone_pickaxe": "craft the final stone pickaxe near a table",
        }.get(skill_id, skill_id)
    if scene == "highway":
        return {
            "keep_lane": "continue in the current lane",
            "lane_left": "change to the left lane if safe",
            "lane_right": "change to the right lane if safe",
            "speed_up": "increase speed when the front gap is safe",
            "slow_down": "reduce speed when the front gap is short",
            "safe_follow": "choose idle or slow down from the front gap",
        }.get(skill_id, skill_id)
    if scene == "ugv_parking":
        return {
            "main_road_driving": "drive along the main road",
            "obstacle_avoidance": "avoid obstacles while staying in drivable regions",
            "target_ball_clear": "approach and clear the yellow target ball",
            "pass_intersection": "pass the signal-controlled intersection safely",
            "enter_street": "enter the street segment",
            "search_parking_slot": "find and approach the target parking slot",
            "parking_control": "align with and park in the target slot",
            "safe_stop": "stop or wait when constraints are active",
        }.get(skill_id, skill_id)
    if scene == "ugv_patrol":
        return {
            "outer_route_follow": "follow the outer patrol route",
            "checkpoint_approach": "approach the next checkpoint in order",
            "pass_intersection": "pass the signal-controlled intersection safely",
            "route_switch": "switch between outer and inner route",
            "inner_route_follow": "follow the inner patrol route",
            "search_parking_slot": "find and approach the final parking slot",
            "parking_control": "complete final parking",
            "safe_stop": "stop or wait when constraints are active",
        }.get(skill_id, skill_id)
    return skill_id
