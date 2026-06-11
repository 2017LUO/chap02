"""Scene-specific high-level skills used by planners and filters."""

from __future__ import annotations

from .data_structures import ActionBehavior


CRAFTER_STONE_PICKAXE_PLAN = [
    ActionBehavior("collect_wood", {"target_count": 4}, expected_phase="collect_wood", max_steps=350),
    ActionBehavior("place_table", {}, expected_phase="place_table", max_steps=80),
    ActionBehavior("make_wood_pickaxe", {}, expected_phase="make_wood_pickaxe", max_steps=80),
    ActionBehavior("collect_stone", {"target_count": 1}, expected_phase="collect_stone", max_steps=350),
    ActionBehavior("make_stone_pickaxe", {}, expected_phase="make_stone_pickaxe", max_steps=80),
]

HIGHWAY_CRUISE_SKILLS = [
    ActionBehavior("safe_follow", {}, expected_phase="driving", max_steps=1),
    ActionBehavior("keep_lane", {}, expected_phase="driving", max_steps=1),
    ActionBehavior("speed_up", {}, expected_phase="driving", max_steps=1),
    ActionBehavior("slow_down", {}, expected_phase="driving", max_steps=1),
]

UGV_PARKING_SKILLS = [
    ActionBehavior("main_road_driving", {}, expected_phase="main_road", max_steps=120),
    ActionBehavior("obstacle_avoidance", {}, expected_phase="obstacle_zone", max_steps=120),
    ActionBehavior("target_ball_clear", {}, expected_phase="target_ball", max_steps=120),
    ActionBehavior("pass_intersection", {}, expected_phase="intersection", max_steps=120),
    ActionBehavior("enter_street", {}, expected_phase="street", max_steps=120),
    ActionBehavior("search_parking_slot", {}, expected_phase="parking_area", max_steps=120),
    ActionBehavior("parking_control", {}, expected_phase="parking", max_steps=160),
    ActionBehavior("safe_stop", {}, expected_phase=None, max_steps=30),
]

UGV_PATROL_SKILLS = [
    ActionBehavior("outer_route_follow", {}, expected_phase="outer_patrol", max_steps=120),
    ActionBehavior("checkpoint_approach", {}, expected_phase="checkpoint", max_steps=100),
    ActionBehavior("pass_intersection", {}, expected_phase="intersection", max_steps=120),
    ActionBehavior("route_switch", {}, expected_phase="route_switch", max_steps=100),
    ActionBehavior("inner_route_follow", {}, expected_phase="inner_patrol", max_steps=120),
    ActionBehavior("search_parking_slot", {}, expected_phase="parking_area", max_steps=120),
    ActionBehavior("parking_control", {}, expected_phase="parking", max_steps=160),
    ActionBehavior("safe_stop", {}, expected_phase=None, max_steps=30),
]


def skill_ids_for_scene(scene: str) -> list[str]:
    if scene == "crafter":
        return [item.skill_id for item in CRAFTER_STONE_PICKAXE_PLAN]
    if scene == "highway":
        return ["keep_lane", "lane_left", "lane_right", "speed_up", "slow_down", "safe_follow"]
    if scene == "ugv_parking":
        return [item.skill_id for item in UGV_PARKING_SKILLS]
    if scene == "ugv_patrol":
        return [item.skill_id for item in UGV_PATROL_SKILLS]
    return []
