"""Region and safety rules used by the chapter-2 planner."""

from __future__ import annotations

from typing import Any


def extract_region_rules(scene: str, domain_knowledge: dict[str, Any]) -> dict[str, Any]:
    rules = dict(domain_knowledge.get("region_rules") or {})
    if rules:
        return rules
    if scene == "crafter":
        return {
            "tree": ["do_collects_wood"],
            "stone": ["wood_pickaxe_required"],
            "table": ["craft_pickaxes_here"],
            "danger": ["avoid_low_health_combat"],
        }
    if scene == "highway":
        return {
            "front_vehicle_close": ["slow_down_or_keep_lane"],
            "left_lane_unsafe": ["forbid_lane_left"],
            "right_lane_unsafe": ["forbid_lane_right"],
        }
    return {}

