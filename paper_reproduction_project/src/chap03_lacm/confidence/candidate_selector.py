"""Confidence scores for candidate compensation behaviors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

from src.common.data_structures import ActionBehavior
from src.common.logger import ensure_dir
from src.common.skill_registry import skill_ids_for_scene


class ConfidenceCandidateSelector:
    def __init__(self, weights: Optional[dict[str, float]] = None) -> None:
        self.weights = weights or {"risk": 0.55, "progress": 0.35, "bias": 0.10}

    def rank(
        self,
        scene: str,
        obs: dict[str, Any],
        current_behavior: Optional[ActionBehavior] = None,
    ) -> list[tuple[ActionBehavior, float]]:
        candidates = [ActionBehavior(skill_id, source="lacm_candidate", max_steps=1) for skill_id in skill_ids_for_scene(scene)]
        scored = [(candidate, score_candidate(scene, obs, candidate, current_behavior, self.weights)) for candidate in candidates]
        return sorted(scored, key=lambda item: item[1], reverse=True)

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        ensure_dir(path.parent)
        path.write_text(json.dumps({"weights": self.weights}, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ConfidenceCandidateSelector":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(weights=dict(data.get("weights") or {}))


def score_candidate(
    scene: str,
    obs: dict[str, Any],
    candidate: ActionBehavior,
    current_behavior: Optional[ActionBehavior],
    weights: Optional[dict[str, float]] = None,
) -> float:
    weights = weights or {"risk": 0.55, "progress": 0.35, "bias": 0.10}
    safety = dict(obs.get("safety_state") or {})
    task = dict(obs.get("task_state") or {})
    risk_score = float(safety.get("risk_score", 0.0) or 0.0)
    progress_score = _progress_score(scene, task, safety, candidate)
    bias_score = 1.0 if current_behavior and current_behavior.skill_id == candidate.skill_id else 0.5
    score = (
        weights.get("risk", 0.55) * (1.0 - risk_score_for_action(scene, safety, candidate.skill_id))
        + weights.get("progress", 0.35) * progress_score
        + weights.get("bias", 0.10) * bias_score
    )
    if risk_score > 0.6 and candidate.skill_id in {"speed_up", "lane_left", "lane_right"}:
        score -= 0.35
    return max(0.0, min(1.0, score))


def risk_score_for_action(scene: str, safety: dict[str, Any], skill_id: str) -> float:
    base = float(safety.get("risk_score", 0.0) or 0.0)
    if scene == "highway":
        if skill_id == "speed_up":
            return min(1.0, base + 0.25)
        if skill_id in {"slow_down", "safe_follow"}:
            return max(0.0, base - 0.25)
        if skill_id == "lane_left" and not safety.get("left_lane_safe", True):
            return 1.0
        if skill_id == "lane_right" and not safety.get("right_lane_safe", True):
            return 1.0
    if scene in {"ugv_parking", "ugv_patrol"}:
        light = str(safety.get("traffic_light_state", "green"))
        region = str(safety.get("current_region_type", safety.get("current_region_id", "")))
        if skill_id == "safe_stop":
            return max(0.0, base - 0.35)
        if skill_id == "pass_intersection" and light != "green":
            return 1.0
        if skill_id in {"parking_control", "search_parking_slot"} and region != "parking_area":
            return min(1.0, base + 0.20)
        if skill_id in {"obstacle_avoidance", "safe_stop"}:
            return max(0.0, base - 0.18)
    if scene == "crafter" and skill_id in {"collect_stone", "make_stone_pickaxe"}:
        return base
    return base


def _progress_score(scene: str, task: dict[str, Any], safety: dict[str, Any], candidate: ActionBehavior) -> float:
    if scene == "crafter":
        phase = str(task.get("phase", ""))
        return 1.0 if phase == candidate.skill_id else 0.45
    if scene == "highway":
        speed = float(task.get("speed", 0.0) or 0.0)
        target = float(task.get("target_speed", 30.0) or 30.0)
        front_dist = safety.get("front_vehicle_dist")
        if front_dist is not None and float(front_dist) < 10.0:
            return 1.0 if candidate.skill_id in {"slow_down", "safe_follow"} else 0.2
        if speed < target * 0.8:
            return 1.0 if candidate.skill_id == "speed_up" else 0.6
        return 0.85 if candidate.skill_id in {"safe_follow", "keep_lane"} else 0.55
    if scene in {"ugv_parking", "ugv_patrol"}:
        phase = str(task.get("phase", ""))
        expected_phase = candidate.expected_phase or _ugv_expected_phase(scene, candidate.skill_id)
        if expected_phase and phase == expected_phase:
            return 1.0
        if candidate.skill_id == "safe_stop":
            risk = float(safety.get("risk_score", 0.0) or 0.0)
            return 0.9 if risk > 0.55 else 0.3
        if phase == candidate.skill_id:
            return 1.0
        return 0.45
    return 0.5


def _ugv_expected_phase(scene: str, skill_id: str) -> Optional[str]:
    if scene == "ugv_parking":
        return {
            "main_road_driving": "main_road",
            "obstacle_avoidance": "obstacle_zone",
            "target_ball_clear": "target_ball",
            "pass_intersection": "intersection",
            "enter_street": "street",
            "search_parking_slot": "parking_area",
            "parking_control": "parking",
        }.get(skill_id)
    if scene == "ugv_patrol":
        return {
            "outer_route_follow": "outer_patrol",
            "checkpoint_approach": "checkpoint",
            "pass_intersection": "intersection",
            "route_switch": "route_switch",
            "inner_route_follow": "inner_patrol",
            "search_parking_slot": "parking_area",
            "parking_control": "parking",
        }.get(skill_id)
    return None
