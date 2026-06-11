"""Create feedback records from environment transitions."""

from __future__ import annotations

from typing import Any

from .feedback_schema import FeedbackRecord


def collect_feedback(scene: str, skill_id: str, obs: dict[str, Any], reward: float, success: bool) -> FeedbackRecord:
    safety = dict(obs.get("safety_state") or {})
    task = dict(obs.get("task_state") or {})
    risk = float(safety.get("risk_score", safety.get("danger", 0.0)) or 0.0)
    violation = bool(safety.get("collision", False) or safety.get("constraint_violation", False) or risk >= 0.85)
    return FeedbackRecord(
        scene=scene,
        skill_id=skill_id,
        risk_score=risk,
        reward=float(reward),
        success=bool(success),
        violation=violation,
        state_signature=state_signature(scene, obs),
        metadata={"phase": task.get("phase"), "done_reason": obs.get("done_reason")},
    )


def state_signature(scene: str, obs: dict[str, Any]) -> str:
    task = dict(obs.get("task_state") or {})
    safety = dict(obs.get("safety_state") or {})
    if scene == "highway":
        return "lane={lane}|front={front}|risk={risk}".format(
            lane=task.get("lane_id"),
            front=round(float(safety.get("front_vehicle_dist", 999.0) or 999.0), 1),
            risk=round(float(safety.get("risk_score", 0.0) or 0.0), 2),
        )
    if scene == "crafter":
        return "phase={phase}|danger={danger}|health={health}".format(
            phase=task.get("phase"),
            danger=round(float(safety.get("danger", 0.0) or 0.0), 2),
            health=safety.get("health"),
        )
    return str(task.get("phase", "unknown"))

