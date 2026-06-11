"""Encode runtime state into a compact numeric feature dictionary."""

from __future__ import annotations

from typing import Any


def encode_runtime_state(obs: dict[str, Any]) -> dict[str, float]:
    task = dict(obs.get("task_state") or {})
    safety = dict(obs.get("safety_state") or {})
    return {
        "time_ratio": _ratio(task.get("time_step", 0), task.get("max_episode_steps", 1)),
        "risk_score": float(safety.get("risk_score", 0.0) or 0.0),
        "danger": float(safety.get("danger", 0.0) or 0.0),
        "front_vehicle_dist": float(safety.get("front_vehicle_dist", 999.0) or 999.0),
        "speed": float(task.get("speed", obs.get("speed", 0.0)) or 0.0),
        "health_ratio": _ratio(safety.get("health", 1), safety.get("max_health", 1)),
    }


def _ratio(value: object, denom: object) -> float:
    try:
        denom_f = max(float(denom), 1e-6)
        return max(0.0, min(1.0, float(value) / denom_f))
    except (TypeError, ValueError):
        return 0.0

