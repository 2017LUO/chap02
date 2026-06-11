"""Latency-aware compensation policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

from src.chap03_lacm.confidence import ConfidenceCandidateSelector
from src.common.data_structures import ActionBehavior
from src.common.logger import ensure_dir


class CompensationPolicy:
    def __init__(
        self,
        selector: Optional[ConfidenceCandidateSelector] = None,
        risk_threshold: float = 0.55,
        max_hold_steps: int = 2,
    ) -> None:
        self.selector = selector or ConfidenceCandidateSelector()
        self.risk_threshold = risk_threshold
        self.max_hold_steps = max_hold_steps

    def select(
        self,
        scene: str,
        obs: dict[str, Any],
        planned_behavior: ActionBehavior,
        waiting_steps: int,
    ) -> tuple[ActionBehavior, dict[str, Any]]:
        safety = dict(obs.get("safety_state") or {})
        risk = float(safety.get("risk_score", 0.0) or 0.0)
        if waiting_steps <= self.max_hold_steps and risk < self.risk_threshold:
            return planned_behavior, {"mode": "hold_plan", "confidence": 1.0 - risk}
        ranked = self.selector.rank(scene, obs, planned_behavior)
        best, confidence = ranked[0]
        return best, {"mode": "confidence_reselect", "confidence": confidence}

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        ensure_dir(path.parent)
        path.write_text(
            json.dumps(
                {
                    "risk_threshold": self.risk_threshold,
                    "max_hold_steps": self.max_hold_steps,
                    "selector_weights": self.selector.weights,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CompensationPolicy":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        selector = ConfidenceCandidateSelector(weights=dict(data.get("selector_weights") or {}))
        return cls(
            selector=selector,
            risk_threshold=float(data.get("risk_threshold", 0.55)),
            max_hold_steps=int(data.get("max_hold_steps", 2)),
        )

