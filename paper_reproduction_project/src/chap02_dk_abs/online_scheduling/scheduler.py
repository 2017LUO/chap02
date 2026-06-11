"""Online scheduler and lightweight trigger model for DK-ABS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

from src.common.data_structures import ActionBehavior
from src.common.logger import ensure_dir

from .runtime_state_encoder import encode_runtime_state


class TriggerModel:
    """A trainable-by-threshold trigger model.

    The current project does not depend on a neural framework. This model keeps
    the save/load surface expected by the docs and uses interpretable features
    from the official environment wrappers.
    """

    def __init__(self, risk_weight: float = 0.6, time_weight: float = 0.2, stall_weight: float = 0.2) -> None:
        self.risk_weight = risk_weight
        self.time_weight = time_weight
        self.stall_weight = stall_weight

    def predict(self, obs: dict[str, Any], planned_behavior: Optional[ActionBehavior] = None) -> float:
        features = encode_runtime_state(obs)
        stall = _phase_stall(obs, planned_behavior)
        score = (
            self.risk_weight * features["risk_score"]
            + self.time_weight * features["time_ratio"]
            + self.stall_weight * stall
        )
        return max(0.0, min(1.0, score))

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        ensure_dir(path.parent)
        path.write_text(
            json.dumps(
                {
                    "risk_weight": self.risk_weight,
                    "time_weight": self.time_weight,
                    "stall_weight": self.stall_weight,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TriggerModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


class DkAbsScheduler:
    def __init__(self, delta_trig: float = 0.65, trigger_model: Optional[TriggerModel] = None) -> None:
        self.delta_trig = delta_trig
        self.trigger_model = trigger_model or TriggerModel()

    def should_reschedule(self, obs: dict[str, Any], behavior: Optional[ActionBehavior]) -> tuple[bool, float]:
        prob = self.trigger_model.predict(obs, behavior)
        return prob >= self.delta_trig, prob

    def choose_next_behavior(self, plan: list[ActionBehavior], obs: dict[str, Any], cursor: int) -> int:
        phase = str((obs.get("task_state") or {}).get("phase", ""))
        for index in range(cursor, len(plan)):
            expected = plan[index].expected_phase
            if expected is None or expected == phase:
                return index
        return min(cursor, max(len(plan) - 1, 0))


def _phase_stall(obs: dict[str, Any], behavior: Optional[ActionBehavior]) -> float:
    if behavior is None or behavior.expected_phase is None:
        return 0.0
    phase = str((obs.get("task_state") or {}).get("phase", ""))
    return 0.0 if phase == behavior.expected_phase else 1.0

