"""Common dataclasses used by the three experiment chapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ActionBehavior:
    """A high-level behavior that can be executed through env.apply_skill."""

    skill_id: str
    args: dict[str, Any] = field(default_factory=dict)
    source: str = "planner"
    expected_phase: Optional[str] = None
    max_steps: Optional[int] = None
    constraints: list[str] = field(default_factory=list)

    def to_skill_args(self) -> dict[str, Any]:
        skill_args = dict(self.args)
        if self.max_steps is not None:
            skill_args.setdefault("max_steps", self.max_steps)
        return skill_args

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "args": dict(self.args),
            "source": self.source,
            "expected_phase": self.expected_phase,
            "max_steps": self.max_steps,
            "constraints": list(self.constraints),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ActionBehavior":
        return cls(
            skill_id=str(value["skill_id"]),
            args=dict(value.get("args") or {}),
            source=str(value.get("source", "planner")),
            expected_phase=value.get("expected_phase"),
            max_steps=value.get("max_steps"),
            constraints=list(value.get("constraints") or []),
        )


@dataclass
class ExecutionRecord:
    """One logged high-level execution event."""

    episode: int
    step_index: int
    scene: str
    method: str
    skill_id: str
    reward: float
    done: bool
    success: bool
    task_state: dict[str, Any]
    safety_state: dict[str, Any]
    event_info: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "step_index": self.step_index,
            "scene": self.scene,
            "method": self.method,
            "skill_id": self.skill_id,
            "reward": self.reward,
            "done": self.done,
            "success": self.success,
            "task_state": self.task_state,
            "safety_state": self.safety_state,
            "event_info": self.event_info,
            **self.extra,
        }


@dataclass
class EpisodeResult:
    """Summary of one experiment episode."""

    scene: str
    method: str
    episode: int
    seed: int
    total_reward: float
    steps: int
    success: bool
    done_reason: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "method": self.method,
            "episode": self.episode,
            "seed": self.seed,
            "total_reward": self.total_reward,
            "steps": self.steps,
            "success": self.success,
            "done_reason": self.done_reason,
            **self.metrics,
        }

