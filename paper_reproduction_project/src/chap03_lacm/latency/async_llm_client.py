"""Local async-plan facade used by LACM experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from src.chap02_dk_abs.planning import generate_action_behavior_plan
from src.common.data_structures import ActionBehavior

from .latency_sampler import LatencySampler
from .request_state import RequestState


@dataclass
class PendingPlanRequest:
    state: RequestState

    def tick(self) -> bool:
        return self.state.tick()

    @property
    def plan(self) -> list[ActionBehavior]:
        return self.state.plan


class AsyncPlanClient:
    """Produces deterministic plans after a sampled number of environment steps."""

    def __init__(self, latency_sampler: LatencySampler) -> None:
        self.latency_sampler = latency_sampler
        self._next_id = 0

    def request_plan(
        self,
        scene: str,
        task_state: dict[str, Any],
        domain_knowledge: dict[str, Any],
        planner_config: Optional[dict[str, Any]] = None,
    ) -> PendingPlanRequest:
        self._next_id += 1
        plan = generate_action_behavior_plan(scene, task_state, domain_knowledge, planner_config)
        latency = self.latency_sampler.sample_latency()
        return PendingPlanRequest(
            RequestState(
                request_id=self._next_id,
                remaining_steps=latency.steps,
                plan=plan,
                metadata={
                    "inference_seconds": latency.inference_seconds,
                    "latency_steps": latency.steps,
                    "control_dt_seconds": latency.control_dt_seconds,
                },
            )
        )
