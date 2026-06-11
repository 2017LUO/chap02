"""State machine for a pending high-level plan request."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common.data_structures import ActionBehavior


@dataclass
class RequestState:
    request_id: int
    remaining_steps: int
    plan: list[ActionBehavior]
    elapsed_steps: int = 0
    finished: bool = False
    metadata: dict = field(default_factory=dict)

    def tick(self) -> bool:
        if self.finished:
            return True
        self.elapsed_steps += 1
        self.remaining_steps -= 1
        if self.remaining_steps <= 0:
            self.finished = True
        return self.finished

