"""Aggregate waiting time during LACM execution."""

from __future__ import annotations


class WaitTimeTracker:
    def __init__(self) -> None:
        self.total_wait_steps = 0
        self.total_wait_seconds = 0.0
        self.total_inference_seconds = 0.0
        self.requests = 0

    def begin_request(self, inference_seconds: float = 0.0) -> None:
        self.requests += 1
        self.total_inference_seconds += float(inference_seconds)

    def add_wait_step(self, seconds: float = 1.0) -> None:
        self.total_wait_steps += 1
        self.total_wait_seconds += float(seconds)

    @property
    def average_wait_steps(self) -> float:
        if self.requests <= 0:
            return 0.0
        return self.total_wait_steps / self.requests

    @property
    def average_wait_seconds(self) -> float:
        if self.requests <= 0:
            return 0.0
        return self.total_wait_seconds / self.requests

    @property
    def average_inference_seconds(self) -> float:
        if self.requests <= 0:
            return 0.0
        return self.total_inference_seconds / self.requests
