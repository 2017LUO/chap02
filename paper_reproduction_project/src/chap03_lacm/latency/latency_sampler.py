"""Latency sampler for high-level decision requests."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LatencySample:
    inference_seconds: float
    steps: int
    control_dt_seconds: float


class LatencySampler:
    def __init__(
        self,
        min_steps: int = 1,
        max_steps: int = 5,
        seed: Optional[int] = None,
        min_seconds: Optional[float] = None,
        max_seconds: Optional[float] = None,
        control_dt_seconds: float = 0.2,
    ) -> None:
        self.min_steps = min_steps
        self.max_steps = max(min_steps, max_steps)
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds if max_seconds is not None else min_seconds
        self.control_dt_seconds = max(float(control_dt_seconds), 1e-6)
        self.rng = random.Random(seed)

    def sample(self) -> int:
        return self.sample_latency().steps

    def sample_latency(self) -> LatencySample:
        if self.min_seconds is None:
            steps = self.rng.randint(self.min_steps, self.max_steps)
            return LatencySample(
                inference_seconds=steps * self.control_dt_seconds,
                steps=steps,
                control_dt_seconds=self.control_dt_seconds,
            )

        min_seconds = float(self.min_seconds)
        max_seconds = max(min_seconds, float(self.max_seconds if self.max_seconds is not None else min_seconds))
        inference_seconds = self.rng.uniform(min_seconds, max_seconds)
        steps = max(1, int(math.ceil(inference_seconds / self.control_dt_seconds)))
        return LatencySample(
            inference_seconds=inference_seconds,
            steps=steps,
            control_dt_seconds=self.control_dt_seconds,
        )

    @classmethod
    def from_config(cls, config: dict, seed: Optional[int] = None) -> "LatencySampler":
        min_seconds = config.get("min_seconds")
        max_seconds = config.get("max_seconds")
        return cls(
            min_steps=int(config.get("min_steps", 1)),
            max_steps=int(config.get("max_steps", 5)),
            seed=seed,
            min_seconds=float(min_seconds) if min_seconds is not None else None,
            max_seconds=float(max_seconds) if max_seconds is not None else None,
            control_dt_seconds=float(config.get("control_dt_seconds", 0.2)),
        )
