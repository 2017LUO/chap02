"""Effective return for compensation training and evaluation."""

from __future__ import annotations

from typing import Optional


def compute_effective_return(
    reward: float,
    wait_steps: int,
    switched: bool,
    risk_score: float,
    wait_seconds: Optional[float] = None,
    wait_cost_per_second: float = 0.02,
) -> float:
    wait_cost = float(wait_seconds) if wait_seconds is not None else float(wait_steps)
    return float(reward) - wait_cost_per_second * wait_cost - (0.03 if switched else 0.0) - 0.25 * float(risk_score)
