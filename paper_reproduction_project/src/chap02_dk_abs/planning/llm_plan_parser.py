"""Parser for structured LLM behavior plans.

The project currently runs deterministic plans against official environments.
This parser is kept so later LLM output can be plugged in without changing the
experiment runner.
"""

from __future__ import annotations

import json
from typing import Any

from src.common.data_structures import ActionBehavior


def parse_plan_text(text: str) -> list[ActionBehavior]:
    value = json.loads(text)
    if isinstance(value, dict):
        value = value.get("plan") or value.get("actions") or []
    if not isinstance(value, list):
        raise ValueError("Plan text must decode to a list or an object with plan/actions.")
    plan: list[ActionBehavior] = []
    for item in value:
        if isinstance(item, str):
            plan.append(ActionBehavior(item))
        elif isinstance(item, dict):
            plan.append(ActionBehavior.from_dict(item))
        else:
            raise ValueError(f"Unsupported plan item: {item!r}")
    return plan

