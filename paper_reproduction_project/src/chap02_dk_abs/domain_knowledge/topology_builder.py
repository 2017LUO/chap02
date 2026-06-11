"""Build a compact topology view from the unified environment interface."""

from __future__ import annotations

from typing import Any


def build_topology(scene: str, domain_knowledge: dict[str, Any]) -> dict[str, Any]:
    topology = dict(domain_knowledge.get("environment_topology") or {})
    if topology:
        return topology
    if scene == "crafter":
        return {
            "regions": ["local_view", "front_cell", "inventory"],
            "transitions": {
                "local_view": ["front_cell"],
                "front_cell": ["inventory"],
            },
        }
    if scene == "highway":
        return {
            "lanes": [0, 1, 2, 3],
            "adjacent_lanes": {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]},
        }
    return {}

