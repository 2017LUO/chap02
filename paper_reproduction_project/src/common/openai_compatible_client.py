"""OpenAI-compatible chat client interfaces.

The default client is local and deterministic. It mirrors the shape of
client.chat.completions.create(...) so a real API client can replace it later
without changing planner code.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class OpenAICompatibleMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class OpenAICompatibleChoice:
    index: int
    message: OpenAICompatibleMessage
    finish_reason: str = "stop"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "message": self.message.to_dict(),
            "finish_reason": self.finish_reason,
        }


@dataclass
class OpenAICompatibleUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class OpenAICompatibleChatCompletion:
    id: str
    model: str
    choices: list[OpenAICompatibleChoice]
    usage: OpenAICompatibleUsage
    object: str = "chat.completion"
    created: int = 0

    def __post_init__(self) -> None:
        if not self.created:
            self.created = int(time.time())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [choice.to_dict() for choice in self.choices],
            "usage": self.usage.to_dict(),
        }

    def model_dump(self) -> dict[str, Any]:
        return self.to_dict()


class SimulatedOpenAIClient:
    """Local OpenAI-style client for deterministic planning experiments."""

    def __init__(self, model: str = "gpt-4o-mini-sim", seed: Optional[int] = None) -> None:
        self.model = model
        self.seed = seed
        self.chat = _SimulatedChat(self)


class _SimulatedChat:
    def __init__(self, client: SimulatedOpenAIClient) -> None:
        self.completions = _SimulatedChatCompletions(client)


class _SimulatedChatCompletions:
    def __init__(self, client: SimulatedOpenAIClient) -> None:
        self.client = client

    def create(
        self,
        model: Optional[str] = None,
        messages: Optional[list[dict[str, str]]] = None,
        temperature: float = 0.0,
        response_format: Optional[dict[str, Any]] = None,
        **_: Any,
    ) -> OpenAICompatibleChatCompletion:
        messages = messages or []
        prompt_text = "\n\n".join(str(message.get("content", "")) for message in messages)
        context = _extract_planning_context(prompt_text)
        content = json.dumps(_simulate_plan_response(context), ensure_ascii=False, indent=2)
        prompt_tokens = max(1, len(prompt_text) // 4)
        completion_tokens = max(1, len(content) // 4)
        return OpenAICompatibleChatCompletion(
            id=f"chatcmpl-sim-{abs(hash(prompt_text)) % 10_000_000}",
            model=model or self.client.model,
            choices=[
                OpenAICompatibleChoice(
                    index=0,
                    message=OpenAICompatibleMessage(role="assistant", content=content),
                )
            ],
            usage=OpenAICompatibleUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )


def create_openai_compatible_client(config: Optional[dict[str, Any]] = None) -> Any:
    config = dict(config or {})
    provider = str(config.get("provider", "simulated")).lower()
    model = str(config.get("model", "gpt-4o-mini-sim"))
    if provider in {"simulated", "mock", "local"}:
        return SimulatedOpenAIClient(model=model, seed=config.get("seed"))
    if provider in {"openai", "real_openai"}:
        from openai import OpenAI  # type: ignore

        kwargs = {key: value for key, value in config.items() if key in {"api_key", "base_url", "organization"}}
        return OpenAI(**kwargs)
    raise ValueError(f"Unsupported LLM provider: {provider}")


def _extract_planning_context(prompt_text: str) -> dict[str, Any]:
    match = re.search(
        r"<PLANNING_CONTEXT_JSON>\s*(.*?)\s*</PLANNING_CONTEXT_JSON>",
        prompt_text,
        flags=re.DOTALL,
    )
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


def _simulate_plan_response(context: dict[str, Any]) -> dict[str, Any]:
    scene = str(context.get("scene", "")).lower()
    planner_config = dict(context.get("planner_config") or {})
    if scene == "crafter":
        max_skill_steps = int(planner_config.get("max_skill_steps", 350))
        plan = [
            {
                "skill_id": "collect_wood",
                "args": {"target_count": int(planner_config.get("wood_target_count", 4))},
                "source": "llm_simulated",
                "expected_phase": "collect_wood",
                "max_steps": max_skill_steps,
                "constraints": ["collect enough wood before table placement"],
            },
            {
                "skill_id": "place_table",
                "args": {},
                "source": "llm_simulated",
                "expected_phase": "place_table",
                "max_steps": 80,
                "constraints": ["place table on a free adjacent cell"],
            },
            {
                "skill_id": "make_wood_pickaxe",
                "args": {},
                "source": "llm_simulated",
                "expected_phase": "make_wood_pickaxe",
                "max_steps": 80,
                "constraints": ["craft near the placed table"],
            },
            {
                "skill_id": "collect_stone",
                "args": {"target_count": int(planner_config.get("stone_target_count", 1))},
                "source": "llm_simulated",
                "expected_phase": "collect_stone",
                "max_steps": max_skill_steps,
                "constraints": ["wood pickaxe is required for stone"],
            },
            {
                "skill_id": "make_stone_pickaxe",
                "args": {},
                "source": "llm_simulated",
                "expected_phase": "make_stone_pickaxe",
                "max_steps": 80,
                "constraints": ["craft near the table after collecting stone"],
            },
        ]
        task_state = dict(context.get("task_state") or {})
        plan = _crop_crafter_plan_by_phase(plan, str(task_state.get("phase", "collect_wood")))
    elif scene == "highway":
        plan = _simulate_highway_action_plan(context, planner_config)
    elif scene in {"ugv_parking", "ugv_patrol"}:
        plan = _simulate_ugv_action_plan(context, planner_config)
    else:
        plan = []
    return {
        "plan": plan,
        "rationale": "Simulated OpenAI-compatible planner generated a deterministic JSON plan from the real prompt context.",
    }


def _simulate_highway_action_plan(context: dict[str, Any], planner_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Choose Highway actions from the five DiscreteMetaAction-equivalent skills."""
    task_state = dict(context.get("task_state") or {})
    safety_state = dict(task_state.get("safety_state") or {})
    ego_state = dict(task_state.get("ego_state") or {})
    available = set(context.get("available_skills") or [])
    five_actions = ["lane_left", "keep_lane", "lane_right", "speed_up", "slow_down"]
    candidates = [item for item in five_actions if not available or item in available]
    if not candidates:
        candidates = five_actions

    speed = _float_or(task_state.get("speed", ego_state.get("speed")), 0.0)
    target_speed = _float_or(task_state.get("target_speed"), 30.0)
    lane_id = _int_or(task_state.get("lane_id", ego_state.get("lane_id")), 0)
    lanes_count = _int_or(task_state.get("lanes_count"), 4)
    front_dist = _float_or_none(safety_state.get("front_vehicle_dist"))
    ttc = _float_or_none(safety_state.get("time_to_collision"))
    risk = _float_or(safety_state.get("risk_score"), 0.0)
    left_safe = bool(safety_state.get("left_lane_safe", lane_id > 0))
    right_safe = bool(safety_state.get("right_lane_safe", lane_id < lanes_count - 1))
    relative_speed_front = _float_or_none(safety_state.get("relative_speed_front"))
    nearby_vehicles = list(task_state.get("nearby_vehicles") or [])
    current_front_gap = front_dist if front_dist is not None else _front_gap_for_lane(nearby_vehicles, lane_id)
    left_front_gap = _front_gap_for_lane(nearby_vehicles, lane_id - 1)
    right_front_gap = _front_gap_for_lane(nearby_vehicles, lane_id + 1)

    front_close = front_dist is not None and front_dist < 18.0
    front_closing = relative_speed_front is not None and relative_speed_front < -1.0
    lane_change_gap = _float_or(planner_config.get("lane_change_front_gap"), 36.0)
    lane_change_gain = _float_or(planner_config.get("lane_change_min_gap_gain"), 10.0)
    left_overtake_score = _lane_overtake_score(
        current_front_gap,
        left_front_gap,
        safe=left_safe and lane_id > 0,
        desired_gap=lane_change_gap,
        min_gain=lane_change_gain,
    )
    right_overtake_score = _lane_overtake_score(
        current_front_gap,
        right_front_gap,
        safe=right_safe and lane_id < lanes_count - 1,
        desired_gap=lane_change_gap,
        min_gain=lane_change_gain,
    )
    urgent = bool(safety_state.get("collision", False)) or risk >= 0.75 or (ttc is not None and ttc < 2.0)

    if urgent:
        chosen = "slow_down"
    elif front_close or (risk >= 0.35 and front_closing):
        if left_safe and lane_id > 0:
            chosen = "lane_left"
        elif right_safe and lane_id < lanes_count - 1:
            chosen = "lane_right"
        else:
            chosen = "slow_down"
    elif max(left_overtake_score, right_overtake_score) > 0.0:
        chosen = "lane_left" if left_overtake_score >= right_overtake_score else "lane_right"
    elif speed < target_speed * 0.90 and risk < 0.30 and (front_dist is None or front_dist > 25.0):
        chosen = "speed_up"
    elif speed > target_speed * 1.10 or (front_dist is not None and front_dist < 25.0 and front_closing):
        chosen = "slow_down"
    else:
        chosen = "keep_lane"

    if chosen not in candidates:
        chosen = "keep_lane" if "keep_lane" in candidates else candidates[0]

    ordered = [chosen] + [item for item in candidates if item != chosen]
    horizon = max(1, int(planner_config.get("planning_horizon", planner_config.get("plan_length", 1))))
    plan = []
    for index in range(horizon):
        skill_id = ordered[min(index, len(ordered) - 1)] if index < len(ordered) else chosen
        plan.append(
            {
                "skill_id": skill_id,
                "args": {},
                "source": "llm_simulated",
                "expected_phase": "driving",
                "max_steps": 1,
                "constraints": [
                    "choose only from lane_left, keep_lane, lane_right, speed_up, slow_down",
                    "use surrounding vehicles, front distance, lane safety, speed, and risk score",
                ],
            }
        )
    return plan


def _simulate_ugv_action_plan(context: dict[str, Any], planner_config: dict[str, Any]) -> list[dict[str, Any]]:
    scene = str(context.get("scene", "")).lower()
    task_state = dict(context.get("task_state") or {})
    phase = str(task_state.get("phase", "") or "")
    available = set(context.get("available_skills") or [])
    if scene == "ugv_parking":
        sequence = [
            ("main_road_driving", "main_road"),
            ("obstacle_avoidance", "obstacle_zone"),
            ("target_ball_clear", "target_ball"),
            ("pass_intersection", "intersection"),
            ("enter_street", "street"),
            ("search_parking_slot", "parking_area"),
            ("parking_control", "parking"),
        ]
    else:
        sequence = [
            ("outer_route_follow", "outer_patrol"),
            ("checkpoint_approach", "checkpoint"),
            ("pass_intersection", "intersection"),
            ("route_switch", "route_switch"),
            ("inner_route_follow", "inner_patrol"),
            ("search_parking_slot", "parking_area"),
            ("parking_control", "parking"),
        ]
    if phase == "task_complete":
        return []
    start = 0
    for index, (skill_id, expected_phase) in enumerate(sequence):
        if phase in {skill_id, expected_phase}:
            start = index
            break
    max_steps = int(planner_config.get("max_skill_steps", 120))
    plan = []
    for skill_id, expected_phase in sequence[start:]:
        if available and skill_id not in available:
            continue
        plan.append(
            {
                "skill_id": skill_id,
                "args": {},
                "source": "llm_simulated",
                "expected_phase": expected_phase,
                "max_steps": max_steps if skill_id != "parking_control" else max(max_steps, 160),
                "constraints": [
                    "obey traffic-light and drivable-region constraints",
                    "use safe_stop if a safety constraint blocks progress",
                ],
            }
        )
    horizon = int(planner_config.get("planning_horizon", len(plan)) or len(plan))
    if plan and horizon > len(plan):
        while len(plan) < horizon:
            plan.append(dict(plan[-1]))
    return plan


def _front_gap_for_lane(vehicles: list[Any], lane_id: int) -> Optional[float]:
    gaps: list[float] = []
    for vehicle in vehicles:
        if not isinstance(vehicle, dict):
            continue
        try:
            if int(vehicle.get("lane_id")) != lane_id:
                continue
            distance = float(vehicle.get("distance", vehicle.get("rel_x")))
        except (TypeError, ValueError):
            continue
        if distance > 0.0:
            gaps.append(distance)
    if not gaps:
        return None
    return min(gaps)


def _lane_overtake_score(
    current_gap: Optional[float],
    target_gap: Optional[float],
    *,
    safe: bool,
    desired_gap: float,
    min_gain: float,
) -> float:
    if not safe or current_gap is None:
        return 0.0
    target = target_gap if target_gap is not None else desired_gap * 2.0
    if current_gap >= desired_gap:
        return 0.0
    gain = target - current_gap
    if gain < min_gain or target < desired_gap:
        return 0.0
    return gain


def _float_or(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _crop_crafter_plan_by_phase(plan: list[dict[str, Any]], phase: str) -> list[dict[str, Any]]:
    if phase == "task_complete":
        return []
    phase_to_skill = {
        "collect_wood": "collect_wood",
        "place_table": "place_table",
        "make_wood_pickaxe": "make_wood_pickaxe",
        "collect_stone": "collect_stone",
        "make_stone_pickaxe": "make_stone_pickaxe",
    }
    target_skill = phase_to_skill.get(phase)
    if target_skill is None:
        return plan
    for index, item in enumerate(plan):
        if item.get("skill_id") == target_skill:
            return plan[index:]
    return plan
