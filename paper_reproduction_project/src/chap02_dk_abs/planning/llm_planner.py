"""OpenAI-compatible LLM planner for DK-ABS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from src.common.data_structures import ActionBehavior
from src.common.openai_compatible_client import create_openai_compatible_client

from .llm_plan_parser import parse_plan_text
from .prompt_builder import build_planning_messages


@dataclass
class LLMPlanningResult:
    plan: list[ActionBehavior]
    messages: list[dict[str, str]]
    raw_response: dict[str, Any]
    response_text: str
    provider: str
    model: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "messages": self.messages,
            "response_text": self.response_text,
            "raw_response": self.raw_response,
            "plan": [item.to_dict() for item in self.plan],
        }


def generate_llm_action_behavior_plan(
    scene: str,
    task_state: dict[str, Any],
    domain_knowledge: dict[str, Any],
    planner_config: Optional[dict[str, Any]] = None,
) -> LLMPlanningResult:
    planner_config = dict(planner_config or {})
    llm_config = dict(planner_config.get("llm") or {})
    llm_config.setdefault("provider", "simulated")
    llm_config.setdefault("model", "gpt-4o-mini-sim")
    messages = build_planning_messages(scene, task_state, domain_knowledge, planner_config)
    client = create_openai_compatible_client(llm_config)
    response = client.chat.completions.create(
        model=llm_config.get("model"),
        messages=messages,
        temperature=float(llm_config.get("temperature", 0.0)),
        response_format={"type": llm_config.get("response_format", "json_object")},
    )
    response_text = response.choices[0].message.content
    plan = parse_plan_text(response_text)
    for item in plan:
        if item.source == "planner":
            item.source = str(llm_config.get("provider", "simulated"))
    return LLMPlanningResult(
        plan=plan,
        messages=messages,
        raw_response=response.model_dump() if hasattr(response, "model_dump") else response.to_dict(),
        response_text=response_text,
        provider=str(llm_config.get("provider", "simulated")),
        model=str(llm_config.get("model", "gpt-4o-mini-sim")),
    )

