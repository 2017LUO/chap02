"""Prompt construction for DK-ABS high-level behavior planning."""

from __future__ import annotations

import json
from typing import Any, Optional


def build_planning_messages(
    scene: str,
    task_state: dict[str, Any],
    domain_knowledge: dict[str, Any],
    planner_config: Optional[dict[str, Any]] = None,
) -> list[dict[str, str]]:
    context = build_planning_context(scene, task_state, domain_knowledge, planner_config)
    system_prompt = (
        "你是一个用于具身智能实验复现的高层动作行为规划器。"
        "你必须严格使用给定的技能集合、领域知识和运行状态，输出可执行的 JSON 计划。"
        "不要输出 Markdown，不要解释额外文本。"
    )
    user_prompt = (
        "请为当前任务生成动作行为序列。要求：\n"
        "1. 只使用 available_skills 中存在的 skill_id。\n"
        "2. 每个计划项必须包含 skill_id、args、source、expected_phase、max_steps、constraints。\n"
        "3. 输出 JSON object，顶层字段为 plan 和 rationale。\n"
        "4. plan 必须能被 env.apply_skill(skill_id, args) 顺序执行。\n\n"
        "<PLANNING_CONTEXT_JSON>\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2)}\n"
        "</PLANNING_CONTEXT_JSON>"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_planning_context(
    scene: str,
    task_state: dict[str, Any],
    domain_knowledge: dict[str, Any],
    planner_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    skill_policies = dict(domain_knowledge.get("skill_policies") or {})
    return {
        "scene": scene,
        "task_state": _jsonable(task_state),
        "domain_knowledge": _jsonable(domain_knowledge),
        "available_skills": sorted(skill_policies),
        "skill_descriptions": _jsonable(skill_policies),
        "planner_config": _jsonable(dict(planner_config or {})),
        "output_schema": {
            "plan": [
                {
                    "skill_id": "string",
                    "args": "object",
                    "source": "llm or llm_simulated",
                    "expected_phase": "string or null",
                    "max_steps": "positive integer",
                    "constraints": ["string"],
                }
            ],
            "rationale": "short string",
        },
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return str(value)
    return value

