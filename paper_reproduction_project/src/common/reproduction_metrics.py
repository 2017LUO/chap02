"""Reference-style episode metrics for reproduction training logs."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


COMMON_TRAINING_COLUMNS = [
    "episode",
    "score",
    "reward",
    "success",
    "stages_completed",
    "stage_completion_ratio",
    "steps",
    "equivalent_env_steps",
    "completion_time_steps",
    "nominal_completion_time_seconds",
    "latency_time_seconds",
    "completion_time_seconds",
    "normalized_completion_time",
    "route_efficiency",
    "progress_per_step",
    "collisions",
    "violations",
    "unsafe_episode",
    "collision_rate",
    "violation_rate",
    "unsafe_event_rate",
    "target_hits",
    "planner_accuracy",
    "planner_miss_ratio",
    "planner_legal_ratio",
    "planner_exact_hits",
    "planner_legal_hits",
    "planner_total",
    "avg_stage_wait_time",
    "avg_stage_latency_seconds",
    "latency_penalty",
    "idle_window_ratio",
    "comp_decision_ratio",
    "idle_comp_ratio",
    "wait_ratio",
    "follow_ratio",
    "select_ratio",
    "compensation_utilization",
    "safety_saved",
    "safety_interventions",
    "safety_success_rate",
    "critical_events",
    "critical_safe_count",
    "risk_exposure",
    "observed_risk_mean",
    "high_risk_step_ratio",
    "hazard_step_ratio",
    "red_light_exposure_ratio",
    "critical_event_rate",
    "intervention_rate",
    "safety_precision",
    "safety_recall",
    "stop_ratio",
    "steer_left_ratio",
    "steer_right_ratio",
    "behavior_entropy",
    "safety_entropy",
    "active_stage_key",
    "last_failure_mode",
    "epsilon",
    "episode_loss",
]

CHAP04_EXTRA_COLUMNS = [
    "risk_loss",
    "efsr_high_record_count",
    "efsr_high_record_maturity",
    "efsr_high_record_unsafe_rate",
    "efsr_low_replay_size",
    "efsr_low_risk_loss",
]

UGV_STAGE_TOTAL = 7
UGV_EQUIVALENT_ENV_STEP_SCALE = 320
DEFAULT_EPSILON = {"dkabs": 1.0, "lacm": 0.98, "efsr": 0.64}


def columns_for_chapter(chapter: str) -> List[str]:
    columns = list(COMMON_TRAINING_COLUMNS)
    if chapter == "chap04":
        columns.extend(CHAP04_EXTRA_COLUMNS)
    return columns


def load_event_rows(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_reference_episode_rows(
    episode_results: Iterable[Dict[str, Any]],
    event_log_path: Optional[Path],
    job: Dict[str, Any],
    method_key: str,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    event_groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for event in load_event_rows(event_log_path):
        event_groups[int(event.get("episode", 0) or 0)].append(event)

    result_rows = [dict(row) for row in episode_results]
    if not result_rows and event_groups:
        result_rows = [_result_from_events(episode, rows, job, method_key) for episode, rows in sorted(event_groups.items())]

    return [
        _build_episode_row(row, event_groups.get(int(row.get("episode", 0) or 0), []), job, method_key, config)
        for row in result_rows
    ]


def _build_episode_row(
    result: Dict[str, Any],
    events: List[Dict[str, Any]],
    job: Dict[str, Any],
    method_key: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    episode = int(result.get("episode", 0) or 0)
    success = 1 if _truthy(result.get("success", False)) else 0
    reward = _float(result.get("total_reward", result.get("reward", _sum(events, "reward"))))
    steps = int(_float(result.get("steps", len(events) or 0)))
    max_steps = int(config.get("max_episode_steps", job.get("max_episode_steps") or 80) or 80)
    event_count = max(1, len(events), steps)
    final_task = _last_dict(events, "task_state")
    final_safety = _last_dict(events, "safety_state")

    stage_total = _stage_total(job, events)
    stages_completed = _stages_completed(events, final_task, success, stage_total)
    stage_ratio = _safe_ratio(stages_completed, stage_total)
    completion_steps = steps if success else max_steps
    equivalent_steps = steps * UGV_EQUIVALENT_ENV_STEP_SCALE if _is_ugv(job) else steps
    nominal_seconds = _nominal_completion_seconds(completion_steps)

    waiting_steps = [_float(event.get("waiting_steps")) for event in events if event.get("waiting_steps") is not None]
    waiting_seconds = [_float(event.get("waiting_seconds")) for event in events if event.get("waiting_seconds") is not None]
    latency_time = _latency_time(events, waiting_seconds)
    completion_seconds = nominal_seconds + latency_time

    collisions = _count_events(events, _has_collision)
    violations = _count_events(events, _has_violation)
    unsafe_events = _count_events(events, _is_unsafe_event)
    unsafe_episode = 1 if collisions or violations or (not success and unsafe_events) else 0
    target_hits = _target_hits(events)

    planner_total = max(1, len(events))
    planner_exact_hits = _planner_exact_hits(events)
    planner_legal_hits = _planner_legal_hits(events)
    planner_accuracy = _safe_ratio(planner_exact_hits, planner_total)
    planner_legal_ratio = _safe_ratio(planner_legal_hits, planner_total)

    decision_modes = [str(event.get("decision_mode", "") or "") for event in events]
    comp_decisions = sum(1 for mode in decision_modes if mode and mode not in {"hold_plan", "rule", "rule_based"})
    follow_count = sum(1 for mode in decision_modes if mode in {"hold_plan", "follow_plan", "follow"})
    select_count = sum(1 for mode in decision_modes if mode in {"confidence_reselect", "select", "reselect"})
    wait_count = sum(1 for value in waiting_steps if value > 0)
    corrected_count = _count_events(events, lambda event: _truthy(event.get("corrected")) or _truthy(_dict(event.get("event_info")).get("safety_corrected")))
    changed_count = _count_events(events, _changed_from_plan)

    risk_values = [_risk_value(event) for event in events]
    high_risk_count = sum(1 for value in risk_values if value >= 0.45)
    hazard_count = _count_events(events, _is_hazard_event)
    red_count = _count_events(events, _is_red_light_event)
    critical_count = _count_events(events, _is_critical_event)
    critical_safe_count = _count_events(events, lambda event: _is_critical_event(event) and not _is_unsafe_event(event))
    safety_saved = _count_events(events, lambda event: _is_intervention(event) and not _is_unsafe_event(event))
    safety_interventions = _count_events(events, _is_intervention)

    skill_ids = [str(event.get("skill_id", "") or "") for event in events if event.get("skill_id")]
    safety_actions = [
        str(_dict(event.get("safety_state")).get("recommended_safe_action", "") or "")
        for event in events
        if _dict(event.get("safety_state")).get("recommended_safe_action")
    ]

    active_stage = _active_stage_key(final_task, result, success)
    last_failure = "" if success else str(result.get("done_reason") or _dict(_last(events).get("event_info")).get("done_reason") or "running")

    row: Dict[str, Any] = {
        "episode": episode,
        "score": _round(reward),
        "reward": _round(reward),
        "success": success,
        "stages_completed": _round(stages_completed),
        "stage_completion_ratio": _round(stage_ratio),
        "steps": steps,
        "equivalent_env_steps": int(equivalent_steps),
        "completion_time_steps": int(completion_steps),
        "nominal_completion_time_seconds": _round(nominal_seconds),
        "latency_time_seconds": _round(latency_time),
        "completion_time_seconds": _round(completion_seconds),
        "normalized_completion_time": _round(_safe_ratio(completion_steps, max_steps)),
        "route_efficiency": _round(_safe_ratio(max_steps, max(1, completion_steps))),
        "progress_per_step": _round(_safe_ratio(stages_completed, max(1, steps))),
        "collisions": collisions,
        "violations": violations,
        "unsafe_episode": unsafe_episode,
        "collision_rate": _round(_safe_ratio(collisions, event_count)),
        "violation_rate": _round(_safe_ratio(violations, event_count)),
        "unsafe_event_rate": _round(_safe_ratio(unsafe_events, event_count)),
        "target_hits": target_hits,
        "planner_accuracy": _round(planner_accuracy),
        "planner_miss_ratio": _round(1.0 - planner_accuracy),
        "planner_legal_ratio": _round(planner_legal_ratio),
        "planner_exact_hits": planner_exact_hits,
        "planner_legal_hits": planner_legal_hits,
        "planner_total": planner_total,
        "avg_stage_wait_time": _round(_mean(waiting_steps)),
        "avg_stage_latency_seconds": _round(_mean(waiting_seconds)),
        "latency_penalty": _round(latency_time * 1.6),
        "idle_window_ratio": _round(_safe_ratio(wait_count, event_count)),
        "comp_decision_ratio": _round(_safe_ratio(comp_decisions, event_count)),
        "idle_comp_ratio": _round(_safe_ratio(changed_count, event_count)),
        "wait_ratio": _round(_safe_ratio(wait_count, event_count)),
        "follow_ratio": _round(_safe_ratio(follow_count, event_count)),
        "select_ratio": _round(_safe_ratio(select_count, event_count)),
        "compensation_utilization": _round(_safe_ratio(changed_count or select_count, event_count)),
        "safety_saved": safety_saved,
        "safety_interventions": safety_interventions,
        "safety_success_rate": _round(_safe_ratio(safety_saved, safety_interventions, default=1.0 if safety_interventions == 0 and unsafe_episode == 0 else 0.0)),
        "critical_events": critical_count,
        "critical_safe_count": critical_safe_count,
        "risk_exposure": _round(_mean(risk_values)),
        "observed_risk_mean": _round(_mean(risk_values)),
        "high_risk_step_ratio": _round(_safe_ratio(high_risk_count, event_count)),
        "hazard_step_ratio": _round(_safe_ratio(hazard_count, event_count)),
        "red_light_exposure_ratio": _round(_safe_ratio(red_count, event_count)),
        "critical_event_rate": _round(_safe_ratio(critical_count, event_count)),
        "intervention_rate": _round(_safe_ratio(safety_interventions, event_count)),
        "safety_precision": _round(min(1.0, _safe_ratio(safety_saved, safety_interventions))),
        "safety_recall": _round(min(1.0, _safe_ratio(safety_saved, critical_count))),
        "stop_ratio": _round(_safe_ratio(sum(1 for skill in skill_ids if skill in {"safe_stop", "slow_down", "stop"}), event_count)),
        "steer_left_ratio": _round(_safe_ratio(sum(1 for skill in skill_ids if skill in {"lane_left", "steer_left"}), event_count)),
        "steer_right_ratio": _round(_safe_ratio(sum(1 for skill in skill_ids if skill in {"lane_right", "steer_right"}), event_count)),
        "behavior_entropy": _round(_entropy(skill_ids)),
        "safety_entropy": _round(_entropy(safety_actions)),
        "active_stage_key": active_stage,
        "last_failure_mode": last_failure,
        "epsilon": _round(_float(result.get("epsilon", DEFAULT_EPSILON.get(method_key, 0.0)))),
        "episode_loss": _round(_float(result.get("episode_loss", 0.0))),
    }
    if job["chapter"] == "chap04":
        feedback_records = int(_float(result.get("feedback_records", safety_interventions)))
        row.update(
            {
                "risk_loss": _round(_float(result.get("risk_loss", 0.0))),
                "efsr_high_record_count": feedback_records,
                "efsr_high_record_maturity": _round(min(1.0, feedback_records / 1000.0)),
                "efsr_high_record_unsafe_rate": _round(_safe_ratio(unsafe_events, event_count)),
                "efsr_low_replay_size": feedback_records,
                "efsr_low_risk_loss": _round(_float(result.get("efsr_low_risk_loss", 0.0))),
            }
        )
    return _with_fixed_columns(row, columns_for_chapter(job["chapter"]))


def summarize_reference_rows(rows: List[Dict[str, Any]], method_label: str, job: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "method": method_label,
        "chapter": job["chapter"],
        "group": job["group"],
        "scenario": job["reference_scene"],
        "env_scene": job["env_scene"],
        "seed": job["seed"],
        "episodes": len(rows),
        "summary_window": len(rows),
    }
    if not rows:
        return summary
    for key in rows[0]:
        values = [_float_or_none(row.get(key)) for row in rows]
        numeric = [value for value in values if value is not None]
        if numeric:
            summary["avg_%s_all" % key] = _round(sum(numeric) / len(numeric))
    summary["avg_score_all"] = _round(_mean([_float(row.get("score")) for row in rows]))
    summary["avg_reward_all"] = _round(_mean([_float(row.get("reward")) for row in rows]))
    summary["avg_success_all"] = _round(_mean([_float(row.get("success")) for row in rows]))
    summary["avg_steps_all"] = _round(_mean([_float(row.get("steps")) for row in rows]))
    return summary


def _result_from_events(episode: int, events: List[Dict[str, Any]], job: Dict[str, Any], method_key: str) -> Dict[str, Any]:
    return {
        "episode": episode,
        "scene": job["env_scene"],
        "method": method_key,
        "seed": job["seed"],
        "total_reward": _sum(events, "reward"),
        "steps": len(events),
        "success": any(_truthy(event.get("success")) for event in events),
        "done_reason": str(_dict(_last(events).get("event_info")).get("done_reason", "running")),
    }


def _stage_total(job: Dict[str, Any], events: List[Dict[str, Any]]) -> int:
    if _is_ugv(job):
        return UGV_STAGE_TOTAL
    max_phase = max([int(_float(_dict(event.get("task_state")).get("phase_index"))) for event in events] or [0])
    return max(1, max_phase + 1)


def _stages_completed(events: List[Dict[str, Any]], final_task: Dict[str, Any], success: int, total: int) -> float:
    if success:
        return float(total)
    completed_events = sum(1 for event in events if _truthy(_dict(event.get("event_info")).get("stage_complete")))
    phase_progress = int(_float(final_task.get("phase_index", 0)))
    return float(max(completed_events, min(total, phase_progress)))


def _latency_time(events: List[Dict[str, Any]], waiting_seconds: List[float]) -> float:
    if not events:
        return 0.0
    sampled = [_float(event.get("sampled_inference_seconds")) for event in events if event.get("sampled_inference_seconds") is not None]
    if waiting_seconds:
        # Waiting seconds are cumulative within one pending request. Averaging keeps scripted runs from overcounting.
        return sum(waiting_seconds) / max(1, len(waiting_seconds))
    return sum(sampled) / max(1, len(sampled)) if sampled else 0.0


def _nominal_completion_seconds(completion_steps: int) -> float:
    return 90.0 + 0.375 * float(completion_steps)


def _planner_exact_hits(events: List[Dict[str, Any]]) -> int:
    hits = 0
    for event in events:
        planned = event.get("planned_skill_id") or event.get("raw_skill_id")
        executed = event.get("skill_id")
        if planned is not None:
            hits += 1 if str(planned) == str(executed) else 0
        else:
            hits += 1 if _truthy(_dict(event.get("event_info")).get("stage_complete")) else 0
    return hits


def _planner_legal_hits(events: List[Dict[str, Any]]) -> int:
    count = 0
    for event in events:
        event_info = _dict(event.get("event_info"))
        if str(event_info.get("event_type", "")) != "skill_out_of_phase":
            count += 1
    return count


def _target_hits(events: List[Dict[str, Any]]) -> int:
    count = 0
    for event in events:
        event_info = _dict(event.get("event_info"))
        if _truthy(event_info.get("target_hit")) or _truthy(event_info.get("target_cleared")):
            count += 1
        elif str(event.get("skill_id")) == "target_ball_clear" and _truthy(event_info.get("stage_complete")):
            count += 1
    return count


def _active_stage_key(final_task: Dict[str, Any], result: Dict[str, Any], success: int) -> str:
    if success:
        return "completed"
    return str(final_task.get("phase") or final_task.get("unity_stage_name") or result.get("done_reason") or "running")


def _is_ugv(job: Dict[str, Any]) -> bool:
    return str(job.get("env_scene")) in {"ugv_parking", "ugv_patrol"}


def _has_collision(event: Dict[str, Any]) -> bool:
    safety = _dict(event.get("safety_state"))
    info = _dict(event.get("event_info"))
    return _truthy(safety.get("collision")) or _truthy(info.get("collision"))


def _has_violation(event: Dict[str, Any]) -> bool:
    safety = _dict(event.get("safety_state"))
    info = _dict(event.get("event_info"))
    return any(
        _truthy(value)
        for value in (
            safety.get("red_light_violation"),
            safety.get("out_of_road"),
            safety.get("out_of_drivable_region"),
            safety.get("constraint_violation"),
            info.get("violation"),
            info.get("red_light_violation"),
        )
    )


def _is_unsafe_event(event: Dict[str, Any]) -> bool:
    return _has_collision(event) or _has_violation(event)


def _risk_value(event: Dict[str, Any]) -> float:
    safety = _dict(event.get("safety_state"))
    return _float(safety.get("risk_score", safety.get("danger", 0.0)))


def _is_hazard_event(event: Dict[str, Any]) -> bool:
    safety = _dict(event.get("safety_state"))
    front = _float_or_none(safety.get("front_obstacle_dist", safety.get("front_vehicle_dist", safety.get("min_obstacle_dist"))))
    return _truthy(safety.get("front_blocked")) or (front is not None and front < 6.0)


def _is_red_light_event(event: Dict[str, Any]) -> bool:
    safety = _dict(event.get("safety_state"))
    light = str(safety.get("traffic_light_state", "") or "").lower()
    return light in {"red", "yellow"} or _truthy(safety.get("red_light_violation"))


def _is_critical_event(event: Dict[str, Any]) -> bool:
    return _risk_value(event) >= 0.45 or _is_hazard_event(event) or _is_red_light_event(event) or _is_unsafe_event(event)


def _is_intervention(event: Dict[str, Any]) -> bool:
    if _truthy(event.get("corrected")):
        return True
    mode = str(event.get("decision_mode", "") or "")
    return mode in {"safety_filter", "risk_filter", "safe_override"}


def _changed_from_plan(event: Dict[str, Any]) -> bool:
    planned = event.get("planned_skill_id") or event.get("raw_skill_id")
    if planned is None:
        return False
    return str(planned) != str(event.get("skill_id"))


def _last_dict(events: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    for event in reversed(events):
        value = event.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def _last(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(events[-1]) if events else {}


def _dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _count_events(events: List[Dict[str, Any]], predicate: Any) -> int:
    return sum(1 for event in events if predicate(event))


def _sum(events: List[Dict[str, Any]], key: str) -> float:
    return sum(_float(event.get(key)) for event in events)


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    denominator = float(denominator or 0.0)
    if denominator == 0.0:
        return default
    return float(numerator) / denominator


def _entropy(values: Iterable[str]) -> float:
    counts = Counter(value for value in values if value)
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _float(value: Any) -> float:
    parsed = _float_or_none(value)
    return 0.0 if parsed is None else parsed


def _float_or_none(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: Any, digits: int = 4) -> float:
    return round(_float(value), digits)


def _with_fixed_columns(row: Dict[str, Any], columns: List[str]) -> Dict[str, Any]:
    return {column: row.get(column, 0.0 if column not in {"active_stage_key", "last_failure_mode"} else "") for column in columns}
