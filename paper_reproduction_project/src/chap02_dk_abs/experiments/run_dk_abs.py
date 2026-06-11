"""Run DK-ABS on official Crafter or highway-env backends."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional, Union

from src.chap02_dk_abs.domain_knowledge import (
    build_topology,
    extract_region_rules,
    extract_skill_descriptors,
)
from src.chap02_dk_abs.online_scheduling import DkAbsScheduler
from src.chap02_dk_abs.planning import (
    generate_action_behavior_plan,
    generate_llm_action_behavior_plan,
    repair_plan,
    validate_plan,
)
from src.common.config_loader import load_config
from src.common.data_structures import EpisodeResult, ExecutionRecord
from src.common.env_tools import make_env
from src.common.logger import JSONLLogger, ensure_dir, write_json
from src.common.metrics import write_summary_artifacts
from src.common.progress_logger import emit_progress


DEFAULT_CONFIG = {
    "scene": "crafter",
    "method": "DK_ABS",
    "episodes": 1,
    "seed": 0,
    "max_episode_steps": 200,
    "output_dir": "results",
    "planner": {
        "mode": "llm_simulated",
        "fallback_to_rule": True,
        "replan_each_step": True,
        "wood_target_count": 4,
        "stone_target_count": 1,
        "max_skill_steps": 350,
        "llm": {"provider": "simulated", "model": "gpt-4o-mini-sim", "temperature": 0.0},
    },
    "scheduler": {"delta_trig": 0.65, "max_plan_length": 80},
}


def run_dk_abs(
    scene: str = "crafter",
    config_path: Optional[Union[str, Path]] = None,
    episodes: Optional[int] = None,
    seed: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    config = load_config(config_path, DEFAULT_CONFIG)
    scene = scene or str(config.get("scene", "crafter"))
    episodes = int(episodes if episodes is not None else config.get("episodes", 1))
    seed = int(seed if seed is not None else config.get("seed", 0))
    output_root = ensure_dir(output_dir or config.get("output_dir", "results"))
    method = str(config.get("method", "DK_ABS"))
    run_dir = ensure_dir(output_root / "chap02_dk_abs" / scene)
    logger = JSONLLogger(run_dir / "episode_events.jsonl")
    results: list[EpisodeResult] = []
    progress_interval = int(config.get("progress_interval", 0) or 0)

    for episode in range(episodes):
        result = _run_episode(scene, method, config, seed + episode, episode, logger, run_dir)
        results.append(result)
        _print_progress(method, scene, episode, episodes, result, progress_interval, config.get("progress_log_path"))

    artifacts = write_summary_artifacts(results, run_dir, "dk_abs")
    artifacts["config_json"] = str(write_json(run_dir / "config.json", config))
    return {"results": [item.to_dict() for item in results], "artifacts": artifacts}


def _print_progress(
    method: str,
    scene: str,
    episode: int,
    episodes: int,
    result: EpisodeResult,
    interval: int,
    log_path: Optional[Union[str, Path]],
) -> None:
    current = episode + 1
    if interval <= 0 and current != episodes:
        return
    if interval > 0 and current != episodes and current % interval != 0:
        return
    emit_progress(
        "[progress] {method} {scene} episode {current}/{total} reward={reward:.4f} success={success}".format(
            method=method,
            scene=scene,
            current=current,
            total=episodes,
            reward=float(result.total_reward),
            success=int(bool(result.success)),
        ),
        log_path,
    )


def _print_step_progress(
    config: dict[str, Any],
    method: str,
    scene: str,
    episode: int,
    step_index: int,
    total_steps: int,
    skill_id: str,
) -> None:
    interval = int(config.get("step_progress_interval", 0) or 0)
    current = step_index + 1
    if interval <= 0:
        return
    if current != 1 and current % interval != 0:
        return
    emit_progress(
        "[step] {method} {scene} episode={episode} step={current}/{total} skill={skill}".format(
            method=method,
            scene=scene,
            episode=episode,
            current=current,
            total=total_steps,
            skill=skill_id,
        ),
        config.get("progress_log_path"),
    )


def _run_episode(
    scene: str,
    method: str,
    config: dict[str, Any],
    seed: int,
    episode: int,
    logger: JSONLLogger,
    run_dir: Path,
) -> EpisodeResult:
    env = make_env(scene, config)
    total_reward = 0.0
    step_index = 0
    done = False
    try:
        obs = env.reset(seed=seed, task_config={"max_episode_steps": config.get("max_episode_steps")})
        domain = env.get_domain_knowledge()
        domain = {
            **domain,
            "environment_topology": build_topology(scene, domain),
            "region_rules": extract_region_rules(scene, domain),
            "skill_policies": extract_skill_descriptors(scene, domain),
        }
        planner_config = {**dict(config.get("planner") or {}), **dict(config.get("environment") or {})}
        replan_each_step = bool(
            planner_config.get("replan_each_step", planner_config.get("replan_each_behavior", True))
        )
        plan, llm_result, planner_mode, valid, errors = _request_plan(
            scene=scene,
            task_state=env.get_task_state(),
            domain=domain,
            planner_config=planner_config,
            scheduler_config=dict(config.get("scheduler") or {}),
            run_dir=run_dir,
            episode=episode,
            request_index=0,
        )
        scheduler = DkAbsScheduler(delta_trig=float((config.get("scheduler") or {}).get("delta_trig", 0.65)))
        initial_plan_length = len(plan)
        llm_request_count = 1 if planner_mode not in {"rule", "rule_based", "deterministic"} else 0

        cursor = 0
        max_behaviors = int(config.get("max_behaviors", len(plan)))
        while cursor < len(plan) and step_index < max_behaviors and not done:
            if replan_each_step and step_index > 0:
                plan, llm_result, planner_mode, valid, errors = _request_plan(
                    scene=scene,
                    task_state=env.get_task_state(),
                    domain=domain,
                    planner_config=planner_config,
                    scheduler_config=dict(config.get("scheduler") or {}),
                    run_dir=run_dir,
                    episode=episode,
                    request_index=step_index,
                )
                if planner_mode not in {"rule", "rule_based", "deterministic"}:
                    llm_request_count += 1
                cursor = 0
                if not plan:
                    break
            behavior = plan[cursor]
            reschedule, trigger_probability = scheduler.should_reschedule(obs, behavior)
            if reschedule:
                cursor = scheduler.choose_next_behavior(plan, obs, cursor)
                behavior = plan[cursor]
            _print_step_progress(config, method, scene, episode, step_index, max_behaviors, behavior.skill_id)
            obs, reward, done, info = env.apply_skill(behavior.skill_id, behavior.to_skill_args())
            total_reward += float(reward)
            success = _is_success(scene, obs, done, info, step_index + 1, max_behaviors)
            logger.log(
                ExecutionRecord(
                    episode=episode,
                    step_index=step_index,
                    scene=scene,
                    method=method,
                    skill_id=behavior.skill_id,
                    reward=float(reward),
                    done=bool(done),
                    success=success,
                    task_state=_jsonable(obs.get("task_state", {})),
                    safety_state=_jsonable(obs.get("safety_state", {})),
                    event_info=_jsonable(info),
                    extra={
                        "trigger_probability": trigger_probability,
                        "rescheduled": reschedule,
                        "plan_valid": valid,
                        "plan_errors": errors,
                        "planner_mode": planner_mode,
                        "llm_provider": llm_result.provider if llm_result else None,
                        "llm_request_index": step_index if replan_each_step else 0,
                        "behavior_source": behavior.source,
                        "vehicle_state": _jsonable(obs.get("vehicle_state", {})),
                        "route_state": _jsonable(obs.get("route_state", {})),
                    },
                ).to_dict()
            )
            step_index += 1
            cursor += 1
            if success:
                done = True

        final_state = env.get_task_state()
        final_info = env.get_event_info()
        success = _is_success(scene, obs, done, final_info, step_index, max_behaviors)
        return EpisodeResult(
            scene=scene,
            method=method,
            episode=episode,
            seed=seed,
            total_reward=total_reward,
            steps=int(final_state.get("time_step", step_index) or step_index),
            success=success,
            done_reason=str(final_info.get("done_reason", "success" if success else "running")),
            metrics={
                "planned_behaviors": initial_plan_length,
                "executed_behaviors": step_index,
                "llm_requests": llm_request_count,
            },
        )
    finally:
        env.close()


def _request_plan(
    scene: str,
    task_state: dict[str, Any],
    domain: dict[str, Any],
    planner_config: dict[str, Any],
    scheduler_config: dict[str, Any],
    run_dir: Path,
    episode: int,
    request_index: int,
) -> tuple[list, Any, str, bool, list[str]]:
    planner_mode = str(planner_config.get("mode", planner_config.get("planner_mode", "llm_simulated"))).lower()
    llm_result = None
    if planner_mode not in {"rule", "rule_based", "deterministic"}:
        try:
            llm_result = generate_llm_action_behavior_plan(scene, task_state, domain, planner_config)
            plan = llm_result.plan
            plan_path = run_dir / f"llm_plan_episode_{episode}_request_{request_index}.json"
            write_json(plan_path, llm_result.to_dict())
            if request_index == 0:
                write_json(run_dir / f"llm_plan_episode_{episode}.json", llm_result.to_dict())
        except Exception:
            if not bool(planner_config.get("fallback_to_rule", True)):
                raise
            plan = generate_action_behavior_plan(scene, task_state, domain, {**planner_config, "mode": "rule"})
    else:
        plan = generate_action_behavior_plan(scene, task_state, domain, planner_config)
    valid, errors = validate_plan(scene, plan, max_plan_length=scheduler_config.get("max_plan_length"))
    if not valid:
        plan = repair_plan(scene, plan, task_state, domain, planner_config)
    return plan, llm_result, planner_mode, valid, errors


def _is_success(scene: str, obs: dict[str, Any], done: bool, info: dict[str, Any], executed: int, horizon: int) -> bool:
    task = dict(obs.get("task_state") or {})
    if bool(task.get("task_success") or info.get("task_success")):
        return True
    if scene == "crafter":
        inventory = dict(obs.get("inventory") or task.get("inventory") or {})
        achievements = dict(obs.get("achievements") or {})
        return bool(inventory.get("stone_pickaxe", 0) or achievements.get("make_stone_pickaxe"))
    if scene == "highway":
        safety = dict(obs.get("safety_state") or {})
        return not bool(safety.get("collision", False)) and executed >= horizon
    return bool(done)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/chap02_dk_abs/dk_abs.yaml")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    result = run_dk_abs(
        scene=args.scene,
        config_path=args.config,
        episodes=args.episodes,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(result["artifacts"])


if __name__ == "__main__":
    main()
