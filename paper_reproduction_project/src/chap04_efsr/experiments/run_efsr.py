"""Run EFSR safety filtering on official environments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional, Union

from src.chap02_dk_abs.planning import generate_action_behavior_plan
from src.chap04_efsr.safety_layer import SafeSkillExecutor, SafetyFilter
from src.common.config_loader import load_config
from src.common.data_structures import ActionBehavior, EpisodeResult, ExecutionRecord
from src.common.env_tools import make_env
from src.common.logger import JSONLLogger, ensure_dir, write_json
from src.common.metrics import write_summary_artifacts
from src.common.progress_logger import emit_progress


DEFAULT_CONFIG = {
    "scene": "highway",
    "method": "EFSR",
    "episodes": 1,
    "seed": 0,
    "max_episode_steps": 80,
    "output_dir": "results",
    "risk": {"threshold": 0.55, "feedback_weight": 0.35},
    "planner": {"planning_horizon": 80},
}


def run_efsr(
    scene: str = "highway",
    config_path: Optional[Union[str, Path]] = None,
    episodes: Optional[int] = None,
    seed: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    config = load_config(config_path, DEFAULT_CONFIG)
    scene = scene or str(config.get("scene", "highway"))
    episodes = int(episodes if episodes is not None else config.get("episodes", 1))
    seed = int(seed if seed is not None else config.get("seed", 0))
    output_root = ensure_dir(output_dir or config.get("output_dir", "results"))
    method = str(config.get("method", "EFSR"))
    run_dir = ensure_dir(output_root / "chap04_efsr" / scene)
    logger = JSONLLogger(run_dir / "episode_events.jsonl")
    progress_interval = int(config.get("progress_interval", 0) or 0)
    results = []
    for episode in range(episodes):
        result = _run_episode(scene, method, config, seed + episode, episode, logger)
        results.append(result)
        _print_progress(method, scene, episode, episodes, result, progress_interval, config.get("progress_log_path"))
    artifacts = write_summary_artifacts(results, run_dir, "efsr")
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


def _run_episode(scene: str, method: str, config: dict[str, Any], seed: int, episode: int, logger: JSONLLogger) -> EpisodeResult:
    env = make_env(scene, config)
    threshold = float((config.get("risk") or {}).get("threshold", 0.55))
    executor = SafeSkillExecutor(SafetyFilter(risk_threshold=threshold))
    total_reward = 0.0
    corrections = 0
    max_steps = int(config.get("max_episode_steps", 80))
    done = False
    try:
        obs = env.reset(seed=seed, task_config={"max_episode_steps": max_steps})
        plan = generate_action_behavior_plan(scene, env.get_task_state(), env.get_domain_knowledge(), config.get("planner"))
        if scene == "highway":
            plan = _stress_highway_plan(max_steps)
        for step_index in range(max_steps):
            if done:
                break
            raw_behavior = plan[step_index % max(1, len(plan))]
            _print_step_progress(config, method, scene, episode, step_index, max_steps, raw_behavior.skill_id)
            obs, reward, done, info = executor.execute(env, scene, obs, raw_behavior)
            total_reward += float(reward)
            safety_info = dict(info.get("safety_info") or {})
            corrections += 1 if safety_info.get("corrected") else 0
            success = _success(scene, obs, done, step_index + 1, max_steps)
            logger.log(
                ExecutionRecord(
                    episode=episode,
                    step_index=step_index,
                    scene=scene,
                    method=method,
                    skill_id=str(info.get("executed_skill_id", raw_behavior.skill_id)),
                    reward=float(reward),
                    done=bool(done),
                    success=success,
                    task_state=_jsonable(obs.get("task_state", {})),
                    safety_state=_jsonable(obs.get("safety_state", {})),
                    event_info=_jsonable(info),
                    extra={
                        "raw_skill_id": raw_behavior.skill_id,
                        "corrected": bool(safety_info.get("corrected", False)),
                        "raw_risk": safety_info.get("raw_risk"),
                        "chosen_risk": safety_info.get("chosen_risk"),
                        "vehicle_state": _jsonable(obs.get("vehicle_state", {})),
                        "route_state": _jsonable(obs.get("route_state", {})),
                    },
                ).to_dict()
            )
            if success:
                done = True
        memory_path = Path(config.get("output_dir", "results")) / "chap04_efsr" / scene / "feedback_memory.json"
        executor.risk_value.memory.save(memory_path)
        final_info = env.get_event_info()
        return EpisodeResult(
            scene=scene,
            method=method,
            episode=episode,
            seed=seed,
            total_reward=total_reward,
            steps=int(env.get_task_state().get("time_step", max_steps) or max_steps),
            success=_success(scene, obs, done, max_steps, max_steps),
            done_reason=str(final_info.get("done_reason", "running")),
            metrics={
                "correction_ratio": corrections / max(1, max_steps),
                "feedback_records": len(executor.risk_value.memory.records),
            },
        )
    finally:
        env.close()


def _stress_highway_plan(max_steps: int) -> list[ActionBehavior]:
    pattern = ["speed_up", "speed_up", "safe_follow", "lane_left", "keep_lane", "lane_right"]
    return [ActionBehavior(pattern[index % len(pattern)], source="efsr_raw_policy", max_steps=1) for index in range(max_steps)]


def _success(scene: str, obs: dict[str, Any], done: bool, executed: int, horizon: int) -> bool:
    task = dict(obs.get("task_state") or {})
    if bool(task.get("task_success")):
        return True
    if scene == "crafter":
        inventory = dict(obs.get("inventory") or {})
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
    parser.add_argument("--config", default="configs/chap04_efsr/efsr.yaml")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    result = run_efsr(args.scene, args.config, args.episodes, args.seed, args.output_dir)
    print(result["artifacts"])


if __name__ == "__main__":
    main()
