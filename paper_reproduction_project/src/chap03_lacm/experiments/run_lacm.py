"""Run LACM with official highway-env or Crafter wrappers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional, Union

from src.chap02_dk_abs.domain_knowledge import build_topology, extract_region_rules, extract_skill_descriptors
from src.chap03_lacm.compensation import CompensationPolicy, compute_effective_return
from src.chap03_lacm.latency import AsyncPlanClient, LatencySampler, WaitTimeTracker
from src.common.config_loader import load_config
from src.common.data_structures import EpisodeResult, ExecutionRecord
from src.common.env_tools import make_env
from src.common.logger import JSONLLogger, ensure_dir, write_json
from src.common.metrics import write_summary_artifacts
from src.common.progress_logger import emit_progress


DEFAULT_CONFIG = {
    "scene": "highway",
    "method": "LACM",
    "episodes": 1,
    "seed": 0,
    "max_episode_steps": 80,
    "output_dir": "results",
    "latency": {
        "min_seconds": 1.0,
        "max_seconds": 1.8,
        "control_dt_seconds": 0.2,
        "request_interval": 6,
    },
    "planner": {
        "mode": "llm_simulated",
        "fallback_to_rule": True,
        "planning_horizon": 80,
        "llm": {"provider": "simulated", "model": "gpt-4o-mini-sim", "temperature": 0.0},
    },
    "compensation": {"risk_threshold": 0.45, "max_hold_steps": 1},
}


def run_lacm(
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
    method = str(config.get("method", "LACM"))
    run_dir = ensure_dir(output_root / "chap03_lacm" / scene)
    logger = JSONLLogger(run_dir / "episode_events.jsonl")
    progress_interval = int(config.get("progress_interval", 0) or 0)
    results = []
    for episode in range(episodes):
        result = _run_episode(scene, method, config, seed + episode, episode, logger)
        results.append(result)
        _print_progress(method, scene, episode, episodes, result, progress_interval, config.get("progress_log_path"))
    artifacts = write_summary_artifacts(results, run_dir, "lacm")
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
    latency_cfg = dict(config.get("latency") or {})
    comp_cfg = dict(config.get("compensation") or {})
    sampler = LatencySampler.from_config(latency_cfg, seed=seed)
    client = AsyncPlanClient(sampler)
    policy = CompensationPolicy(
        risk_threshold=float(comp_cfg.get("risk_threshold", 0.55)),
        max_hold_steps=int(comp_cfg.get("max_hold_steps", 2)),
    )
    wait_tracker = WaitTimeTracker()
    total_reward = 0.0
    effective_return = 0.0
    compensation_steps = 0
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
        pending = client.request_plan(scene, env.get_task_state(), domain, config.get("planner"))
        wait_tracker.begin_request(float(pending.state.metadata.get("inference_seconds", 0.0) or 0.0))
        active_plan = pending.plan
        cursor = 0
        max_steps = int(config.get("max_episode_steps", 80))
        request_interval = int(latency_cfg.get("request_interval", 6))

        for step_index in range(max_steps):
            if done:
                break
            if pending.tick():
                active_plan = pending.plan
                cursor = 0
                pending = client.request_plan(scene, env.get_task_state(), domain, config.get("planner"))
                wait_tracker.begin_request(float(pending.state.metadata.get("inference_seconds", 0.0) or 0.0))
            else:
                wait_tracker.add_wait_step(float(pending.state.metadata.get("control_dt_seconds", sampler.control_dt_seconds)))

            planned = active_plan[cursor % max(1, len(active_plan))]
            chosen, decision = policy.select(scene, obs, planned, pending.state.elapsed_steps)
            if chosen.skill_id != planned.skill_id:
                compensation_steps += 1
            _print_step_progress(config, method, scene, episode, step_index, max_steps, chosen.skill_id)
            obs, reward, done, info = env.apply_skill(chosen.skill_id, chosen.to_skill_args())
            risk = float((obs.get("safety_state") or {}).get("risk_score", 0.0) or 0.0)
            control_dt = float(pending.state.metadata.get("control_dt_seconds", sampler.control_dt_seconds))
            waiting_seconds = pending.state.elapsed_steps * control_dt
            effective = compute_effective_return(
                reward,
                pending.state.elapsed_steps,
                chosen.skill_id != planned.skill_id,
                risk,
                wait_seconds=waiting_seconds,
                wait_cost_per_second=float(comp_cfg.get("wait_cost_per_second", latency_cfg.get("wait_cost_per_second", 0.02))),
            )
            total_reward += float(reward)
            effective_return += effective
            success = _success(scene, obs, done, step_index + 1, max_steps)
            logger.log(
                ExecutionRecord(
                    episode=episode,
                    step_index=step_index,
                    scene=scene,
                    method=method,
                    skill_id=chosen.skill_id,
                    reward=float(reward),
                    done=bool(done),
                    success=success,
                    task_state=_jsonable(obs.get("task_state", {})),
                    safety_state=_jsonable(obs.get("safety_state", {})),
                    event_info=_jsonable(info),
                    extra={
                        "planned_skill_id": planned.skill_id,
                        "decision_mode": decision["mode"],
                        "confidence": decision["confidence"],
                        "waiting_steps": pending.state.elapsed_steps,
                        "waiting_seconds": waiting_seconds,
                        "sampled_inference_seconds": pending.state.metadata.get("inference_seconds"),
                        "sampled_latency_steps": pending.state.metadata.get("latency_steps"),
                        "control_dt_seconds": control_dt,
                        "effective_return": effective,
                        "vehicle_state": _jsonable(obs.get("vehicle_state", {})),
                        "route_state": _jsonable(obs.get("route_state", {})),
                    },
                ).to_dict()
            )
            cursor += 1
            if (step_index + 1) % request_interval == 0:
                pending = client.request_plan(scene, env.get_task_state(), domain, config.get("planner"))
                wait_tracker.begin_request(float(pending.state.metadata.get("inference_seconds", 0.0) or 0.0))
            if success:
                done = True

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
                "avg_wait_steps": wait_tracker.average_wait_steps,
                "avg_wait_seconds": wait_tracker.average_wait_seconds,
                "avg_inference_seconds": wait_tracker.average_inference_seconds,
                "compensation_ratio": compensation_steps / max(1, max_steps),
                "effective_return": effective_return,
            },
        )
    finally:
        env.close()


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
    parser.add_argument("--config", default="configs/chap03_lacm/lacm.yaml")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    result = run_lacm(args.scene, args.config, args.episodes, args.seed, args.output_dir)
    print(result["artifacts"])


if __name__ == "__main__":
    main()
