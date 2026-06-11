"""Command-line manual and code control for UGV Unity scenes.

Examples:
    conda run -n DT python scripts/ugv_control.py --backend scripted --mode code
    conda run -n DT python scripts/ugv_control.py --backend unity --mode manual
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
UNITY_PROJECT = WORKSPACE_ROOT / "rlenvironments"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.unity_ugv_env import UnityUGVEnv  # noqa: E402
from src.chap02_dk_abs.domain_knowledge import build_topology, extract_region_rules, extract_skill_descriptors  # noqa: E402
from src.chap02_dk_abs.planning import generate_llm_action_behavior_plan, validate_plan  # noqa: E402


KEY_ACTIONS = {
    "w": (0.0, 0.65, "forward"),
    "up": (0.0, 0.65, "forward"),
    "s": (0.0, -0.45, "reverse"),
    "down": (0.0, -0.45, "reverse"),
    "a": (-0.75, 0.35, "left"),
    "left": (-0.75, 0.35, "left"),
    "d": (0.75, 0.35, "right"),
    "right": (0.75, 0.35, "right"),
    "x": (0.0, 0.0, "stop"),
    "space": (0.0, 0.0, "stop"),
}


def main() -> None:
    args = parse_args()
    config = build_env_config(args)

    env: Optional[UnityUGVEnv] = None
    try:
        env = create_env(args, config)
        obs = env.reset(seed=args.seed)
        print_summary("reset", obs, 0.0, False)

        if args.mode == "manual":
            run_manual(env, args)
        elif args.mode == "code":
            run_code(env, args)
        elif args.mode == "route":
            run_route(env, args)
        else:
            run_smoke(env, args)
    finally:
        if env is not None:
            env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control UGV Unity scenes from Python.")
    parser.add_argument("--scene", choices=["ugv_parking", "ugv_patrol"], default="ugv_parking")
    parser.add_argument("--backend", choices=["unity", "udp", "scripted"], default="unity")
    parser.add_argument("--mode", choices=["manual", "code", "smoke", "route"], default="manual")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--base-port", type=int, default=5004)
    parser.add_argument("--udp-port", type=int, default=5055)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration-steps", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--route-id", default="parking_demo_01")
    parser.add_argument("--route-loop", action="store_true")
    parser.add_argument("--route-reach-radius", type=float, default=1.2)
    parser.add_argument("--route-max-motor", type=float, default=0.45)
    parser.add_argument("--route-no-llm", action="store_true")
    parser.add_argument("--route-llm-min", type=float, default=1.0)
    parser.add_argument("--route-llm-max", type=float, default=1.6)
    parser.add_argument("--route-llm-skip-order", type=int, default=0)
    parser.add_argument("--route-pivot-angle", type=float, default=70.0)
    parser.add_argument("--route-pivot-distance", type=float, default=8.0)
    parser.add_argument("--route-decision-timeout", type=float, default=3.0)
    parser.add_argument("--player-path", default="")
    parser.add_argument("--return-arrays", action="store_true")
    parser.add_argument("--launch-unity", action="store_true")
    parser.add_argument("--unity-editor", default=os.environ.get("UNITY_EDITOR_PATH", ""))
    parser.add_argument("--unity-project", default=str(UNITY_PROJECT))
    return parser.parse_args()


def build_env_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {
        "backend": args.backend,
        "worker_id": args.worker_id,
        "base_port": args.base_port,
        "udp_port": args.udp_port,
        "seed": args.seed,
        "timeout_wait": args.timeout,
        "control_mode": args.mode if args.mode in {"manual", "code"} else "code",
        "return_observation_arrays": args.return_arrays,
    }
    if args.player_path:
        config["file_name"] = args.player_path
    return config


def create_env(args: argparse.Namespace, config: dict[str, Any]) -> UnityUGVEnv:
    if args.backend != "unity" or args.player_path or not args.launch_unity:
        print_unity_wait_hint(args)
        return UnityUGVEnv(args.scene, config)

    editor = find_unity_editor(args.unity_editor, Path(args.unity_project))
    if editor is None:
        raise RuntimeError("Unity Editor was not found. Pass --unity-editor or set UNITY_EDITOR_PATH.")

    print("Python is waiting for Unity. Launching Unity Editor now...", flush=True)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(UnityUGVEnv, args.scene, config)
        time.sleep(1.0)
        launch_unity_editor(editor, Path(args.unity_project), args.scene)
        return future.result(timeout=max(args.timeout + 30, 60))


def print_unity_wait_hint(args: argparse.Namespace) -> None:
    if args.backend != "unity" or args.player_path:
        return
    print(
        "\n"
        "=== Unity 无人车联调入口 ===\n"
        "当前脚本是命令行控制器，不会弹出图形窗口。\n"
        "如果你想要按钮、状态面板和键盘说明界面，请运行：\n"
        "  conda run -n DT python tests\\ugv_manual_control.py\n\n"
        "当前命令行联调顺序：\n"
        "1. 保持这个 Python 窗口等待连接。\n"
        "2. 到 Unity 打开无人车场景并点击 Play。\n"
        "3. 确认 Unity 的 Project Settings > ML-Agents > Connect Trainer 已开启。\n",
        flush=True,
    )


def find_unity_editor(explicit: str, project_path: Path) -> Optional[Path]:
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None

    version_file = project_path / "ProjectSettings" / "ProjectVersion.txt"
    version = ""
    if version_file.exists():
        for line in version_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("m_EditorVersion:"):
                version = line.split(":", 1)[1].strip()
                break

    candidates = []
    for root in [Path(os.environ.get("ProgramFiles", "")), Path(os.environ.get("ProgramFiles(x86)", ""))]:
        if not root:
            continue
        hub = root / "Unity" / "Hub" / "Editor"
        if version:
            candidates.append(hub / version / "Editor" / "Unity.exe")
        candidates.extend(hub.glob("*/Editor/Unity.exe") if hub.exists() else [])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def launch_unity_editor(editor: Path, project_path: Path, scene: str) -> None:
    subprocess.Popen(
        [
            str(editor),
            "-projectPath",
            str(project_path),
            "-executeMethod",
            "UGVLiveControlLauncher.OpenSelectedSceneAndPlay",
            "-ugvScene",
            "UGVR2Patrol" if scene == "ugv_patrol" else "UGVParking",
        ],
        cwd=str(project_path),
    )


def run_manual(env: UnityUGVEnv, args: argparse.Namespace) -> None:
    env.set_control_mode("manual")
    print(
        "\n"
        "=== 手动控制 ===\n"
        "W / 上方向键：前进\n"
        "S / 下方向键：倒车\n"
        "A / 左方向键：左转\n"
        "D / 右方向键：右转\n"
        "X / 空格：停车\n"
        "C：执行一次代码规划技能\n"
        "Q：退出\n",
        flush=True,
    )
    step = 0
    while step < args.max_steps:
        key = read_key()
        if key is None:
            time.sleep(0.05)
            continue
        if key == "q":
            break
        if key == "c":
            obs, reward, done, _ = run_one_code_skill(env, args.scene)
            print_summary("code_skill", obs, reward, done)
        elif key in KEY_ACTIONS:
            steer, motor, label = KEY_ACTIONS[key]
            obs, reward, done, _ = env.manual_step(steer, motor, duration_steps=args.duration_steps, label=label)
            print_summary(label, obs, reward, done)
        else:
            continue
        step += 1
        if done:
            break


def run_code(env: UnityUGVEnv, args: argparse.Namespace) -> None:
    env.set_control_mode("code")
    for step in range(args.max_steps):
        obs, reward, done, info = run_one_code_skill(env, args.scene)
        print_summary(f"code_step_{step + 1}:{info.get('skill_id', '')}", obs, reward, done)
        if done:
            break
        time.sleep(0.15)


def run_smoke(env: UnityUGVEnv, args: argparse.Namespace) -> None:
    env.set_control_mode("manual")
    for index, (steer, motor, label) in enumerate([(0.0, 0.35, "forward"), (-0.5, 0.25, "left"), (0.5, 0.25, "right"), (0.0, 0.0, "stop")]):
        obs, reward, done, _ = env.manual_step(steer, motor, duration_steps=args.duration_steps, label=label)
        print_summary(f"smoke_{index + 1}:{label}", obs, reward, done)
        if done:
            break


def run_route(env: UnityUGVEnv, args: argparse.Namespace) -> None:
    env.set_control_mode("code")
    env.send_route(
        args.route_id,
        loop=args.route_loop,
        reach_radius=args.route_reach_radius,
        max_motor=args.route_max_motor,
        llm_on_waypoints=not args.route_no_llm,
        skip_llm_order_index=args.route_llm_skip_order,
        pivot_turn_angle_deg=args.route_pivot_angle,
        pivot_turn_distance=args.route_pivot_distance,
        local_inference_min_seconds=args.route_llm_min,
        local_inference_max_seconds=args.route_llm_max,
    )
    print(
        "route_sent | route_id={route_id} loop={loop} llm_wait={llm_wait} skip_order={skip_order}".format(
            route_id=args.route_id,
            loop=args.route_loop,
            llm_wait="unity_local" if not args.route_no_llm else "off",
            skip_order=args.route_llm_skip_order,
        ),
        flush=True,
    )
    for step in range(args.max_steps):
        obs, reward, done, _ = env.code_step(
            {"steer": 0.0, "motor": 0.0, "duration_steps": 1, "action_name": "route_keepalive", "source": "route"}
        )
        print_summary(f"route_keepalive_{step + 1}", obs, reward, done)
        if done:
            break
        time.sleep(0.12)


def service_route_llm_request(
    env: UnityUGVEnv,
    args: argparse.Namespace,
    obs: dict[str, Any],
    answered_requests: dict[str, dict[str, Any]],
    rng: random.Random,
) -> tuple[dict[str, Any], bool]:
    route = dict(obs.get("route_state") or {})
    if not route.get("waitingForDecision"):
        return obs, bool(obs.get("done"))

    request_id = str(route.get("llmRequestId") or "")
    if not request_id:
        return obs, bool(obs.get("done"))

    if request_id in answered_requests:
        previous = answered_requests[request_id]
        env.send_route_decision(
            request_id=request_id,
            decision=str(previous.get("decision", "keep_lane")),
            reason=str(previous.get("reason", "resend previous decision")),
            prompt=str(previous.get("prompt", "")),
            response=str(previous.get("response", "")),
            inference_seconds=float(previous.get("inference_seconds", 0.0) or 0.0),
        )
        latest_obs, reward, done, _ = env.code_step(
            {
                "steer": 0.0,
                "motor": 0.0,
                "duration_steps": 1,
                "action_name": "llm_decision_resend",
                "source": "route_llm",
            }
        )
        print_summary("llm_decision_resend", latest_obs, reward, done)
        return latest_obs, bool(done)

    inference_seconds = rng.uniform(float(args.route_llm_min), float(args.route_llm_max))
    prompt = build_route_decision_prompt(obs, route)
    route_rule = route_stage_rule(route.get("completedWaypointOrder") or route.get("currentWaypointOrder"))
    phase_label = route.get("currentStageName") or route_rule.get("stage") or "unknown"
    print(
        "llm_request | request_id={request_id} phase={phase} delay={delay:.2f}s".format(
            request_id=request_id,
            phase=phase_label,
            delay=inference_seconds,
        ),
        flush=True,
    )

    deadline = time.time() + inference_seconds
    latest_obs = obs
    wait_step = 0
    while time.time() < deadline:
        wait_step += 1
        latest_obs, reward, done, _ = env.code_step(
            {
                "steer": 0.0,
                "motor": 0.0,
                "duration_steps": 1,
                "action_name": "llm_inference_wait",
                "source": "route_llm",
            }
        )
        if done:
            print_summary(f"llm_wait_{wait_step}", latest_obs, reward, done)
            return latest_obs, True
        time.sleep(min(0.08, max(0.0, deadline - time.time())))

    decision, reason = choose_route_decision(latest_obs)
    latest_route = dict(latest_obs.get("route_state") or {})
    decision_rule = route_stage_rule(latest_route.get("completedWaypointOrder") or latest_route.get("currentWaypointOrder"))
    response = json.dumps(
        {
            "decision": decision,
            "reason": reason,
            "stage": decision_rule.get("stage"),
            "action_behavior": decision_rule.get("action_behavior"),
            "target": decision_rule.get("target"),
            "request_id": request_id,
            "inference_seconds": round(inference_seconds, 3),
        },
        ensure_ascii=False,
    )
    env.send_route_decision(
        request_id=request_id,
        decision=decision,
        reason=reason,
        prompt=prompt,
        response=response,
        inference_seconds=inference_seconds,
    )
    answered_requests[request_id] = {
        "decision": decision,
        "reason": reason,
        "prompt": prompt,
        "response": response,
        "inference_seconds": inference_seconds,
    }
    print(
        "llm_decision | request_id={request_id} decision={decision} reason={reason}".format(
            request_id=request_id,
            decision=decision,
            reason=reason,
        ),
        flush=True,
    )
    for confirm_step in range(3):
        latest_obs, reward, done, _ = env.code_step(
            {
                "steer": 0.0,
                "motor": 0.0,
                "duration_steps": 1,
                "action_name": "llm_decision_confirm",
                "source": "route_llm",
            }
        )
        route_after = dict(latest_obs.get("route_state") or {})
        print_summary(f"llm_decision_confirm_{confirm_step + 1}", latest_obs, reward, done)
        if done or not route_after.get("waitingForDecision") or str(route_after.get("llmRequestId") or "") != request_id:
            return latest_obs, bool(done)
        env.send_route_decision(
            request_id=request_id,
            decision=decision,
            reason=reason,
            prompt=prompt,
            response=response,
            inference_seconds=inference_seconds,
        )

    return latest_obs, bool(latest_obs.get("done"))


def build_route_decision_prompt(obs: dict[str, Any], route: dict[str, Any]) -> str:
    packet = dict(obs.get("unity_packet") or {})
    vehicle = dict(obs.get("vehicle_state") or {})
    safety = dict(obs.get("safety_state") or {})
    task = dict(obs.get("task_state") or {})
    entities = list(packet.get("entities") or [])
    route_prompt = str(route.get("llmPrompt") or "")
    route_order = route.get("completedWaypointOrder") or route.get("currentWaypointOrder")
    route_rule = route_stage_rule(route_order)

    compact_entities = []
    for entity in sorted(entities, key=lambda item: float(item.get("distance", 999.0) or 999.0))[:10]:
        compact_entities.append(
            {
                "label": entity.get("label"),
                "type": entity.get("type"),
                "distance": entity.get("distance"),
                "bearingDeg": entity.get("bearingDeg"),
                "relativeDirection": entity.get("relativeDirection"),
                "risk": entity.get("risk"),
            }
        )

    prompt_payload = {
        "route": {
            "activeRouteId": route.get("activeRouteId"),
            "currentStageName": route.get("currentStageName") or route_rule.get("stage"),
            "currentActionBehavior": route.get("currentActionBehavior") or route_rule.get("action_behavior"),
            "currentGoalLabel": route.get("currentGoalLabel") or route_rule.get("target"),
            "distanceToTarget": route.get("distanceToTarget"),
        },
        "domain_knowledge": {
            "stage_flow": [
                {"stage": "车道保持", "action_behavior": "车道保持", "target": "道路通行"},
                {"stage": "避障通行", "action_behavior": "避障通行", "target": "避开障碍物"},
                {"stage": "撞击目标", "action_behavior": "撞击目标", "target": "高价值球体"},
                {"stage": "进入街道", "action_behavior": "进入街道", "target": "街道入口"},
                {"stage": "寻找停车位置", "action_behavior": "寻找停车位置", "target": "停车位"},
                {"stage": "停车", "action_behavior": "停车", "target": "停车位"},
            ],
            "current_stage_rule": {
                "stage": route_rule.get("stage"),
                "action_behavior": route_rule.get("action_behavior"),
                "target": route_rule.get("target"),
            },
        },
        "vehicle_state": {
            "position": vehicle.get("position"),
            "forward": vehicle.get("forward"),
            "euler_angles": vehicle.get("euler_angles"),
            "speed": vehicle.get("speed"),
            "heading_deg": vehicle.get("heading_deg"),
            "forward_speed": vehicle.get("forward_speed"),
            "lateral_speed": vehicle.get("lateral_speed"),
            "angular_speed": vehicle.get("angular_speed"),
            "motor_command": vehicle.get("motor_command"),
            "steer_command": vehicle.get("steer_command"),
        },
        "task": {
            "phase": task.get("phase"),
            "unity_stage_name": task.get("unity_stage_name"),
            "stage_action_hint": task.get("stage_action_hint"),
            "action_behavior": task.get("stage_action_hint") or route_rule.get("action_behavior"),
            "goal": task.get("goal_label") or route_rule.get("target"),
        },
        "environment_observation": {
            "safety": safety,
            "entities": compact_entities,
        },
        "rules": [
            "lane keeping stage -> keep_lane",
            "obstacle passing stage -> avoid_obstacle",
            "target impact stage -> hit_target",
            "street entry stage -> enter_street",
            "parking search stage -> search_parking",
            "parking stage -> park",
        ],
        "unity_prompt": route_prompt,
    }
    return json.dumps(prompt_payload, ensure_ascii=False, indent=2)


def choose_route_decision(obs: dict[str, Any]) -> tuple[str, str]:
    packet = dict(obs.get("unity_packet") or {})
    safety = dict(obs.get("safety_state") or {})
    task = dict(obs.get("task_state") or {})
    entities = list(packet.get("entities") or [])

    traffic_light = str(safety.get("traffic_light_state") or "").lower()
    phase_text = " ".join(
        str(value or "").lower()
        for value in [
            task.get("phase"),
            task.get("unity_stage_name"),
            task.get("stage_action_hint"),
            safety.get("current_region_type"),
            safety.get("current_region_id"),
        ]
    )
    front_entities = [entity for entity in entities if is_front_entity(entity, max_distance=14.0)]
    nearest_front = min(front_entities, key=lambda item: float(item.get("distance", 999.0) or 999.0), default=None)

    front_light = first_entity(front_entities, {"trafficlight", "traffic_light"})
    if front_light is not None and traffic_light not in {"green", "go"}:
        return "wait_green", "front traffic light is not green"

    if "intersection" in phase_text or "gate" in phase_text:
        if traffic_light and traffic_light not in {"green", "go"}:
            return "wait_green", "intersection requires green light"
        return "enter_street", "vehicle is at the intersection and can enter the street"

    parking_slot = first_entity(entities, {"parkingslot", "parking_slot", "parking"})
    if parking_slot is not None and is_street_or_parking_area(phase_text):
        distance = float(parking_slot.get("distance", 999.0) or 999.0)
        if distance <= 5.0:
            return "park", "parking slot is close enough to stop"
        return "search_parking", "parking slot exists but is not close enough"

    front_target = first_entity(front_entities, {"highvaluetarget", "target"})
    if front_target is not None or looks_like_ball(nearest_front):
        return "hit_target", "front sphere or high value target detected"

    front_obstacle = first_entity(front_entities, {"obstacle", "wall", "staticobstacle"})
    if front_obstacle is not None or bool(safety.get("front_blocked")):
        return "avoid_obstacle", "front obstacle blocks the route"

    if is_street_or_parking_area(phase_text):
        return "search_parking", "vehicle is in street or parking area"

    return "keep_lane", "no blocking front object"


def route_stage_rules() -> list[dict[str, Any]]:
    return [
        {"orders": [1, 2], "stage": "车道保持", "action_behavior": "车道保持", "target": "道路通行", "action_id": "keep_lane"},
        {"orders": [3, 4, 5], "stage": "避障通行", "action_behavior": "避障通行", "target": "避开障碍物", "action_id": "avoid_obstacle"},
        {"orders": [6, 7, 8, 9], "stage": "撞击目标", "action_behavior": "撞击目标", "target": "高价值球体", "action_id": "hit_target"},
        {"orders": [10, 11, 12], "stage": "进入街道", "action_behavior": "进入街道", "target": "街道入口", "action_id": "enter_street"},
        {"orders": [13, 14, 15], "stage": "寻找停车位置", "action_behavior": "寻找停车位置", "target": "停车位", "action_id": "search_parking"},
        {"orders": [16], "stage": "停车", "action_behavior": "停车", "target": "停车位", "action_id": "park"},
    ]


def route_stage_rule(order_value: Any) -> dict[str, Any]:
    try:
        order = int(order_value)
    except (TypeError, ValueError):
        return {}

    for rule in route_stage_rules():
        if order in rule["orders"]:
            return {**rule, "order": order}
    return {}


def is_front_entity(entity: dict[str, Any], max_distance: float) -> bool:
    direction = str(entity.get("relativeDirection") or "").lower()
    if "front" not in direction:
        return False
    try:
        return float(entity.get("distance", 999.0) or 999.0) <= max_distance
    except (TypeError, ValueError):
        return False


def first_entity(entities: list[Any], types: set[str]) -> Optional[dict[str, Any]]:
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        entity_type = normalize_entity_type(entity)
        if entity_type in types:
            return entity
    return None


def normalize_entity_type(entity: dict[str, Any]) -> str:
    return str(entity.get("type") or entity.get("tag") or "").strip().lower().replace("_", "")


def looks_like_ball(entity: Optional[dict[str, Any]]) -> bool:
    if not isinstance(entity, dict):
        return False
    label = str(entity.get("label") or entity.get("id") or "").lower()
    return "ball" in label or "sphere" in label


def is_street_or_parking_area(phase_text: str) -> bool:
    return any(token in phase_text for token in ["street", "parking", "innerstreet", "roadarea"])


def run_one_code_skill(env: UnityUGVEnv, scene: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
    task_state = env.get_task_state()
    domain = env.get_domain_knowledge()
    domain = {
        **domain,
        "environment_topology": build_topology(scene, domain),
        "region_rules": extract_region_rules(scene, domain),
        "skill_policies": extract_skill_descriptors(scene, domain),
    }
    result = generate_llm_action_behavior_plan(
        scene,
        task_state,
        domain,
        {
            "mode": "llm_simulated",
            "fallback_to_rule": True,
            "planning_horizon": 8,
            "llm": {"provider": "simulated", "model": "gpt-4o-mini-sim", "temperature": 0.0},
        },
    )
    ok, errors = validate_plan(scene, result.plan, max_plan_length=16)
    if not ok or not result.plan:
        raise RuntimeError(f"Invalid plan: {errors}")

    behavior = result.plan[0]
    obs, reward, done, info = env.apply_skill(behavior.skill_id, behavior.to_skill_args())
    return obs, reward, done, {**info, "skill_id": behavior.skill_id}


def print_summary(label: str, obs: dict[str, Any], reward: float, done: bool) -> None:
    task = obs.get("task_state", {})
    safety = obs.get("safety_state", {})
    vehicle = obs.get("vehicle_state", {})
    print(
        "{label} | phase={phase} unity_stage={unity_stage} speed={speed} motor={motor} steer={steer} "
        "risk={risk} rec={rec} reward={reward:.3f} done={done}".format(
            label=label,
            phase=task.get("phase"),
            unity_stage=task.get("unity_stage_name", ""),
            speed=fmt(vehicle.get("speed")),
            motor=fmt(vehicle.get("motor_command")),
            steer=fmt(vehicle.get("steer_command")),
            risk=fmt(safety.get("risk_score")),
            rec=safety.get("recommended_safe_action"),
            reward=reward,
            done=done,
        ),
        flush=True,
    )


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def read_key() -> Optional[str]:
    try:
        import msvcrt
    except ImportError:
        text = input("key> ").strip().lower()
        return text or None

    if not msvcrt.kbhit():
        return None
    key = msvcrt.getwch()
    if key in ("\x00", "\xe0"):
        code = msvcrt.getwch()
        return {"H": "up", "P": "down", "K": "left", "M": "right"}.get(code)
    if key == " ":
        return "space"
    return key.lower()


if __name__ == "__main__":
    main()
