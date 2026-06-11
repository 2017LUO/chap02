"""Unified UGV environments with scripted and optional ML-Agents backends."""

from __future__ import annotations

import base64
import io
import json
import socket
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from .base_env import BaseExperimentEnv


STATE_CHANNEL_ID = "9b95f441-6f23-4d61-bb23-5df476f26f25"


@dataclass(frozen=True)
class UGVStage:
    phase: str
    skill_id: str
    region: str
    reward: float


UGV_PARKING_STAGES = [
    UGVStage("main_road", "main_road_driving", "main_road", 1.0),
    UGVStage("obstacle_zone", "obstacle_avoidance", "main_road", 1.2),
    UGVStage("target_ball", "target_ball_clear", "main_road", 1.2),
    UGVStage("intersection", "pass_intersection", "intersection", 1.4),
    UGVStage("street", "enter_street", "street", 1.4),
    UGVStage("parking_area", "search_parking_slot", "parking_area", 1.0),
    UGVStage("parking", "parking_control", "parking_area", 5.0),
]


UGV_PATROL_STAGES = [
    UGVStage("outer_patrol", "outer_route_follow", "outer_route", 1.0),
    UGVStage("checkpoint", "checkpoint_approach", "outer_route", 1.0),
    UGVStage("intersection", "pass_intersection", "intersection", 1.3),
    UGVStage("route_switch", "route_switch", "switch_area", 1.2),
    UGVStage("inner_patrol", "inner_route_follow", "inner_route", 1.0),
    UGVStage("parking_area", "search_parking_slot", "parking_area", 1.0),
    UGVStage("parking", "parking_control", "parking_area", 5.0),
]


SKILL_ALIASES = {
    "outer_loop_driving": "main_road_driving",
    "target_tracking": "target_ball_clear",
    "patrol_follow": "outer_route_follow",
}


UNITY_STAGE_ALIASES = {
    "车道保持": "main_road",
    "lane_keeping": "main_road",
    "lanekeeping": "main_road",
    "main_road_driving": "main_road",
    "避障通行": "obstacle_zone",
    "obstacles_avoiding": "obstacle_zone",
    "obstaclesavoiding": "obstacle_zone",
    "obstacle_avoidance": "obstacle_zone",
    "目标跟踪": "target_ball",
    "目标清除": "target_ball",
    "撞击目标": "target_ball",
    "hitting_target": "target_ball",
    "hitting target": "target_ball",
    "target_ball_clear": "target_ball",
    "路口通行": "intersection",
    "intersection": "intersection",
    "pass_intersection": "intersection",
    "进入街道": "street",
    "entering_street": "street",
    "enteringstreet": "street",
    "enter_street": "street",
    "搜索停车位": "parking_area",
    "寻找停车位置": "parking_area",
    "reaching_park": "parking_area",
    "reachingpark": "parking_area",
    "search_parking_slot": "parking_area",
    "执行泊车": "parking",
    "停车": "parking",
    "parking_control": "parking",
    "outer_route_follow": "outer_patrol",
    "outer_patrol": "outer_patrol",
    "checkpoint_approach": "checkpoint",
    "checkpoint": "checkpoint",
    "signal_wait": "intersection",
    "dynamic_obstacle_avoidance": "intersection",
    "route_switch": "route_switch",
    "inner_route_follow": "inner_patrol",
    "inner_patrol": "inner_patrol",
}


class UnityBackendUnavailable(RuntimeError):
    """Raised when the Unity backend is requested but dependencies are absent."""


class UnityUGVEnv(BaseExperimentEnv):
    """UGV environment facade used by chapter methods.

    The default backend is deterministic ``scripted`` so experiments and tests
    can run without a Unity player. Set ``backend: unity`` plus a Unity player
    path or an already running editor/player to use ML-Agents when available.
    """

    def __init__(self, scene_name: str, env_config: Optional[dict[str, Any]] = None) -> None:
        self.scene_name = scene_name.strip().lower()
        if self.scene_name not in {"ugv_parking", "ugv_patrol"}:
            raise ValueError(f"Unsupported UGV scene: {scene_name}")

        self.env_config = dict(env_config or {})
        self.backend_name = str(self.env_config.get("backend", "scripted")).strip().lower()
        if self.backend_name in {"unity_pending", "pending", "stub"}:
            self.backend_name = "scripted"

        self._unity_backend: Optional[_MLAgentsUGVBackend] = None
        self._udp_backend: Optional[_UdpUGVBackend] = None
        self._stages = UGV_PARKING_STAGES if self.scene_name == "ugv_parking" else UGV_PATROL_STAGES
        self._stage_index = 0
        self._time_step = 0
        self._done = False
        self._task_success = False
        self._risk_score = 0.15
        self._last_reward = 0.0
        self._last_event: dict[str, Any] = {"event_type": "created", "done_reason": "not_started"}
        self._last_unity_packet: dict[str, Any] = {}
        self._control_mode = self._normalize_control_mode(self.env_config.get("control_mode", "code"))
        self._last_control_action: dict[str, Any] = {}

        if self.backend_name in {"unity", "mlagents", "ml-agents"}:
            self._unity_backend = _MLAgentsUGVBackend(self.scene_name, self.env_config)
        elif self.backend_name in {"udp", "unity_udp", "unity-udp"}:
            self.backend_name = "udp"
            self._udp_backend = _UdpUGVBackend(self.scene_name, self.env_config)
        elif self.backend_name != "scripted":
            raise ValueError(f"Unsupported UGV backend: {self.backend_name}")

    def reset(self, seed: Optional[int] = None, task_config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        self._stage_index = 0
        self._time_step = 0
        self._done = False
        self._task_success = False
        self._risk_score = 0.15
        self._last_reward = 0.0
        self._last_event = {
            "event_type": "reset",
            "done_reason": "running",
            "seed": seed,
            "backend": self.backend_name,
        }
        if self._unity_backend is not None:
            self._last_unity_packet = self._unity_backend.reset(seed=seed, task_config=task_config)
            self._sync_from_unity_packet(self._last_unity_packet)
        elif self._udp_backend is not None:
            self._udp_backend.reset(seed=seed, task_config=task_config)
        return self._build_observation()

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self._ensure_started()
        if self._unity_backend is not None:
            self._last_unity_packet, reward, done, info = self._unity_backend.step(action)
            self._sync_from_unity_packet(self._last_unity_packet)
            self._last_reward = reward
            self._done = bool(done)
            self._last_control_action = self._action_payload(action, source=self._control_mode)
            self._last_event = {**dict(info), "control_mode": self._control_mode}
            return self._build_observation(), float(reward), bool(done), dict(info)
        if self._udp_backend is not None:
            self._udp_backend.step(action)

        action_name = self._normalize_action_name(action)
        self._last_control_action = self._action_payload(action, source=self._control_mode)
        if action_name in {"stop", "safe_stop", "slow_down"}:
            reward = -0.02
            self._risk_score = max(0.02, self._risk_score - 0.08)
        elif action_name in {"accelerate", "speed_up"}:
            reward = 0.05
            self._risk_score = min(1.0, self._risk_score + 0.04)
        else:
            reward = 0.03
            self._risk_score = max(0.02, self._risk_score - 0.01)

        self._time_step += 1
        self._last_reward = reward
        self._last_event = {
            "event_type": "step",
            "action_name": action_name,
            "done_reason": "running",
            "backend": "scripted",
            "control_mode": self._control_mode,
            "control_action": dict(self._last_control_action),
        }
        return self._build_observation(), float(reward), self._done, dict(self._last_event)

    def apply_skill(
        self, skill_id: str, skill_args: Optional[dict[str, Any]] = None
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self._ensure_started()
        skill_args = dict(skill_args or {})
        skill_id = self._canonical_skill(skill_id)

        if self._unity_backend is not None:
            action = {"skill_id": skill_id, "skill_args": skill_args}
            self._last_unity_packet, reward, done, info = self._unity_backend.step(action)
            self._sync_from_unity_packet(self._last_unity_packet)
            self._last_reward = reward
            self._done = bool(done)
            self._last_control_action = self._action_payload(action, source=self._control_mode)
            self._last_event = {**dict(info), "control_mode": self._control_mode}
            return self._build_observation(), float(reward), bool(done), dict(info)
        if self._udp_backend is not None:
            self._udp_backend.step({"skill_id": skill_id, "skill_args": skill_args})

        if skill_id == "safe_stop":
            reward = -0.01
            self._risk_score = max(0.02, self._risk_score - 0.12)
            event_type = "safe_stop"
            stage_complete = False
        else:
            current = self._current_stage()
            valid_skills = set(self._skill_sequence()) | {"safe_stop"}
            if skill_id not in valid_skills:
                raise ValueError(f"Unknown skill for {self.scene_name}: {skill_id}")

            stage_complete = skill_id == current.skill_id
            if stage_complete:
                reward = current.reward
                self._stage_index += 1
                self._risk_score = self._risk_after_success(skill_id)
                event_type = "skill_complete"
                if self._stage_index >= len(self._stages):
                    self._done = True
                    self._task_success = True
            else:
                reward = -0.2
                self._risk_score = min(1.0, self._risk_score + 0.18)
                event_type = "skill_out_of_phase"

        self._time_step += max(1, int(skill_args.get("simulated_steps", 1)))
        self._last_reward = reward
        self._last_event = {
            "event_type": event_type,
            "skill_id": skill_id,
            "skill_args": skill_args,
            "skill_steps": int(skill_args.get("max_steps", 1) or 1),
            "stage_complete": stage_complete,
            "task_success": self._task_success,
            "done_reason": "success" if self._task_success else "running",
            "backend": "scripted",
            "control_mode": self._control_mode,
        }
        return self._build_observation(), float(reward), self._done, dict(self._last_event)

    def set_control_mode(self, mode: str) -> None:
        """Switch between algorithmic code control and manual control."""
        self._control_mode = self._normalize_control_mode(mode)
        if self._unity_backend is not None:
            self._unity_backend.send_control_mode(self._control_mode)
        if self._udp_backend is not None:
            self._udp_backend.send_control_mode(self._control_mode)
        self._last_event = {
            **dict(self._last_event),
            "event_type": "control_mode_changed",
            "control_mode": self._control_mode,
        }

    def get_control_mode(self) -> str:
        """Return the active control mode."""
        return self._control_mode

    def manual_step(
        self,
        steer: float,
        motor: float,
        duration_steps: int = 1,
        label: str = "manual_control",
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Send a manual continuous control action through the normal action channel."""
        previous_mode = self._control_mode
        if previous_mode != "manual":
            self.set_control_mode("manual")
        action = {
            "steer": self._clip_unit(steer),
            "motor": self._clip_unit(motor),
            "duration_steps": max(1, int(duration_steps)),
            "action_name": label,
            "source": "manual",
        }
        obs, reward, done, info = self.step(action)
        info = {**info, "control_mode": "manual", "control_action": dict(action)}
        self._last_event = {**dict(self._last_event), **info}
        return obs, reward, done, info

    def code_step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Send an algorithm-produced low-level action through the normal action channel."""
        if self._control_mode != "code":
            self.set_control_mode("code")
        return self.step(action)

    def send_route(
        self,
        route_id: str,
        loop: bool = False,
        reach_radius: Optional[float] = None,
        max_motor: Optional[float] = None,
        min_motor: Optional[float] = None,
        llm_on_waypoints: bool = True,
        skip_llm_order_index: int = 0,
        pivot_turn_angle_deg: Optional[float] = None,
        pivot_turn_distance: Optional[float] = None,
        decision_timeout_seconds: Optional[float] = None,
        local_inference_min_seconds: Optional[float] = None,
        local_inference_max_seconds: Optional[float] = None,
    ) -> None:
        """Ask Unity to follow waypoints whose groupId matches route_id."""
        payload: dict[str, Any] = {
            "type": "route",
            "scene": self.scene_name,
            "route_id": route_id,
            "loop": bool(loop),
            "llm_on_waypoints": bool(llm_on_waypoints),
            "llm_on_waypoints_set": True,
            "skip_llm_order_index": int(skip_llm_order_index),
        }
        if reach_radius is not None:
            payload["reach_radius"] = float(reach_radius)
        if max_motor is not None:
            payload["max_motor"] = float(max_motor)
        if min_motor is not None:
            payload["min_motor"] = float(min_motor)
        if pivot_turn_angle_deg is not None:
            payload["pivot_turn_angle_deg"] = float(pivot_turn_angle_deg)
        if pivot_turn_distance is not None:
            payload["pivot_turn_distance"] = float(pivot_turn_distance)
        if decision_timeout_seconds is not None:
            payload["decision_timeout_seconds"] = float(decision_timeout_seconds)
        if local_inference_min_seconds is not None:
            payload["local_inference_min_seconds"] = float(local_inference_min_seconds)
        if local_inference_max_seconds is not None:
            payload["local_inference_max_seconds"] = float(local_inference_max_seconds)

        if self._unity_backend is not None:
            self._unity_backend.send_route(payload)
        elif self._udp_backend is not None:
            self._udp_backend.send_route(payload)
        else:
            self._last_event = {
                **dict(self._last_event),
                "event_type": "scripted_route_sent",
                "route_id": route_id,
                "loop": bool(loop),
            }

    def send_route_decision(
        self,
        request_id: str,
        decision: str,
        reason: str = "",
        prompt: str = "",
        response: str = "",
        inference_seconds: Optional[float] = None,
    ) -> None:
        """Reply to Unity after a waypoint-triggered high-level decision."""
        payload: dict[str, Any] = {
            "type": "route_decision",
            "scene": self.scene_name,
            "request_id": request_id,
            "decision": decision,
            "reason": reason,
            "prompt": prompt,
            "response": response or reason,
        }
        if inference_seconds is not None:
            payload["inference_seconds"] = float(inference_seconds)

        if self._unity_backend is not None:
            self._unity_backend.send_route(payload)
        elif self._udp_backend is not None:
            self._udp_backend.send_route(payload)
        else:
            self._last_event = {
                **dict(self._last_event),
                "event_type": "scripted_route_decision",
                "request_id": request_id,
                "decision": decision,
                "reason": reason,
            }

    def stop_route(self) -> None:
        """Stop Unity waypoint following if it is active."""
        payload = {"type": "route_stop", "scene": self.scene_name}
        if self._unity_backend is not None:
            self._unity_backend.send_route(payload)
        elif self._udp_backend is not None:
            self._udp_backend.send_route(payload)

    def get_task_state(self) -> dict[str, Any]:
        stage = self._current_stage()
        phase_index = min(self._stage_index, len(self._stages) - 1)
        base = {
            "scene_name": self.scene_name,
            "phase": "task_complete" if self._task_success else stage.phase,
            "phase_index": phase_index,
            "time_step": self._time_step,
            "stage_complete": self._task_success,
            "task_success": self._task_success,
            "distance_to_goal": self._distance_to_goal(),
            "traffic_light_state": self._traffic_light_state(),
            "backend": self.backend_name,
        }
        if self.scene_name == "ugv_parking":
            base.update(
                {
                    "target_id": "target_ball_01",
                    "road_segment_id": stage.region,
                    "parking_slot_id": str(self.env_config.get("target_slot_id", "slot_01")),
                    "is_in_parking_area": stage.region == "parking_area",
                }
            )
        else:
            base.update(
                {
                    "route_mode": "inner" if stage.region == "inner_route" else "outer",
                    "checkpoint_id": f"cp_{phase_index + 1:02d}",
                    "next_checkpoint_id": f"cp_{phase_index + 2:02d}",
                    "checkpoint_order_ok": True,
                    "switch_area_id": "switch_01",
                    "final_parking_slot": str(self.env_config.get("final_parking_slot", "slot_01")),
                    "distance_to_checkpoint": self._distance_to_goal(),
                }
            )
        return self._merge_unity_task_state(base)

    def get_domain_knowledge(self) -> dict[str, Any]:
        if self.scene_name == "ugv_parking":
            topology = {
                "nodes": ["start", "main_road", "intersection", "street", "parking_area"],
                "edges": [
                    ["start", "main_road"],
                    ["main_road", "intersection"],
                    ["intersection", "street"],
                    ["street", "parking_area"],
                ],
            }
            rules = {
                "main_road": ["lane_keep_required"],
                "intersection": ["green_light_required"],
                "grass_area": ["forbidden"],
                "parking_area": ["low_speed_required", "slot_alignment_required"],
            }
            policies = {
                "main_road_driving": "drive along the main road and keep lane",
                "obstacle_avoidance": "avoid static obstacles while staying on road",
                "target_ball_clear": "approach and clear the yellow target ball",
                "pass_intersection": "pass the intersection only when signal permits",
                "enter_street": "enter the street segment after intersection",
                "search_parking_slot": "search and approach the target parking slot",
                "parking_control": "align and park in the target slot",
                "safe_stop": "stop or wait when safety constraints are active",
            }
        else:
            topology = {
                "nodes": [
                    "outer_cp_01",
                    "outer_cp_02",
                    "intersection",
                    "inner_cp_01",
                    "inner_cp_02",
                    "parking_area",
                ],
                "edges": [
                    ["outer_cp_01", "outer_cp_02"],
                    ["outer_cp_02", "intersection"],
                    ["intersection", "inner_cp_01"],
                    ["inner_cp_01", "inner_cp_02"],
                    ["inner_cp_02", "parking_area"],
                ],
            }
            rules = {
                "outer_route": ["checkpoint_order_required"],
                "inner_route": ["checkpoint_order_required"],
                "intersection": ["green_light_required"],
                "parking_area": ["low_speed_required", "slot_alignment_required"],
            }
            policies = {
                "outer_route_follow": "follow the outer patrol route",
                "inner_route_follow": "follow the inner patrol route",
                "checkpoint_approach": "approach and trigger the next checkpoint",
                "pass_intersection": "pass the signal-controlled intersection safely",
                "route_switch": "switch between outer and inner route when allowed",
                "search_parking_slot": "search and approach the final parking slot",
                "parking_control": "complete final parking",
                "safe_stop": "stop or wait when safety constraints are active",
            }
        return {
            "environment_topology": topology,
            "region_rules": rules,
            "skill_policies": policies,
            "backend": self.backend_name,
            "unity_state_channel_id": STATE_CHANNEL_ID,
        }

    def get_safety_state(self) -> dict[str, Any]:
        stage = self._current_stage()
        light = self._traffic_light_state()
        front = max(1.0, 16.0 - self._stage_index * 1.5)
        if stage.skill_id in {"obstacle_avoidance", "target_ball_clear", "pass_intersection"}:
            front = min(front, 5.5)
        collision = self._risk_score >= 0.98
        red_violation = light != "green" and stage.region == "intersection" and self._last_event.get("skill_id") != "safe_stop"
        recommended = "keep"
        if red_violation or (light != "green" and stage.region == "intersection"):
            recommended = "safe_stop"
        elif front < 6.0:
            recommended = "obstacle_avoidance" if self.scene_name == "ugv_parking" else "safe_stop"
        elif stage.region == "parking_area":
            recommended = "parking_control"
        base = {
            "collision": collision,
            "out_of_road": False,
            "out_of_drivable_region": False,
            "red_light_violation": red_violation,
            "front_blocked": front < 4.0,
            "risk_score": round(float(self._risk_score), 4),
            "recommended_safe_action": recommended,
            "min_obstacle_dist": front,
            "front_obstacle_dist": front,
            "left_obstacle_dist": 8.0,
            "right_obstacle_dist": 7.5,
            "front_wall_dist": 12.0,
            "traffic_light_state": light,
            "current_region_id": stage.region,
            "current_region_type": stage.region,
        }
        return self._merge_unity_safety_state(base)

    def get_event_info(self) -> dict[str, Any]:
        return dict(self._last_event)

    def close(self) -> None:
        if self._unity_backend is not None:
            self._unity_backend.close()
        if self._udp_backend is not None:
            self._udp_backend.close()

    def _build_observation(self) -> dict[str, Any]:
        task = self.get_task_state()
        safety = self.get_safety_state()
        stage = self._current_stage()
        vehicle = {
            "position": [float(self._stage_index * 4), 0.0, 0.0],
            "heading_deg": 0.0 if stage.region != "parking_area" else 90.0,
            "speed": 0.0 if self._done else max(0.0, 4.0 - self._risk_score),
            "forward_speed": 0.0 if self._done else max(0.0, 3.6 - self._risk_score),
            "motor_command": 0.0 if self._done else 0.45,
            "steer_command": 0.0,
        }
        vehicle = self._merge_unity_vehicle_state(vehicle)
        return {
            "scene_name": self.scene_name,
            "backend": self.backend_name,
            "task_state": task,
            "safety_state": safety,
            "vehicle_state": vehicle,
            "control_state": {
                "action_space": "continuous[steer,motor]",
                "control_mode": self._control_mode,
                "last_control_action": dict(self._last_control_action),
                "available_skills": self._skill_sequence() + ["safe_stop"],
            },
            "domain_knowledge": self.get_domain_knowledge(),
            "event_info": self.get_event_info(),
            "route_state": dict(self._last_unity_packet.get("route") or {}),
            "reward": self._last_reward,
            "done": self._done,
            "visual_observations": list(self._last_unity_packet.get("visuals") or []),
            "mlagents_observations": dict(self._last_unity_packet.get("mlagents_observations") or {}),
            "unity_packet": dict(self._last_unity_packet),
            "vector": [
                float(self._stage_index),
                float(self._risk_score),
                float(task.get("distance_to_goal", 0.0) or 0.0),
            ],
        }

    def _current_stage(self) -> UGVStage:
        if self._stage_index >= len(self._stages):
            return self._stages[-1]
        return self._stages[self._stage_index]

    def _skill_sequence(self) -> list[str]:
        return [stage.skill_id for stage in self._stages]

    def _canonical_skill(self, skill_id: str) -> str:
        normalized = str(skill_id).strip()
        return SKILL_ALIASES.get(normalized, normalized)

    def _ensure_started(self) -> None:
        if self._last_event.get("event_type") == "created":
            self.reset()

    def _distance_to_goal(self) -> float:
        remaining = max(0, len(self._stages) - self._stage_index)
        return float(remaining * 8.0)

    def _traffic_light_state(self) -> str:
        stage = self._current_stage()
        if stage.region != "intersection":
            return "green"
        return str(self.env_config.get("traffic_light_state", "green"))

    def _risk_after_success(self, skill_id: str) -> float:
        if skill_id in {"obstacle_avoidance", "target_ball_clear", "pass_intersection"}:
            return 0.18
        if skill_id == "parking_control":
            return 0.03
        return max(0.05, self._risk_score - 0.04)

    def _normalize_action_name(self, action: Any) -> str:
        if isinstance(action, dict):
            return str(action.get("action_name") or action.get("skill_id") or "continuous_control")
        if isinstance(action, (list, tuple)):
            return "continuous_control"
        return str(action)

    def _normalize_control_mode(self, mode: Any) -> str:
        normalized = str(mode).strip().lower()
        if normalized in {"manual", "human", "keyboard"}:
            return "manual"
        if normalized in {"code", "auto", "algorithm", "policy", "planner"}:
            return "code"
        raise ValueError(f"Unsupported control mode: {mode}; expected 'code' or 'manual'.")

    @staticmethod
    def _clip_unit(value: float) -> float:
        return max(-1.0, min(1.0, float(value)))

    def _action_payload(self, action: Any, source: str) -> dict[str, Any]:
        if isinstance(action, dict):
            payload = dict(action)
        elif isinstance(action, (list, tuple)) and len(action) >= 2:
            payload = {"steer": float(action[0]), "motor": float(action[1])}
        else:
            payload = {"action_name": str(action)}
        payload.setdefault("source", source)
        return payload

    def _sync_from_unity_packet(self, packet: dict[str, Any]) -> None:
        task = dict(packet.get("task") or {})
        safety = dict(packet.get("safety") or {})
        phase = self._resolve_unity_phase(task)
        stage_index = self._stage_index_for_phase(phase)
        if stage_index is not None:
            self._stage_index = stage_index
        self._time_step = int(task.get("stepCount", self._time_step) or self._time_step)
        if "riskScore" in safety:
            self._risk_score = float(safety.get("riskScore") or 0.0)
        elif "risk_score" in safety:
            self._risk_score = float(safety.get("risk_score") or 0.0)
        if self._bool_from_any(task.get("taskSuccess") or task.get("task_success")):
            self._task_success = True
            self._done = True

    def _merge_unity_task_state(self, base: dict[str, Any]) -> dict[str, Any]:
        task = dict(self._last_unity_packet.get("task") or {})
        route = dict(self._last_unity_packet.get("route") or {})
        if not task:
            task = {}

        phase = self._resolve_unity_phase(task)
        route_stage = str(route.get("currentStageName") or "").strip()
        if route_stage:
            phase = self._resolve_unity_phase({"currentStageName": route_stage}) or phase
        stage_index = self._stage_index_for_phase(phase)
        task_success = self._bool_from_any(task.get("taskSuccess") or task.get("task_success"))

        live = dict(base)
        live["unity_scene_name"] = task.get("sceneName", task.get("scene_name"))
        live["unity_stage_name"] = route_stage or task.get("currentStageName", task.get("phase"))
        live["phase"] = "task_complete" if task_success else phase or base.get("phase")
        if stage_index is not None:
            live["phase_index"] = stage_index
        if "stepCount" in task:
            live["time_step"] = int(task.get("stepCount") or 0)
        if "maxStep" in task:
            live["max_episode_steps"] = int(task.get("maxStep") or 0)
        if "distanceToGoal" in task:
            live["distance_to_goal"] = float(task.get("distanceToGoal") or 0.0)
        if "goalLabel" in task:
            live["goal_label"] = task.get("goalLabel") or ""
        if "focusLabel" in task:
            live["focus_label"] = task.get("focusLabel") or ""
        if "stageActionHint" in task:
            live["stage_action_hint"] = task.get("stageActionHint") or ""
        if route.get("currentActionBehavior"):
            live["stage_action_hint"] = route.get("currentActionBehavior") or live.get("stage_action_hint", "")
        if route.get("currentGoalLabel"):
            live["goal_label"] = route.get("currentGoalLabel") or live.get("goal_label", "")
        if "goalPosition" in task:
            live["goal_position"] = self._vector3_to_list(task.get("goalPosition"))
        if "routeMode" in task:
            live["route_mode"] = task.get("routeMode") or live.get("route_mode")
        if "checkpointId" in task:
            live["checkpoint_id"] = task.get("checkpointId") or live.get("checkpoint_id")
        if "nextCheckpointId" in task:
            live["next_checkpoint_id"] = task.get("nextCheckpointId") or live.get("next_checkpoint_id")
        live["task_success"] = task_success or bool(base.get("task_success"))
        live["stage_complete"] = live["task_success"]
        return live

    def _merge_unity_safety_state(self, base: dict[str, Any]) -> dict[str, Any]:
        safety = dict(self._last_unity_packet.get("safety") or {})
        if not safety:
            return base

        live = dict(base)
        mapping = {
            "collision": "collision",
            "out_of_road": "out_of_road",
            "out_of_drivable_region": "out_of_drivable_region",
            "outOfDrivableRegion": "out_of_drivable_region",
            "red_light_violation": "red_light_violation",
            "redLightViolation": "red_light_violation",
            "front_blocked": "front_blocked",
            "frontBlocked": "front_blocked",
            "current_region_id": "current_region_id",
            "currentRegionId": "current_region_id",
            "current_region_type": "current_region_type",
            "currentRegionType": "current_region_type",
            "traffic_light_state": "traffic_light_state",
            "trafficLightState": "traffic_light_state",
            "min_obstacle_dist": "min_obstacle_dist",
            "minObstacleDistance": "min_obstacle_dist",
            "front_obstacle_dist": "front_obstacle_dist",
            "frontObstacleDistance": "front_obstacle_dist",
            "left_obstacle_dist": "left_obstacle_dist",
            "leftObstacleDistance": "left_obstacle_dist",
            "right_obstacle_dist": "right_obstacle_dist",
            "rightObstacleDistance": "right_obstacle_dist",
            "front_wall_dist": "front_wall_dist",
            "frontWallDistance": "front_wall_dist",
            "risk_score": "risk_score",
            "riskScore": "risk_score",
            "recommended_safe_action": "recommended_safe_action",
            "recommendedSafeAction": "recommended_safe_action",
            "waiting_area": "waiting_area",
            "waitingArea": "waiting_area",
            "before_light_area": "before_light_area",
            "beforeLightArea": "before_light_area",
            "corner_area": "corner_area",
            "cornerArea": "corner_area",
            "danger_area": "danger_area",
            "dangerArea": "danger_area",
        }
        for unity_key, local_key in mapping.items():
            if unity_key in safety:
                live[local_key] = safety[unity_key]
        if "out_of_drivable_region" in live:
            live["out_of_road"] = bool(live["out_of_drivable_region"])
        return live

    def _merge_unity_vehicle_state(self, base: dict[str, Any]) -> dict[str, Any]:
        vehicle = dict(self._last_unity_packet.get("vehicle") or {})
        if not vehicle:
            return base

        live = dict(base)
        mapping = {
            "position": "position",
            "forward": "forward",
            "eulerAngles": "euler_angles",
            "headingDeg": "heading_deg",
            "speed": "speed",
            "forwardSpeed": "forward_speed",
            "lateralSpeed": "lateral_speed",
            "angularSpeed": "angular_speed",
            "motorCommand": "motor_command",
            "steerCommand": "steer_command",
        }
        for unity_key, local_key in mapping.items():
            if unity_key not in vehicle:
                continue
            value = vehicle[unity_key]
            if unity_key in {"position", "forward", "eulerAngles"}:
                value = self._vector3_to_list(value)
            live[local_key] = value
        return live

    def _resolve_unity_phase(self, task: dict[str, Any]) -> str:
        raw = str(task.get("currentStageName") or task.get("phase") or "").strip()
        if not raw:
            return ""
        normalized = raw.lower().replace("-", "_")
        return UNITY_STAGE_ALIASES.get(raw, UNITY_STAGE_ALIASES.get(normalized, raw))

    def _stage_index_for_phase(self, phase: str) -> Optional[int]:
        if not phase:
            return None
        normalized = phase.lower().replace("-", "_")
        for index, stage in enumerate(self._stages):
            if normalized in {stage.phase, stage.skill_id, stage.region}:
                return index
        return None

    @staticmethod
    def _vector3_to_list(value: Any) -> Any:
        if isinstance(value, dict):
            return [float(value.get(axis, 0.0) or 0.0) for axis in ("x", "y", "z")]
        return value

    @staticmethod
    def _bool_from_any(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)


class _UnityStateChannel:
    def __init__(self, raw_channel_cls: Any, channel_id: str) -> None:
        self._channel = raw_channel_cls(uuid.UUID(channel_id))

    @property
    def channel(self) -> Any:
        return self._channel

    def pop_latest(self) -> dict[str, Any]:
        receiver = getattr(self._channel, "get_and_clear_received_messages", None)
        messages = list(receiver() if receiver is not None else [])
        if not messages:
            return {}
        raw_data = messages[-1]
        raw = raw_data.decode("utf-8") if isinstance(raw_data, (bytes, bytearray)) else str(raw_data)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}

    def send_json(self, payload: dict[str, Any]) -> None:
        sender = getattr(self._channel, "send_raw_data", None) or getattr(self._channel, "send_raw_bytes", None)
        if sender is not None:
            sender(bytearray(json.dumps(payload, ensure_ascii=False).encode("utf-8")))


class _MLAgentsUGVBackend:
    def __init__(self, scene_name: str, config: dict[str, Any]) -> None:
        try:
            from mlagents_envs.environment import UnityEnvironment  # type: ignore
            from mlagents_envs.side_channel.raw_bytes_channel import RawBytesChannel  # type: ignore
        except ImportError as exc:
            raise UnityBackendUnavailable(
                "Unity backend requires the mlagents_envs package. "
                "Use backend='scripted' for local smoke tests or install ML-Agents for Unity debugging."
            ) from exc

        self.scene_name = scene_name
        self.config = dict(config)
        self._UnityEnvironment = UnityEnvironment
        self._state_channel = _UnityStateChannel(RawBytesChannel, STATE_CHANNEL_ID)
        self._return_observation_arrays = bool(self.config.get("return_observation_arrays", False))
        self._env = UnityEnvironment(
            file_name=self.config.get("file_name") or self.config.get("unity_executable"),
            worker_id=int(self.config.get("worker_id", 0)),
            base_port=int(self.config.get("base_port", 5004) or 5004),
            seed=int(self.config.get("seed", 0) or 0),
            no_graphics=bool(self.config.get("no_graphics", False)),
            timeout_wait=int(self.config.get("timeout_wait", 60) or 60),
            side_channels=[self._state_channel.channel],
        )
        self._behavior_name: Optional[str] = None

    def reset(self, seed: Optional[int] = None, task_config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        self._state_channel.send_json({"type": "reset", "scene": self.scene_name, "seed": seed, "task_config": task_config or {}})
        self._env.reset()
        self._behavior_name = self._resolve_behavior_name()
        packet = self._state_channel.pop_latest()
        self._attach_mlagents_observations(packet)
        self._attach_visual_observation_arrays(packet)
        return packet

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        duration_steps = self._duration_steps(action)
        packet: dict[str, Any] = {}
        for _ in range(duration_steps):
            self._state_channel.send_json({"type": "action", "scene": self.scene_name, "payload": action})
            self._send_action(action)
            self._env.step()
            packet = self._state_channel.pop_latest() or packet
        self._attach_mlagents_observations(packet)
        self._attach_visual_observation_arrays(packet)
        reward = float(packet.get("reward", 0.0) or 0.0)
        task = dict(packet.get("task") or {})
        done = bool(packet.get("done", False) or task.get("task_success", False) or task.get("taskSuccess", False))
        info = {
            "event_type": "unity_step",
            "done_reason": "success" if done else "running",
            "backend": "unity",
            "packet_type": packet.get("packetType"),
            "duration_steps": duration_steps,
        }
        return packet, reward, done, info

    def send_control_mode(self, mode: str) -> None:
        self._state_channel.send_json({"type": "control_mode", "scene": self.scene_name, "mode": mode})

    def send_route(self, payload: dict[str, Any]) -> None:
        self._state_channel.send_json(payload)

    def close(self) -> None:
        self._env.close()

    def _attach_mlagents_observations(self, packet: dict[str, Any]) -> None:
        if not packet:
            return
        try:
            if self._behavior_name is None:
                self._behavior_name = self._resolve_behavior_name()
            decision_steps, terminal_steps = self._env.get_steps(self._behavior_name)
        except Exception as exc:
            packet["mlagents_observation_error"] = str(exc)
            return

        source_name = "decision" if len(decision_steps) > 0 else "terminal"
        source_steps = decision_steps if len(decision_steps) > 0 else terminal_steps
        observations = []
        for obs in getattr(source_steps, "obs", []):
            item = {
                "shape": list(getattr(obs, "shape", [])),
                "dtype": str(getattr(obs, "dtype", "")),
            }
            if self._return_observation_arrays:
                item["array"] = obs.copy()
            observations.append(item)

        packet["mlagents_observations"] = {
            "behavior_name": self._behavior_name,
            "step_type": source_name,
            "agent_count": int(len(source_steps)),
            "observations": observations,
        }

    def _attach_visual_observation_arrays(self, packet: dict[str, Any]) -> None:
        visuals = packet.get("visuals")
        if not isinstance(visuals, list):
            return

        for visual in visuals:
            if not isinstance(visual, dict):
                continue
            width = int(visual.get("width", 0) or 0)
            height = int(visual.get("height", 0) or 0)
            channels = int(visual.get("channels", 3) or 3)
            if width > 0 and height > 0:
                visual.setdefault("shape", [height, width, channels])
            if not self._return_observation_arrays:
                continue

            encoded = visual.get("data")
            if not encoded:
                continue
            try:
                from PIL import Image  # type: ignore
                import numpy as np

                image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
                array = np.asarray(image)
                visual["array"] = array
                visual["shape"] = list(array.shape)
                visual["dtype"] = str(array.dtype)
            except Exception as exc:
                visual["decode_error"] = str(exc)

    def _resolve_behavior_name(self) -> str:
        behavior_specs = getattr(self._env, "behavior_specs")
        names = list(behavior_specs.keys())
        if not names:
            raise UnityBackendUnavailable("No ML-Agents behavior was found in the Unity scene.")
        return names[0]

    def _send_action(self, action: Any) -> None:
        if self._behavior_name is None:
            self._behavior_name = self._resolve_behavior_name()
        try:
            import numpy as np
            from mlagents_envs.base_env import ActionTuple  # type: ignore
        except ImportError:
            return

        spec = self._env.behavior_specs[self._behavior_name]
        action_spec = spec.action_spec
        steer, motor = self._continuous_action(action)
        decision_steps, _ = self._env.get_steps(self._behavior_name)
        agent_count = len(decision_steps)
        if agent_count <= 0:
            return
        if getattr(action_spec, "continuous_size", 0) > 0:
            continuous_size = int(getattr(action_spec, "continuous_size", 2) or 2)
            continuous = np.zeros((agent_count, continuous_size), dtype=np.float32)
            continuous[:, 0] = steer
            if continuous_size > 1:
                continuous[:, 1] = motor
            action_tuple = ActionTuple(continuous=continuous)
        else:
            steer_index = 0 if steer < -0.2 else 2 if steer > 0.2 else 1
            motor_index = 0 if motor < -0.2 else 2 if motor > 0.2 else 1
            discrete = np.array([[steer_index, motor_index]] * agent_count, dtype=np.int32)
            action_tuple = ActionTuple(discrete=discrete)
        self._env.set_actions(self._behavior_name, action_tuple)

    def _continuous_action(self, action: Any) -> tuple[float, float]:
        if isinstance(action, dict):
            if "steer" in action or "motor" in action:
                return float(action.get("steer", 0.0)), float(action.get("motor", 0.0))
            skill_id = str(action.get("skill_id", ""))
            if skill_id == "safe_stop":
                return 0.0, 0.0
            if skill_id in {"parking_control", "search_parking_slot"}:
                return 0.15, 0.25
            return 0.0, 0.45
        if isinstance(action, (list, tuple)) and len(action) >= 2:
            return float(action[0]), float(action[1])
        return 0.0, 0.4

    @staticmethod
    def _duration_steps(action: Any) -> int:
        if isinstance(action, dict):
            return max(1, int(action.get("duration_steps", 1) or 1))
        return 1


class _UdpUGVBackend:
    def __init__(self, scene_name: str, config: dict[str, Any]) -> None:
        self.scene_name = scene_name
        self.config = dict(config)
        self.host = str(self.config.get("udp_host", "127.0.0.1"))
        self.port = int(self.config.get("udp_port", 5055) or 5055)
        self.step_seconds = float(self.config.get("udp_step_seconds", 0.05) or 0.05)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def reset(self, seed: Optional[int] = None, task_config: Optional[dict[str, Any]] = None) -> None:
        self._send({"type": "reset", "scene": self.scene_name, "seed": seed, "task_config": task_config or {}})

    def step(self, action: Any) -> None:
        steer, motor = self._continuous_action(action)
        duration_steps = self._duration_steps(action)
        hold_seconds = max(0.08, duration_steps * self.step_seconds)
        payload = {
            "type": "action",
            "scene": self.scene_name,
            "steer": steer,
            "motor": motor,
            "duration": hold_seconds,
            "label": self._label(action),
        }
        self._send(payload)
        time.sleep(min(hold_seconds, 0.12))

    def send_control_mode(self, mode: str) -> None:
        self._send({"type": "control_mode", "scene": self.scene_name, "mode": mode})

    def send_route(self, payload: dict[str, Any]) -> None:
        self._send(payload)

    def close(self) -> None:
        self._send({"type": "action", "scene": self.scene_name, "steer": 0.0, "motor": 0.0, "duration": 0.2, "label": "close"})
        self._socket.close()

    def _send(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._socket.sendto(data, (self.host, self.port))

    def _continuous_action(self, action: Any) -> tuple[float, float]:
        if isinstance(action, dict):
            if "steer" in action or "motor" in action:
                return float(action.get("steer", 0.0)), float(action.get("motor", 0.0))
            skill_id = str(action.get("skill_id", ""))
            if skill_id == "safe_stop":
                return 0.0, 0.0
            if skill_id in {"parking_control", "search_parking_slot"}:
                return 0.15, 0.25
            if skill_id in {"obstacle_avoidance", "route_switch"}:
                return 0.35, 0.35
            return 0.0, 0.45
        if isinstance(action, (list, tuple)) and len(action) >= 2:
            return float(action[0]), float(action[1])
        return 0.0, 0.4

    @staticmethod
    def _duration_steps(action: Any) -> int:
        if isinstance(action, dict):
            return max(1, int(action.get("duration_steps", 1) or 1))
        return 1

    @staticmethod
    def _label(action: Any) -> str:
        if isinstance(action, dict):
            return str(action.get("action_name") or action.get("skill_id") or "continuous_control")
        return "continuous_control"
