"""Highway 高速驾驶任务的统一环境封装。"""

from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Any, Iterable, Optional, Union

from .base_env import BaseExperimentEnv


# 默认动作编号来自 highway-env 的 5 个离散高层动作。
ACTION_ID_TO_NAME: dict[int, str] = {
    0: "lane_left",
    1: "idle",
    2: "lane_right",
    3: "faster",
    4: "slower",
}

ACTION_NAME_TO_ID: dict[str, int] = {name: action_id for action_id, name in ACTION_ID_TO_NAME.items()}


class _ScriptedHighwayBackend:
    """无 highway-env 依赖时使用的轻量交通后端。"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.rng = random.Random()
        self.lanes_count = int(config.get("lanes_count", 4))
        self.vehicles_count = int(config.get("vehicles_count", 50))
        self.duration = int(config.get("duration", config.get("max_episode_steps", 100)))
        self.safe_distance = float(config.get("safe_distance", 10.0))
        self.time_step = 0
        self.lane_id = min(self.lanes_count - 1, max(0, self.lanes_count // 2))
        self.speed = float(config.get("initial_speed", 25.0))
        self.distance_travelled = 0.0
        self.collision = False
        self.vehicles: list[dict[str, Any]] = []

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """重置脚本交通流。"""
        if seed is not None:
            self.rng.seed(seed)
        options = options or {}
        self.lanes_count = int(options.get("lanes_count", self.config.get("lanes_count", 4)))
        self.vehicles_count = int(options.get("vehicles_count", self.config.get("vehicles_count", 50)))
        self.duration = int(options.get("duration", options.get("max_episode_steps", self.config.get("duration", 100))))
        self.time_step = 0
        self.lane_id = min(self.lanes_count - 1, max(0, self.lanes_count // 2))
        self.speed = float(options.get("initial_speed", self.config.get("initial_speed", 25.0)))
        self.distance_travelled = 0.0
        self.collision = False
        self.vehicles = self._spawn_vehicles()
        return self._build_obs(), self._build_info(action_name="reset", lane_change=False, done_reason="running")

    def step(self, action_id: int) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """执行离散驾驶动作并推进脚本交通流。"""
        self.time_step += 1
        old_lane = self.lane_id
        action_name = ACTION_ID_TO_NAME[action_id]

        if action_name == "lane_left":
            self.lane_id = max(0, self.lane_id - 1)
        elif action_name == "lane_right":
            self.lane_id = min(self.lanes_count - 1, self.lane_id + 1)
        elif action_name == "faster":
            self.speed = min(float(self.config.get("max_speed", 35.0)), self.speed + 2.0)
        elif action_name == "slower":
            self.speed = max(float(self.config.get("min_speed", 10.0)), self.speed - 2.0)

        lane_change = old_lane != self.lane_id
        self.distance_travelled += self.speed
        self._advance_traffic()
        self.collision = self._detect_collision()

        done_reason = "running"
        if self.collision:
            done_reason = "collision"
        elif self.time_step >= self.duration:
            done_reason = "max_episode_steps"
        done = done_reason != "running"

        # 奖励只用于脚本后端跑通流程；真实后端保留原始奖励。
        reward = self.speed / max(float(self.config.get("target_speed", 30.0)), 1.0)
        if lane_change:
            reward -= 0.05
        if self.collision:
            reward -= 5.0

        info = self._build_info(action_name=action_name, lane_change=lane_change, done_reason=done_reason)
        return self._build_obs(), reward, done, info

    def close(self) -> None:
        """脚本后端没有外部连接。"""
        return None

    def _spawn_vehicles(self) -> list[dict[str, Any]]:
        """生成约 50 辆干扰车辆的相对位置和速度。"""
        vehicles: list[dict[str, Any]] = []
        for index in range(self.vehicles_count):
            lane_id = index % self.lanes_count
            direction = -1 if index % 5 == 0 else 1
            base_distance = 18.0 + index * 7.0
            rel_x = direction * (base_distance + self.rng.uniform(-2.0, 2.0))
            # 避免初始状态同车道过近，便于测试稳定通过。
            if lane_id == self.lane_id and abs(rel_x) < self.safe_distance * 1.5:
                rel_x += self.safe_distance * 2.0
            vehicles.append(
                {
                    "id": f"veh_{index:02d}",
                    "lane_id": lane_id,
                    "rel_x": rel_x,
                    "speed": self.rng.uniform(18.0, 32.0),
                }
            )
        return vehicles

    def _advance_traffic(self) -> None:
        """根据相对速度更新周围车辆位置。"""
        for vehicle in self.vehicles:
            vehicle["rel_x"] += float(vehicle["speed"]) - self.speed
            if vehicle["rel_x"] < -150.0:
                vehicle["rel_x"] = 300.0 + self.rng.uniform(0.0, 80.0)
                vehicle["lane_id"] = self.rng.randrange(self.lanes_count)
                vehicle["speed"] = self.rng.uniform(18.0, 32.0)

    def _detect_collision(self) -> bool:
        """检测同车道近距离碰撞。"""
        for vehicle in self.vehicles:
            if vehicle["lane_id"] == self.lane_id and abs(float(vehicle["rel_x"])) < 2.5:
                return True
        return False

    def _nearby_vehicles(self) -> list[dict[str, Any]]:
        """返回距离自车最近的若干车辆。"""
        sorted_vehicles = sorted(self.vehicles, key=lambda item: abs(float(item["rel_x"])))
        return [
            {
                "id": vehicle["id"],
                "lane_id": vehicle["lane_id"],
                "distance": float(vehicle["rel_x"]),
                "speed": float(vehicle["speed"]),
                "relative_speed": float(vehicle["speed"]) - self.speed,
            }
            for vehicle in sorted_vehicles[:12]
        ]

    def _build_obs(self) -> dict[str, Any]:
        """构造 highway wrapper 可解析的脚本观测。"""
        return {
            "ego_state": {
                "lane_id": self.lane_id,
                "speed": self.speed,
                "position": [self.distance_travelled, float(self.lane_id)],
                "crashed": self.collision,
            },
            "nearby_vehicles": self._nearby_vehicles(),
        }

    def _build_info(self, action_name: str, lane_change: bool, done_reason: str) -> dict[str, Any]:
        """构造统一事件信息。"""
        return {
            "action_name": action_name,
            "lane_id": self.lane_id,
            "speed": self.speed,
            "collision": self.collision,
            "lane_change": lane_change,
            "time_step": self.time_step,
            "distance_travelled": self.distance_travelled,
            "task_success": False,
            "done_reason": done_reason,
        }


class HighwayEnvWrapper(BaseExperimentEnv):
    """Highway 场景统一接口。"""

    def __init__(self, env_config: Optional[dict[str, Any]] = None, backend_env: Optional[Any] = None) -> None:
        self.env_config = {
            "backend": "scripted",
            "env_id": "highway-v0",
            "render_mode": "rgb_array",
            "lanes_count": 4,
            "vehicles_count": 50,
            "duration": 100,
            "max_episode_steps": 100,
            "target_speed": 30.0,
            "safe_distance": 10.0,
            **(env_config or {}),
        }
        self.action_id_to_name = self._load_action_mapping(self.env_config)
        self.action_name_to_id = {name: action_id for action_id, name in self.action_id_to_name.items()}
        self.task_config: dict[str, Any] = {}
        self._last_obs: Optional[dict[str, Any]] = None
        self._raw_obs: Any = None
        self._last_info: dict[str, Any] = {"done_reason": "not_started"}
        self._ego_state: dict[str, Any] = {}
        self._nearby_vehicles: list[dict[str, Any]] = []
        self._last_action_name: Optional[str] = None
        self._waiting_for_high_level_decision = False
        self._elapsed_wait_steps = 0
        self._step_count = 0
        self._backend = backend_env or self._create_backend()
        self._configure_real_backend_if_possible()

    def reset(self, seed: Optional[int] = None, task_config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """重置 Highway 环境，返回统一观测。"""
        self.task_config = {
            "scene_name": "highway",
            "lanes_count": self.env_config.get("lanes_count", 4),
            "vehicles_count": self.env_config.get("vehicles_count", 50),
            "max_episode_steps": self.env_config.get("max_episode_steps", 100),
            "duration": self.env_config.get("duration", 100),
            **(task_config or {}),
        }
        raw_obs, info = self._call_reset(seed=seed, task_config=self.task_config)
        self._last_action_name = None
        self._waiting_for_high_level_decision = False
        self._elapsed_wait_steps = 0
        self._step_count = 0
        obs = self._normalize_observation(raw_obs, info)
        self._last_obs = obs
        return obs

    def step(self, action: Union[int, str]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """执行 Highway 原始离散动作。"""
        self._ensure_started()
        action_id = self._coerce_action_id(action)
        action_name = self.action_id_to_name[action_id]
        result = self._backend.step(action_id)
        raw_obs, reward, done, info = self._unpack_step_result(result)
        self._step_count += 1
        info = {**(info or {}), "action_name": action_name}
        info.setdefault("time_step", self._step_count)
        if done and "done_reason" not in info:
            info["done_reason"] = "terminated"
        obs = self._normalize_observation(raw_obs, info)
        self._last_obs = obs
        self._last_action_name = action_name
        return obs, float(reward), bool(done), info

    def apply_skill(
        self, skill_id: str, skill_args: Optional[dict[str, Any]] = None
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """将高层技能策略映射为 Highway 离散动作或短动作序列。"""
        self._ensure_started()
        skill_args = skill_args or {}
        skill_id = skill_id.strip()
        max_steps = int(skill_args.get("max_steps", 1))

        if skill_id == "basic_action_sequence":
            action_ids = self._resolve_action_sequence(skill_args)
        elif skill_id == "keep_lane":
            action_ids = [self.action_name_to_id["idle"]]
        elif skill_id == "lane_left":
            action_ids = [self.action_name_to_id["lane_left"]]
        elif skill_id == "lane_right":
            action_ids = [self.action_name_to_id["lane_right"]]
        elif skill_id == "speed_up":
            action_ids = [self.action_name_to_id["faster"]]
        elif skill_id == "slow_down":
            action_ids = [self.action_name_to_id["slower"]]
        elif skill_id == "safe_follow":
            action_ids = self._safe_follow_actions(max_steps=max_steps)
        else:
            raise KeyError(f"Highway 未知技能策略：{skill_id}")

        obs, total_reward, done, action_trace = self._run_action_sequence(action_ids, max_steps=max_steps)
        info = {
            **self.get_event_info(),
            "skill_id": skill_id,
            "skill_args": deepcopy(skill_args),
            "skill_steps": len(action_trace),
            "action_sequence": action_trace,
            "skill_reward": total_reward,
            "task_success": self.get_task_state().get("task_success", False),
            "done_reason": self.get_event_info().get("done_reason", "running") if done else "running",
        }
        return obs, total_reward, done, info

    def get_task_state(self) -> dict[str, Any]:
        """获取车道、速度、步数和当前驾驶目标。"""
        max_steps = int(self.task_config.get("max_episode_steps", self.env_config.get("max_episode_steps", 100)))
        time_step = int(self._last_info.get("time_step", self._step_count) or self._step_count)
        lane_id = self._extract_lane_id()
        speed = float(self._ego_state.get("speed", self._last_info.get("speed", 0.0)) or 0.0)
        task_success = bool(self._last_info.get("task_success", False))
        return {
            "scene_name": "highway",
            "phase": "driving",
            "time_step": time_step,
            "max_episode_steps": max_steps,
            "lane_id": lane_id,
            "target_lane_id": self.task_config.get("target_lane_id"),
            "speed": speed,
            "target_speed": float(self.env_config.get("target_speed", 30.0)),
            "distance_travelled": float(self._last_info.get("distance_travelled", 0.0) or 0.0),
            "task_success": task_success,
        }

    def get_domain_knowledge(self) -> dict[str, Any]:
        """返回 Highway 车道拓扑、规则和技能说明。"""
        lanes_count = int(self.task_config.get("lanes_count", self.env_config.get("lanes_count", 4)))
        lanes = list(range(lanes_count))
        adjacent_lanes = {
            lane: [candidate for candidate in (lane - 1, lane + 1) if 0 <= candidate < lanes_count]
            for lane in lanes
        }
        return {
            "environment_topology": {
                "lanes": lanes,
                "adjacent_lanes": adjacent_lanes,
            },
            "region_rules": {
                "leftmost_lane": ["no_left_lane_change"],
                "rightmost_lane": ["no_right_lane_change"],
                "dense_traffic": ["safe_distance_required"],
                "front_vehicle_close": ["slow_down_preferred"],
            },
            "skill_policies": {
                "keep_lane": "保持当前车道行驶",
                "lane_left": "在左侧车道安全时向左变道",
                "lane_right": "在右侧车道安全时向右变道",
                "speed_up": "在前方安全时加速",
                "slow_down": "在前车较近或风险较高时减速",
                "safe_follow": "根据前车距离调整速度",
                "basic_action_sequence": "按给定原始动作序列执行",
            },
            "action_id_to_name": deepcopy(self.action_id_to_name),
        }

    def get_safety_state(self) -> dict[str, Any]:
        """获取碰撞风险、车距、相对速度和推荐安全动作。"""
        lane_id = self._extract_lane_id()
        speed = float(self._ego_state.get("speed", self._last_info.get("speed", 0.0)) or 0.0)
        collision = bool(self._last_info.get("collision", self._ego_state.get("crashed", False)))
        front_vehicle = self._closest_vehicle(lane_id=lane_id, ahead=True)
        rear_vehicle = self._closest_vehicle(lane_id=lane_id, ahead=False)
        front_dist = self._vehicle_distance(front_vehicle)
        rear_dist = abs(self._vehicle_distance(rear_vehicle)) if rear_vehicle else None
        relative_speed_front = None
        if front_vehicle is not None:
            relative_speed_front = float(front_vehicle.get("relative_speed", front_vehicle.get("speed", 0.0) - speed))

        ttc = math.inf
        if front_dist is not None and relative_speed_front is not None:
            closing_speed = max(0.0, -relative_speed_front)
            if closing_speed > 1e-6:
                ttc = max(0.0, front_dist / closing_speed)

        left_lane_safe = self._lane_safe(lane_id - 1)
        right_lane_safe = self._lane_safe(lane_id + 1)
        risk_score = self._compute_risk_score(collision, front_dist, ttc)
        recommended = "slow_down" if risk_score >= float(self.env_config.get("risk_slowdown_threshold", 0.35)) else "keep"

        return {
            "collision": collision,
            "front_vehicle_dist": front_dist,
            "rear_vehicle_dist": rear_dist,
            "left_lane_safe": left_lane_safe,
            "right_lane_safe": right_lane_safe,
            "relative_speed_front": relative_speed_front,
            "time_to_collision": None if math.isinf(ttc) else ttc,
            "speed": speed,
            "risk_score": risk_score,
            "recommended_safe_action": recommended,
        }

    def get_event_info(self) -> dict[str, Any]:
        """获取碰撞、变道、速度变化和回合结束原因。"""
        return {
            "event_type": self._last_info.get("event_type", self._infer_event_type()),
            "action_name": self._last_info.get("action_name"),
            "lane_change": bool(self._last_info.get("lane_change", False)),
            "collision": bool(self._last_info.get("collision", False)),
            "reward": self._last_info.get("reward"),
            "done_reason": self._last_info.get("done_reason", "running"),
        }

    def get_compensation_candidates(self) -> list[str]:
        """返回第三章等待高层决策期间可用的补偿动作行为。"""
        return ["keep_lane", "safe_follow", "slow_down", "lane_left", "lane_right"]

    def close(self) -> None:
        """结束环境会话。"""
        if hasattr(self._backend, "close"):
            self._backend.close()

    def render_rgb(self) -> Optional[Any]:
        """返回真实 highway-env 的 RGB 渲染帧；脚本后端返回 None。"""
        if isinstance(self._backend, _ScriptedHighwayBackend):
            return None
        render = getattr(self._backend, "render", None)
        if not callable(render):
            return None
        try:
            return render()
        except Exception:
            return None

    def _create_backend(self) -> Any:
        """根据配置创建真实 highway-env 后端或脚本后端。"""
        backend = str(self.env_config.get("backend", "scripted")).lower()
        if backend == "scripted":
            return _ScriptedHighwayBackend(self.env_config)
        if backend in {"gym", "gymnasium", "highway_env", "highway-env"}:
            return self._make_gym_backend()
        raise ValueError(f"不支持的 Highway backend：{backend}")

    def _make_gym_backend(self) -> Any:
        """创建 Gym/Gymnasium highway-env 后端。"""
        env_id = str(self.env_config.get("env_id", "highway-v0"))
        try:
            import highway_env  # noqa: F401  # type: ignore
        except ImportError:
            # 有些项目会自行注册 highway-v0；未安装 highway_env 时仍尝试 gym.make。
            pass
        try:
            import gymnasium as gym  # type: ignore
        except ImportError:
            try:
                import gym  # type: ignore
            except ImportError as exc:
                raise ImportError("未安装 gymnasium/gym，无法创建 Highway 真实后端。") from exc
        render_mode = self.env_config.get("render_mode")
        if render_mode is not None:
            try:
                return gym.make(env_id, render_mode=render_mode)
            except TypeError:
                pass
        return gym.make(env_id)

    def _configure_real_backend_if_possible(self) -> None:
        """将论文实验默认参数写入 highway-env 配置。"""
        if isinstance(self._backend, _ScriptedHighwayBackend):
            return
        unwrapped = getattr(self._backend, "unwrapped", self._backend)
        configure = getattr(unwrapped, "configure", None)
        if not callable(configure):
            return
        configure(
            {
                "lanes_count": int(self.env_config.get("lanes_count", 4)),
                "vehicles_count": int(self.env_config.get("vehicles_count", 50)),
                "duration": int(self.env_config.get("duration", 100)),
                "action": {"type": "DiscreteMetaAction"},
                **{
                    key: self.env_config[key]
                    for key in (
                        "vehicles_density",
                        "controlled_vehicles",
                        "policy_frequency",
                        "simulation_frequency",
                        "reward_speed_range",
                        "collision_reward",
                        "right_lane_reward",
                        "high_speed_reward",
                    )
                    if key in self.env_config
                },
            }
        )

    def _call_reset(
        self, seed: Optional[int], task_config: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        """兼容脚本后端、Gymnasium 和旧 Gym 的 reset。"""
        try:
            result = self._backend.reset(seed=seed, options=task_config)
        except TypeError:
            if seed is not None and hasattr(self._backend, "seed"):
                self._backend.seed(seed)
            result = self._backend.reset()

        if isinstance(result, tuple) and len(result) == 2:
            raw_obs, info = result
            return raw_obs, dict(info or {})
        return result, {}

    def _unpack_step_result(self, result: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        """兼容 Gymnasium 五元组和旧 Gym 四元组。"""
        if not isinstance(result, tuple):
            raise TypeError("环境 step 必须返回 tuple。")
        if len(result) == 5:
            raw_obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
            info = dict(info or {})
            if done and "done_reason" not in info:
                info["done_reason"] = "terminated" if terminated else "truncated"
            info["reward"] = float(reward)
            return raw_obs, float(reward), done, info
        if len(result) == 4:
            raw_obs, reward, done, info = result
            info = dict(info or {})
            info["reward"] = float(reward)
            return raw_obs, float(reward), bool(done), info
        raise ValueError("环境 step 返回值长度必须为 4 或 5。")

    def _normalize_observation(self, raw_obs: Any, info: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """把真实或脚本 Highway 输出整理成统一字段。"""
        info = dict(info or {})
        raw_dict = raw_obs if isinstance(raw_obs, dict) else {}
        self._raw_obs = raw_obs
        self._ego_state = self._extract_ego_state(raw_dict, info)
        info.setdefault("time_step", self._step_count)
        info.setdefault("collision", bool(self._ego_state.get("crashed", False)))
        self._nearby_vehicles = self._extract_nearby_vehicles(raw_dict, info)
        self._last_info = {**self._last_info, **info}
        task_state = self.get_task_state()
        safety_state = self.get_safety_state()
        event_info = self.get_event_info()
        latency_state = {
            "time_step": task_state["time_step"],
            "last_high_level_action": self._last_action_name,
            "waiting_for_high_level_decision": self._waiting_for_high_level_decision,
            "elapsed_wait_steps": self._elapsed_wait_steps,
            "candidate_compensation_actions": self.get_compensation_candidates(),
        }
        obs = {
            "raw_obs": raw_obs,
            "ego_state": deepcopy(self._ego_state),
            "nearby_vehicles": deepcopy(self._nearby_vehicles),
            "lane_id": task_state["lane_id"],
            "speed": task_state["speed"],
            "time_step": task_state["time_step"],
            "collision": safety_state["collision"],
            "reward": info.get("reward"),
            "done_reason": event_info["done_reason"],
            "task_state": task_state,
            "safety_state": safety_state,
            "event_info": event_info,
            "latency_state": latency_state,
        }
        return obs

    def _extract_ego_state(self, raw_dict: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        """优先从 info/raw_obs 提取自车状态，否则从 highway-env 对象读取。"""
        ego_state = dict(raw_dict.get("ego_state") or info.get("ego_state") or {})
        if ego_state:
            return ego_state

        vehicle = getattr(getattr(self._backend, "unwrapped", self._backend), "vehicle", None)
        if vehicle is not None:
            lane_index = getattr(vehicle, "lane_index", None)
            lane_id = lane_index[-1] if isinstance(lane_index, tuple) and lane_index else info.get("lane_id", 0)
            position = getattr(vehicle, "position", [0.0, 0.0])
            return {
                "lane_id": lane_id,
                "speed": float(getattr(vehicle, "speed", info.get("speed", 0.0))),
                "position": self._to_list(position),
                "crashed": bool(getattr(vehicle, "crashed", info.get("collision", False))),
            }

        return {
            "lane_id": int(info.get("lane_id", 0) or 0),
            "speed": float(info.get("speed", 0.0) or 0.0),
            "position": [float(info.get("distance_travelled", 0.0) or 0.0), 0.0],
            "crashed": bool(info.get("collision", False)),
        }

    def _extract_nearby_vehicles(self, raw_dict: dict[str, Any], info: dict[str, Any]) -> list[dict[str, Any]]:
        """从 raw_obs/info/highway-env 对象提取周围车辆列表。"""
        raw_nearby = raw_dict.get("nearby_vehicles") or info.get("nearby_vehicles")
        if isinstance(raw_nearby, list):
            return [dict(vehicle) for vehicle in raw_nearby]

        unwrapped = getattr(self._backend, "unwrapped", self._backend)
        road = getattr(unwrapped, "road", None)
        ego_vehicle = getattr(unwrapped, "vehicle", None)
        road_vehicles = getattr(road, "vehicles", None)
        if road_vehicles is not None and ego_vehicle is not None:
            ego_position = self._to_list(getattr(ego_vehicle, "position", [0.0, 0.0]))
            ego_speed = float(getattr(ego_vehicle, "speed", 0.0))
            nearby: list[dict[str, Any]] = []
            for index, vehicle in enumerate(road_vehicles):
                if vehicle is ego_vehicle:
                    continue
                position = self._to_list(getattr(vehicle, "position", [0.0, 0.0]))
                lane_index = getattr(vehicle, "lane_index", None)
                lane_id = lane_index[-1] if isinstance(lane_index, tuple) and lane_index else None
                distance = float(position[0] - ego_position[0]) if position else 0.0
                speed = float(getattr(vehicle, "speed", 0.0))
                nearby.append(
                    {
                        "id": f"veh_{index}",
                        "lane_id": lane_id,
                        "distance": distance,
                        "position": position,
                        "speed": speed,
                        "relative_speed": speed - ego_speed,
                    }
                )
            return sorted(nearby, key=lambda item: abs(float(item.get("distance", 0.0))))[:12]

        return self._nearby_from_array_like(raw_dict)

    def _nearby_from_array_like(self, raw_dict: dict[str, Any]) -> list[dict[str, Any]]:
        """尽量从数组观测中提取车辆信息，字段不足时返回空列表。"""
        raw_obs = raw_dict or self._raw_obs
        rows = raw_obs
        if hasattr(rows, "tolist"):
            rows = rows.tolist()
        if not isinstance(rows, list) or not rows:
            return []
        nearby: list[dict[str, Any]] = []
        for index, row in enumerate(rows[1:13], start=1):
            if not isinstance(row, (list, tuple)) or len(row) < 5:
                continue
            presence = float(row[0])
            if presence <= 0.0:
                continue
            nearby.append(
                {
                    "id": f"obs_{index}",
                    "lane_id": None,
                    "distance": float(row[1]),
                    "speed": float(row[3]),
                    "relative_speed": float(row[3]) - float(rows[0][3] if isinstance(rows[0], (list, tuple)) and len(rows[0]) > 3 else 0.0),
                }
            )
        return nearby

    def _safe_follow_actions(self, max_steps: int) -> list[int]:
        """根据前车距离选择保持或减速补偿动作。"""
        safety_state = self.get_safety_state()
        front_dist = safety_state.get("front_vehicle_dist")
        safe_distance = float(self.env_config.get("safe_distance", 10.0))
        action_name = "slower" if front_dist is not None and float(front_dist) < safe_distance else "idle"
        return [self.action_name_to_id[action_name] for _ in range(max(1, max_steps))]

    def _run_action_sequence(
        self, action_ids: Iterable[int], max_steps: int
    ) -> tuple[dict[str, Any], float, bool, list[str]]:
        """执行一段离散动作序列并累计奖励。"""
        total_reward = 0.0
        done = False
        action_trace: list[str] = []
        obs = self._last_obs or {}
        for action_id in list(action_ids)[:max_steps]:
            obs, reward, done, info = self.step(action_id)
            total_reward += reward
            action_trace.append(info.get("action_name", self.action_id_to_name[action_id]))
            if done:
                break
        return obs, total_reward, done, action_trace

    def _resolve_action_sequence(self, skill_args: dict[str, Any]) -> list[int]:
        """解析 basic_action_sequence 的动作编号或动作名称。"""
        raw_sequence = skill_args.get("action_ids", skill_args.get("actions", []))
        return [self._coerce_action_id(action) for action in raw_sequence]

    def _closest_vehicle(self, lane_id: int, ahead: bool) -> Optional[dict[str, Any]]:
        """查找同车道前车或后车。"""
        candidates = []
        for vehicle in self._nearby_vehicles:
            vehicle_lane = vehicle.get("lane_id")
            if vehicle_lane is not None and int(vehicle_lane) != lane_id:
                continue
            distance = self._vehicle_distance(vehicle)
            if distance is None:
                continue
            if ahead and distance > 0:
                candidates.append((distance, vehicle))
            if not ahead and distance < 0:
                candidates.append((abs(distance), vehicle))
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: item[0])[0][1]

    def _lane_safe(self, lane_id: int) -> bool:
        """判断相邻车道是否存在以及是否满足安全距离。"""
        lanes_count = int(self.task_config.get("lanes_count", self.env_config.get("lanes_count", 4)))
        if lane_id < 0 or lane_id >= lanes_count:
            return False
        safe_distance = float(self.env_config.get("safe_distance", 10.0))
        for vehicle in self._nearby_vehicles:
            vehicle_lane = vehicle.get("lane_id")
            if vehicle_lane is None or int(vehicle_lane) != lane_id:
                continue
            distance = self._vehicle_distance(vehicle)
            if distance is not None and abs(distance) < safe_distance:
                return False
        return True

    def _compute_risk_score(self, collision: bool, front_dist: Optional[float], ttc: float) -> float:
        """把碰撞、前车距离和 TTC 压缩成 0 到 1 的风险分数。"""
        if collision:
            return 1.0
        risk = 0.0
        safe_distance = float(self.env_config.get("safe_distance", 10.0))
        if front_dist is not None:
            risk = max(risk, max(0.0, (safe_distance - front_dist) / max(safe_distance, 1e-6)))
        if not math.isinf(ttc):
            ttc_threshold = float(self.env_config.get("ttc_threshold", 4.0))
            risk = max(risk, max(0.0, (ttc_threshold - ttc) / max(ttc_threshold, 1e-6)))
        return min(1.0, risk)

    def _infer_event_type(self) -> str:
        """根据最近 info 推断事件类型。"""
        if bool(self._last_info.get("collision", False)):
            return "collision"
        if bool(self._last_info.get("lane_change", False)):
            return "lane_change"
        action_name = self._last_info.get("action_name")
        if action_name in {"faster", "slower"}:
            return "speed_change"
        return "driving"

    def _vehicle_distance(self, vehicle: Optional[dict[str, Any]]) -> Optional[float]:
        """统一读取车辆相对纵向距离。"""
        if vehicle is None:
            return None
        if "distance" in vehicle:
            return float(vehicle["distance"])
        position = vehicle.get("position")
        ego_position = self._ego_state.get("position")
        if position is not None and ego_position is not None:
            return float(position[0] - ego_position[0])
        return None

    def _extract_lane_id(self) -> int:
        """从自车状态或 info 中读取车道编号。"""
        lane_id = self._ego_state.get("lane_id", self._last_info.get("lane_id", 0))
        return int(lane_id or 0)

    def _coerce_action_id(self, action: Union[int, str]) -> int:
        """把动作名称或动作编号统一转成动作编号。"""
        if isinstance(action, str):
            if action not in self.action_name_to_id:
                raise KeyError(f"Highway 未知动作名称：{action}")
            return self.action_name_to_id[action]
        action_id = int(action)
        if action_id not in self.action_id_to_name:
            raise KeyError(f"Highway 未知动作编号：{action_id}")
        return action_id

    def _ensure_started(self) -> None:
        """避免上层在 reset 前直接 step。"""
        if self._last_obs is None:
            raise RuntimeError("请先调用 reset() 再执行 step() 或 apply_skill()。")

    @staticmethod
    def _load_action_mapping(env_config: dict[str, Any]) -> dict[int, str]:
        """允许项目真实 Highway 动作编号覆盖默认映射。"""
        raw_mapping = env_config.get("action_id_to_name")
        if not raw_mapping:
            return dict(ACTION_ID_TO_NAME)
        return {int(action_id): str(name) for action_id, name in raw_mapping.items()}

    @staticmethod
    def _to_list(value: Any) -> list[float]:
        """把 numpy/list 等位置字段转为普通列表。"""
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return [float(item) for item in value]
        return [0.0, 0.0]
