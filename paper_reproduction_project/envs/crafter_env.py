"""Crafter 制作石镐任务的统一环境封装。"""

from __future__ import annotations

import importlib
import random
import sys
from collections import deque
from copy import deepcopy
from typing import Any, Iterable, Optional, Union

from .base_env import BaseExperimentEnv


# 说明文档中的简化动作编号，用于脚本后端和统一接口测试。
ACTION_ID_TO_NAME: dict[int, str] = {
    0: "noop",
    1: "move_left",
    2: "move_right",
    3: "move_up",
    4: "move_down",
    5: "interact",
    6: "place_table",
    7: "make_wood_pickaxe",
    8: "make_stone_pickaxe",
}

# Gym Crafter 原生动作表来自 crafter/data.yaml。
GYM_ACTION_ID_TO_NAME: dict[int, str] = {
    0: "noop",
    1: "move_left",
    2: "move_right",
    3: "move_up",
    4: "move_down",
    5: "do",
    6: "sleep",
    7: "place_stone",
    8: "place_table",
    9: "place_furnace",
    10: "place_plant",
    11: "make_wood_pickaxe",
    12: "make_stone_pickaxe",
    13: "make_iron_pickaxe",
    14: "make_wood_sword",
    15: "make_stone_sword",
    16: "make_iron_sword",
}

ACTION_NAME_TO_ID: dict[str, int] = {name: action_id for action_id, name in ACTION_ID_TO_NAME.items()}

DEFAULT_STAGE_LIST = [
    "collect_wood",
    "place_table",
    "make_wood_pickaxe",
    "collect_stone",
    "make_stone_pickaxe",
]


class _ScriptedCrafterBackend:
    """无第三方依赖时使用的轻量后端，用于验证统一接口流程。"""

    def __init__(self, config: dict[str, Any], action_id_to_name: dict[int, str]) -> None:
        self.config = config
        self.action_id_to_name = action_id_to_name
        self.rng = random.Random()
        self.time_step = 0
        self.max_episode_steps = int(config.get("max_episode_steps", 10000))
        self.max_health = int(config.get("max_health", 8))
        self.health = self.max_health
        self.inventory: dict[str, int] = {}
        self.achievements: dict[str, bool] = {}
        self.last_event: dict[str, Any] = {}

    def reset(
        self, seed: Optional[int] = None, task_config: Optional[dict[str, Any]] = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """重置脚本状态，返回近似 Crafter 观测。"""
        if seed is not None:
            self.rng.seed(seed)
        task_config = task_config or {}
        self.max_episode_steps = int(task_config.get("max_episode_steps", self.config.get("max_episode_steps", 10000)))
        self.time_step = 0
        self.health = self.max_health
        self.inventory = {
            "wood": 0,
            "stone": 0,
            "table": 0,
            "wood_pickaxe": 0,
            "stone_pickaxe": 0,
        }
        self.achievements = {
            "collect_wood": False,
            "place_table": False,
            "make_wood_pickaxe": False,
            "collect_stone": False,
            "make_stone_pickaxe": False,
        }
        self.last_event = {
            "event_type": "reset",
            "achievement_unlocked": [],
            "inventory_delta": {},
            "health_delta": 0.0,
            "task_success": False,
            "done_reason": "running",
        }
        return self._build_obs(), deepcopy(self.last_event)

    def step(self, action_id: int) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """执行脚本化原子动作，模拟资源采集和工具制作。"""
        self.time_step += 1
        action_name = self.action_id_to_name[action_id]
        reward = 0.0
        unlocked: list[str] = []
        inventory_delta: dict[str, int] = {}
        health_delta = 0.0
        event_type = action_name

        if action_name in {"move_left", "move_right", "move_up", "move_down"}:
            event_type = "explore"
            reward += 0.01
            health_delta = self._maybe_apply_hazard()
        elif action_name == "noop":
            event_type = "noop"
        elif action_name == "interact":
            event_type, reward, unlocked, inventory_delta = self._handle_interact()
        elif action_name == "place_table":
            event_type, reward, unlocked, inventory_delta = self._place_table()
        elif action_name == "make_wood_pickaxe":
            event_type, reward, unlocked, inventory_delta = self._make_wood_pickaxe()
        elif action_name == "make_stone_pickaxe":
            event_type, reward, unlocked, inventory_delta = self._make_stone_pickaxe()

        task_success = self.inventory.get("stone_pickaxe", 0) > 0
        done_reason = "running"
        if task_success:
            done_reason = "task_success"
        elif self.health <= 0:
            done_reason = "dead"
        elif self.time_step >= self.max_episode_steps:
            done_reason = "max_episode_steps"

        done = done_reason != "running"
        self.last_event = {
            "action_name": action_name,
            "achievement_unlocked": unlocked,
            "inventory_delta": inventory_delta,
            "health_delta": health_delta,
            "event_type": event_type,
            "task_success": task_success,
            "done_reason": done_reason,
        }
        return self._build_obs(), reward, done, deepcopy(self.last_event)

    def close(self) -> None:
        """脚本后端没有外部连接。"""
        return None

    def _handle_interact(self) -> tuple[str, float, list[str], dict[str, int]]:
        """交互动作根据当前阶段采集木材或石头。"""
        stone_required = int(self.config.get("stone_required_for_stone_pickaxe", 3))
        if self.inventory.get("wood_pickaxe", 0) > 0 and self.inventory.get("stone", 0) < stone_required:
            self.inventory["stone"] += 1
            unlocked = self._unlock("collect_stone")
            return "collect_stone", 0.30, unlocked, {"stone": 1}

        self.inventory["wood"] += 1
        unlocked = self._unlock("collect_wood")
        return "collect_wood", 0.20, unlocked, {"wood": 1}

    def _place_table(self) -> tuple[str, float, list[str], dict[str, int]]:
        """放置工作台，脚本后端要求至少已经采集 1 个木材。"""
        if self.inventory.get("wood", 0) <= 0:
            return "invalid_action", -0.05, [], {}
        if self.inventory.get("table", 0) > 0:
            return "place_table", 0.0, [], {}
        self.inventory["table"] = 1
        unlocked = self._unlock("place_table")
        return "place_table", 0.15, unlocked, {"table": 1}

    def _make_wood_pickaxe(self) -> tuple[str, float, list[str], dict[str, int]]:
        """制作木镐，作为采集石头的前置工具。"""
        wood_required = int(self.config.get("wood_required_for_wood_pickaxe", 2))
        if self.inventory.get("table", 0) <= 0 or self.inventory.get("wood", 0) < wood_required:
            return "invalid_action", -0.05, [], {}
        if self.inventory.get("wood_pickaxe", 0) > 0:
            return "make_wood_pickaxe", 0.0, [], {}
        self.inventory["wood_pickaxe"] = 1
        unlocked = self._unlock("make_wood_pickaxe")
        return "make_wood_pickaxe", 0.50, unlocked, {"wood_pickaxe": 1}

    def _make_stone_pickaxe(self) -> tuple[str, float, list[str], dict[str, int]]:
        """制作石镐，完成本封装默认任务。"""
        stone_required = int(self.config.get("stone_required_for_stone_pickaxe", 3))
        has_precondition = (
            self.inventory.get("table", 0) > 0
            and self.inventory.get("wood_pickaxe", 0) > 0
            and self.inventory.get("stone", 0) >= stone_required
        )
        if not has_precondition:
            return "invalid_action", -0.05, [], {}
        if self.inventory.get("stone_pickaxe", 0) > 0:
            return "make_stone_pickaxe", 0.0, [], {}
        self.inventory["stone_pickaxe"] = 1
        unlocked = self._unlock("make_stone_pickaxe")
        return "make_stone_pickaxe", 1.00, unlocked, {"stone_pickaxe": 1}

    def _unlock(self, achievement_name: str) -> list[str]:
        """记录首次达成的制作或采集成就。"""
        if not self.achievements.get(achievement_name, False):
            self.achievements[achievement_name] = True
            return [achievement_name]
        return []

    def _maybe_apply_hazard(self) -> float:
        """可选危险事件，用于后续安全字段验证。"""
        if not bool(self.config.get("enable_scripted_hazards", False)):
            return 0.0
        if self.rng.random() >= float(self.config.get("hazard_probability", 0.02)):
            return 0.0
        self.health = max(0, self.health - 1)
        return -1.0

    def _build_obs(self) -> dict[str, Any]:
        """构造统一封装可解析的脚本观测。"""
        image_value = min(255, self.time_step)
        return {
            "image": [[[image_value, 0, 0] for _ in range(4)] for _ in range(4)],
            "inventory": deepcopy(self.inventory),
            "achievements": deepcopy(self.achievements),
            "player_state": {
                "health": self.health,
                "hunger": 0,
                "thirst": 0,
                "time_step": self.time_step,
            },
            "event_info": deepcopy(self.last_event),
        }


class CrafterEnv(BaseExperimentEnv):
    """Crafter 制作石镐任务的统一接口。"""

    def __init__(self, env_config: Optional[dict[str, Any]] = None, backend_env: Optional[Any] = None) -> None:
        self.env_config = {
            "backend": "scripted",
            "goal": "make_stone_pickaxe",
            "max_episode_steps": 10000,
            "wood_required_for_wood_pickaxe": 2,
            "wood_target_count": 4,
            "stone_required_for_stone_pickaxe": 1,
            **(env_config or {}),
        }
        self.action_id_to_name = self._load_action_mapping(self.env_config)
        self.action_name_to_id = {name: action_id for action_id, name in self.action_id_to_name.items()}
        self.task_config: dict[str, Any] = {}
        self._last_obs: Optional[dict[str, Any]] = None
        self._last_inventory: dict[str, int] = {}
        self._last_achievements: dict[str, bool] = {}
        self._last_player_state: dict[str, Any] = {}
        self._last_event: dict[str, Any] = {"event_type": "init", "done_reason": "not_started"}
        self._backend = backend_env or self._create_backend()
        self._sync_action_mapping_from_backend()

    def reset(self, seed: Optional[int] = None, task_config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """重置环境并返回包含任务、安全和事件字段的观测。"""
        self.task_config = {
            "scene_name": "crafter",
            "goal": self.env_config.get("goal", "make_stone_pickaxe"),
            "max_episode_steps": self.env_config.get("max_episode_steps", 10000),
            "use_achievement_reward": self.env_config.get("use_achievement_reward", True),
            **(task_config or {}),
        }
        raw_obs, info = self._call_reset(seed=seed, task_config=self.task_config)
        obs = self._normalize_observation(raw_obs, info)
        self._last_obs = obs
        return obs

    def step(self, action: Union[int, str]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """执行 Crafter 原始离散动作。"""
        self._ensure_started()
        action_id = self._coerce_action_id(action)
        result = self._backend.step(action_id)
        raw_obs, reward, done, info = self._unpack_step_result(result)
        info = {**(info or {}), "action_name": self.action_id_to_name[action_id]}
        obs = self._normalize_observation(raw_obs, info)
        self._last_obs = obs
        return obs, float(reward), bool(done), info

    def apply_skill(
        self, skill_id: str, skill_args: Optional[dict[str, Any]] = None
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """执行制作石镐任务中的高层技能策略。"""
        self._ensure_started()
        skill_args = skill_args or {}
        skill_id = skill_id.strip()

        total_reward = 0.0
        done = False
        action_trace: list[str] = []
        max_steps = int(skill_args.get("max_steps", self.env_config.get("default_skill_max_steps", 50)))

        if skill_id == "basic_action_sequence":
            action_ids = self._resolve_action_sequence(skill_args)
            obs, total_reward, done, action_trace = self._run_action_sequence(action_ids, max_steps=max_steps)
        elif skill_id == "explore":
            sequence = [self.action_name_to_id[name] for name in ("move_up", "move_right", "move_down", "move_left")]
            obs, total_reward, done, action_trace = self._run_repeating_sequence(sequence, max_steps=max_steps)
        elif skill_id == "avoid_zombie":
            sequence = [self.action_name_to_id["move_left"], self.action_name_to_id["move_up"]]
            obs, total_reward, done, action_trace = self._run_repeating_sequence(sequence, max_steps=min(max_steps, 8))
        elif skill_id == "collect_wood":
            target = int(skill_args.get("target_count", self.env_config.get("wood_target_count", 3)))
            if self._is_official_backend():
                obs, total_reward, done, action_trace = self._official_collect_until("tree", "wood", target, max_steps)
            else:
                obs, total_reward, done, action_trace = self._run_until_inventory("wood", target, max_steps=max_steps)
        elif skill_id == "place_table":
            if self._is_official_backend():
                obs, total_reward, done, action_trace = self._official_place("table", max_steps)
            else:
                obs, total_reward, done, action_trace = self._run_action_sequence(
                    [self.action_name_to_id["place_table"]], max_steps=1
                )
        elif skill_id == "make_wood_pickaxe":
            if self._is_official_backend():
                obs, total_reward, done, action_trace = self._official_make("wood_pickaxe", max_steps)
            else:
                obs, total_reward, done, action_trace = self._run_action_sequence(
                    [self.action_name_to_id["make_wood_pickaxe"]], max_steps=1
                )
        elif skill_id == "collect_stone":
            target = int(skill_args.get("target_count", self.env_config.get("stone_required_for_stone_pickaxe", 3)))
            if self._is_official_backend():
                obs, total_reward, done, action_trace = self._official_collect_until("stone", "stone", target, max_steps)
            else:
                obs, total_reward, done, action_trace = self._run_until_inventory("stone", target, max_steps=max_steps)
        elif skill_id == "make_stone_pickaxe":
            if self._is_official_backend():
                obs, total_reward, done, action_trace = self._official_make("stone_pickaxe", max_steps)
            else:
                obs, total_reward, done, action_trace = self._run_action_sequence(
                    [self.action_name_to_id["make_stone_pickaxe"]], max_steps=1
                )
        else:
            raise KeyError(f"Crafter 未知技能策略：{skill_id}")

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
        """获取制作阶段、资源状态和石镐任务成功标记。"""
        inventory = deepcopy(self._last_inventory)
        achievements = deepcopy(self._last_achievements)
        phase = self._infer_phase(inventory, achievements)
        task_success = phase == "task_complete"
        phase_index = len(DEFAULT_STAGE_LIST) if task_success else DEFAULT_STAGE_LIST.index(phase)
        return {
            "scene_name": "crafter",
            "goal": self.task_config.get("goal", self.env_config.get("goal", "make_stone_pickaxe")),
            "phase": phase,
            "phase_index": phase_index,
            "inventory": inventory,
            "achievements": achievements,
            "stage_complete": task_success,
            "task_success": task_success,
        }

    def get_domain_knowledge(self) -> dict[str, Any]:
        """返回 Crafter 制作石镐任务需要的领域知识。"""
        return {
            "environment_topology": {
                "regions": ["forest", "stone_area", "safe_area", "danger_area", "workbench_area"],
                "objects": ["tree", "stone", "workbench", "zombie"],
            },
            "region_rules": {
                "forest": ["wood_available"],
                "stone_area": ["stone_available", "wood_pickaxe_required"],
                "danger_area": ["zombie_risk"],
                "workbench_area": ["tool_crafting_available"],
            },
            "craft_rules": {
                "place_table": ["wood_required"],
                "make_wood_pickaxe": ["wood_required", "workbench_required"],
                "make_stone_pickaxe": ["stone_required", "wood_pickaxe_required", "workbench_required"],
            },
            "skill_policies": {
                "explore": "探索附近区域，寻找树木、石头和安全工作位置",
                "collect_wood": "采集木材，为制作工具准备资源",
                "place_table": "放置工作台，支持后续工具制作",
                "make_wood_pickaxe": "制作木镐，用于采集石头",
                "collect_stone": "采集石头，为制作石镐准备资源",
                "make_stone_pickaxe": "制作石镐，完成任务目标",
                "avoid_zombie": "远离危险对象，减少生命值损失",
                "basic_action_sequence": "按给定原子动作序列执行",
            },
            "action_id_to_name": deepcopy(self.action_id_to_name),
        }

    def get_safety_state(self) -> dict[str, Any]:
        """获取生命值、危险对象和推荐安全动作。"""
        health = int(self._last_player_state.get("health", self.env_config.get("max_health", 8)))
        max_health = int(self.env_config.get("max_health", 8))
        nearby_zombie = bool(self._last_event.get("nearby_zombie", False))
        health_delta = float(self._last_event.get("health_delta", 0.0) or 0.0)
        danger_level = 0.0
        if health <= 0:
            danger_level = 1.0
        elif nearby_zombie:
            danger_level = 0.7
        else:
            danger_level = min(0.6, max(0.0, (max_health - health) / max(max_health, 1)))

        return {
            "health": health,
            "health_delta": health_delta,
            "nearby_zombie": nearby_zombie,
            "zombie_distance": self._last_event.get("zombie_distance"),
            "danger_level": danger_level,
            "is_dead": health <= 0,
            "recommended_safe_action": "avoid_zombie" if nearby_zombie else "keep",
        }

    def get_event_info(self) -> dict[str, Any]:
        """返回最近一次动作或技能产生的事件。"""
        return deepcopy(self._last_event)

    def close(self) -> None:
        """结束环境会话。"""
        if hasattr(self._backend, "close"):
            self._backend.close()

    def _create_backend(self) -> Any:
        """根据配置创建真实后端或脚本后端。"""
        backend = str(self.env_config.get("backend", "scripted")).lower()
        if backend == "scripted":
            return _ScriptedCrafterBackend(self.env_config, self.action_id_to_name)
        if backend in {"gym", "gymnasium", "crafter"}:
            return self._make_gym_backend()
        raise ValueError(f"不支持的 Crafter backend：{backend}")

    def _make_gym_backend(self) -> Any:
        """创建 Gym/Gymnasium Crafter 后端。"""
        env_id = str(self.env_config.get("env_id", "CrafterReward-v1"))
        try:
            crafter = self._import_official_crafter()
        except Exception as exc:
            raise ImportError(f"crafter 已安装但导入失败，无法创建真实 Crafter 后端：{exc}") from exc
        # Crafter 包实现的是旧版 Gym API，直接用 crafter.Env 可避免新版 Gym
        # TimeLimit 包装器要求 5 元组 step 返回值导致手动控制按钮失效。
        reward_enabled = env_id != "CrafterNoReward-v1"
        return crafter.Env(
            reward=reward_enabled,
            length=int(self.env_config.get("max_episode_steps", 10000)),
            seed=self.env_config.get("seed"),
        )

    @staticmethod
    def _import_official_crafter() -> Any:
        """Import the installed crafter package even when tests/crafter shadows it."""
        module = importlib.import_module("crafter")
        if hasattr(module, "Env"):
            return module
        original_path = list(sys.path)
        sys.modules.pop("crafter", None)
        try:
            sys.path = [
                path
                for path in original_path
                if path and not path.replace("\\", "/").rstrip("/").endswith("/tests")
            ]
            module = importlib.import_module("crafter")
        finally:
            sys.path = original_path
        if not hasattr(module, "Env"):
            raise ImportError("Imported module named 'crafter' does not expose crafter.Env.")
        return module

    def _call_reset(
        self, seed: Optional[int], task_config: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        """兼容 Gymnasium、旧 Gym 和脚本后端的 reset 返回格式。"""
        try:
            result = self._backend.reset(seed=seed, task_config=task_config)
        except TypeError:
            try:
                result = self._backend.reset(seed=seed, options=task_config)
            except TypeError:
                if seed is not None and hasattr(self._backend, "_seed"):
                    self._backend._seed = seed
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
            return raw_obs, float(reward), done, info
        if len(result) == 4:
            raw_obs, reward, done, info = result
            return raw_obs, float(reward), bool(done), dict(info or {})
        raise ValueError("环境 step 返回值长度必须为 4 或 5。")

    def _normalize_observation(self, raw_obs: Any, info: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """把真实或脚本后端输出整理成论文实验统一字段。"""
        info = dict(info or {})
        official_context = self._extract_official_context()
        raw_dict = raw_obs if isinstance(raw_obs, dict) else {}
        inventory = self._coerce_int_dict(
            raw_dict.get("inventory") or info.get("inventory") or official_context.get("inventory") or self._last_inventory
        )
        achievements = self._coerce_bool_dict(
            raw_dict.get("achievements") or info.get("achievements") or official_context.get("achievements") or self._last_achievements
        )
        player_state = dict(
            raw_dict.get("player_state") or info.get("player_state") or official_context.get("player_state") or self._last_player_state
        )
        if "health" not in player_state:
            player_state["health"] = (
                info.get("health")
                or inventory.get("health")
                or player_state.get("health")
                or self.env_config.get("max_health", 8)
            )
        if "hunger" not in player_state:
            player_state["hunger"] = inventory.get("food", player_state.get("hunger", 0))
        if "thirst" not in player_state:
            player_state["thirst"] = inventory.get("drink", player_state.get("thirst", 0))
        event_info = {
            **official_context,
            **dict(raw_dict.get("event_info") or {}),
            **info,
        }
        event_info.setdefault("event_type", info.get("event_type", "running"))
        event_info.setdefault("done_reason", info.get("done_reason", "running"))

        self._last_inventory = inventory
        self._last_achievements = achievements
        self._last_player_state = player_state
        self._last_event = event_info

        obs = {
            "image": raw_dict.get("image", raw_obs),
            "inventory": deepcopy(inventory),
            "achievements": deepcopy(achievements),
            "player_state": deepcopy(player_state),
            "task_state": self.get_task_state(),
            "safety_state": self.get_safety_state(),
            "event_info": self.get_event_info(),
        }
        if bool(self.env_config.get("include_raw_obs", True)):
            obs["raw_obs"] = raw_obs
        return obs

    def _infer_phase(self, inventory: dict[str, int], achievements: dict[str, bool]) -> str:
        """根据背包和成就推断制作石镐任务下一阶段。"""
        if inventory.get("stone_pickaxe", 0) > 0 or achievements.get("make_stone_pickaxe", False):
            return "task_complete"
        if (
            inventory.get("wood", 0) < int(self.env_config.get("wood_target_count", 4))
            and not achievements.get("place_table", False)
        ):
            return "collect_wood"
        if inventory.get("table", 0) <= 0 and not achievements.get("place_table", False):
            return "place_table"
        if inventory.get("wood_pickaxe", 0) <= 0 and not achievements.get("make_wood_pickaxe", False):
            return "make_wood_pickaxe"
        if (
            inventory.get("stone", 0) < int(self.env_config.get("stone_required_for_stone_pickaxe", 1))
            and not achievements.get("collect_stone", False)
        ):
            return "collect_stone"
        return "make_stone_pickaxe"

    def _run_until_inventory(
        self, item_name: str, target_count: int, max_steps: int
    ) -> tuple[dict[str, Any], float, bool, list[str]]:
        """重复执行 interact，直到目标资源数量达标或步数耗尽。"""
        total_reward = 0.0
        done = False
        action_trace: list[str] = []
        obs = self._last_obs or {}
        interact_id = self._semantic_action_id("interact")
        while len(action_trace) < max_steps and self._last_inventory.get(item_name, 0) < target_count and not done:
            obs, reward, done, info = self.step(interact_id)
            total_reward += reward
            action_trace.append(info.get("action_name", "interact"))
        return obs, total_reward, done, action_trace

    def _run_action_sequence(
        self, action_ids: Iterable[int], max_steps: int
    ) -> tuple[dict[str, Any], float, bool, list[str]]:
        """按给定原子动作序列执行。"""
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

    def _run_repeating_sequence(
        self, action_ids: list[int], max_steps: int
    ) -> tuple[dict[str, Any], float, bool, list[str]]:
        """循环执行一组探索或避险动作。"""
        repeated = [action_ids[index % len(action_ids)] for index in range(max_steps)]
        return self._run_action_sequence(repeated, max_steps=max_steps)

    def _resolve_action_sequence(self, skill_args: dict[str, Any]) -> list[int]:
        """解析 basic_action_sequence 的动作编号或动作名称。"""
        raw_sequence = skill_args.get("action_ids", skill_args.get("actions", []))
        return [self._coerce_action_id(action) for action in raw_sequence]

    def _official_collect_until(
        self, material_name: str, item_name: str, target_count: int, max_steps: int
    ) -> tuple[dict[str, Any], float, bool, list[str]]:
        """在官方 Crafter 中移动到目标材料旁边并执行 do 采集。"""
        total_reward = 0.0
        done = False
        action_trace: list[str] = []
        obs = self._last_obs or {}
        while len(action_trace) < max_steps and self._last_inventory.get(item_name, 0) < target_count and not done:
            target = self._find_nearest_material(material_name)
            if target is None:
                obs, reward, done, trace = self._official_wander(max_steps=1)
            else:
                obs, reward, done, trace = self._official_move_adjacent_and_do(target, max_steps - len(action_trace))
            total_reward += reward
            action_trace.extend(trace)
            if not trace:
                break
        return obs, total_reward, done, action_trace

    def _official_place(self, place_name: str, max_steps: int) -> tuple[dict[str, Any], float, bool, list[str]]:
        """在官方 Crafter 中面向可放置地块并放置工作台等对象。"""
        total_reward = 0.0
        done = False
        action_trace: list[str] = []
        obs = self._last_obs or {}
        direction = self._find_place_direction()
        if direction is None:
            obs, total_reward, done, action_trace = self._official_wander(max_steps=max(1, min(4, max_steps)))
            direction = self._find_place_direction()
        if direction is None or done or len(action_trace) >= max_steps:
            return obs, total_reward, done, action_trace
        self._set_official_facing(direction)
        obs, reward, done, info = self.step(f"place_{place_name}")
        total_reward += reward
        action_trace.append(info.get("action_name", f"place_{place_name}"))
        return obs, total_reward, done, action_trace

    def _official_make(self, item_name: str, max_steps: int) -> tuple[dict[str, Any], float, bool, list[str]]:
        """在官方 Crafter 中靠近工作台并执行 make_* 动作。"""
        total_reward = 0.0
        done = False
        action_trace: list[str] = []
        obs = self._last_obs or {}
        if "table" not in self._nearby_material_names(distance=1):
            table_pos = self._find_nearest_material("table")
            if table_pos is not None:
                obs, reward, done, trace = self._official_move_adjacent(table_pos, max_steps=max_steps)
                total_reward += reward
                action_trace.extend(trace)
        if done or len(action_trace) >= max_steps:
            return obs, total_reward, done, action_trace
        obs, reward, done, info = self.step(f"make_{item_name}")
        total_reward += reward
        action_trace.append(info.get("action_name", f"make_{item_name}"))
        return obs, total_reward, done, action_trace

    def _official_move_adjacent_and_do(
        self, target: tuple[int, int], max_steps: int
    ) -> tuple[dict[str, Any], float, bool, list[str]]:
        """移动到目标材料旁边，面向目标后执行 do。"""
        obs, total_reward, done, action_trace = self._official_move_adjacent(target, max_steps=max_steps)
        if done or len(action_trace) >= max_steps:
            return obs, total_reward, done, action_trace
        direction = self._direction_to_target(target)
        if direction is not None:
            self._set_official_facing(direction)
        obs, reward, done, info = self.step("interact")
        total_reward += reward
        action_trace.append(info.get("action_name", "do"))
        return obs, total_reward, done, action_trace

    def _official_move_adjacent(
        self, target: tuple[int, int], max_steps: int
    ) -> tuple[dict[str, Any], float, bool, list[str]]:
        """沿网格路径移动到目标格子的四邻域。"""
        total_reward = 0.0
        done = False
        action_trace: list[str] = []
        obs = self._last_obs or {}
        path = self._path_to_adjacent(target)
        for action_name in path[:max_steps]:
            obs, reward, done, info = self.step(action_name)
            total_reward += reward
            action_trace.append(info.get("action_name", action_name))
            if done:
                break
        return obs, total_reward, done, action_trace

    def _official_wander(self, max_steps: int) -> tuple[dict[str, Any], float, bool, list[str]]:
        """找不到目标时做少量探索移动。"""
        actions = ["move_up", "move_right", "move_down", "move_left"]
        return self._run_action_sequence([self._coerce_action_id(a) for a in actions[:max_steps]], max_steps=max_steps)

    def _find_nearest_material(self, material_name: str) -> Optional[tuple[int, int]]:
        """从官方 Crafter 世界网格中查找最近材料位置。"""
        world, player = self._official_world_and_player()
        if world is None or player is None:
            return None
        start = tuple(int(x) for x in player.pos)
        best: Optional[tuple[int, int]] = None
        best_dist = 10**9
        for x in range(int(world.area[0])):
            for y in range(int(world.area[1])):
                material, _ = world[(x, y)]
                if material != material_name:
                    continue
                dist = abs(x - start[0]) + abs(y - start[1])
                if dist < best_dist:
                    best = (x, y)
                    best_dist = dist
        return best

    def _path_to_adjacent(self, target: tuple[int, int]) -> list[str]:
        """BFS 搜索到目标四邻域的动作序列。"""
        world, player = self._official_world_and_player()
        if world is None or player is None:
            return []
        start = tuple(int(x) for x in player.pos)
        goals = {
            (target[0] - 1, target[1]),
            (target[0] + 1, target[1]),
            (target[0], target[1] - 1),
            (target[0], target[1] + 1),
        }
        goals = {pos for pos in goals if self._official_is_free(pos) or pos == start}
        if start in goals:
            return []
        moves = [
            ("move_left", (-1, 0)),
            ("move_right", (1, 0)),
            ("move_up", (0, -1)),
            ("move_down", (0, 1)),
        ]
        queue: deque[tuple[tuple[int, int], list[str]]] = deque([(start, [])])
        seen = {start}
        while queue:
            pos, path = queue.popleft()
            for action, delta in moves:
                nxt = (pos[0] + delta[0], pos[1] + delta[1])
                if nxt in seen or not self._official_is_free(nxt):
                    continue
                next_path = path + [action]
                if nxt in goals:
                    return next_path
                seen.add(nxt)
                queue.append((nxt, next_path))
        return []

    def _official_is_free(self, pos: tuple[int, int]) -> bool:
        """判断官方 Crafter 网格是否可站立。"""
        world, player = self._official_world_and_player()
        if world is None or player is None:
            return False
        material, obj = world[pos]
        if material is None:
            return False
        return obj is None and material in player.walkable

    def _direction_to_target(self, target: tuple[int, int]) -> Optional[tuple[int, int]]:
        """返回玩家当前位置到相邻目标格子的朝向。"""
        _, player = self._official_world_and_player()
        if player is None:
            return None
        px, py = [int(x) for x in player.pos]
        dx, dy = target[0] - px, target[1] - py
        if abs(dx) + abs(dy) != 1:
            return None
        return dx, dy

    def _set_official_facing(self, direction: tuple[int, int]) -> None:
        """直接设置官方 Crafter 玩家朝向，用于可靠执行 do/place。"""
        _, player = self._official_world_and_player()
        if player is not None:
            player.facing = tuple(direction)

    def _find_place_direction(self) -> Optional[tuple[int, int]]:
        """查找玩家四周可放置工作台的方向。"""
        world, player = self._official_world_and_player()
        if world is None or player is None:
            return None
        px, py = [int(x) for x in player.pos]
        for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            target = (px + direction[0], py + direction[1])
            material, obj = world[target]
            if obj is None and material in {"grass", "sand", "path"}:
                return direction
        return None

    def _nearby_material_names(self, distance: int) -> set[str]:
        """读取玩家附近材料名称集合。"""
        world, player = self._official_world_and_player()
        if world is None or player is None:
            return set()
        materials, _ = world.nearby(player.pos, distance)
        return {str(material) for material in materials if material is not None}

    def _extract_official_context(self) -> dict[str, Any]:
        """从官方 Crafter 内部读取领域知识和局部环境状态。"""
        world, player = self._official_world_and_player()
        if world is None or player is None:
            return {}
        px, py = [int(x) for x in player.pos]
        facing = tuple(int(x) for x in player.facing)
        front = (px + facing[0], py + facing[1])
        front_material, front_obj = world[front]
        local_materials: dict[str, int] = {}
        local_objects: dict[str, int] = {}
        for x in range(px - 4, px + 5):
            for y in range(py - 4, py + 5):
                material, obj = world[(x, y)]
                if material is not None:
                    local_materials[str(material)] = local_materials.get(str(material), 0) + 1
                if obj is not None and obj is not player:
                    name = type(obj).__name__
                    local_objects[name] = local_objects.get(name, 0) + 1
        return {
            "inventory": dict(player.inventory),
            "achievements": dict(player.achievements),
            "player_state": {
                "health": int(player.inventory.get("health", 0)),
                "hunger": int(player.inventory.get("food", 0)),
                "thirst": int(player.inventory.get("drink", 0)),
                "energy": int(player.inventory.get("energy", 0)),
                "position": [px, py],
                "facing": list(facing),
            },
            "player_pos": [px, py],
            "facing": list(facing),
            "front_material": front_material,
            "front_object": type(front_obj).__name__ if front_obj is not None else None,
            "nearby_materials": local_materials,
            "nearby_objects": local_objects,
            "nearby_zombie": local_objects.get("Zombie", 0) > 0,
        }

    def _official_world_and_player(self) -> tuple[Optional[Any], Optional[Any]]:
        """返回官方 Crafter 的 world/player 句柄。"""
        backend = getattr(self._backend, "unwrapped", self._backend)
        world = getattr(backend, "_world", None)
        player = getattr(backend, "_player", None)
        return world, player

    def _is_official_backend(self) -> bool:
        """判断当前是否为真实 Crafter 后端。"""
        return not isinstance(self._backend, _ScriptedCrafterBackend)

    def _coerce_action_id(self, action: Union[int, str]) -> int:
        """把动作名称或动作编号统一转成动作编号。"""
        if isinstance(action, str):
            if action == "interact" and action not in self.action_name_to_id and "do" in self.action_name_to_id:
                return self.action_name_to_id["do"]
            if action not in self.action_name_to_id:
                raise KeyError(f"Crafter 未知动作名称：{action}")
            return self.action_name_to_id[action]
        action_id = int(action)
        if action_id not in self.action_id_to_name:
            raise KeyError(f"Crafter 未知动作编号：{action_id}")
        return action_id

    def _ensure_started(self) -> None:
        """避免上层在 reset 前直接 step。"""
        if self._last_obs is None:
            raise RuntimeError("请先调用 reset() 再执行 step() 或 apply_skill()。")

    @staticmethod
    def _load_action_mapping(env_config: dict[str, Any]) -> dict[int, str]:
        """允许项目真实 Crafter 动作编号覆盖默认映射。"""
        raw_mapping = env_config.get("action_id_to_name")
        if not raw_mapping:
            if str(env_config.get("backend", "scripted")).lower() in {"gym", "gymnasium", "crafter"}:
                return dict(GYM_ACTION_ID_TO_NAME)
            return dict(ACTION_ID_TO_NAME)
        return {int(action_id): str(name) for action_id, name in raw_mapping.items()}

    def _sync_action_mapping_from_backend(self) -> None:
        """真实 Crafter 后端存在 action_names 时，以后端动作表为准。"""
        action_names = getattr(self._backend, "action_names", None)
        if action_names is None:
            action_names = getattr(getattr(self._backend, "unwrapped", None), "action_names", None)
        if action_names is None:
            return
        self.action_id_to_name = {index: str(name) for index, name in enumerate(list(action_names))}
        self.action_name_to_id = {name: action_id for action_id, name in self.action_id_to_name.items()}

    def _semantic_action_id(self, action_name: str) -> int:
        """把统一语义动作映射到当前后端的实际动作编号。"""
        if action_name in self.action_name_to_id:
            return self.action_name_to_id[action_name]
        if action_name == "interact" and "do" in self.action_name_to_id:
            return self.action_name_to_id["do"]
        raise KeyError(f"Crafter 当前后端不支持动作：{action_name}")

    @staticmethod
    def _coerce_int_dict(value: Any) -> dict[str, int]:
        """把背包字段整理成 str -> int。"""
        if not isinstance(value, dict):
            return {}
        return {str(key): int(val) for key, val in value.items()}

    @staticmethod
    def _coerce_bool_dict(value: Any) -> dict[str, bool]:
        """把成就字段整理成 str -> bool。"""
        if not isinstance(value, dict):
            return {}
        return {str(key): bool(val) for key, val in value.items()}
