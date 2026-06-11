"""Unity 环境尚未接入时使用的显式占位实现。"""

from __future__ import annotations

from typing import Any, Optional

from .base_env import BaseExperimentEnv


class UnityEnvironmentNotConnected(RuntimeError):
    """调用未接入的 Unity 环境时抛出的清晰错误。"""


class UnityStubEnv(BaseExperimentEnv):
    """保留统一接口，但不伪造 Unity 底层能力。"""

    def __init__(self, scene_name: str, env_config: Optional[dict[str, Any]] = None) -> None:
        self.scene_name = scene_name
        self.env_config = env_config or {}
        self._last_event = {
            "event_type": "unity_not_connected",
            "done_reason": "unity_not_connected",
            "message": f"{scene_name} Unity 底层环境尚未接入。",
        }

    def _raise_not_connected(self) -> None:
        raise UnityEnvironmentNotConnected(
            f"{self.scene_name} Unity 底层环境尚未接入；请在后续接入 Unity 通信层后替换该占位类。"
        )

    def reset(self, seed: Optional[int] = None, task_config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Unity 未接入前不允许启动真实回合。"""
        self._raise_not_connected()

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Unity 未接入前不允许发送底层控制。"""
        self._raise_not_connected()

    def apply_skill(
        self, skill_id: str, skill_args: Optional[dict[str, Any]] = None
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Unity 未接入前不允许执行技能策略。"""
        self._raise_not_connected()

    def get_task_state(self) -> dict[str, Any]:
        """返回占位任务状态，方便上层识别当前不可运行。"""
        return {
            "scene_name": self.scene_name,
            "phase": "unity_not_connected",
            "stage_complete": False,
            "task_success": False,
        }

    def get_domain_knowledge(self) -> dict[str, Any]:
        """占位类只声明连接状态，不提供虚假的场景拓扑。"""
        return {
            "environment_topology": {},
            "region_rules": {},
            "skill_policies": {},
            "connection_state": "unity_not_connected",
        }

    def get_safety_state(self) -> dict[str, Any]:
        """Unity 未接入时无法返回真实安全字段。"""
        return {
            "collision": False,
            "out_of_road": False,
            "red_light_violation": False,
            "risk_score": None,
            "recommended_safe_action": "wait_for_unity_connection",
        }

    def get_event_info(self) -> dict[str, Any]:
        """返回最近一次占位事件。"""
        return dict(self._last_event)

    def close(self) -> None:
        """占位类没有外部资源需要释放。"""
        return None

