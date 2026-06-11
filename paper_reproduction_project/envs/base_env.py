"""实验环境统一接口基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseExperimentEnv(ABC):
    """四类实验环境都需要实现的统一接口。"""

    @abstractmethod
    def reset(self, seed: Optional[int] = None, task_config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """重置环境并返回统一观测。"""

    @abstractmethod
    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """执行一个底层动作，返回 obs, reward, done, info。"""

    @abstractmethod
    def apply_skill(
        self, skill_id: str, skill_args: Optional[dict[str, Any]] = None
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """执行一个高层技能策略或有限长度原子动作序列。"""

    @abstractmethod
    def get_task_state(self) -> dict[str, Any]:
        """返回任务阶段、目标和进度字段。"""

    @abstractmethod
    def get_domain_knowledge(self) -> dict[str, Any]:
        """返回环境拓扑、区域规则和技能策略说明。"""

    @abstractmethod
    def get_safety_state(self) -> dict[str, Any]:
        """返回风险预测和安全约束需要的字段。"""

    @abstractmethod
    def get_event_info(self) -> dict[str, Any]:
        """返回当前步或最近一次技能执行产生的事件反馈。"""

    @abstractmethod
    def close(self) -> None:
        """释放环境资源。"""

