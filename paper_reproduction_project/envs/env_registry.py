"""实验环境注册与创建入口。"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .base_env import BaseExperimentEnv

EnvFactory = Callable[[Optional[Dict[str, Any]]], BaseExperimentEnv]


_REGISTRY: dict[str, EnvFactory] = {}
_DEFAULTS_LOADED = False


def register_env(name: str, factory: EnvFactory) -> None:
    """注册环境工厂，供章节方法按名称创建环境。"""
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("环境名称不能为空。")
    _REGISTRY[normalized] = factory


def available_envs() -> list[str]:
    """返回当前已注册的环境名称。"""
    _ensure_default_envs()
    return sorted(_REGISTRY)


def create_env(name: str, env_config: Optional[dict[str, Any]] = None) -> BaseExperimentEnv:
    """按统一名称创建环境实例。"""
    _ensure_default_envs()
    normalized = name.strip().lower()
    if normalized not in _REGISTRY:
        raise KeyError(f"未知环境：{name}；可用环境为：{available_envs()}")
    return _REGISTRY[normalized](env_config or {})


def _ensure_default_envs() -> None:
    """延迟导入默认环境，避免模块互相导入时产生循环。"""
    global _DEFAULTS_LOADED
    if _DEFAULTS_LOADED:
        return

    from .crafter_env import CrafterEnv
    from .highway_env_wrapper import HighwayEnvWrapper
    from .ugv_parking_env import UGVParkingEnv
    from .ugv_patrol_env import UGVPatrolEnv

    register_env("crafter", CrafterEnv)
    register_env("highway", HighwayEnvWrapper)
    register_env("ugv_parking", UGVParkingEnv)
    register_env("ugv_patrol", UGVPatrolEnv)
    _DEFAULTS_LOADED = True
