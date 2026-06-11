"""Environment creation helpers for scripted, Gym/Gymnasium, and Unity backends."""

from __future__ import annotations

from typing import Any, Optional

from envs import create_env


def supported_official_scenes() -> tuple[str, ...]:
    return ("crafter", "highway", "ugv_parking", "ugv_patrol")


def make_env(scene: str, config: Optional[dict[str, Any]] = None):
    scene = scene.strip().lower()
    config = dict(config or {})
    if scene == "crafter":
        env_config = {
            "backend": "gym",
            "env_id": "CrafterReward-v1",
            "wood_target_count": 4,
            "max_episode_steps": config.get("max_episode_steps", 10000),
            **dict(config.get("environment") or {}),
            **dict(config.get("env_config") or {}),
        }
        return _create_with_scripted_fallback("crafter", env_config, config)
    if scene == "highway":
        env_config = {
            "backend": "gymnasium",
            "env_id": "highway-v0",
            "render_mode": "rgb_array",
            "duration": config.get("max_episode_steps", config.get("duration", 100)),
            "max_episode_steps": config.get("max_episode_steps", 100),
            **dict(config.get("environment") or {}),
            **dict(config.get("env_config") or {}),
        }
        return _create_with_scripted_fallback("highway", env_config, config)
    if scene in {"ugv_parking", "ugv_patrol"}:
        env_config = {
            "backend": "scripted",
            "max_episode_steps": config.get("max_episode_steps", 100),
            **dict(config.get("environment") or {}),
            **dict(config.get("env_config") or {}),
        }
        return _create_with_scripted_fallback(scene, env_config, config)
    raise KeyError(f"Unsupported scene: {scene}; available official scenes: {supported_official_scenes()}")


def _create_with_scripted_fallback(scene: str, env_config: dict[str, Any], root_config: dict[str, Any]):
    try:
        return create_env(scene, env_config)
    except (ImportError, ModuleNotFoundError):
        if not _allow_scripted_fallback(env_config, root_config):
            raise
        fallback = dict(env_config)
        fallback["backend"] = "scripted"
        return create_env(scene, fallback)


def _allow_scripted_fallback(env_config: dict[str, Any], root_config: dict[str, Any]) -> bool:
    env_value = env_config.get("allow_scripted_fallback")
    root_value = root_config.get("allow_scripted_fallback")
    if env_value is not None:
        return bool(env_value)
    if root_value is not None:
        return bool(root_value)
    return True
