"""UGV Patrol unified environment entry."""

from __future__ import annotations

from typing import Any, Optional

from .unity_ugv_env import UnityUGVEnv


class UGVPatrolEnv(UnityUGVEnv):
    """UGV Patrol scene wrapper."""

    def __init__(self, env_config: Optional[dict[str, Any]] = None) -> None:
        super().__init__("ugv_patrol", env_config)
