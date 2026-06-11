"""UGV Parking unified environment entry."""

from __future__ import annotations

from typing import Any, Optional

from .unity_ugv_env import UnityUGVEnv


class UGVParkingEnv(UnityUGVEnv):
    """UGV Parking scene wrapper."""

    def __init__(self, env_config: Optional[dict[str, Any]] = None) -> None:
        super().__init__("ugv_parking", env_config)
