"""Smoke-test a live Unity Editor or player connection for UGV scenes.

Run:
    conda run -n DT python tests/ugv_unity_connection_smoke.py --scene ugv_parking

For the Unity Editor, start this script first, then press Play in Unity.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.unity_ugv_env import UnityUGVEnv  # noqa: E402


def _summarize_observations(obs: dict[str, Any]) -> str:
    ml_obs = obs.get("mlagents_observations") or {}
    shapes = []
    for item in ml_obs.get("observations", []):
        shapes.append(f"{item.get('shape')}:{item.get('dtype')}")
    visuals = []
    for item in obs.get("visual_observations", []):
        visuals.append(
            f"{item.get('name') or item.get('cameraName')}:{item.get('shape') or [item.get('height'), item.get('width'), item.get('channels')]}"
        )
    return (
        f"behavior={ml_obs.get('behavior_name')} "
        f"step_type={ml_obs.get('step_type')} "
        f"agents={ml_obs.get('agent_count')} "
        f"obs={shapes} "
        f"visuals={visuals}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["ugv_parking", "ugv_patrol"], default="ugv_parking")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--base-port", type=int, default=5004)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--player-path", default="")
    parser.add_argument("--return-arrays", action="store_true")
    args = parser.parse_args()

    config: dict[str, Any] = {
        "backend": "unity",
        "worker_id": args.worker_id,
        "base_port": args.base_port,
        "timeout_wait": args.timeout,
        "seed": args.seed,
        "return_observation_arrays": args.return_arrays,
    }
    if args.player_path:
        config["file_name"] = args.player_path

    print(f"PY_READY scene={args.scene} worker={args.worker_id} timeout={args.timeout}", flush=True)
    env = UnityUGVEnv(args.scene, config)
    try:
        print("CONNECTED backend-created", flush=True)
        obs = env.reset(seed=args.seed)
        print("RESET_OK", _summarize_observations(obs), flush=True)
        obs, reward, done, info = env.manual_step(0.0, 0.25, duration_steps=3, label="smoke_forward")
        print(f"STEP_OK reward={reward:.4f} done={done} info={info}", flush=True)
        print("STEP_OBS", _summarize_observations(obs), flush=True)
    finally:
        env.close()
        print("CLOSED", flush=True)


if __name__ == "__main__":
    main()
