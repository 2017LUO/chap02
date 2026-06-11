"""Train/save the lightweight LACM compensation policy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.chap03_lacm.compensation import CompensationPolicy
from src.common.config_loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/chap03_lacm/lacm.yaml")
    parser.add_argument("--scene", default="highway")
    parser.add_argument("--output", default="results/metrics/chap03_compensation_policy.json")
    args = parser.parse_args()
    config = load_config(args.config, {})
    comp = dict(config.get("compensation") or {})
    policy = CompensationPolicy(
        risk_threshold=float(comp.get("risk_threshold", 0.55)),
        max_hold_steps=int(comp.get("max_hold_steps", 2)),
    )
    path = policy.save(args.output)
    print({"scene": args.scene, "policy": str(path)})


if __name__ == "__main__":
    main()

