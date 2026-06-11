"""Train/save the lightweight DK-ABS trigger model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.chap02_dk_abs.online_scheduling import TriggerModel
from src.common.config_loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/chap02_dk_abs/dk_abs.yaml")
    parser.add_argument("--scene", default="crafter")
    parser.add_argument("--output", default="results/metrics/chap02_trigger_model.json")
    args = parser.parse_args()
    config = load_config(args.config, {})
    scheduler_cfg = dict(config.get("scheduler") or {})
    model = TriggerModel(
        risk_weight=float(scheduler_cfg.get("risk_weight", 0.6)),
        time_weight=float(scheduler_cfg.get("time_weight", 0.2)),
        stall_weight=float(scheduler_cfg.get("stall_weight", 0.2)),
    )
    path = model.save(args.output)
    print({"scene": args.scene, "model": str(path)})


if __name__ == "__main__":
    main()

