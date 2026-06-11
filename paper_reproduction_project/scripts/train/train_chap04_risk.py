"""Train/save the lightweight EFSR risk value function."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.chap04_efsr.risk_prediction import RiskValueFunction
from src.common.config_loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/chap04_efsr/efsr.yaml")
    parser.add_argument("--scene", default="highway")
    parser.add_argument("--output", default="results/metrics/chap04_risk_model.json")
    args = parser.parse_args()
    config = load_config(args.config, {})
    risk_cfg = dict(config.get("risk") or {})
    model = RiskValueFunction(feedback_weight=float(risk_cfg.get("feedback_weight", 0.35)))
    path = model.save(args.output)
    print({"scene": args.scene, "model": str(path)})


if __name__ == "__main__":
    main()

