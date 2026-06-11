"""Train/save the lightweight LACM confidence selector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.chap03_lacm.confidence import ConfidenceCandidateSelector
from src.common.config_loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/chap03_lacm/lacm.yaml")
    parser.add_argument("--scene", default="highway")
    parser.add_argument("--output", default="results/metrics/chap03_confidence_model.json")
    args = parser.parse_args()
    config = load_config(args.config, {})
    weights = dict((config.get("confidence") or {}).get("weights") or {})
    selector = ConfidenceCandidateSelector(weights=weights or None)
    path = selector.save(args.output)
    print({"scene": args.scene, "model": str(path)})


if __name__ == "__main__":
    main()

