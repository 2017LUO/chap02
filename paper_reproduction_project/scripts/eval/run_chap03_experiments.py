"""CLI wrapper for chapter-3 LACM experiments."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.chap03_lacm.experiments.run_lacm import main


if __name__ == "__main__":
    main()

