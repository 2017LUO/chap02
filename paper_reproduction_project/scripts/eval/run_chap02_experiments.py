"""CLI wrapper for chapter-2 DK-ABS experiments."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.chap02_dk_abs.experiments.run_dk_abs import main


if __name__ == "__main__":
    main()

