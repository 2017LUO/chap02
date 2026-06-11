"""Metric aggregation for reproducible experiment logs."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Union

from .data_structures import EpisodeResult
from .logger import ensure_dir, write_json


def summarize_episode_results(results: Iterable[EpisodeResult]) -> dict[str, Any]:
    rows = [item.to_dict() for item in results]
    if not rows:
        return {"episodes": 0}
    rewards = [float(row["total_reward"]) for row in rows]
    steps = [int(row["steps"]) for row in rows]
    successes = [1.0 if row["success"] else 0.0 for row in rows]
    summary: dict[str, Any] = {
        "episodes": len(rows),
        "scene": rows[0]["scene"],
        "method": rows[0]["method"],
        "avg_reward": mean(rewards),
        "avg_steps": mean(steps),
        "success_rate": mean(successes),
    }
    metric_keys = sorted({key for row in rows for key in row if key not in summary and key not in _BASE_KEYS})
    for key in metric_keys:
        values = [row.get(key) for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            summary[f"avg_{key}"] = mean(values)
    return summary


def write_summary_artifacts(results: list[EpisodeResult], output_dir: Union[str, Path], prefix: str) -> dict[str, str]:
    output_dir = ensure_dir(output_dir)
    rows = [item.to_dict() for item in results]
    summary = summarize_episode_results(results)
    json_path = write_json(output_dir / f"{prefix}_summary.json", summary)
    csv_path = output_dir / f"{prefix}_episodes.csv"
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    return {"summary_json": str(json_path), "episodes_csv": str(csv_path)}


_BASE_KEYS = {
    "scene",
    "method",
    "episode",
    "seed",
    "total_reward",
    "steps",
    "success",
    "done_reason",
}

