"""Unified three-seed reproduction training entry."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.chap02_dk_abs.experiments.run_dk_abs import run_dk_abs
from src.chap03_lacm.experiments.run_lacm import run_lacm
from src.chap04_efsr.experiments.run_efsr import run_efsr
from src.common.config_loader import deep_update, load_config
from src.common.logger import ensure_dir, write_json
from src.common.progress_logger import emit_progress
from src.common.reference_simulation import (
    CHAPTER_METHOD_KEYS,
    DEFAULT_REFERENCE_ROOT,
    REFERENCE_SCENES,
    REFERENCE_SEEDS,
    env_scene_name,
    find_reference_item,
    normalize_chapter,
    normalize_group,
    reference_scene_name,
)
from src.common.reproduction_metrics import (
    build_reference_episode_rows,
    columns_for_chapter,
    summarize_reference_rows,
)


Runner = Callable[..., Dict[str, Any]]

CHAPTER_RUNNERS: Dict[str, Tuple[Runner, str, str, str]] = {
    "chap02": (run_dk_abs, "dkabs", "DK-ABS", "configs/chap02_dk_abs/dk_abs.yaml"),
    "chap03": (run_lacm, "lacm", "LACM", "configs/chap03_lacm/lacm.yaml"),
    "chap04": (run_efsr, "efsr", "EFSR", "configs/chap04_efsr/efsr.yaml"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chapter methods with the copied three-seed reference layout.")
    parser.add_argument("--chapter", default="all", help="chap02, chap03, chap04, or all.")
    parser.add_argument("--group", default="comparison", help="comparison or ablation.")
    parser.add_argument("--scene", default="all", help="road_parking, patrol_parking, ugv_parking, ugv_patrol, or all.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(REFERENCE_SEEDS))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--backend", default="scripted", help="scripted, unity, udp, gym, or gymnasium.")
    parser.add_argument("--output-dir", default="results/reproduction_training")
    parser.add_argument("--reference-root", default=str(DEFAULT_REFERENCE_ROOT))
    parser.add_argument("--config", default=None, help="Optional single-chapter config override.")
    parser.add_argument("--progress-interval", type=int, default=100, help="Print progress every N episodes; 0 disables progress lines.")
    parser.add_argument("--step-progress-interval", type=int, default=0, help="Write a heartbeat every N high-level steps; 0 disables step heartbeats.")
    parser.add_argument("--test-run", action="store_true", help="Use a temporary output directory and delete it after a successful run.")
    parser.add_argument("--keep-test-output", action="store_true", help="Keep --test-run output for debugging instead of deleting it.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the planned jobs.")
    args = parser.parse_args()

    if args.test_run:
        args.output_dir = "results/_tmp_reproduction_training"
        _remove_test_output(ROOT / args.output_dir)

    jobs = list(_build_jobs(args))
    if args.dry_run:
        print(json.dumps([_job_preview(job) for job in jobs], ensure_ascii=False, indent=2))
        return

    run_rows = []
    for job in jobs:
        run_rows.append(_run_job(job))

    output_root = ensure_dir(ROOT / args.output_dir)
    write_json(output_root / "manifest.json", {"runs": run_rows})
    print(json.dumps({"manifest": str(output_root / "manifest.json"), "runs": len(run_rows)}, ensure_ascii=False))
    if args.test_run and not args.keep_test_output:
        _remove_test_output(output_root)
        print(json.dumps({"test_run": True, "output_removed": str(output_root)}, ensure_ascii=False))


def _remove_test_output(path: Path) -> None:
    resolved = path.resolve()
    allowed_root = (ROOT / "results").resolve()
    if resolved == allowed_root or allowed_root not in resolved.parents:
        raise ValueError("Refusing to remove non-test output directory: %s" % resolved)
    if resolved.name != "_tmp_reproduction_training":
        raise ValueError("Refusing to remove unexpected output directory: %s" % resolved)
    shutil.rmtree(resolved, ignore_errors=True)


def _build_jobs(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    chapters = list(CHAPTER_RUNNERS) if str(args.chapter).lower() == "all" else [normalize_chapter(args.chapter)]
    group = normalize_group(args.group)
    if str(args.scene).lower() == "all":
        scenes = list(REFERENCE_SCENES)
    else:
        scenes = [reference_scene_name(args.scene)]

    if args.config and len(chapters) != 1:
        raise ValueError("--config can only be used with a single chapter.")

    for chapter in chapters:
        for scene in scenes:
            for seed in args.seeds:
                yield {
                    "chapter": chapter,
                    "group": group,
                    "reference_scene": scene,
                    "env_scene": env_scene_name(scene),
                    "seed": int(seed),
                    "episodes": int(args.episodes),
                    "backend": args.backend,
                    "output_dir": args.output_dir,
                    "reference_root": Path(args.reference_root),
                    "config": args.config,
                    "max_episode_steps": args.max_episode_steps,
                    "progress_interval": int(args.progress_interval),
                    "step_progress_interval": int(args.step_progress_interval),
                }


def _job_preview(job: Dict[str, Any]) -> Dict[str, Any]:
    runner, method_key, _, default_config = CHAPTER_RUNNERS[job["chapter"]]
    reference_item = find_reference_item(
        job["chapter"],
        job["group"],
        job["reference_scene"],
        job["reference_root"],
    )
    output_dir = _workspace_dir(job, method_key)
    return {
        "chapter": job["chapter"],
        "group": job["group"],
        "scene": job["reference_scene"],
        "env_scene": job["env_scene"],
        "seed": job["seed"],
        "episodes": job["episodes"],
        "method": method_key,
        "config": job["config"] or default_config,
        "output_dir": str(output_dir),
        "reference_dir": reference_item.destination if reference_item else None,
    }


def _run_job(job: Dict[str, Any]) -> Dict[str, Any]:
    runner, method_key, method_label, default_config = CHAPTER_RUNNERS[job["chapter"]]
    workspace_dir = ensure_dir(_workspace_dir(job, method_key))
    native_dir = ensure_dir(workspace_dir / "_native")
    progress_log = workspace_dir / "progress.log"
    progress_log.write_text("", encoding="utf-8")
    resolved_config = _write_resolved_config(job, default_config, native_dir, method_label)
    emit_progress(
        "[start] {chapter} {group} {scene} seed={seed} episodes={episodes} method={method}".format(
            chapter=job["chapter"],
            group=job["group"],
            scene=job["reference_scene"],
            seed=job["seed"],
            episodes=job["episodes"],
            method=method_key,
        ),
        progress_log,
    )
    emit_progress("[logs] progress heartbeat: %s" % progress_log, progress_log)
    emit_progress("[logs] native events will be under: %s" % native_dir, progress_log)

    result = runner(
        scene=job["env_scene"],
        config_path=resolved_config,
        episodes=job["episodes"],
        seed=job["seed"],
        output_dir=native_dir,
    )
    raw_rows = [dict(item) for item in result.get("results", [])]
    resolved_config_data = json.loads(Path(resolved_config).read_text(encoding="utf-8"))
    event_log = _event_log_path(result, native_dir)
    rows = build_reference_episode_rows(raw_rows, event_log, job, method_key, resolved_config_data)
    train_log = _write_train_episode_log(workspace_dir / "train_episode_log.csv", rows, job["chapter"])
    summary = summarize_reference_rows(rows, method_label, job)
    write_json(workspace_dir / "summary.json", summary)
    _write_metric_csv(workspace_dir / "summary.csv", summary)

    seed_dir = workspace_dir.parent.parent
    suite_summary = _write_suite_summary(seed_dir, method_key, summary)
    run_manifest = {
        "chapter": job["chapter"],
        "group": job["group"],
        "scene": job["reference_scene"],
        "env_scene": job["env_scene"],
        "seed": job["seed"],
        "method": method_key,
        "workspace": str(workspace_dir),
        "train_episode_log": str(train_log),
        "summary_json": str(workspace_dir / "summary.json"),
        "summary_csv": str(workspace_dir / "summary.csv"),
        "suite_summary_csv": str(suite_summary),
        "native_artifacts": result.get("artifacts", {}),
        "native_event_log": str(event_log) if event_log is not None else None,
        "reference": _reference_pointer(job),
    }
    write_json(seed_dir / "run_manifest.json", run_manifest)
    emit_progress("[done] train_episode_log: %s" % train_log, progress_log)
    return run_manifest


def _workspace_dir(job: Dict[str, Any], method_key: str) -> Path:
    return (
        ROOT
        / job["output_dir"]
        / job["chapter"]
        / job["group"]
        / job["reference_scene"]
        / "three_seed_runs"
        / ("seed_%d" % job["seed"])
        / "_workspace"
        / method_key
    )


def _write_resolved_config(
    job: Dict[str, Any],
    default_config: str,
    native_dir: Path,
    method_label: str,
) -> Path:
    config_path = job["config"] or default_config
    config = load_config(ROOT / config_path, {})
    overrides: Dict[str, Any] = {
        "scene": job["env_scene"],
        "method": method_label,
        "episodes": job["episodes"],
        "seed": job["seed"],
        "output_dir": str(native_dir),
        "allow_scripted_fallback": True,
        "progress_interval": int(job.get("progress_interval", 100)),
        "step_progress_interval": int(job.get("step_progress_interval", 10)),
        "progress_log_path": str(_workspace_dir(job, CHAPTER_RUNNERS[job["chapter"]][1]) / "progress.log"),
    }
    if job.get("max_episode_steps") is not None:
        overrides["max_episode_steps"] = int(job["max_episode_steps"])

    # UGV 参考场景默认用 scripted 后端；真实复现时可通过 --backend unity/udp 切换。
    if job["env_scene"] in {"ugv_parking", "ugv_patrol"}:
        overrides["environment"] = {"backend": str(job["backend"])}
    elif job.get("backend"):
        overrides["environment"] = {"backend": str(job["backend"])}

    config = deep_update(config, overrides)
    resolved = native_dir / "resolved_config.json"
    write_json(resolved, config)
    return resolved


def _event_log_path(result: Dict[str, Any], native_dir: Path) -> Optional[Path]:
    artifacts = dict(result.get("artifacts") or {})
    summary_json = artifacts.get("summary_json")
    if summary_json:
        candidate = Path(summary_json).parent / "episode_events.jsonl"
        if candidate.exists():
            return candidate
    matches = list(native_dir.glob("**/episode_events.jsonl"))
    return matches[0] if matches else None


def _write_train_episode_log(path: Path, rows: List[Dict[str, Any]], chapter: str) -> Path:
    ensure_dir(path.parent)
    columns = columns_for_chapter(chapter)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _write_metric_csv(path: Path, summary: Dict[str, Any]) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key in sorted(summary):
            writer.writerow([key, summary[key]])
    return path


def _write_suite_summary(seed_dir: Path, method_key: str, summary: Dict[str, Any]) -> Path:
    path = seed_dir / "suite_summary.csv"
    suite_json = seed_dir / "suite_summary.json"
    rows: List[Dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [dict(row) for row in reader if row.get("method_key") != method_key]
    row = dict(summary)
    row["method_key"] = method_key
    row["scenario_key"] = summary.get("scenario")
    rows.append(row)
    columns = sorted({key for row_item in rows for key in row_item})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row_item in rows:
            writer.writerow(row_item)
    write_json(suite_json, rows)
    return path


def _reference_pointer(job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    item = find_reference_item(job["chapter"], job["group"], job["reference_scene"], job["reference_root"])
    if item is None:
        return None
    method_key = CHAPTER_METHOD_KEYS.get(job["chapter"], "")
    return {
        "suite_dir": item.destination,
        "seed_dir": "%s\\seed_%d" % (item.destination, job["seed"]),
        "method_dir": "%s\\seed_%d\\_workspace\\%s" % (item.destination, job["seed"], method_key),
    }


_BASE_RESULT_KEYS = {
    "scene",
    "method",
    "episode",
    "seed",
    "total_reward",
    "steps",
    "success",
    "done_reason",
}


if __name__ == "__main__":
    main()
