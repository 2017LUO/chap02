"""Helpers for the copied three-seed simulation reference data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_ROOT = PROJECT_ROOT / "results" / "reference_simulation"
REFERENCE_SEEDS: Tuple[int, ...] = (42, 43, 44)
REFERENCE_SCENES: Tuple[str, ...] = ("road_parking", "patrol_parking")
CHAPTERS: Tuple[str, ...] = ("chap02", "chap03", "chap04")
GROUPS: Tuple[str, ...] = ("comparison", "ablation")

REFERENCE_TO_ENV_SCENE = {
    "road_parking": "ugv_parking",
    "patrol_parking": "ugv_patrol",
    "ugv_parking": "ugv_parking",
    "ugv_patrol": "ugv_patrol",
}

ENV_TO_REFERENCE_SCENE = {
    "ugv_parking": "road_parking",
    "ugv_patrol": "patrol_parking",
    "road_parking": "road_parking",
    "patrol_parking": "patrol_parking",
}

CHAPTER_METHOD_KEYS = {
    "chap02": "dkabs",
    "chap03": "lacm",
    "chap04": "efsr",
}


@dataclass(frozen=True)
class ReferenceItem:
    """One copied reference suite entry."""

    chapter: str
    group: str
    scene: str
    source: str
    destination: str
    seeds: Tuple[str, ...]
    file_count: int
    size_mb: float

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "ReferenceItem":
        return cls(
            chapter=normalize_chapter(str(row.get("chapter", ""))),
            group=normalize_group(str(row.get("group", ""))),
            scene=reference_scene_name(str(row.get("scene", ""))),
            source=str(row.get("source", "")),
            destination=str(row.get("destination", "")),
            seeds=tuple(str(item) for item in row.get("seeds", [])),
            file_count=int(row.get("file_count", 0) or 0),
            size_mb=float(row.get("size_mb", 0.0) or 0.0),
        )

    def destination_path(self, project_root: Path = PROJECT_ROOT) -> Path:
        raw_path = Path(self.destination)
        if raw_path.is_absolute():
            return raw_path
        if raw_path.parts and raw_path.parts[0] == project_root.name:
            return project_root.parent / raw_path
        return project_root / raw_path


def normalize_chapter(chapter: str) -> str:
    value = chapter.strip().lower().replace("chapter", "chap")
    if value in {"2", "02", "chap2"}:
        return "chap02"
    if value in {"3", "03", "chap3"}:
        return "chap03"
    if value in {"4", "04", "chap4"}:
        return "chap04"
    if value in CHAPTERS:
        return value
    raise ValueError("Unsupported chapter: %s" % chapter)


def normalize_group(group: str) -> str:
    value = group.strip().lower()
    if value not in GROUPS:
        raise ValueError("Unsupported experiment group: %s" % group)
    return value


def env_scene_name(scene: str) -> str:
    value = scene.strip().lower()
    if value in REFERENCE_TO_ENV_SCENE:
        return REFERENCE_TO_ENV_SCENE[value]
    if value in {"crafter", "highway"}:
        return value
    raise ValueError("Unsupported scene: %s" % scene)


def reference_scene_name(scene: str) -> str:
    value = scene.strip().lower()
    if value in ENV_TO_REFERENCE_SCENE:
        return ENV_TO_REFERENCE_SCENE[value]
    if value in {"crafter", "highway"}:
        return value
    raise ValueError("Unsupported reference scene: %s" % scene)


def load_reference_manifest(reference_root: Optional[Path] = None) -> Dict[str, Any]:
    root = Path(reference_root or DEFAULT_REFERENCE_ROOT)
    manifest_path = root / "reference_manifest.json"
    if not manifest_path.exists():
        return {"items": []}
    return json.loads(manifest_path.read_text(encoding="utf-8-sig"))


def iter_reference_items(reference_root: Optional[Path] = None) -> Iterable[ReferenceItem]:
    manifest = load_reference_manifest(reference_root)
    for row in manifest.get("items", []):
        yield ReferenceItem.from_dict(row)


def find_reference_item(
    chapter: str,
    group: str,
    scene: str,
    reference_root: Optional[Path] = None,
) -> Optional[ReferenceItem]:
    chapter = normalize_chapter(chapter)
    group = normalize_group(group)
    scene = reference_scene_name(scene)
    for item in iter_reference_items(reference_root):
        if item.chapter == chapter and item.group == group and item.scene == scene:
            return item
    return None


def reference_suite_dir(
    chapter: str,
    group: str,
    scene: str,
    reference_root: Optional[Path] = None,
) -> Optional[Path]:
    item = find_reference_item(chapter, group, scene, reference_root)
    if item is None:
        return None
    return item.destination_path()


def validate_reference_snapshot(reference_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return one validation row per copied reference suite."""

    rows: List[Dict[str, Any]] = []
    expected = tuple("seed_%d" % seed for seed in REFERENCE_SEEDS)
    for item in iter_reference_items(reference_root):
        path = item.destination_path()
        existing = tuple(sorted(child.name for child in path.glob("seed_*") if child.is_dir()))
        rows.append(
            {
                "chapter": item.chapter,
                "group": item.group,
                "scene": item.scene,
                "path": str(path),
                "expected_seeds": expected,
                "existing_seeds": existing,
                "exists": path.exists(),
                "seed_set_complete": existing == expected,
            }
        )
    return rows
