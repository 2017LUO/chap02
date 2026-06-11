"""registry.py – Task–Skill registry (static for 5-skill CLC_Rl)
=============================================================
Maintains a persistent mapping **task_id → metadata** where metadata =
    {
        "desc"      : str  # natural-language description
        "skill_ckpt": str  # relative/absolute path to frozen Actor weights (.pth)
        "extra"     : dict # optional misc info
    }
Registry is stored in JSON (default `tasks.json`) sitting next to this script.
This version ships with **five pre-defined skills**:
    0  Lane Keeping              → ckpts/avoid_policy.pth
    1  Dynamic Obstacle Avoidance→ ckpts/avoid_policy.pth
    2  Hit Target Sphere         → ckpts/hit_sphere_policy.pth
    3  Enter Street              → ckpts/enter_street.pth
    4  Precision Parking         → ckpts/park_policy.pth
"""
from __future__ import annotations

import json
import pathlib
from typing import Dict, Any, List

__all__ = ["TaskRegistry", "RegistryItem"]
RegistryItem = Dict[str, Any]


class TaskRegistry:
    """JSON-backed task registry.  Read-only by default in deployment."""

    def __init__(self, path: str | pathlib.Path | None = None, *, readonly: bool = True):
        default_json = pathlib.Path(__file__).with_name("tasks.json")
        self.path = pathlib.Path(path) if path is not None else default_json
        self.readonly = readonly
        self._load()

        # auto-seed the five skills if JSON missing/empty and not readonly
        if not self.readonly and len(self.db) == 0:
            self._seed_defaults()

    # -----------------------------------------------------------
    def _load(self) -> None:
        if self.path.exists():
            try:
                self.db: Dict[str, RegistryItem] = json.loads(self.path.read_text("utf-8"))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Registry file {self.path} is corrupted: {e}")
        else:
            self.db = {}

    def _save(self) -> None:
        if self.readonly:
            raise PermissionError("Registry is readonly; cannot modify file.")
        self.path.write_text(json.dumps(self.db, indent=2, ensure_ascii=False), encoding="utf-8")

    # -----------------------------------------------------------
    # Public API
    # -----------------------------------------------------------
    def list_ids(self) -> List[int]:
        return sorted(int(k) for k in self.db.keys())

    def list_tasks(self) -> List[tuple[int, str]]:
        return [(tid, self.db[str(tid)]["desc"]) for tid in self.list_ids()]

    def get(self, task_id: int) -> RegistryItem:
        return self.db[str(task_id)]

    def skill_ckpt(self, task_id: int) -> str:
        return self.get(task_id)["skill_ckpt"]

    def __len__(self):
        return len(self.db)

    def __contains__(self, task_id: int):
        return str(task_id) in self.db

    # -----------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------
    def _seed_defaults(self):
        defaults = {
            0: ("Lane Keeping", "ckpts/avoid_policy.pth"),
            1: ("Dynamic Obstacle Avoidance", "ckpts/avoid_policy.pth"),
            2: ("Hit Target Sphere", "ckpts/hit_sphere_policy.pth"),
            3: ("Enter Street", "ckpts/enter_street.pth"),
            4: ("Precision Parking", "ckpts/park_policy.pth"),
        }
        for tid, (desc, ckpt) in defaults.items():
            self.db[str(tid)] = {"desc": desc, "skill_ckpt": ckpt, "extra": {}}
        self._save()

    # -----------------------------------------------------------
    def _pretty_print(self):
        print("TaskRegistry  (id → desc → ckpt)")
        for tid in self.list_ids():
            meta = self.get(tid)
            print(f"  {tid:2d}  {meta['desc']:<30}  {meta['skill_ckpt']}")


if __name__ == "__main__":
    reg = TaskRegistry(readonly=False)  # allow writing for demo
    reg._pretty_print()