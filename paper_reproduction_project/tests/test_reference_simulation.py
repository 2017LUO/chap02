"""Tests for the copied simulation reference snapshot."""

from __future__ import annotations

import unittest

from src.common.reference_simulation import (
    env_scene_name,
    find_reference_item,
    reference_scene_name,
    validate_reference_snapshot,
)


class ReferenceSimulationTest(unittest.TestCase):
    def test_scene_aliases_match_reproduction_envs(self) -> None:
        self.assertEqual(env_scene_name("road_parking"), "ugv_parking")
        self.assertEqual(env_scene_name("patrol_parking"), "ugv_patrol")
        self.assertEqual(reference_scene_name("ugv_parking"), "road_parking")
        self.assertEqual(reference_scene_name("ugv_patrol"), "patrol_parking")

    def test_reference_manifest_contains_three_seed_suites(self) -> None:
        item = find_reference_item("chap03", "comparison", "road_parking")
        self.assertIsNotNone(item)
        assert item is not None
        self.assertEqual(item.seeds, ("seed_42", "seed_43", "seed_44"))
        self.assertTrue(item.destination_path().exists())

    def test_all_reference_suites_keep_seed_set(self) -> None:
        rows = validate_reference_snapshot()
        self.assertEqual(len(rows), 12)
        missing = [row for row in rows if not row["seed_set_complete"]]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
