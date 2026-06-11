"""Smoke tests for chapter experiment modules."""

from __future__ import annotations

import unittest

from src.chap02_dk_abs.planning import generate_action_behavior_plan, validate_plan
from src.chap02_dk_abs.planning.llm_planner import generate_llm_action_behavior_plan
from src.chap02_dk_abs.planning.prompt_builder import build_planning_messages
from src.chap03_lacm.compensation import CompensationPolicy
from src.chap03_lacm.latency import LatencySampler
from src.chap04_efsr.risk_prediction import RiskValueFunction
from src.common.data_structures import ActionBehavior


class MethodModuleSmokeTest(unittest.TestCase):
    def test_crafter_plan_validates(self) -> None:
        plan = generate_action_behavior_plan("crafter", {}, {}, {"wood_target_count": 4})
        ok, errors = validate_plan("crafter", plan)
        self.assertTrue(ok, errors)
        self.assertEqual([item.skill_id for item in plan][-1], "make_stone_pickaxe")

    def test_ugv_parking_plan_validates(self) -> None:
        plan = generate_action_behavior_plan("ugv_parking", {"phase": "main_road"}, {}, {"planning_horizon": 7})
        ok, errors = validate_plan("ugv_parking", plan)
        self.assertTrue(ok, errors)
        self.assertEqual(plan[0].skill_id, "main_road_driving")
        self.assertIn("parking_control", [item.skill_id for item in plan])

    def test_lacm_policy_selects_conservative_behavior_on_risk(self) -> None:
        obs = {"safety_state": {"risk_score": 0.8}, "task_state": {"speed": 30.0, "target_speed": 30.0}}
        policy = CompensationPolicy(risk_threshold=0.45, max_hold_steps=1)
        behavior, info = policy.select("highway", obs, ActionBehavior("speed_up", max_steps=1), waiting_steps=2)
        self.assertIn(behavior.skill_id, {"safe_follow", "slow_down", "keep_lane"})
        self.assertEqual(info["mode"], "confidence_reselect")

    def test_latency_sampler_converts_real_inference_seconds_to_steps(self) -> None:
        sampler = LatencySampler.from_config(
            {"min_seconds": 1.0, "max_seconds": 1.8, "control_dt_seconds": 0.2},
            seed=0,
        )
        sample = sampler.sample_latency()
        self.assertGreaterEqual(sample.inference_seconds, 1.0)
        self.assertLessEqual(sample.inference_seconds, 1.8)
        self.assertGreaterEqual(sample.steps, 5)
        self.assertLessEqual(sample.steps, 9)

    def test_risk_value_uses_rule_risk_without_feedback(self) -> None:
        obs = {
            "safety_state": {
                "risk_score": 0.7,
                "front_vehicle_dist": 6.0,
                "left_lane_safe": False,
                "right_lane_safe": True,
            }
        }
        risk = RiskValueFunction().predict("highway", obs, "speed_up")
        self.assertGreaterEqual(risk, 0.9)

    def test_simulated_openai_planner_uses_real_prompt(self) -> None:
        domain = {
            "skill_policies": {
                "collect_wood": "collect wood",
                "place_table": "place table",
                "make_wood_pickaxe": "craft wood pickaxe",
                "collect_stone": "collect stone",
                "make_stone_pickaxe": "craft stone pickaxe",
            }
        }
        config = {
            "mode": "llm_simulated",
            "wood_target_count": 4,
            "stone_target_count": 1,
            "llm": {"provider": "simulated", "model": "gpt-4o-mini-sim"},
        }
        messages = build_planning_messages("crafter", {"phase": "collect_wood"}, domain, config)
        self.assertIn("<PLANNING_CONTEXT_JSON>", messages[1]["content"])
        result = generate_llm_action_behavior_plan("crafter", {"phase": "collect_wood"}, domain, config)
        self.assertEqual(result.provider, "simulated")
        self.assertEqual(result.plan[0].skill_id, "collect_wood")
        self.assertEqual(result.plan[-1].skill_id, "make_stone_pickaxe")

    def test_simulated_openai_planner_starts_from_current_crafter_phase(self) -> None:
        domain = {
            "skill_policies": {
                "collect_wood": "collect wood",
                "place_table": "place table",
                "make_wood_pickaxe": "craft wood pickaxe",
                "collect_stone": "collect stone",
                "make_stone_pickaxe": "craft stone pickaxe",
            }
        }
        result = generate_llm_action_behavior_plan(
            "crafter",
            {"phase": "collect_stone"},
            domain,
            {"mode": "llm_simulated", "llm": {"provider": "simulated"}},
        )
        self.assertEqual(result.plan[0].skill_id, "collect_stone")
        self.assertEqual(result.plan[1].skill_id, "make_stone_pickaxe")

    def test_simulated_openai_highway_uses_safety_state_for_five_actions(self) -> None:
        domain = {
            "skill_policies": {
                "lane_left": "change left if safe",
                "keep_lane": "keep lane",
                "lane_right": "change right if safe",
                "speed_up": "accelerate",
                "slow_down": "decelerate",
            }
        }
        task_state = {
            "phase": "driving",
            "lane_id": 1,
            "lanes_count": 4,
            "speed": 28.0,
            "target_speed": 30.0,
            "safety_state": {
                "front_vehicle_dist": 8.0,
                "relative_speed_front": -4.0,
                "left_lane_safe": True,
                "right_lane_safe": False,
                "risk_score": 0.55,
            },
            "nearby_vehicles": [{"lane_id": 1, "distance": 8.0, "relative_speed": -4.0}],
        }
        result = generate_llm_action_behavior_plan(
            "highway",
            task_state,
            domain,
            {"mode": "llm_simulated", "planning_horizon": 5, "llm": {"provider": "simulated"}},
        )
        self.assertEqual(result.plan[0].skill_id, "lane_left")
        self.assertEqual(len(result.plan), 5)
        self.assertLessEqual({item.skill_id for item in result.plan}, set(domain["skill_policies"]))

    def test_simulated_openai_highway_overtakes_when_adjacent_gap_is_better(self) -> None:
        domain = {
            "skill_policies": {
                "lane_left": "change left if safe",
                "keep_lane": "keep lane",
                "lane_right": "change right if safe",
                "speed_up": "accelerate",
                "slow_down": "decelerate",
            }
        }
        task_state = {
            "phase": "driving",
            "lane_id": 2,
            "lanes_count": 4,
            "speed": 29.0,
            "target_speed": 30.0,
            "safety_state": {
                "front_vehicle_dist": 28.0,
                "relative_speed_front": -0.5,
                "left_lane_safe": True,
                "right_lane_safe": False,
                "risk_score": 0.15,
            },
            "nearby_vehicles": [
                {"lane_id": 2, "distance": 28.0, "relative_speed": -0.5},
                {"lane_id": 1, "distance": 72.0, "relative_speed": 0.0},
                {"lane_id": 3, "distance": 18.0, "relative_speed": 0.0},
            ],
        }
        result = generate_llm_action_behavior_plan(
            "highway",
            task_state,
            domain,
            {"mode": "llm_simulated", "planning_horizon": 5, "llm": {"provider": "simulated"}},
        )
        self.assertEqual(result.plan[0].skill_id, "lane_left")

    def test_simulated_openai_ugv_starts_from_current_phase(self) -> None:
        domain = {
            "skill_policies": {
                "main_road_driving": "drive",
                "obstacle_avoidance": "avoid",
                "target_ball_clear": "clear target",
                "pass_intersection": "pass light",
                "enter_street": "enter street",
                "search_parking_slot": "search slot",
                "parking_control": "park",
                "safe_stop": "stop",
            }
        }
        result = generate_llm_action_behavior_plan(
            "ugv_parking",
            {"phase": "intersection"},
            domain,
            {"mode": "llm_simulated", "llm": {"provider": "simulated"}},
        )
        self.assertEqual(result.plan[0].skill_id, "pass_intersection")


if __name__ == "__main__":
    unittest.main()
