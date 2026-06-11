"""环境统一接口基础测试。"""

from __future__ import annotations

import unittest

from envs import CrafterEnv, HighwayEnvWrapper, available_envs, create_env


class EnvRegistryTest(unittest.TestCase):
    """验证四类环境都能通过统一注册器访问。"""

    def test_available_envs(self) -> None:
        self.assertEqual(available_envs(), ["crafter", "highway", "ugv_parking", "ugv_patrol"])

    def test_create_env(self) -> None:
        self.assertIsInstance(create_env("crafter"), CrafterEnv)
        self.assertIsInstance(create_env("highway"), HighwayEnvWrapper)


class CrafterEnvTest(unittest.TestCase):
    """验证 Crafter 石镐任务可以通过技能接口跑通。"""

    def test_make_stone_pickaxe_flow(self) -> None:
        env = CrafterEnv({"backend": "scripted"})
        obs = env.reset(seed=0)
        self.assertEqual(obs["task_state"]["phase"], "collect_wood")

        env.apply_skill("collect_wood", {"target_count": 4, "max_steps": 10})
        self.assertEqual(env.get_task_state()["phase"], "place_table")

        env.apply_skill("place_table")
        env.apply_skill("make_wood_pickaxe")
        env.apply_skill("collect_stone", {"target_count": 3, "max_steps": 10})
        obs, reward, done, info = env.apply_skill("make_stone_pickaxe")

        self.assertTrue(done)
        self.assertGreater(reward, 0.0)
        self.assertTrue(info["task_success"])
        self.assertTrue(obs["task_state"]["task_success"])
        self.assertEqual(obs["inventory"]["stone_pickaxe"], 1)


class HighwayEnvTest(unittest.TestCase):
    """验证 Highway 封装返回第三章需要的统一字段。"""

    def test_highway_step_and_compensation_fields(self) -> None:
        env = HighwayEnvWrapper({"backend": "scripted", "lanes_count": 4, "vehicles_count": 50, "duration": 100})
        obs = env.reset(seed=1)
        self.assertEqual(obs["task_state"]["max_episode_steps"], 100)
        self.assertIn("safe_follow", obs["latency_state"]["candidate_compensation_actions"])

        obs, reward, done, info = env.step("faster")
        self.assertIsInstance(reward, float)
        self.assertIn(info["action_name"], {"faster"})
        self.assertIn("front_vehicle_dist", obs["safety_state"])
        self.assertFalse(done)

        obs, _, _, info = env.apply_skill("safe_follow", {"max_steps": 3})
        self.assertEqual(info["skill_id"], "safe_follow")
        self.assertLessEqual(info["skill_steps"], 3)
        self.assertIn("latency_state", obs)


class UGVScriptedEnvTest(unittest.TestCase):
    """UGV scripted backends should expose the same interface as Unity."""

    def test_ugv_parking_skill_flow(self) -> None:
        env = create_env("ugv_parking", {"backend": "scripted"})
        obs = env.reset(seed=0)
        self.assertEqual(obs["task_state"]["phase"], "main_road")
        self.assertIn("main_road_driving", env.get_domain_knowledge()["skill_policies"])

        for skill_id in [
            "main_road_driving",
            "obstacle_avoidance",
            "target_ball_clear",
            "pass_intersection",
            "enter_street",
            "search_parking_slot",
            "parking_control",
        ]:
            obs, reward, done, info = env.apply_skill(skill_id)
            self.assertGreaterEqual(reward, 0.0)
            self.assertTrue(info["stage_complete"])

        self.assertTrue(done)
        self.assertTrue(obs["task_state"]["task_success"])

    def test_ugv_manual_control_mode_sends_continuous_action(self) -> None:
        env = create_env("ugv_parking", {"backend": "scripted"})
        env.reset(seed=0)
        env.set_control_mode("manual")
        obs, _, _, info = env.manual_step(steer=-0.5, motor=0.75, duration_steps=2)

        self.assertEqual(env.get_control_mode(), "manual")
        self.assertEqual(info["control_mode"], "manual")
        self.assertEqual(obs["control_state"]["control_mode"], "manual")
        self.assertAlmostEqual(obs["control_state"]["last_control_action"]["steer"], -0.5)
        self.assertAlmostEqual(obs["control_state"]["last_control_action"]["motor"], 0.75)

    def test_ugv_patrol_skill_flow(self) -> None:
        env = create_env("ugv_patrol", {"backend": "scripted"})
        obs = env.reset(seed=0)
        self.assertEqual(obs["task_state"]["phase"], "outer_patrol")

        for skill_id in [
            "outer_route_follow",
            "checkpoint_approach",
            "pass_intersection",
            "route_switch",
            "inner_route_follow",
            "search_parking_slot",
            "parking_control",
        ]:
            obs, _, done, _ = env.apply_skill(skill_id)

        self.assertTrue(done)
        self.assertTrue(obs["task_state"]["task_success"])


if __name__ == "__main__":
    unittest.main()
