# Highway 环境接口与控制说明

本文档用于指导 Codex 实现 Highway 场景的 Python 接口。接口建议放在：

```text
envs/highway_env_wrapper.py
```

Highway 通常基于 highway-env 或同类 Gym 接口，不要求修改原始环境核心逻辑。Codex 需要新增统一封装层，使章节方法可以通过统一接口控制环境。

## 1. 场景任务

Highway 场景为多车道高速公路驾驶任务。实验设置建议保持：

- 4 条车道。
- 约 50 辆干扰车辆。
- 每回合 100 步。
- 使用原始 5 个离散高层动作。
- 主要用于第三章高层决策延迟和补偿决策实验。

## 2. 推荐类定义

```python
class HighwayEnvWrapper(BaseExperimentEnv):
    """Highway 场景统一接口。"""

    def __init__(self, env_config: dict):
        """初始化 Highway 环境和动作映射。"""

    def reset(self, seed: int = 0, task_config: dict | None = None) -> dict:
        """重置环境，返回初始观测。"""

    def step(self, action_id: int) -> tuple[dict, float, bool, dict]:
        """执行一个 Highway 原始离散动作。"""

    def apply_skill(self, skill_id: str, skill_args: dict | None = None) -> tuple[dict, float, bool, dict]:
        """将技能策略映射为 Highway 离散动作或短动作序列。"""

    def get_task_state(self) -> dict:
        """获取车道、速度、步数和当前行驶目标。"""

    def get_domain_knowledge(self) -> dict:
        """获取车道拓扑、驾驶规则和动作说明。"""

    def get_safety_state(self) -> dict:
        """获取碰撞风险、车距和相对速度等字段。"""

    def get_event_info(self) -> dict:
        """获取碰撞、变道、速度变化和回合结束原因。"""

    def get_compensation_candidates(self) -> list[str]:
        """返回高层决策等待期间可选择的补偿动作行为。"""

    def close(self) -> None:
        """结束环境会话。"""
```

## 3. 动作映射

```python
ACTION_ID_TO_NAME = {
    0: "lane_left",
    1: "idle",
    2: "lane_right",
    3: "faster",
    4: "slower"
}
```

如果项目中的 Highway 动作编号不同，以项目实际编号为准，并在接口注释中写明。

## 4. reset 输入输出

```python
obs = env.reset(seed=0, task_config={
    "scene_name": "highway",
    "lanes_count": 4,
    "vehicles_count": 50,
    "max_episode_steps": 100,
    "duration": 100
})
```

`obs` 至少包含：

```python
obs = {
    "raw_obs": raw_obs,
    "ego_state": {},
    "nearby_vehicles": [],
    "task_state": {},
    "safety_state": {},
    "event_info": {}
}
```

## 5. step 控制

```python
obs, reward, done, info = env.step(action_id=3)
```

`info` 建议包含：

```python
info = {
    "action_name": "faster",
    "lane_id": 1,
    "speed": 28.0,
    "collision": False,
    "lane_change": False,
    "time_step": 10,
    "task_success": False,
    "done_reason": "running"
}
```

如果 Gymnasium 环境返回 `terminated` 和 `truncated`，接口中合并为统一 `done`，并在 `done_reason` 中说明原因。

## 6. apply_skill 控制

建议支持技能策略：

| skill_id | 功能 | 对应动作 |
|---|---|---|
| `keep_lane` | 保持车道 | `idle` |
| `lane_left` | 向左变道 | `lane_left` |
| `lane_right` | 向右变道 | `lane_right` |
| `speed_up` | 加速 | `faster` |
| `slow_down` | 减速 | `slower` |
| `safe_follow` | 安全跟车 | `idle` 或 `slower` |

示例：

```python
obs, reward, done, info = env.apply_skill(
    skill_id="safe_follow",
    skill_args={"target_speed": 25.0, "max_steps": 5}
)
```

## 7. get_task_state 字段

```python
task_state = {
    "scene_name": "highway",
    "phase": "driving",
    "time_step": 10,
    "max_episode_steps": 100,
    "lane_id": 1,
    "target_lane_id": None,
    "speed": 28.0,
    "target_speed": 30.0,
    "distance_travelled": 120.0,
    "task_success": False
}
```

## 8. get_domain_knowledge 字段

```python
domain_knowledge = {
    "environment_topology": {
        "lanes": [0, 1, 2, 3],
        "adjacent_lanes": {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    },
    "region_rules": {
        "leftmost_lane": ["no_left_lane_change"],
        "rightmost_lane": ["no_right_lane_change"],
        "dense_traffic": ["safe_distance_required"],
        "front_vehicle_close": ["slow_down_preferred"]
    },
    "skill_policies": {
        "keep_lane": "保持当前车道行驶",
        "lane_left": "在左侧车道安全时向左变道",
        "lane_right": "在右侧车道安全时向右变道",
        "speed_up": "在前方安全时加速",
        "slow_down": "在前车较近或风险较高时减速",
        "safe_follow": "根据前车距离调整速度"
    }
}
```

## 9. get_safety_state 字段

```python
safety_state = {
    "collision": False,
    "front_vehicle_dist": 18.5,
    "rear_vehicle_dist": 12.0,
    "left_lane_safe": True,
    "right_lane_safe": False,
    "relative_speed_front": -2.0,
    "time_to_collision": 6.2,
    "speed": 28.0,
    "risk_score": 0.21,
    "recommended_safe_action": "keep"
}
```

## 10. 第三章延迟实验字段

第三章需要模拟高层决策延迟，接口建议返回：

```python
latency_state = {
    "time_step": t,
    "last_high_level_action": last_action,
    "waiting_for_high_level_decision": waiting,
    "elapsed_wait_steps": wait_steps,
    "candidate_compensation_actions": ["keep_lane", "safe_follow", "slow_down", "lane_left", "lane_right"]
}
```

## 11. 验收标准

- 可以通过统一接口启动和重置 Highway 环境。
- 可以固定 4 条车道、约 50 辆干扰车、每回合 100 步的实验设置。
- 可以执行原始 5 个离散动作。
- 可以返回自车状态、周围车辆状态、车道信息、奖励、碰撞和回合结束原因。
- 可以支持第三章高层决策延迟和补偿决策实验。
- 新增代码包含中文注释。
