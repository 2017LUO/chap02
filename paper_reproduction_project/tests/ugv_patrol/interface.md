# UGV Patrol 环境接口与控制说明

本文档用于指导 Codex 实现 UGV Patrol 场景的 Python 接口。接口建议放在：

```text
envs/ugv_patrol_env.py
```

章节方法代码只能通过该接口控制 Patrol 环境。

## 1. 场景任务

UGV Patrol 场景包括外圈巡逻、检查点按序通过、绿灯条件下进入内圈、内圈巡逻、内外圈切换和最终泊车。

## 2. 推荐类定义

```python
class UGVPatrolEnv(BaseExperimentEnv):
    """UGV Patrol 场景统一接口。"""

    def __init__(self, env_config: dict):
        """初始化 Unity 连接、巡逻路线配置和技能策略配置。"""

    def reset(self, seed: int = 0, task_config: dict | None = None) -> dict:
        """重置巡逻场景并返回初始观测。"""

    def step(self, action) -> tuple[dict, float, bool, dict]:
        """执行一个底层连续控制动作。"""

    def apply_skill(self, skill_id: str, skill_args: dict | None = None) -> tuple[dict, float, bool, dict]:
        """执行一个技能策略或有限长度原子动作序列。"""

    def get_task_state(self) -> dict:
        """获取巡逻阶段、检查点和路线状态。"""

    def get_domain_knowledge(self) -> dict:
        """获取路线拓扑、区域规则和技能策略说明。"""

    def get_safety_state(self) -> dict:
        """获取安全相关状态。"""

    def get_event_info(self) -> dict:
        """获取当前步事件反馈。"""

    def close(self) -> None:
        """结束 Unity 连接。"""
```

## 3. reset 输入输出

```python
obs = env.reset(seed=0, task_config={
    "scene_name": "ugv_patrol",
    "start_route": "outer",
    "checkpoint_order": ["cp_01", "cp_02", "cp_03", "cp_04"],
    "traffic_light_mode": "dynamic",
    "final_parking_slot": "slot_01",
    "max_episode_steps": 1200
})
```

`obs` 至少包含：

```python
obs = {
    "image": image_array,
    "lidar": lidar_array,
    "low_dim": [speed, angular_speed, distance_to_goal, time_since_action, fault_flag],
    "vehicle_state": {},
    "task_state": {},
    "safety_state": {},
    "event_info": {}
}
```

## 4. step 控制

```python
action = {"steer": 0.1, "throttle": 0.4, "brake": 0.0}
obs, reward, done, info = env.step(action)
```

`info` 建议包含：

```python
info = {
    "route_mode": "outer",
    "checkpoint_id": "cp_01",
    "checkpoint_order_ok": True,
    "checkpoint_reached": False,
    "switch_success": False,
    "parking_success": False,
    "collision": False,
    "red_light_violation": False,
    "done_reason": "running",
    "reward_items": {}
}
```

## 5. apply_skill 控制

建议支持技能策略：

| skill_id | 功能 |
|---|---|
| `outer_route_follow` | 外圈巡逻 |
| `inner_route_follow` | 内圈巡逻 |
| `checkpoint_approach` | 接近检查点 |
| `pass_intersection` | 通过红绿灯路口 |
| `route_switch` | 内外圈切换 |
| `search_parking_slot` | 搜索泊车位 |
| `parking_control` | 最终泊车 |
| `safe_stop` | 安全停车 |

示例：

```python
obs, reward, done, info = env.apply_skill(
    skill_id="checkpoint_approach",
    skill_args={"checkpoint_id": "cp_02", "max_steps": 100}
)
```

## 6. get_task_state 字段

```python
task_state = {
    "scene_name": "ugv_patrol",
    "phase": "outer_patrol",
    "phase_index": 0,
    "route_mode": "outer",
    "checkpoint_id": "cp_01",
    "next_checkpoint_id": "cp_02",
    "checkpoint_order_ok": True,
    "distance_to_checkpoint": 12.4,
    "traffic_light_state": "green",
    "switch_area_id": "switch_01",
    "final_parking_slot": "slot_01",
    "stage_complete": False,
    "task_success": False
}
```

## 7. get_domain_knowledge 字段

```python
domain_knowledge = {
    "environment_topology": {
        "nodes": ["outer_cp_01", "outer_cp_02", "intersection", "inner_cp_01", "inner_cp_02", "parking_area"],
        "edges": [["outer_cp_01", "outer_cp_02"], ["outer_cp_02", "intersection"], ["intersection", "inner_cp_01"], ["inner_cp_01", "inner_cp_02"], ["inner_cp_02", "parking_area"]]
    },
    "region_rules": {
        "outer_route": ["checkpoint_order_required"],
        "inner_route": ["checkpoint_order_required"],
        "intersection": ["green_light_required"],
        "parking_area": ["low_speed_required", "slot_alignment_required"]
    },
    "skill_policies": {
        "outer_route_follow": "沿外圈路线行驶并接近检查点",
        "inner_route_follow": "沿内圈路线行驶并接近检查点",
        "checkpoint_approach": "接近并触发指定检查点",
        "route_switch": "在允许区域完成路线切换",
        "parking_control": "完成最终泊车"
    }
}
```

## 8. get_safety_state 字段

```python
safety_state = {
    "collision": False,
    "out_of_road": False,
    "red_light_violation": False,
    "min_obstacle_dist": 4.8,
    "front_obstacle_dist": 6.5,
    "speed": 3.2,
    "route_deviation": 0.4,
    "traffic_light_state": "green",
    "risk_score": 0.18,
    "recommended_safe_action": "keep"
}
```

## 9. 验收标准

- 可以通过 Python 接口启动 Patrol Unity 环境。
- 可以读取图像、雷达、车辆低维状态、路线模式、检查点、红绿灯和泊车状态。
- 可以执行连续控制和技能策略控制。
- 可以返回路线拓扑、区域规则和技能策略说明。
- 可以返回碰撞、越界、路线偏差、红灯和障碍物距离字段。
- 新增代码包含中文注释。
