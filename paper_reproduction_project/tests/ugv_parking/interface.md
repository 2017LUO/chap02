# UGV Parking 环境接口与控制说明

本文档用于指导 Codex 实现 UGV Parking 场景的 Python 接口。接口建议放在：

```text
envs/ugv_parking_env.py
```

章节方法代码只能调用该接口，不直接读取 Unity 场景对象。

## 1. 场景任务

UGV Parking 场景包括主干道行驶、障碍物避让、黄色目标球体清除、绿灯路口通过、街道进入、泊车位搜索和指定车位泊车。

## 2. 推荐类定义

```python
class UGVParkingEnv(BaseExperimentEnv):
    """UGV Parking 场景统一接口。"""

    def __init__(self, env_config: dict):
        """初始化 Unity 连接、场景参数和技能策略配置。"""

    def reset(self, seed: int = 0, task_config: dict | None = None) -> dict:
        """按随机种子和任务配置重置环境，返回初始观测。"""

    def step(self, action) -> tuple[dict, float, bool, dict]:
        """执行一个底层连续动作。"""

    def apply_skill(self, skill_id: str, skill_args: dict | None = None) -> tuple[dict, float, bool, dict]:
        """执行一个技能策略或有限长度原子动作序列。"""

    def get_task_state(self) -> dict:
        """获取当前任务阶段和目标状态。"""

    def get_domain_knowledge(self) -> dict:
        """获取环境拓扑、区域规则和技能策略说明。"""

    def get_safety_state(self) -> dict:
        """获取风险预测和安全约束所需字段。"""

    def get_event_info(self) -> dict:
        """获取当前步事件反馈。"""

    def close(self) -> None:
        """结束 Unity 连接并释放资源。"""
```

## 3. reset 输入输出

```python
obs = env.reset(seed=0, task_config={
    "scene_name": "ugv_parking",
    "start_pose_id": "default",
    "target_slot_id": "slot_01",
    "obstacle_density": "normal",
    "traffic_light_mode": "dynamic",
    "max_episode_steps": 1000
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
action = {
    "steer": 0.0,
    "throttle": 0.3,
    "brake": 0.0
}
obs, reward, done, info = env.step(action)
```

`info` 建议包含：

```python
info = {
    "phase": "main_road",
    "stage_success": False,
    "task_success": False,
    "collision": False,
    "red_light_violation": False,
    "parking_success": False,
    "done_reason": "running",
    "reward_items": {}
}
```

## 5. apply_skill 控制

```python
obs, reward, done, info = env.apply_skill(
    skill_id="pass_intersection",
    skill_args={"target_id": "intersection_01", "max_steps": 120}
)
```

建议支持技能策略：

| skill_id | 功能 |
|---|---|
| `main_road_driving` | 主干道通行 |
| `obstacle_avoidance` | 障碍物避让 |
| `target_ball_clear` | 黄色目标球体清除 |
| `pass_intersection` | 绿灯条件下通过路口 |
| `enter_street` | 街道进入 |
| `search_parking_slot` | 搜索泊车位 |
| `parking_control` | 泊车控制 |
| `safe_stop` | 安全停车 |

## 6. get_task_state 字段

```python
task_state = {
    "scene_name": "ugv_parking",
    "phase": "main_road",
    "phase_index": 0,
    "target_id": "target_ball_01",
    "target_position": [x, y, z],
    "distance_to_goal": dist,
    "traffic_light_state": "green",
    "road_segment_id": "road_main_01",
    "parking_slot_id": "slot_01",
    "is_in_parking_area": False,
    "stage_complete": False,
    "task_success": False
}
```

## 7. get_domain_knowledge 字段

```python
domain_knowledge = {
    "environment_topology": {
        "nodes": ["start", "main_road", "intersection", "street", "parking_area"],
        "edges": [["start", "main_road"], ["main_road", "intersection"], ["intersection", "street"], ["street", "parking_area"]]
    },
    "region_rules": {
        "intersection": ["green_light_required"],
        "grass_area": ["forbidden"],
        "parking_area": ["low_speed_required", "slot_alignment_required"]
    },
    "skill_policies": {
        "main_road_driving": "沿主干道行驶",
        "obstacle_avoidance": "避开障碍物",
        "target_ball_clear": "接近并清除黄色目标球体",
        "pass_intersection": "在绿灯条件下通过路口",
        "parking_control": "完成指定车位泊车"
    }
}
```

## 8. get_safety_state 字段

```python
safety_state = {
    "collision": False,
    "out_of_road": False,
    "red_light_violation": False,
    "min_obstacle_dist": 5.2,
    "front_obstacle_dist": 8.0,
    "left_obstacle_dist": 3.5,
    "right_obstacle_dist": 4.1,
    "speed": 3.0,
    "traffic_light_state": "green",
    "risk_score": 0.12,
    "recommended_safe_action": "keep"
}
```

## 9. 验收标准

- 可以通过 Python 接口启动 Parking Unity 环境。
- 可以读取图像、雷达、低维状态、任务阶段、红绿灯、泊车状态和事件反馈。
- 可以执行连续控制和技能策略控制。
- 可以返回环境拓扑、区域规则和技能策略说明。
- 可以返回风险预测和安全约束所需字段。
- 新增代码包含中文注释。
