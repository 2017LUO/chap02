# 环境侧接口补充说明

本文档只说明实验环境侧需要补充哪些数据接口和控制入口。三章方法代码不在本文件中实现。

## 1. 环境范围

当前实验环境有四类：

1. UGV Parking，无人车复杂道路通行与街道泊车，Unity 环境。
2. UGV Patrol，无人车街道巡逻与泊车，Unity 环境。
3. Crafter，制作石镐任务，Python 环境。
4. Highway，高速公路驾驶任务，Python 环境。

两个无人车环境需要在 Unity 与 Python 通信侧补充数据字段。Crafter 和 Highway 主要新增统一 Python 封装层，不修改原始环境核心逻辑。

## 2. 环境侧和实验代码侧边界

### 2.1 环境侧负责

- 返回当前观测，包括图像、雷达、车辆状态或环境状态。
- 返回任务阶段、目标、进度、检查点、泊车位、制作进度等状态。
- 返回事件信息，例如碰撞、闯红灯、通过检查点、采集资源、制作成功。
- 返回安全相关字段，例如障碍物距离、是否越界、红绿灯状态、风险字段。
- 接收 Python 端控制命令，包括连续控制量、离散动作和技能策略编号。
- 支持按随机种子和任务配置重置环境。

### 2.2 实验代码侧负责

- 实现第二章 DK-ABS、第三章 LACM、第四章 EFSR。
- 根据环境接口返回的数据构建领域知识、在线调度输入、补偿决策输入和安全约束输入。
- 调用环境接口执行技能策略或原子动作。
- 统计论文实验指标并保存日志、曲线数据和表格结果。

## 3. Unity 无人车通用观测字段

建议 Unity 侧或 Python 解析层统一输出如下结构：

```python
observation = {
    "image": image_array,              # 前视图像或环境相机图像
    "lidar": lidar_array,              # 雷达距离数组
    "low_dim": low_dim_array,          # 在线调度网络输入的低维状态
    "vehicle_state": {
        "position": [x, y, z],
        "rotation": [roll, pitch, yaw],
        "speed": speed,
        "angular_speed": angular_speed,
        "distance_to_goal": distance_to_goal,
        "time_since_action": time_since_action,
        "fault_flag": fault_flag
    },
    "task_state": {
        "scene_name": scene_name,
        "phase": phase,
        "target_id": target_id,
        "progress": progress
    },
    "safety_state": {
        "collision": collision,
        "out_of_road": out_of_road,
        "red_light_violation": red_light_violation,
        "min_obstacle_dist": min_obstacle_dist,
        "risk_score": risk_score
    },
    "event_info": {
        "event_type": event_type,
        "reward_items": reward_items,
        "done_reason": done_reason
    }
}
```

其中 `low_dim` 建议固定为：

```python
low_dim = [
    speed,                 # 当前速度
    angular_speed,         # 当前角速度
    distance_to_goal,      # 到当前阶段目标距离
    time_since_action,     # 当前动作行为持续时间
    fault_flag             # 异常标记
]
```

## 4. Unity 无人车通用控制字段

### 4.1 连续控制

```python
control = {
    "steer": steer,        # 转向，范围 [-1, 1]
    "throttle": throttle,  # 油门，范围 [0, 1]
    "brake": brake         # 刹车，范围 [0, 1]
}
```

### 4.2 技能策略控制

```python
skill_command = {
    "skill_id": "pass_intersection",
    "skill_args": {
        "target_id": "intersection_01",
        "max_steps": 120
    }
}
```

若技能策略在 Python 端实现，Unity 侧只需要执行每一步连续控制命令。

## 5. UGV Parking 专有字段

```python
parking_task_state = {
    "road_segment_id": road_id,
    "traffic_light_state": light_state,
    "target_ball_visible": visible,
    "target_ball_distance": ball_dist,
    "parking_slot_id": slot_id,
    "parking_slot_pose": slot_pose,
    "is_in_parking_area": in_area,
    "parking_success": success
}
```

## 6. UGV Patrol 专有字段

```python
patrol_task_state = {
    "route_mode": route_mode,             # outer / inner / parking
    "checkpoint_id": checkpoint_id,
    "next_checkpoint_id": next_id,
    "checkpoint_order_ok": order_ok,
    "traffic_light_state": light_state,
    "switch_area_id": switch_area_id,
    "parking_success": success
}
```

## 7. Crafter 封装字段

```python
crafter_state = {
    "image": image_array,
    "inventory": inventory_dict,
    "achievements": achievements,
    "health": health,
    "hunger": hunger,
    "thirst": thirst,
    "current_goal": goal,
    "event_info": event_info
}
```

Crafter 需要把离散动作编号和动作名称对应起来，例如移动、交互、放置工作台、制作木镐和制作石镐。

## 8. Highway 封装字段

```python
highway_state = {
    "ego_state": ego_state,
    "nearby_vehicles": nearby_vehicles,
    "lane_id": lane_id,
    "speed": speed,
    "time_step": t,
    "collision": collision,
    "reward": reward,
    "done_reason": done_reason
}
```

Highway 实验设置建议保持：4 条车道、约 50 辆干扰车、每回合 100 步、原始 5 个离散高层动作。

## 9. 环境侧验收标准

- Python 端可以通过统一接口启动四个环境。
- 两个 Unity 无人车环境可以返回图像、雷达、车辆低维状态、任务阶段、事件反馈和安全字段。
- Crafter 和 Highway 封装层可以返回统一结构的观测和任务状态。
- 四个环境都支持 `reset`、`step`、`apply_skill`、`get_task_state`、`get_event_info`、`get_safety_state`、`close`。
- 三章方法代码不直接读取 Unity 内部对象。
- 所有新增代码保留中文注释。
