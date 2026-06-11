# Crafter 环境接口与控制说明

本文档用于指导 Codex 实现 Crafter 场景的 Python 接口。接口建议放在：

```text
envs/crafter_env.py
```

Crafter 通常已经是 Python 环境，不要求修改原始环境核心逻辑。Codex 需要新增统一封装层，使其能被第二章、第三章和第四章代码以相同方式调用。

## 1. 场景任务

Crafter 任务目标是制作石镐。任务过程包括探索、采集木材、放置工作台、制作木镐、采集石头、制作石镐，以及避开僵尸等危险对象。

## 2. 推荐类定义

```python
class CrafterEnv(BaseExperimentEnv):
    """Crafter 场景统一接口。"""

    def __init__(self, env_config: dict):
        """初始化 Crafter 环境、动作映射和任务配置。"""

    def reset(self, seed: int = 0, task_config: dict | None = None) -> dict:
        """重置环境并返回初始观测。"""

    def step(self, action_id: int) -> tuple[dict, float, bool, dict]:
        """执行一个原始离散动作。"""

    def apply_skill(self, skill_id: str, skill_args: dict | None = None) -> tuple[dict, float, bool, dict]:
        """执行一个技能策略或原子动作序列。"""

    def get_task_state(self) -> dict:
        """获取制作阶段和资源状态。"""

    def get_domain_knowledge(self) -> dict:
        """获取资源、工具、制作规则和技能策略说明。"""

    def get_safety_state(self) -> dict:
        """获取生命值、危险对象和受伤事件等字段。"""

    def get_event_info(self) -> dict:
        """获取采集、制作、受伤、死亡等事件反馈。"""

    def close(self) -> None:
        """结束环境会话。"""
```

## 3. 动作映射

接口需要显式保存动作编号和动作名称，便于日志记录。

```python
ACTION_ID_TO_NAME = {
    0: "noop",
    1: "move_left",
    2: "move_right",
    3: "move_up",
    4: "move_down",
    5: "interact",
    6: "place_table",
    7: "make_wood_pickaxe",
    8: "make_stone_pickaxe"
}
```

如果当前项目中的 Crafter 动作编号不同，以已有环境为准，并在接口注释中写明。

## 4. reset 输入输出

```python
obs = env.reset(seed=0, task_config={
    "scene_name": "crafter",
    "goal": "make_stone_pickaxe",
    "max_episode_steps": 10000,
    "use_achievement_reward": True
})
```

`obs` 至少包含：

```python
obs = {
    "image": image_array,
    "inventory": {},
    "achievements": {},
    "player_state": {},
    "task_state": {},
    "safety_state": {},
    "event_info": {}
}
```

## 5. step 控制

```python
obs, reward, done, info = env.step(action_id=5)
```

`info` 建议包含：

```python
info = {
    "action_name": "interact",
    "achievement_unlocked": [],
    "inventory_delta": {"wood": 1},
    "health_delta": 0.0,
    "event_type": "collect_wood",
    "task_success": False,
    "done_reason": "running"
}
```

奖励规则建议保持原环境设置，不要随意修改奖励含义。

## 6. apply_skill 控制

建议支持技能策略：

| skill_id | 功能 |
|---|---|
| `explore` | 探索附近区域 |
| `collect_wood` | 采集木材 |
| `place_table` | 放置工作台 |
| `make_wood_pickaxe` | 制作木镐 |
| `collect_stone` | 采集石头 |
| `make_stone_pickaxe` | 制作石镐 |
| `avoid_zombie` | 避开僵尸 |
| `basic_action_sequence` | 执行原子动作序列 |

示例：

```python
obs, reward, done, info = env.apply_skill(
    skill_id="collect_wood",
    skill_args={"target_count": 3, "max_steps": 200}
)
```

## 7. get_task_state 字段

```python
task_state = {
    "scene_name": "crafter",
    "goal": "make_stone_pickaxe",
    "phase": "collect_wood",
    "phase_index": 0,
    "inventory": {
        "wood": 1,
        "stone": 0,
        "wood_pickaxe": 0,
        "stone_pickaxe": 0
    },
    "achievements": {
        "collect_wood": True,
        "make_wood_pickaxe": False,
        "collect_stone": False,
        "make_stone_pickaxe": False
    },
    "stage_complete": False,
    "task_success": False
}
```

## 8. get_domain_knowledge 字段

```python
domain_knowledge = {
    "environment_topology": {
        "regions": ["forest", "stone_area", "safe_area", "danger_area"],
        "objects": ["tree", "stone", "workbench", "zombie"]
    },
    "region_rules": {
        "forest": ["wood_available"],
        "stone_area": ["stone_available", "wood_pickaxe_required"],
        "danger_area": ["zombie_risk"],
        "workbench_area": ["tool_crafting_available"]
    },
    "craft_rules": {
        "make_wood_pickaxe": ["wood_required", "workbench_required"],
        "make_stone_pickaxe": ["stone_required", "wood_pickaxe_required", "workbench_required"]
    },
    "skill_policies": {
        "collect_wood": "采集木材，为制作工具准备资源",
        "place_table": "放置工作台，支持后续工具制作",
        "make_wood_pickaxe": "制作木镐，用于采集石头",
        "collect_stone": "采集石头，为制作石镐准备资源",
        "make_stone_pickaxe": "制作石镐，完成任务目标",
        "avoid_zombie": "远离危险对象，减少生命值损失"
    }
}
```

## 9. get_safety_state 字段

```python
safety_state = {
    "health": 8,
    "health_delta": 0.0,
    "nearby_zombie": False,
    "zombie_distance": None,
    "danger_level": 0.0,
    "is_dead": False,
    "recommended_safe_action": "keep"
}
```

## 10. 验收标准

- 可以通过统一接口启动和重置 Crafter 环境。
- 可以执行原始离散动作和技能策略。
- 可以返回图像、背包、成就、生命值、制作阶段和事件反馈。
- 可以返回资源规则、制作规则和技能策略说明。
- 可以支持制作石镐任务的阶段推进统计和整体任务成功率统计。
- 新增代码包含中文注释。
