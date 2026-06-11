# 论文复现实验项目

本项目按 `docs/` 中的说明组织代码、配置和测试。当前已完成第一阶段环境统一封装：

- `envs/CrafterEnv`：支持制作石镐任务，提供动作映射、技能策略、任务阶段、安全字段和事件反馈。
- `envs/HighwayEnvWrapper`：支持 4 车道、约 50 辆干扰车、100 步回合设置，提供第三章延迟补偿所需字段。
- `envs/UGVParkingEnv` 和 `envs/UGVPatrolEnv`：提供 scripted 调试后端，并预留 ML-Agents Unity 通信后端。
- `tests/ugv_manual_control.py`：支持 Unity/scripted 后端下的手动键盘控制与代码规划控制切换。

## 快速验证

```bash
conda run -n DT python -m unittest tests.test_env_interfaces
```

本项目默认使用 Conda 环境 `DT`，当前按 `Python 3.8.19` 兼容维护；不要按 Python 3.10+ 语法新增代码。

## 参考数据与训练入口

已将 `Paper_Code` 中的三种子模拟训练结果复制到 `results/reference_simulation/`，其中每个套件保留 `seed_42`、`seed_43`、`seed_44`。复现实验统一入口为：

```bash
conda run -n DT python scripts/train/run_reproduction_training.py --chapter chap03 --group comparison --scene road_parking --seeds 42 43 44 --episodes 4000
```

先检查目录规划而不运行训练：

```bash
conda run -n DT python scripts/train/run_reproduction_training.py --chapter all --group comparison --scene all --dry-run
```

测试训练但不保留测试数据：

```bash
conda run --no-capture-output -n DT python scripts/train/run_reproduction_training.py --chapter chap03 --group comparison --scene road_parking --seeds 42 --episodes 10 --test-run
```

## 当前环境入口

```python
from envs import create_env

crafter = create_env("crafter")
obs = crafter.reset(seed=0)
crafter.apply_skill("collect_wood", {"target_count": 3})
crafter.apply_skill("place_table")
crafter.apply_skill("make_wood_pickaxe")
crafter.apply_skill("collect_stone", {"target_count": 3})
obs, reward, done, info = crafter.apply_skill("make_stone_pickaxe")

highway = create_env("highway")
obs = highway.reset(seed=0)
obs, reward, done, info = highway.apply_skill("safe_follow", {"max_steps": 3})
```

真实 Crafter 或 highway-env 接入后，可在 `configs/envs/*.yaml` 中把 `backend` 从 `scripted` 改为 `gym`/`highway_env`，或在实例化时传入已有后端对象。
