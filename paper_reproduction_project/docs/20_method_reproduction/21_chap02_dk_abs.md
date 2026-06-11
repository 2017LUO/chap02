# 第二章 DK-ABS 方法复现说明

本文档用于指导 Codex 复现第二章基于领域知识约束的动作行为规划与在线调度方法。本文件只说明实验代码侧需要新增或修改的模块，不负责修改环境源码。

## 1. 复现目标

第二章代码需要实现从任务配置、领域知识、规划动作行为序列到在线调度执行的完整流程。

核心目标：

- 建立环境拓扑、区域规则和技能策略三类领域知识。
- 根据任务配置筛选任务相关领域知识。
- 调用大语言模型生成规划动作行为序列，序列长度不超过 `K_max=12`。
- 将规划动作行为映射到底层技能策略或有限长度原子动作序列。
- 使用图像、雷达和低维运行状态训练在线调度网络。
- 当当前动作行为与运行时状态不一致时，触发局部调度并更新后续动作行为。
- 输出高层规划正确合规率、平均阶段完成率、整体任务成功率和平均累计回报。

## 2. 依赖环境接口

第二章至少需要支持：

| 环境 | 接口类建议 | 用途 |
|---|---|---|
| UGV Parking | `UGVParkingEnv` | 无人车复杂道路通行与街道泊车实验 |
| UGV Patrol | `UGVPatrolEnv` | 无人车街道巡逻与泊车实验 |
| Crafter | `CrafterEnv` | 制作石镐任务实验 |

本章代码从环境接口获得：

```text
image                 # 图像观测
lidar                 # 雷达或占据信息
low_dim               # 速度、角速度、目标距离、动作持续时间、异常标记等
high_level_state      # 当前阶段、可见对象、异常标记、任务进度等
task_progress         # 阶段完成状态、是否成功、回合奖励等
available_skills      # 当前场景可用技能策略集合
```

## 3. 建议代码目录

```text
src/chap02_dk_abs/
├── README.md
├── domain_knowledge/
│   ├── topology_builder.py
│   ├── region_rules.py
│   ├── skill_descriptors.py
│   ├── knowledge_schema.py
│   └── task_relevance.py
├── planning/
│   ├── prompt_builder.py
│   ├── plan_generator.py
│   ├── plan_parser.py
│   ├── plan_validator.py
│   └── plan_repair.py
├── online_scheduling/
│   ├── trigger_dataset.py
│   ├── trigger_network.py
│   ├── trigger_trainer.py
│   ├── runtime_state_encoder.py
│   └── scheduler.py
├── baselines/
│   ├── hsd3_baseline.py
│   ├── saycan_baseline.py
│   └── dilu_baseline.py
└── experiments/
    ├── run_dk_abs.py
    ├── run_baselines.py
    ├── run_ablation.py
    ├── evaluate_metrics.py
    └── export_tables.py
```

## 4. 核心数据结构

建议在 `src/common/data_structures.py` 中统一定义：

| 数据结构 | 建议字段 | 说明 |
|---|---|---|
| `TaskConfig` | `scene`、`task_type`、`goal`、`stage_list`、`success_condition` | 任务配置 |
| `DomainKnowledge` | `topology_graph`、`region_rules`、`skill_set`、`entity_list` | 完整领域知识 |
| `TaskKnowledge` | `G_tau`、`R_tau`、`S_tau`、`selected_items` | 筛选后的领域知识 |
| `ActionBehavior` | `name`、`target`、`region`、`stage`、`skill_id`、`precondition`、`terminal_condition` | 高层动作行为 |
| `HighLevelState` | `visible_objects`、`stage_label`、`abnormal_flag`、`progress` | 高层状态 |
| `RuntimeObs` | `image`、`lidar`、`low_dim`、`event_flags` | 在线调度网络输入 |

## 5. 领域知识模块

### 5.1 环境拓扑

```python
def build_topology(scene_name: str, env_meta: dict) -> dict:
    """根据场景名称和环境元信息构建拓扑图。"""


def query_neighbors(topology: dict, region_id: str) -> list[str]:
    """查询某一区域的相邻区域。"""


def find_reachable_path(topology: dict, start: str, goal: str) -> list[str]:
    """查询起点到目标区域的可达路径。"""
```

### 5.2 区域规则

```python
def load_region_rules(scene_name: str, config_path: str) -> dict:
    """读取区域规则配置。"""


def check_action_allowed(action_behavior, high_level_state, region_rules: dict) -> tuple[bool, str]:
    """判断当前动作行为是否满足区域规则。"""
```

### 5.3 技能策略说明

无人车技能策略建议包括：主干道通行、障碍物避让、目标球体清除、路口通过、街道进入、搜索泊车位、泊车控制、巡逻检查点通过、内外圈切换。

Crafter 技能策略建议包括：木材采集、石头采集、避开僵尸、放置工作台、制作木镐、制作石镐、基础移动和交互动作。

### 5.4 任务相关度筛选

```python
def select_task_knowledge(task_config, domain_knowledge, k_rel: int = 12):
    """根据任务配置筛选与当前任务最相关的领域知识。"""
```

## 6. 大语言模型规划模块

规划输入包括：任务描述、当前场景、任务目标、环境拓扑摘要、区域规则摘要、候选技能策略集合、当前高层状态和任务阶段列表。

LLM 输出必须解析为结构化动作行为序列：

```json
{
  "plan": [
    {
      "action_name": "主干道通行",
      "target": "路口区域",
      "region": "main_road",
      "stage": "approach_intersection",
      "skill_id": "main_road_driving"
    }
  ]
}
```

`plan_validator.py` 需要检查：

- 动作行为是否在候选技能集合内。
- 目标区域是否存在于拓扑图。
- 动作行为是否违反区域规则。
- 动作顺序是否符合任务阶段。
- 序列长度是否超过 `K_max=12`。

检查失败时进行一次修复；仍失败时回退到规则允许的默认动作行为。

## 7. 在线调度网络

在线调度网络输入最近 `H=8` 步运行时观测序列。

输入字段：

```text
image_t
lidar_t
low_dim_t = [速度, 角速度, 到目标距离, 当前动作持续时间, 最近失败标记]
current_action_id
stage_id
```

建议网络结构：

```text
图像编码器 CNN
雷达编码器 MLP
低维状态编码 MLP
拼接特征
GRU，隐层维度 128
线性层 + Sigmoid
输出触发概率 p_t
```

触发阈值：

```text
delta_trig = 0.60
```

当 `p_t >= delta_trig` 时，调用在线调度模块更新当前或后续动作行为。

## 8. 运行入口

```bash
python scripts/eval/run_chap02_experiments.py \
  --config configs/chap02_dk_abs/dk_abs.yaml \
  --scene ugv_parking \
  --method dk_abs \
  --seed 0
```

训练在线调度网络：

```bash
python scripts/train/train_chap02_trigger.py \
  --config configs/chap02_dk_abs/dk_abs.yaml \
  --scene ugv_parking
```

## 9. 日志字段

至少记录：

```text
episode_id, seed, scene, method, step, stage_id,
plan_action_id, current_action_id, trigger_prob, triggered,
skill_id, stage_completed, task_success, episode_return, plan_valid
```

## 10. 指标

本章至少计算：

```text
高层规划正确合规率
平均阶段完成率
整体任务成功率
平均累计回报
训练奖励曲线
```

## 11. 验收标准

- DK-ABS 能在 UGV Parking、UGV Patrol 和 Crafter 中跑通完整回合。
- 领域知识模块能输出拓扑、规则和技能描述。
- 大语言模型输出能被解析为结构化动作行为序列。
- 在线调度网络可以训练、保存和加载。
- 指标脚本可以从日志自动输出论文表格数据。
- 新增代码均有中文注释。
