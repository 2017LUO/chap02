# 第四章 EFSR 方法复现说明

本文档用于指导 Codex 复现第四章基于执行反馈与风险预测的动作行为安全约束方法。本文件只说明实验代码侧修改，默认第二章 DK-ABS 和第三章 LACM 的代码已经可以运行。

## 1. 复现目标

第四章代码需要在高层规划调度和高层决策补偿基础上加入两部分内容：一是将历史执行结果整理为执行反馈知识，并提供给后续高层规划和调度使用；二是在底层技能策略输出和环境执行之间加入风险预测与安全修正。

核心目标：

- 记录高层动作行为在不同场景、区域和阶段下的执行结果。
- 将执行结果转化为结构化执行反馈知识，供后续规划和调度检索。
- 训练风险价值函数，预测当前状态下原子动作的短期风险。
- 当原始原子动作风险较高时，从候选修正动作中选择风险较低且偏差较小的原子动作。
- 输出不安全回合率、整体任务成功率、平均累计回报和安全约束成功率。

## 2. 依赖环境接口

第四章主要用于两个无人车场景：

| 环境 | 接口类建议 | 用途 |
|---|---|---|
| UGV Parking | `UGVParkingEnv` | 复杂道路通行与街道泊车安全约束实验 |
| UGV Patrol | `UGVPatrolEnv` | 街道巡逻与泊车安全约束实验 |

环境接口必须额外提供安全相关信息：

```text
collision_flag          # 是否发生碰撞
red_light_violation     # 是否闯红灯
grass_violation         # 是否驶入草坪区域
out_of_lane_flag        # 是否越界或偏离有效道路
unsafe_event            # 是否发生任意不安全事件
risk_related_state      # 风险预测所需低维状态
raw_atomic_action        # 技能策略输出的原始原子动作
executed_atomic_action   # 实际发送给环境的原子动作
```

## 3. 与前两章代码关系

第四章不重新实现第二章和第三章内容，应直接复用已有模块。

| 依赖模块 | 使用方式 |
|---|---|
| 第二章领域知识 | 作为高层规划和调度的基础约束 |
| 第二章动作行为序列 | 作为当前执行目标 |
| 第二章在线调度 | 与执行反馈知识共同影响后续调度 |
| 第三章补偿决策 | 作为存在高层决策等待时的执行基础 |
| 共用技能策略接口 | 技能策略输出原始原子动作，安全层进行修正 |
| 共用日志模块 | 记录安全事件、风险值和修正动作 |

## 4. 建议代码目录

```text
src/chap04_efsr/
├── README.md
├── execution_feedback/
│   ├── feedback_schema.py
│   ├── feedback_collector.py
│   ├── feedback_memory.py
│   ├── feedback_retriever.py
│   └── prompt_feedback_adapter.py
├── risk_prediction/
│   ├── risk_dataset.py
│   ├── risk_value_network.py
│   ├── risk_trainer.py
│   └── risk_replay_buffer.py
├── safety_layer/
│   ├── atomic_action_space.py
│   ├── action_candidates.py
│   ├── safety_filter.py
│   └── safe_skill_executor.py
├── baselines/
│   ├── lacm_baseline.py
│   ├── efsr_no_feedback.py
│   └── efsr_no_risk_prediction.py
└── experiments/
    ├── run_efsr.py
    ├── run_efsr_ablation.py
    ├── evaluate_safety_metrics.py
    └── export_tables.py
```

## 5. 执行反馈知识模块

### 5.1 反馈数据结构

```python
@dataclass
class FeedbackDescriptor:
    scene: str
    task_type: str
    stage_id: str
    region_id: str
    action_behavior_id: str
    skill_id: str

@dataclass
class ExecutionFeedback:
    descriptor: FeedbackDescriptor
    stage_reward: float
    duration: float
    completed: bool
    unsafe_event: bool
    fail_type: str | None
```

### 5.2 反馈统计记录

`feedback_memory.py` 需要按描述符聚合统计：

```text
count
success_count
unsafe_count
avg_return
avg_duration
common_fail_type
last_update_step
```

### 5.3 反馈检索

```python
def retrieve_feedback(current_context: dict, top_k: int = 5) -> list[dict]:
    """检索与当前情形相近的执行反馈记录。"""
```

### 5.4 提示词适配

`prompt_feedback_adapter.py` 负责把结构化反馈记录转成高层规划和调度可使用的简短描述。描述中包含相似情形下的成功频率、不安全事件频率和常见失败类型。

## 6. 执行反馈更新规则

每个高层动作行为结束时都要生成执行反馈。触发条件包括：

```text
阶段完成
阶段失败
发生不安全事件
动作行为超时
高层调度切换到下一动作行为
```

建议规则：

```text
样本数小于 N_min=30 时，只记录，不作为强约束使用
安全完成频率大于 0.83 时，标记为较可靠
长期安全完成频率低于 0.51 时，标记为高风险
不安全事件频率较高时，后续规划和调度降低该动作行为优先级
```

执行反馈知识只作为高层规划和调度的补充输入，不替代环境拓扑、区域规则和技能策略领域知识。

## 7. 风险价值函数

输入：

```text
当前状态 x_t
原始原子动作 e_t
环境规则状态
短期不安全事件标记
```

输出：

```text
Q_risk(x_t, e_t)
```

数值越大表示执行该原子动作后的短期风险越高。

建议网络：

```text
状态编码 MLP
动作编码 Embedding 或 one-hot
状态动作拼接
两层 MLP，每层 256 维
输出一维风险值
```

训练配置示例：

```yaml
risk_prediction:
  gamma_risk: 0.95
  learning_rate: 0.0001
  batch_size: 128
  replay_capacity: 100000
  target_update_interval: 2000
  risk_threshold: 0.35
```

## 8. 安全约束层

安全约束层位于技能策略输出和环境执行之间。

流程：

```text
当前高层动作行为
对应技能策略输出原始原子动作
风险价值函数评估原始原子动作风险
若风险不超过阈值，直接执行原始原子动作
若风险超过阈值，构造候选修正原子动作集合
从候选集合中选择风险较低且偏差较小的原子动作
将修正后的原子动作发送给环境
记录 raw_action、safe_action、risk_value 和 safety_triggered
```

候选修正动作建议包括：保持原动作、减速、制动、轻微左偏修正、轻微右偏修正和保持车道中心。当所有候选动作风险都较高时，默认执行制动动作。

安全修正只作用于即将发送到环境的原子动作，不改变当前高层动作行为目标。

## 9. 运行入口

训练风险价值函数：

```bash
python scripts/train/train_chap04_risk.py \
  --config configs/chap04_efsr/efsr.yaml \
  --scene ugv_parking
```

运行 EFSR：

```bash
python scripts/eval/run_chap04_experiments.py \
  --config configs/chap04_efsr/efsr.yaml \
  --scene ugv_patrol \
  --method efsr \
  --seed 0
```

运行消融：

```bash
python scripts/eval/run_chap04_experiments.py \
  --config configs/chap04_efsr/efsr.yaml \
  --scene ugv_parking \
  --method efsr_no_risk_prediction \
  --seed 0
```

## 10. 日志字段

至少记录：

```text
episode_id, seed, scene, method, stage_id,
action_behavior_id, skill_id,
raw_action, safe_action, risk_value,
safety_triggered, safety_success, unsafe_event,
collision_flag, red_light_violation, grass_violation,
task_success, episode_return, feedback_descriptor
```

## 11. 指标

第四章至少计算：

```text
Score，平均累计回报
UVR，不安全回合率
SR，整体任务成功率
SCSR，安全约束成功率
```

`evaluate_safety_metrics.py` 必须能从日志中独立计算上述指标，并输出论文表格和绘图所需 csv。

## 12. 对比与消融方法

建议实现：

```text
LACM
EFSR
EFSR 去执行反馈
EFSR 去风险预测
```

所有方法应使用相同环境接口、相同随机种子和相同底层技能策略，差异只来自第四章新增部分。

## 13. 验收标准

- `feedback_collector.py` 能从回合日志中生成执行反馈记录。
- `feedback_memory.py` 能按描述符聚合历史执行结果。
- `feedback_retriever.py` 能返回当前情形下的历史反馈摘要。
- `risk_value_network.py` 能输入状态和原子动作，输出风险值。
- `safety_filter.py` 能在风险超过阈值时选择安全修正动作。
- `safe_skill_executor.py` 能把技能策略输出、安全层修正和环境执行串起来。
- 运行脚本能跑通两个无人车场景。
- 指标脚本能输出 UVR、SR、Score 和 SCSR。
- 新增代码均有中文注释。
