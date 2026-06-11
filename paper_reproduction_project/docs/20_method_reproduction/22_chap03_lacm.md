# 第三章 LACM 方法复现说明

本文档用于指导 Codex 复现第三章基于延迟感知与动作行为置信度评估的高层决策补偿方法。本文件只说明实验代码侧修改，默认第二章 DK-ABS 的领域知识、规划动作行为序列、技能策略接口和环境接口已经可以使用。

## 1. 复现目标

第三章代码需要在第二章基础上加入高层决策等待建模、动作行为置信度评估和补偿决策训练，使智能体在大语言模型推理结果尚未返回时仍能继续推进任务。

核心目标：

- 把大语言模型请求改为异步调用，记录每次高层请求等待时间。
- 在等待期间根据当前状态选择补偿动作行为。
- 训练动作行为置信度评估网络，判断候选动作行为与当前状态的匹配程度。
- 训练补偿决策策略，在等待、沿用规划动作行为和置信度重选之间进行选择。
- 输出平均任务完成时间、平均高层决策等待时间、补偿决策占比、平均累计回报和平均阶段完成率。

## 2. 依赖环境接口

第三章需要支持：

| 环境 | 接口类建议 | 用途 |
|---|---|---|
| UGV Parking | `UGVParkingEnv` | 延迟补偿在泊车任务中的验证 |
| UGV Patrol | `UGVPatrolEnv` | 延迟补偿在巡逻泊车任务中的验证 |
| Highway | `HighwayEnvWrapper` | 高速公路驾驶任务验证 |

本章需要从环境接口获取：

```text
runtime_observation       # 当前图像、雷达、低维状态等
high_level_state          # 当前阶段、目标、异常标记和任务进度
candidate_actions         # 当前可选动作行为或技能策略集合
task_progress             # 阶段完成、回合成功、累计回报和时间统计
```

## 3. 与第二章代码关系

第三章不重新实现领域知识和规划动作行为序列生成，应复用第二章模块。

| 第二章模块 | 第三章使用方式 |
|---|---|
| `TaskConfig` | 读取场景、任务目标和阶段信息 |
| `DomainKnowledge` | 限制候选动作行为范围 |
| `ActionBehavior` | 作为补偿决策选择的基本单位 |
| `skill_registry.py` | 将补偿动作行为映射到底层技能策略 |
| `plan_validator.py` | 检查补偿动作行为是否满足当前约束 |
| `logger.py` | 记录等待时间、补偿决策和任务结果 |

## 4. 建议代码目录

```text
src/chap03_lacm/
├── README.md
├── latency/
│   ├── async_llm_client.py
│   ├── latency_sampler.py
│   ├── request_state.py
│   └── wait_time_tracker.py
├── confidence/
│   ├── confidence_dataset.py
│   ├── confidence_network.py
│   ├── confidence_trainer.py
│   └── candidate_selector.py
├── compensation/
│   ├── compensation_action_space.py
│   ├── compensation_policy.py
│   ├── compensation_trainer.py
│   ├── replay_buffer.py
│   └── reward_model.py
├── baselines/
│   ├── oracle_policy.py
│   ├── dk_abs_delay_no_comp.py
│   ├── saycan_delay.py
│   └── dilu_delay.py
└── experiments/
    ├── run_lacm.py
    ├── run_delay_baselines.py
    ├── run_lacm_ablation.py
    ├── evaluate_lacm_metrics.py
    └── export_tables.py
```

## 5. 异步高层请求

`async_llm_client.py` 负责模拟或真实调用本地大语言模型服务。需要支持两种模式：

```text
real_vllm        # 调用真实本地 vLLM 服务
simulated_delay  # 不调用模型，只按配置采样等待时间
```

建议接口：

```python
class AsyncLLMClient:
    def submit_request(self, request_payload: dict) -> str:
        """提交高层决策请求，返回 request_id。"""

    def poll_result(self, request_id: str) -> tuple[bool, dict | None]:
        """查询请求是否完成，完成时返回动作行为结果。"""

    def cancel_request(self, request_id: str) -> None:
        """取消已经不需要的请求。"""
```

`wait_time_tracker.py` 记录请求发起时间、请求返回时间、阶段等待时间、回合累计等待时间和平均高层决策等待时间。

## 6. 延迟配置

```yaml
latency:
  mode: simulated_delay
  min_seconds: 1.0
  max_seconds: 1.8
  control_dt_seconds: 0.2
  request_interval: 6
compensation:
  wait_cost_per_second: 0.02
  switch_cost: 0.10
```

模拟 LLM 请求先采样真实推理耗时 `inference_seconds`，再按 `control_dt_seconds` 换算成环境等待步数。日志中同时保留 `sampled_inference_seconds`、`sampled_latency_steps` 和 `waiting_seconds`。如果使用真实 vLLM 服务，则用真实请求耗时覆盖采样时延，同时保留这些字段。

## 7. 置信度评估网络

输入：

```text
当前运行时状态编码
当前高层状态编码
任务阶段编号
候选动作行为编码
候选技能策略编码
```

输出：

```text
confidence_score[action_id]
```

训练数据从第二章和第三章日志构造：阶段成功推进的动作行为作为正样本；长时间等待、切换后失败或阶段无法推进的动作行为作为负样本。

建议网络：

```text
状态编码 MLP 或 CNN+MLP
动作行为编码 Embedding
状态动作拼接
两层 MLP，隐层维度 256
Sigmoid 或 Softmax 输出置信度
```

## 8. 补偿决策策略

代码中可以用英文常量，论文写作中不需要展开这些英文枚举。

| 代码枚举 | 含义 |
|---|---|
| `WAIT_HOLD` | 短时间保持等待或执行安全保持动作 |
| `FOLLOW_PLAN` | 沿用当前规划动作行为序列 |
| `SELECT_CONFIDENT` | 选择置信度更高的候选动作行为 |

策略输入建议包含：当前状态编码、当前阶段编号、已等待时间、候选动作行为最高置信度、当前规划动作行为置信度、是否接近阶段终止条件。

## 9. 有效回报计算

```python
def compute_effective_reward(task_reward: float, wait_time: float, switch_flag: bool, config: dict) -> float:
    """计算补偿决策训练使用的有效回报。"""
```

配置示例：

```yaml
reward:
  wait_cost: 0.20
  switch_cost: 0.10
  confidence_bonus: 0.05
```

## 10. 运行入口

训练置信度网络：

```bash
python scripts/train/train_chap03_confidence.py \
  --config configs/chap03_lacm/lacm.yaml \
  --scene ugv_parking
```

训练补偿策略：

```bash
python scripts/train/train_chap03_compensation.py \
  --config configs/chap03_lacm/lacm.yaml \
  --scene highway
```

运行 LACM：

```bash
python scripts/eval/run_chap03_experiments.py \
  --config configs/chap03_lacm/lacm.yaml \
  --scene ugv_patrol \
  --method lacm \
  --seed 0
```

## 11. 日志字段

至少记录：

```text
episode_id, seed, scene, method, stage_id,
request_id, request_start_time, request_end_time,
stage_wait_time, episode_wait_time,
compensation_action, selected_action_id, confidence_score,
switch_flag, stage_completed, task_success,
episode_return, episode_duration
```

## 12. 指标

第三章至少计算：

```text
平均任务完成时间
平均高层决策等待时间
补偿决策占比
平均累计回报
平均阶段完成率
```

## 13. 对比方法

建议实现：

```text
Oracle
DK-ABS
DK-ABS+Delay/NoComp
SayCan+Delay
DiLu+Delay
LACM
LACM 去沿用规划补偿
LACM 去置信度重选补偿
```

所有延迟相关方法必须使用相同的等待设置。

## 14. 验收标准

- `AsyncLLMClient` 支持真实请求和采样延迟两种模式。
- 置信度网络可以训练、保存和加载。
- 补偿策略可以在等待期间输出补偿决策，并映射到动作行为或技能策略。
- 运行脚本能在 Highway、UGV Parking 和 UGV Patrol 中跑通完整回合。
- 指标脚本能输出平均任务完成时间、平均高层决策等待时间、补偿决策占比和平均累计回报。
- 新增代码均有中文注释。
