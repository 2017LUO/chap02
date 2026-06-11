# 项目目录结构说明

本文档规定论文方法复现实验代码的推荐目录结构。Codex 修改代码时应先整理目录，再实现环境接口和三章方法代码。

## 1. 推荐根目录

建议项目根目录命名为：

```text
paper_reproduction_project/
```

推荐结构如下：

```text
paper_reproduction_project/
├── README.md
├── requirements.txt
├── docs/                                # Codex 说明文件
│   ├── 00_docs_index.md
│   ├── 01_project_structure.md
│   ├── 10_environment_interfaces/
│   └── 20_method_reproduction/
├── envs/                                # 四个实验环境的统一封装
│   ├── __init__.py
│   ├── base_env.py
│   ├── env_registry.py
│   ├── unity_bridge/
│   │   ├── unity_client.py
│   │   ├── observation_parser.py
│   │   └── command_builder.py
│   ├── ugv_parking_env.py
│   ├── ugv_patrol_env.py
│   ├── crafter_env.py
│   └── highway_env_wrapper.py
├── src/                                 # 模型与方法源程序
│   ├── common/
│   ├── chap02_dk_abs/
│   ├── chap03_lacm/
│   └── chap04_efsr/
├── configs/                             # 所有实验配置
│   ├── envs/
│   ├── llm/
│   ├── chap02_dk_abs/
│   ├── chap03_lacm/
│   └── chap04_efsr/
├── scripts/                             # 可执行训练、评估和绘图入口
│   ├── train/
│   ├── eval/
│   └── plot/
├── checkpoints/                         # 模型权重
│   ├── chap02_dk_abs/
│   ├── chap03_lacm/
│   └── chap04_efsr/
├── results/                             # 原始日志、指标、图片和表格
│   ├── raw_logs/
│   ├── metrics/
│   ├── figures/
│   └── tables/
└── tests/                               # 基础测试
    ├── test_env_interfaces.py
    ├── crafter/
    │   ├── interface.md
    │   └── manual_control.py
    ├── highway/
    │   ├── interface.md
    │   └── manual_control.py
    ├── ugv_parking/
    │   └── interface.md
    └── ugv_patrol/
        └── interface.md
```

## 2. `docs/` 目录

`docs/` 只放说明文件，不放训练代码、模型权重和实验结果。根目录只保留两个入口文件：

```text
docs/
├── 00_docs_index.md
├── 01_project_structure.md
├── 02_python_environment.md
├── 10_environment_interfaces/
└── 20_method_reproduction/
```

这样可以避免多个 README 同时存在导致 Codex 不知道先读哪个文件。

## 2.1 Python 运行环境

本项目后续 Python 命令固定使用 Conda 环境 `DT`。测试、训练、评估和可视化脚本都应按下面形式运行：

```bash
conda run -n DT python <script_or_module>
```

除非用户明确指定，不要切换到其他 Python 解释器或 Conda 环境。

## 3. `envs/` 目录

`envs/` 只负责封装四个实验环境。三章方法代码都通过该目录调用环境。

建议统一基类：

```python
class BaseExperimentEnv:
    """四类实验环境的统一接口基类。"""

    def reset(self, seed: int | None = None, task_config: dict | None = None) -> dict:
        """重置环境并返回初始观测。"""
        raise NotImplementedError

    def step(self, action):
        """执行一个底层动作，返回 obs, reward, done, info。"""
        raise NotImplementedError

    def apply_skill(self, skill_id: str, skill_args: dict | None = None):
        """执行一个技能策略或有限长度原子动作序列。"""
        raise NotImplementedError

    def get_task_state(self) -> dict:
        """返回当前任务阶段、目标和进度。"""
        raise NotImplementedError

    def get_domain_knowledge(self) -> dict:
        """返回环境拓扑、区域规则和技能策略说明。"""
        raise NotImplementedError

    def get_safety_state(self) -> dict:
        """返回风险预测和安全约束所需字段。"""
        raise NotImplementedError

    def get_event_info(self) -> dict:
        """返回当前步发生的事件。"""
        raise NotImplementedError

    def close(self) -> None:
        """释放环境资源。"""
        raise NotImplementedError
```

## 4. `src/common/` 目录

`src/common/` 放三章共用代码，避免每章重复实现。

```text
src/common/
├── data_structures.py       # TaskConfig、ActionBehavior、RuntimeObs 等统一数据结构
├── skill_registry.py        # 高层动作行为到技能策略的映射
├── llm_client.py            # 本地 vLLM 或模拟 LLM 调用
├── llm_output_parser.py     # 解析大语言模型输出
├── logger.py                # jsonl/csv 日志
├── metrics.py               # 通用指标计算
├── seed.py                  # 随机种子设置
└── config_loader.py         # yaml/json 配置读取
```

## 5. 三章代码目录

### 5.1 第二章 DK-ABS

```text
src/chap02_dk_abs/
├── domain_knowledge/
├── planning/
├── online_scheduling/
├── baselines/
└── experiments/
```

### 5.2 第三章 LACM

```text
src/chap03_lacm/
├── latency/
├── confidence/
├── compensation/
├── baselines/
└── experiments/
```

### 5.3 第四章 EFSR

```text
src/chap04_efsr/
├── execution_feedback/
├── risk_prediction/
├── safety_layer/
├── baselines/
└── experiments/
```

## 6. 配置目录

```text
configs/
├── envs/
│   ├── ugv_parking.yaml
│   ├── ugv_patrol.yaml
│   ├── crafter.yaml
│   └── highway.yaml
├── llm/
│   └── llama33_70b_vllm.yaml
├── chap02_dk_abs/
│   └── dk_abs.yaml
├── chap03_lacm/
│   └── lacm.yaml
└── chap04_efsr/
    └── efsr.yaml
```

论文中的关键参数应写入配置文件，例如 `K_rel=12`、`K_max=12`、`H=8`、GRU 隐层维度 `128`、触发阈值 `0.60`、风险阈值和延迟采样范围。

## 7. 脚本目录

```text
scripts/
├── train/
│   ├── train_chap02_trigger.py
│   ├── train_chap03_confidence.py
│   ├── train_chap03_compensation.py
│   └── train_chap04_risk.py
├── eval/
│   ├── run_chap02_experiments.py
│   ├── run_chap03_experiments.py
│   └── run_chap04_experiments.py
└── plot/
    ├── plot_chap02_figures.py
    ├── plot_chap03_figures.py
    └── plot_chap04_figures.py
```

## 8. 验收标准

- 根目录存在 `docs/`、`envs/`、`src/`、`configs/`、`scripts/`、`checkpoints/`、`results/` 和 `tests/`。
- 三章方法分别放在 `src/chap02_dk_abs/`、`src/chap03_lacm/` 和 `src/chap04_efsr/`。
- 四类环境通过 `envs/` 统一调用。
- 三章方法代码不直接访问 Unity 对象或第三方环境内部变量。
- 所有配置文件集中放在 `configs/`。
- 所有 Python 命令默认使用 Conda 环境 `DT`。
- 所有新增代码有中文注释。
