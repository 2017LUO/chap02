# Codex 说明文件目录

本目录用于指导 Codex 在已有实验代码项目上复现论文第二章、第三章和第四章方法。说明文件分成两类，避免环境接口修改和方法代码修改混在一起。

## 1. 目录结构

```text
docs/
├── 00_docs_index.md                         # 本文件，说明所有 md 的用途和阅读顺序
├── 01_project_structure.md                  # 项目总目录结构，规定代码、配置、日志和结果怎么放
├── 02_python_environment.md                  # Python 运行环境说明，固定使用 conda 环境 DT
├── 10_environment_interfaces/               # 实验环境接口说明，只说明环境给 Python 端哪些接口
│   ├── 10_env_side.md                       # 环境侧需要补充的数据字段和控制入口
│   ├── 11_ugv_parking.md                    # UGV Parking 接口与控制说明
│   ├── 12_ugv_patrol.md                     # UGV Patrol 接口与控制说明
│   ├── 13_crafter.md                        # Crafter 接口与控制说明
│   └── 14_highway.md                        # Highway 接口与控制说明
└── 20_method_reproduction/                  # 三章方法复现说明，只说明实验代码怎么写
    ├── 21_chap02_dk_abs.md                  # 第二章 DK-ABS 方法复现说明
    ├── 22_chap03_lacm.md                    # 第三章 LACM 方法复现说明
    └── 23_chap04_efsr.md                    # 第四章 EFSR 方法复现说明
```

## 2. 两类说明文件的区别

### 2.1 `10_environment_interfaces/`

该目录只负责说明实验环境接口。Codex 需要根据这些文件完成四个环境的 Python 封装，使实验代码可以统一调用环境。

四个实验环境包括：

- UGV Parking，无人车复杂道路通行与街道泊车，Unity 环境。
- UGV Patrol，无人车街道巡逻与泊车，Unity 环境。
- Crafter，制作石镐任务，Python 环境。
- Highway，高速公路驾驶任务，Python 环境。

环境接口目录不写三章算法细节，只说明：

- `reset` 返回什么观测。
- `step` 接收什么动作。
- `apply_skill` 如何执行技能策略。
- `get_task_state` 返回什么任务阶段信息。
- `get_domain_knowledge` 返回什么领域知识。
- `get_safety_state` 返回什么安全相关字段。
- `get_event_info` 返回什么执行事件。

### 2.2 `20_method_reproduction/`

该目录只负责说明第二章、第三章和第四章实验代码怎么修改。方法代码只能通过 `envs/` 下的环境封装控制环境，不直接访问 Unity 场景对象或第三方环境内部对象。

三章代码分别放在：

```text
src/chap02_dk_abs/
src/chap03_lacm/
src/chap04_efsr/
```

## 3. 推荐阅读顺序

Codex 修改项目时按下面顺序阅读：

1. `00_docs_index.md`，了解说明文件目录。
2. `01_project_structure.md`，先整理项目目录。
3. `02_python_environment.md`，确认后续 Python 命令固定使用 conda 环境 `DT`。
4. `10_environment_interfaces/10_env_side.md`，明确环境侧需要提供哪些字段。
5. `10_environment_interfaces/11_ugv_parking.md` 到 `14_highway.md`，实现四个环境的统一接口。
6. `20_method_reproduction/21_chap02_dk_abs.md`，实现第二章 DK-ABS。
7. `20_method_reproduction/22_chap03_lacm.md`，在第二章基础上实现第三章 LACM。
8. `20_method_reproduction/23_chap04_efsr.md`，在前两章基础上实现第四章 EFSR。

## 4. Codex 总体要求

- 环境接口和方法代码分开修改。
- 代码按章节组织，不把三章方法写到同一个大文件中。
- 共用数据结构、日志、指标计算和 LLM 调用放在 `src/common/`。
- 所有训练参数写入 `configs/`，不要直接写死在脚本里。
- 所有实验结果写入 `results/`，并保留原始日志。
- 新增 Python 代码需要中文注释，尤其是观测字段、任务阶段、动作行为、奖励和安全字段。
- 所有 Python 命令默认通过 `conda run -n DT python ...` 执行，除非用户明确要求更换环境。

## 5. 可直接给 Codex 的提示词

```text
请先阅读 docs/00_docs_index.md 和 docs/01_project_structure.md。当前任务分为两部分：第一部分按照 docs/10_environment_interfaces 中的说明实现四个实验环境的统一 Python 接口；第二部分按照 docs/20_method_reproduction 中的说明复现第二章 DK-ABS、第三章 LACM 和第四章 EFSR。请保持环境接口和方法代码分开，新增代码保留中文注释，配置写入 configs，日志和结果写入 results。
```
