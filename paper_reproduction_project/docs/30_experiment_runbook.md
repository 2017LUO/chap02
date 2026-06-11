# 三章实验运行说明

本项目 Python 命令统一使用本机 Conda 环境 `DT`：

```bash
conda run -n DT python <script>
```

当前已经接入并验证的官方环境：

- `crafter`：使用官方 `crafter.Env`，用于第二章 DK-ABS 的制作石镐任务。
- `highway`：使用官方 `highway-env` / Gymnasium `highway-v0`，用于第三章 LACM 和第四章 EFSR。

Unity 环境（`ugv_parking`、`ugv_patrol`）现在可先用 scripted 后端跑通三章方法 smoke test；真实 Unity 联调时把环境配置中的 `backend` 改为 `unity`，并通过 ML-Agents side-channel 接收运行状态。

手动/代码控制切换调试入口：

```bash
conda run -n DT python tests/ugv_manual_control.py
```

窗口中 `manual` 模式会把键盘输入转换成连续 `steer/motor` 动作发送给 Unity；`code` 模式会调用同一个环境接口执行规划器技能，两种模式可随时切换。

## 大模型调用层

第二章和第三章的规划入口已经使用 OpenAI 风格接口：

```python
client.chat.completions.create(...)
```

当前默认 provider 是 `simulated`，不会联网，也不会消耗 API。提示词是真实构造的，包含任务状态、领域知识、可用技能和 JSON 输出 schema；第二章默认每个高层技能执行前重新请求一次模拟大模型，并在结果目录保存 `llm_plan_episode_*_request_*.json`，方便检查每一步的 prompt 和模拟响应。

后续接真实 API 时，优先只修改配置中的 `planner.llm.provider/model/api_key/base_url`，实验主流程不需要改。

## 第二章 DK-ABS

运行官方 Crafter 制作石镐实验：

```bash
conda run -n DT python scripts/eval/run_chap02_experiments.py --config configs/chap02_dk_abs/dk_abs.yaml --scene crafter
```

训练/保存轻量触发模型：

```bash
conda run -n DT python scripts/train/train_chap02_trigger.py --config configs/chap02_dk_abs/dk_abs.yaml --scene crafter
```

## 第三章 LACM

运行官方 Highway 延迟补偿实验：

```bash
conda run -n DT python scripts/eval/run_chap03_experiments.py --config configs/chap03_lacm/lacm.yaml --scene highway
```

保存置信度选择器和补偿策略：

```bash
conda run -n DT python scripts/train/train_chap03_confidence.py --config configs/chap03_lacm/lacm.yaml --scene highway
conda run -n DT python scripts/train/train_chap03_compensation.py --config configs/chap03_lacm/lacm.yaml --scene highway
```

## 第四章 EFSR

运行官方 Highway 风险安全层实验：

```bash
conda run -n DT python scripts/eval/run_chap04_experiments.py --config configs/chap04_efsr/efsr.yaml --scene highway
```

保存风险价值函数：

```bash
conda run -n DT python scripts/train/train_chap04_risk.py --config configs/chap04_efsr/efsr.yaml --scene highway
```

## 验证

```bash
conda run -n DT python -m unittest discover -s tests
```

实验日志默认写入 `results/<chapter>/<scene>/`，包含 `episode_events.jsonl`、`*_episodes.csv` 和 `*_summary.json`。
