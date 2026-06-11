# Unity UGV 实时联调说明

本文用于启动 Unity 无人车场景，并用 Python 做手动控制或代码控制。

## 1. 环境

Python 使用 Conda 环境 `DT`：

```bash
conda run -n DT python <script>
```

Unity 工程路径：

```text
F:\毕业材料\5-模型源程序\rlenvironments
```

可联调场景：

- `ugv_parking`：`Assets/Environments/UGV/UGVParking/UGVParking.unity`
- `ugv_patrol`：`Assets/Environments/UGV/UGVPatrol/Scenes/UGVR2Patrol.unity`

## 2. 推荐启动顺序

先启动 Python，让 Python 等 Unity：

```bash
cd /d F:\毕业材料\5-模型源程序\paper_reproduction_project
conda run -n DT python scripts/ugv_control.py --backend unity --scene ugv_parking --mode manual
```

再打开 Unity 工程，在菜单里选择：

```text
Tools > UGV Live Control > Play Parking Scene
```

或手动打开 `UGVParking.unity` 后点击 Play。Unity 需要启用：

```text
Project Settings > ML-Agents > Connect Trainer
```

同时确认 `Editor Training Port` 是 `5004`。Python 控制器里的 `base port` 默认也是 `5004`。

## 3. Python 手动控制

终端控制：

```bash
conda run -n DT python scripts/ugv_control.py --backend unity --scene ugv_parking --mode manual
```

按键：

- `W` / 上方向键：前进
- `S` / 下方向键：倒车
- `A` / 左方向键：左转
- `D` / 右方向键：右转
- `X` / 空格：停车
- `C`：临时执行一次代码规划技能
- `Q`：退出

图形界面控制：

```bash
conda run -n DT python tests/ugv_manual_control.py
```

窗口里选择 `backend=unity`，点击 `Connect / Reset`，再到 Unity 按 Play。`Manual keyboard` 是键盘控制，`Code planner` 是代码规划控制。

如果 ML-Agents 一直连接不上，改选 `backend=udp`。UDP 模式不需要等待连接；Unity 场景在 Play 状态时会监听 `127.0.0.1:5055`，Python 会直接发送 `steer/motor` 控制量。

## 4. Python 代码控制

自动规划并执行技能：

```bash
conda run -n DT python scripts/ugv_control.py --backend unity --scene ugv_parking --mode code --max-steps 40
```

Patrol 场景：

```bash
conda run -n DT python scripts/ugv_control.py --backend unity --scene ugv_patrol --mode code --max-steps 40
```

没有打开 Unity 时，可先用 scripted 后端验证 Python 控制链路：

```bash
conda run -n DT python scripts/ugv_control.py --backend scripted --scene ugv_parking --mode code --max-steps 8
conda run -n DT python scripts/ugv_control.py --backend scripted --scene ugv_parking --mode smoke
```

## 5. 自动打开 Unity

如果本机能找到 Unity Editor，Python 可以同时拉起 Unity：

```bash
conda run -n DT python scripts/ugv_control.py --backend unity --scene ugv_parking --mode manual --launch-unity
```

如果找不到 Unity，请显式传入 Editor 路径：

```bash
conda run -n DT python scripts/ugv_control.py --backend unity --scene ugv_parking --mode manual --launch-unity --unity-editor "C:\Program Files\Unity\Hub\Editor\2022.3.53f1\Editor\Unity.exe"
```

当前机器上如果 Unity LicensingClient 未启动或许可证不可用，Unity 会启动失败；此时先从 Unity Hub 正常打开一次工程，再按第 2 节顺序联调。
