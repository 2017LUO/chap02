# Python 运行环境说明

本项目后续所有 Python 代码、测试脚本、训练脚本和可视化手动控制脚本，默认使用本机 Conda 环境：

```text
DT
```

当前确认的解释器版本：

```text
Python 3.8.19
```

Codex 后续执行 Python 命令时，应优先使用：

```bash
conda run -n DT python <script_or_module>
```

例如：

```bash
conda run -n DT python -m unittest tests.test_env_interfaces
conda run -n DT python tests/crafter/manual_control.py
conda run -n DT python tests/highway/manual_control.py
```

除非用户明确说明更换环境，不要再临时查找、猜测或切换到其他 Python 解释器。

## Python 版本约束

项目运行目标按 Python 3.8 处理，不按 Python 3.10+ 编写。后续新增代码应避免使用 3.10 专属类型写法，例如：

```python
# 不推荐
value: int | None

# 推荐
from typing import Optional

value: Optional[int]
```

如果需要联合类型，使用 `typing.Union`；如果需要可选类型，使用 `typing.Optional`。

## Crafter / Pillow 修复记录

DT 环境中的 Crafter 依赖 Pillow。若出现如下错误：

```text
ImportError: DLL load failed while importing _imaging: 拒绝访问
```

优先在 DT 环境中重装 Pillow wheel：

```bash
conda run -n DT python -m pip install --force-reinstall --no-cache-dir Pillow==10.4.0 -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
```

修复后可用下面命令确认真实 Gym Crafter 能启动：

```bash
conda run -n DT python -c "from envs import CrafterEnv; env=CrafterEnv({'backend':'gym'}); obs=env.reset(seed=0); print(obs['image'].shape); env.close()"
```
