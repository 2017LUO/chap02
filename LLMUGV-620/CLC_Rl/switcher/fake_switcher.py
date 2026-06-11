# fake_switcher.py  —— 兼容 Python 3.8，前两次 KEEP，之后 WAIT
# -------------------------------------------------------------
from typing import Optional
from CLC_Rl.switcher.constants import ACT_SELECTOR, ACT_WAIT, IDX2NAME


class FakeSwitcher:
    """
    最简 Switcher：
      • 第 1、2 次 select() 返回 "KEEP"
      • 第 3 次及以后返回 "WAIT"
    其余接口留空或简单实现，保证可直接替换真正的 OnlineSwitcher。
    """

    def __init__(self, ckpt_path: Optional[str] = None, *args, **kwargs):
        self.ckpt_path = ckpt_path
        self._calls: int = 0          # 统计 select() 调用次数

    # ------------------------------------------------------------------ #
    # 必需接口
    # ------------------------------------------------------------------ #
    def select(self, img, lidar=None, conf_prob: float = 0.0) -> str:
        """
        Parameters
        ----------
        img, lidar : 占位参数，保持与真 Switcher 签名一致
        conf_prob  : 占位参数

        Returns
        -------
        str
            前两次 "KEEP"，之后 "WAIT"
        """
        self._calls += 1
        if self._calls <= 2:
            return IDX2NAME[ACT_SELECTOR]
        return IDX2NAME[ACT_WAIT]

    # ------------------------------------------------------------------ #
    # 可选接口：reload/close
    # ------------------------------------------------------------------ #
    def reload(self) -> None:
        """
        如需在热加载权重时恢复计数，可取消下一行注释：
        """
        # self._calls = 0
        pass

    def close(self) -> None:
        """占位，保持接口兼容。"""
        pass
