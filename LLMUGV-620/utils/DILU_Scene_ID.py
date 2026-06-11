"""
用于根据 (pos_x, pos_z) 坐标判断子任务 ID。

* 第一次调用 → 0
* 第二次及以后 → 1–5（符合区间即可）
  - ID = 5 仅当之前已经命中过 1–4 中任意一个
"""


class SceneID:

    def __init__(self) -> None:
        self._first_call: bool = True   # 是否首次调用
        self._cur_id: int = 0           # 当前保存的 ID

    def reset(self) -> None:
        """显式重置（等同于重新创建实例）。"""
        self._first_call = True
        self._cur_id = 0

    def __call__(self, pos_x: float, pos_z: float) -> int:
        # ---------- 第一次调用：直接返回 0 ----------
        if self._first_call:
            self._first_call = False
            return self._cur_id               # 此时 _cur_id == 0

        # ---------- 判断并更新 ----------
        if pos_x > 0.6 and 0.3 < pos_z < 0.5:               # ID 1 ok
            self._cur_id = 0
        elif 0.3 < pos_x < 0.6 and pos_z < -0.6:                   # ID 2 ok
            self._cur_id = 2
        elif pos_x < -0.60 and 0.0 < pos_z < 0.5:          # ID 3 ok
            self._cur_id = 3
        elif -0.55 < pos_x < -0.45 and 0.3 < pos_z < 0.6:          # ID 4 ok
            self._cur_id = 4
        elif (-0.45 < pos_x < -0.10 and -0.3 < pos_z < 0.1
              and self._cur_id != 0):                         # ID 5
            self._cur_id = 5
        # 若没有任何条件满足，则 _cur_id 保持不变

        return self._cur_id

