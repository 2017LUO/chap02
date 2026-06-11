# unity_img_utils.py
# ---------------------------------------------------------------
# 把 Unity ML-Agents 视觉观测转成 uint8-HWC 并保存为 PNG
# ---------------------------------------------------------------

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


def arr_to_uint8_hwc(arr: np.ndarray) -> np.ndarray:
    """
    将 Unity 观测数组规范化为 (H,W,C) uint8 0-255。
    支持输入：
        • (C,H,W) 或 (H,W,C)
        • float32 in [0,1]  或 [-1,1]
        • uint8 in [0,255]
    """
    # 1) 通道顺序 —— 若是 CHW 就转 HWC
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    # 2) 数值范围 / 类型 —— float → uint8
    if arr.dtype != np.uint8:
        v_min, v_max = float(arr.min()), float(arr.max())
        if v_min >= -1.0 and v_max <= 1.0:            # 0-1 或 −1-1
            arr = ((arr + 1.0) * 127.5) if v_min < 0 else (arr * 255.0)
        else:
            raise ValueError("数组值域超出 [-1,1] 且 dtype ≠ uint8，无法自动转换。")

        arr = arr.clip(0, 255).astype(np.uint8)

    return arr


def save_unity_image(arr: np.ndarray, out_path: Union[str, Path]) -> None:
    """
    将观测数组保存为 PNG。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)  # 自动建目录

    img_arr = arr_to_uint8_hwc(arr)
    Image.fromarray(img_arr).save(out_path)
