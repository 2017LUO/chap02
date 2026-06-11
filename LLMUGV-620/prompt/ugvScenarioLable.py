# road_analyzer.py  ——  Robust version (0–1 float RGB, Py 3.8)
from __future__ import annotations
from typing import Dict, Tuple, Union
import numpy as np

ColorStats = Dict[str, Dict[str, Union[bool, float]]]


class RoadAnalyzer:
    # ───────── 单像素阈值 ───────── #
    T_BLACK = 0.20
    T_WHITE = 0.80

    T_RED = 0.60;  DELTA_RED = 0.15
    T_GREEN = 0.35;  DELTA_GREEN = 0.15
    T_BLUE = 0.60;  DELTA_BLUE = 0.15

    T_YELLOW = 0.50; DELTA_RG = 0.25; DELTA_YB = 0.25

    # ───────── 面积阈值 ───────── #
    BLACK_RATIO_TH = 0.02   # >25 % → 进入/停车
    GREEN_RATIO_TH = 0.01   # >10 % → 绿色障碍物
    YELLOW_RATIO_TH = 0.001   # >10 % → 黄色气球
    SMALL_TH = 0.02   # <2 %  → 近似无
    SMALLER_TH = 0.001

    def __init__(self) -> None:
        self.Parking = 0
        self.CountPark = 0
        self.parking_frames = 0

    # ───────────────── 主接口 ───────────────── #
    def analyze_road(self, img: np.ndarray) -> Tuple[str, ColorStats]:
        """
        img : (84,84,3) - float32/64, 值域 [0,1]
        返回 (场景描述, 颜色统计)
        """
        if img.shape != (84, 84, 3):
            raise ValueError("图像尺寸必须是 (84,84,3)")
        if img.dtype.kind not in ('f', 'c'):
            raise TypeError("请先把 RGB 归一化到 0-1 浮点")

        stats = self._calc_color_stats(img)
        scene = self._classify(stats)
        return scene, stats

    # ───────────── 颜色统计 ───────────── #
    def _calc_color_stats(self, img: np.ndarray) -> ColorStats:
        total = img.size // 3
        r, g, b = img[..., 0], img[..., 1], img[..., 2]

        # 掩码定义
        mask_black = (np.maximum(np.maximum(r, g), b) < self.T_BLACK)
        mask_white = (np.minimum(np.minimum(r, g), b) > self.T_WHITE)

        mask_red = (r > self.T_RED) & (r - np.maximum(g, b) > self.DELTA_RED)
        mask_green = (g > self.T_GREEN) & (g - np.maximum(r, b) > self.DELTA_GREEN)
        mask_blue = (b > self.T_BLUE) & (b - np.maximum(r, g) > self.DELTA_BLUE)

        mask_yellow = (
            (r > self.T_YELLOW) & (g > self.T_YELLOW) &
            (np.abs(r - g) < self.DELTA_RG) &
            (np.maximum(r, g) - b > self.DELTA_YB)
        )

        def to_stat(mask):
            return {"present": bool(mask.any()), "ratio": float(mask.sum()) / total}

        return {
            "black": to_stat(mask_black),
            "white": to_stat(mask_white),
            "red": to_stat(mask_red),
            "green": to_stat(mask_green),
            "blue": to_stat(mask_blue),
            "yellow": to_stat(mask_yellow),
        }

    # ───────────── 场景判定 ───────────── #
    def _classify(self, s: ColorStats) -> str:
        # 1) 停车区 / 停车 —— 按 4+4 硬规则
        if self.CountPark > 0:
            self.CountPark -= 1
            return "停车"

        if s["black"]["ratio"] > self.BLACK_RATIO_TH:
            self.parking_frames += 1  # 连续黑区帧 +1
            if self.parking_frames == 4:
                self.CountPark = 4
            if self.parking_frames <= 4:
                return "进入停车区域"  # 第 1–4 帧

        # 2) 绿色障碍物
        if (s["green"]["ratio"] > self.GREEN_RATIO_TH and
            s["black"]["ratio"] < self.SMALL_TH):
            return "前方道路有多个绿色障碍物"

        # 3) 黄色气球
        if (s["yellow"]["ratio"] > self.YELLOW_RATIO_TH and
            s["green"]["ratio"] < self.SMALLER_TH):
            return "前方道路有多个黄色气球"

        # 4) 右转进入街道标志
        if (s["red"]["present"] and s["green"]["present"] and
            s["yellow"]["present"] and s["blue"]["present"]):
            return "前方道路有右转进入街道标志"

        # 5) 默认平坦
        if s["black"]["ratio"] < self.SMALLER_TH:
            return "前方道路平坦没有物体"
