# online_switcher.py
"""
OnlineSwitcher
====================================================
• 每次调用 select():
      - ε‑greedy 探索三动作
      - 贪婪时基于 Q + α·置信度 + β·连续保持
• legal 动作永久三元 [WAIT, KEEP, SELECTOR]
  外部主程序负责把 ACT_SELECTOR 映射为
  "置信度最高的技能 ID"。
"""
import os, time, math, random
from typing import Optional
import torch

from .delay_q import DelayQNetwork
from .constants import (
    ACT_WAIT, ACT_KEEP, ACT_SELECTOR,
    EPS_START, EPS_END, EPS_DECAY, IDX2NAME,
)

# ------------------------------------------------
WARMUP_STEPS = 120  # 前 120 帧屏蔽 ACT_SELECTOR
EXPLORE_P_SEL = 0.70  # 随机分支抽 ACT_SELECTOR 的目标概率

class OnlineSwitcher:
    def __init__(self,
                 ckpt_path: str,
                 *,
                 lidar_dim: int = 802,
                 device: str = "cuda",
                 reload_sec: float = 2.0,
                 eps_start: float = EPS_START,
                 eps_end: float = EPS_END,
                 eps_decay: int = EPS_DECAY):
        # ------- 网络与权重热加载 -------
        self.net = DelayQNetwork(lidar_dim=lidar_dim).to(device)
        self.device = torch.device(device)
        self.ckpt_path, self.reload_sec = ckpt_path, reload_sec
        self.last_mtime = 0.0

        # ------- ε‑greedy 参数 -------
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.global_step = 0

    # --------- 权重自动热加载 ---------
    def _reload_ckpt(self):
        if time.time() - self.last_mtime < self.reload_sec:
            return
        if not os.path.exists(self.ckpt_path):
            return
        mtime = os.path.getmtime(self.ckpt_path)
        if mtime != self.last_mtime:
            self.net.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
            self.net.eval()
            self.last_mtime = mtime

    # --------- 核心决策函数 ---------

    @torch.no_grad()
    def select(self,
               img: torch.Tensor,  # (3,84,84) float32
               lidar: torch.Tensor  # (lidar_dim,)
               ) -> str:
        """返回 "WAIT" / "KEEP" / "SELECTOR"（无 conf_prob / prev_skill / selector 偏置）"""
        # ---- 热加载 & ε 衰减 ----
        self._reload_ckpt()
        self.global_step += 1
        eps = self.eps_end + (self.eps_start - self.eps_end) * \
              math.exp(-self.global_step / self.eps_decay)

        # ---- 合法动作集合 ----
        if self.global_step <= WARMUP_STEPS:
            actions = [ACT_WAIT, ACT_KEEP]  # 屏蔽 SELECTOR
            weights = [0.5, 0.5]  # 随机分支均匀
        else:
            actions = [ACT_WAIT, ACT_KEEP, ACT_SELECTOR]
            weights = [(1 - EXPLORE_P_SEL) / 2,
                       (1 - EXPLORE_P_SEL) / 2,
                       EXPLORE_P_SEL]  # 随机分支偏向 SELECTOR

        # ---- ε-greedy 随机 ----
        if random.random() < eps:
            return IDX2NAME[random.choices(actions, weights=weights, k=1)[0]]

        # ---- 贪婪 argmax ----
        q, _ = self.net(
            img.unsqueeze(0).to(self.device),
            lidar.unsqueeze(0).to(self.device),
        )
        q = q[0]  # (3,)

        local_best = torch.argmax(q[actions]).item()  # 在合法动作子张量上 argmax
        best_int = actions[local_best]  # 映回全局 ACT_* 编号

        return IDX2NAME[best_int]


