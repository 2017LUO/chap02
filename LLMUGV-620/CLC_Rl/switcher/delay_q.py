# delay_q.py
"""
Tri‑Action Dueling Q‑Network
====================================================
输入:
    • RGB 图像  (B,3,84,84)
    • LiDAR 向量 (B,lidar_dim)   (可设 0 禁用)

输出:
    • Q(s,a) ∈ ℝ^(B,3)   (Wait / Keep / Selector)
    • V(s)   ∈ ℝ^(B,)
训练:
    • IQL (expectile) + Advantage‑Weighted BC
"""
from typing import Optional
import torch, torch.nn as nn
from .constants import N_SWITCH_ACT


class DelayQNetwork(nn.Module):
    def __init__(self, *, lidar_dim: int = 802, hidden_dim: int = 256):
        super().__init__()

        # -------- 图像分支 (3×84×84 → 256) --------
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2),  # 84→41
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                            # 41→20
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # 20→9
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),                            # 9→4
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 256),
        )
        self.norm = nn.LayerNorm(256)

        # -------- LiDAR 分支 (→256) --------
        self.use_lidar = lidar_dim > 0
        if self.use_lidar:
            self.lidar_enc = nn.Sequential(
                nn.Linear(lidar_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            )

        # -------- 融合 + Dueling 头 --------
        fusion_dim = 256 + (hidden_dim if self.use_lidar else 0)
        self.backbone = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.v_head = nn.Linear(256, 1)               # 状态值 V(s)
        self.a_head = nn.Linear(256, N_SWITCH_ACT)    # 优势 A(s,a)

    @staticmethod
    def _norm(x: torch.Tensor) -> torch.Tensor:
        """把 uint8 图像归一化到 [0,1]"""
        return x.float() / 255.0 if x.dtype == torch.uint8 else x.float()

    # -------- 前向传播 --------
    def forward(self, img: torch.Tensor,
                lidar: Optional[torch.Tensor] = None):
        """
        img   : (B,3,84,84)
        lidar : (B,lidar_dim) or None
        """
        x_img = self.norm(self.cnn(self._norm(img)))

        if self.use_lidar:
            if lidar is None:
                raise ValueError("LiDAR tensor missing!")
            x = torch.cat([x_img, self.lidar_enc(lidar)], dim=1)
        else:
            x = x_img

        x = self.backbone(x)
        v = self.v_head(x).squeeze(-1)                # (B,)
        a = self.a_head(x)                            # (B,3)
        q = v.unsqueeze(1) + (a - a.mean(1, keepdim=True))
        return q, v
