# =============================================================
# tiny_cnn_top4_classifier.py  (Python 3.8+, RGB Tensor input)
# =============================================================
from __future__ import annotations
import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class CNN_Class(nn.Module):
    """(B,3,84,84) → (B,256) 轻量特征提取器"""
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2),  # 84→41
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                            # 41→20
            nn.Conv2d(16, 32, kernel_size=4, stride=2), # 20→9
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),                            # 9→4
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 256),
        )
        self.norm = nn.LayerNorm(256)

    def forward(self, x: torch.Tensor):
        return self.norm(self.cnn(x))


class Head(nn.Module):
    def __init__(self, n_cls: int = 6):
        super().__init__()
        self.fc = nn.Linear(256, n_cls)

    def forward(self, z: torch.Tensor):
        return self.fc(z)


class Net(nn.Module):
    """返回 `(logits, features)` 方便调试/可视化"""
    def __init__(self, n_cls: int = 6):
        super().__init__()
        self.backbone, self.head = CNN_Class(), Head(n_cls)

    def forward(self, x: torch.Tensor):
        z = self.backbone(x)
        logits = self.head(z)
        return logits, z
