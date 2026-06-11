"""skill_manager.py – load five skills for CLC_Rl package
===================================================
* Located inside CLC_Rl/   ;   ckpts/  is **sibling** to CLC_Rl/   →  project tree::

    project_root/
    ├── ckpts/
    │   └── *.pth
    └── CLC_Rl/
        ├── registry.py
        └── skill_manager.py  (this file)

* This revision resolves relative ckpt paths ("ckpts/…") against project_root.
* Public API:

      lib = SkillLibrary(device="cuda")
      action = lib.act(task_id, img_tensor, lidar_tensor)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from numpy._typing import ArrayLike

# Local imports (CLC_Rl package)
from CLC_Rl.skills_manager.registry import TaskRegistry
from agent.sac.sac_net import Actor  # adjust import path if Actor lives elsewhere

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # one level up from CLC_Rl/


class SkillWrapper(nn.Module):
    """Wrap a frozen Actor policy."""

    def __init__(self, ckpt_path: str | Path, *, device: str,
                 action_dim: int = 2, max_action: float = 1.0,
                 lidar_dim: int = 802, hidden_dim: int = 256):
        super().__init__()

        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_absolute():
            ckpt_path = (PROJECT_ROOT / ckpt_path).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

        self.model = Actor(action_dim, max_action, lidar_dim, hidden_dim)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.max_action = max_action

    @torch.no_grad()
    def act(
            self,
            img: ArrayLike,  # numpy / list / torch.Tensor
            lidar: ArrayLike,  # numpy / list / torch.Tensor
    ) -> np.ndarray:
        """
        与 Agent.select_action() 行为一致：
          1. 输入可为 numpy / list / Tensor
          2. 单帧时自动补 batch 维
          3. 前向推理 → tanh 限幅
          4. 始终返回 CPU numpy；若批量=1 则 squeeze
        """
        # ① 转张量 & 搬到目标设备
        img_t = torch.as_tensor(img, dtype=torch.float32, device=self.device)
        lidar_t = torch.as_tensor(lidar, dtype=torch.float32, device=self.device)

        # ② 补批量维
        if img_t.dim() == 3:  # (C,H,W)
            img_t = img_t.unsqueeze(0)  # → (1,C,H,W)
            lidar_t = lidar_t.unsqueeze(0)  # → (1,802)

        # ③ 推理
        mu, _ = self.model(img_t, lidar_t)
        action = torch.tanh(mu) * self.max_action  # [-max,max]

        # ④ 回 CPU numpy；单帧时 squeeze
        action_np = action.cpu().numpy()
        # if action_np.shape[0] == 1:
        #     action_np = action_np.squeeze(0)  # → (action_dim,)
        return action_np


class SkillLibrary:
    """Load all skills defined in tasks.json and expose act()."""

    def __init__(self, *, registry_path: str | Path | None = None,
                 device: str,
                 action_dim: int = 2, max_action: float = 1.0,
                 lidar_dim: int = 802, hidden_dim: int = 256):
        self.device = device
        self.registry = TaskRegistry(registry_path, readonly=True)
        self.skills: Dict[int, SkillWrapper] = {}

        for tid in self.registry.list_ids():
            ckpt_rel = self.registry.skill_ckpt(tid)
            self.skills[tid] = SkillWrapper(
                ckpt_rel, device=device,
                action_dim=action_dim, max_action=max_action,
                lidar_dim=lidar_dim, hidden_dim=hidden_dim)
        print(f"[SkillLibrary] Loaded {len(self.skills)} skills from {self.registry.path.name} → {device}")

    # ------------------------------------------------------------
    def act(self, task_id: int, img: ArrayLike, lidar: ArrayLike) -> np.ndarray:
        if task_id not in self.skills:
            raise KeyError(f"task_id {task_id} not in SkillLibrary")
        return self.skills[task_id].act(img, lidar)

    def __len__(self):
        return len(self.skills)


# CLI demo -----------------------------------------------------
if __name__ == "__main__":

    lib = SkillLibrary(device="cpu")
    img_demo = torch.rand(1, 4, 84, 84)
    lidar_demo = torch.rand(1, 802)
    for tid in lib.registry.list_ids():
        act = lib.act(tid, img_demo, lidar_demo)
        print(f"task {tid} action →", act)