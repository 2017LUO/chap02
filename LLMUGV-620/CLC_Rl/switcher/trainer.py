# trainer.py
"""
LCMTrainer – IQL + Advantage‑Weighted BC
-----------------------------------------------------------
• 无 τ_step，按子任务粒度结算
• 目标值: V_target = r + γ·V_next
"""
import logging
from pathlib import Path
import torch, torch.nn.functional as F, torch.multiprocessing as mp

from .delay_q import DelayQNetwork
from ..memory.switcher_replay_buffer import ReplayBuffer


class LCMTrainer(mp.Process):
    def __init__(self,
                 sample_q: mp.Queue,
                 ckpt_path: str,
                 *,
                 gamma: float = 0.99,
                 batch_size: int = 32,
                 lr: float = 2.5e-4,
                 update_target_every: int = 1_000,
                 log_every: int = 500,
                 device: str = "cuda",
                 lidar_dim: int = 802,
                 tau: float = 0.7,      # expectile
                 beta: float = 3.0,     # AW‑BC 温度
                 awbc_clip: float = 10.0):
        super().__init__(daemon=True)
        self.q_in = sample_q
        self.gamma, self.batch_size = gamma, batch_size
        self.device = torch.device(device)

        self.online = DelayQNetwork(lidar_dim=lidar_dim).to(self.device)
        self.target = DelayQNetwork(lidar_dim=lidar_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())

        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buf = None

        self.tau, self.beta, self.clip = tau, beta, awbc_clip
        self.step = 0
        self.update_target_every, self.log_every = update_target_every, log_every

        self.ckpt = Path(ckpt_path)
        self.ckpt.parent.mkdir(parents=True, exist_ok=True)

    # ---------------- 主循环 ----------------
    def run(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [LCMTrainer] %(message)s",
        )
        if self.buf is None:  # 避免重复初始化
            self.buf = ReplayBuffer()

        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index or 0)

        while True:
            t = self.q_in.get()
            if t is None:
                break
            self.buf.push(t)
            if len(self.buf) >= self.batch_size:
                self._train_once()

        self._save_ckpt()

    # -------------- 单批梯度 --------------
    def _train_once(self):
        self.step += 1
        batch = self.buf.sample(self.batch_size)

        stack = lambda lst: torch.stack(lst).to(self.device)
        img, lidar, act, rew, img2, lidar2, done = zip(*batch)

        img, img2 = stack(img), stack(img2)
        lidar, lidar2 = stack(lidar), stack(lidar2)
        act = torch.tensor(act, dtype=torch.long, device=self.device)
        rew = torch.tensor(rew, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)

        # online & target
        q_all, v_s = self.online(img, lidar)
        _, v_next = self.target(img2, lidar2)

        v_target = rew + self.gamma * v_next * (1 - done)
        diff = v_target - v_s
        weight = torch.where(diff > 0, self.tau, 1 - self.tau)
        v_loss = (weight * diff.pow(2)).mean()

        q_a = q_all.gather(1, act.unsqueeze(1)).squeeze(1)
        q_loss = F.smooth_l1_loss(q_a, v_target)

        adv = (v_target - v_s).detach()
        logp = F.log_softmax(q_all, dim=1).gather(1, act.unsqueeze(1)).squeeze(1)
        aw = (adv / self.beta).clamp(max=self.clip).exp()
        awbc_loss = -(aw * logp).mean()

        loss = v_loss + q_loss + awbc_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if self.step % self.update_target_every == 0:
            self.target.load_state_dict(self.online.state_dict())
            self._save_ckpt()

        if self.step % self.log_every == 0:
            logging.info(
                f"step {self.step:>6}  V={v_loss:.3f}  Q={q_loss:.3f}  "
                f"AWBC={awbc_loss:.3f}  buf={len(self.buf)}"
            )

    # ---------------- 保存权重 ----------------
    def _save_ckpt(self):
        tmp = self.ckpt.with_suffix(".tmp")
        torch.save(self.online.state_dict(), tmp)
        tmp.replace(self.ckpt)
