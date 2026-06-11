# =============================================================
# online_trainer.py  (Python 3.8+)  ——  带日志 & 可选 TensorBoard
# =============================================================
from __future__ import annotations
import multiprocessing as mp, os, time, logging
import queue
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn


# ---------------- 你的 ReplayBuffer ----------------
from CLC_Rl.memory.selector_replay_buffer import SelectorReplayBuffer
from CLC_Rl.selector.tiny_cnn_top4_classifier import Net


# ---------------------------------------------------

class OnlineTrainer(mp.Process):
    def __init__(
        self,
        q: mp.Queue,
        checkpoint_path: Union[str, os.PathLike],
        *,
        device: str = "cuda",
        n_cls: int = 6,
        capacity: int = 50000,
        batch_size: int = 32,
        min_buffer: int = 64,
        train_every: int = 2,
        iters_per_session: int = 512,
        lr: float = 1e-3,
        log_file: Union[str, os.PathLike] = "trainer.log",
        tb_dir: Optional[Union[str, os.PathLike]] = None,
    ):
        super().__init__(daemon=True)
        # --- core params ---
        self.q, self.ckpt = q, Path(checkpoint_path)
        self.device = torch.device(device)
        self.n_cls = n_cls
        self.capacity, self.batch_size = capacity, batch_size
        self.min_buffer, self.train_every = min_buffer, train_every
        self.iters_per_session, self.lr = iters_per_session, lr

        # --- logging ---
        self.log_file = Path(log_file)
        self.tb_dir = Path(tb_dir) if tb_dir else None

        # --- runtime ---
        self.buf: Optional[SelectorReplayBuffer] = None
        self.net: Optional[Net] = None
        self.opt: Optional[torch.optim.Optimizer] = None
        self.loss_fn, self.new_samples, self.step = nn.CrossEntropyLoss(), 0, 0

    # ---------- helpers ----------
    def _setup(self):
        # 1) logger
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [Trainer] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(self.log_file, encoding="utf-8"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("Trainer")

        # 2) replay, net, optim
        self.buf = SelectorReplayBuffer(self.capacity)
        self.net = Net(self.n_cls).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        self.net.train()

        # 3) warm-start
        self.ckpt.parent.mkdir(parents=True, exist_ok=True)
        if self.ckpt.exists():
            try:
                self.net.load_state_dict(torch.load(self.ckpt, map_location=self.device))
                self.logger.info("Warm-started from %s", self.ckpt)
            except Exception as e:
                self.logger.warning("Could not load ckpt: %s", e)

        # 4) optional TensorBoard
        self.tb_writer = None
        if self.tb_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(self.tb_dir)
                self.logger.info("TensorBoard log dir: %s", self.tb_dir)
            except ImportError:
                self.logger.warning("TensorBoard not installed; skip tb logging")

    def _train_once(self):
        if len(self.buf) < self.min_buffer:
            return
        loss_sum = 0.0
        for _ in range(self.iters_per_session):
            imgs, lbls = self.buf.sample(self.batch_size, self.device)
            logits, _ = self.net(imgs)
            loss = self.loss_fn(logits, lbls)
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            loss_sum += loss.item()
            self.step += 1

        avg_loss = loss_sum / self.iters_per_session
        # save ckpt atomically
        tmp = self.ckpt.with_suffix(".tmp")
        torch.save(self.net.state_dict(), tmp)
        tmp.replace(self.ckpt)

        self.logger.info(
            "step=%d  buffer=%d  avg_loss=%.6f  last_loss=%.6f  ckpt→%s",
            self.step, len(self.buf), avg_loss, loss.item(), self.ckpt.name)

        # tensorboard
        if self.tb_writer:
            self.tb_writer.add_scalar("loss/avg",  avg_loss,  self.step)
            self.tb_writer.add_scalar("loss/last", loss.item(), self.step)
            self.tb_writer.add_scalar("meta/buffer", len(self.buf), self.step)

    # ---------- main loop ----------
    def run(self):
        mp.set_start_method("spawn", force=True)
        self._setup()
        self.logger.info("Online trainer started …")
        while True:
            try:
                item = self.q.get(timeout=1.0)
            except queue.Empty:  # ← 只捕获超时
                continue  # 队列暂时没东西，继续等
            # 到这里 item 一定是取到的对象
            if item is None:  # ← 真正的关机哨兵
                self.logger.info("Shutdown signal; final train")
                self._train_once()
                if self.tb_writer: self.tb_writer.close()
                return

            img, lbl = item
            self.buf.push(img, lbl)
            self.new_samples += 1

            if self.new_samples >= self.train_every:
                self.new_samples = 0
                self._train_once()
