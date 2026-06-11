# =============================================================
# selector_replay_buffer.py
# =============================================================
"""A thread-/process-agnostic replay buffer for (image, label) pairs.

This buffer **lives inside the online trainer process**; the main LCM
process pushes new samples through a `multiprocessing.Queue`, so the
buffer itself does **not** need to be shared across processes.

Python 3.8 compatible — no PEP 604 ``|`` unions are used.
"""
from collections import deque
import random
from typing import Deque, Tuple, List
import torch


class SelectorReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.capacity: int = capacity
        self.buf: Deque[Tuple[torch.Tensor, int]] = deque(maxlen=capacity)

    def push(self, img: torch.Tensor, label: int):
        """Store a *detached* copy to avoid later in‑place ops corrupting data."""
        self.buf.append((img.clone(), label))

    def sample(self, batch_size: int, device: str = "cpu"):
        size = min(batch_size, len(self.buf))
        batch = random.sample(self.buf, size)
        imgs, labels = zip(*batch)
        imgs = torch.stack(imgs).to(device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        return imgs, labels

    def __len__(self):
        return len(self.buf)