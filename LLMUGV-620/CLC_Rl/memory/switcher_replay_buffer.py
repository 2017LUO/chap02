# switcher_replay_buffer.py
"""
线程安全循环缓冲区
-------------------------------------------------
每个子任务结算一次 Transition:
(img, lidar, act, reward, next_img, next_lidar, done)
"""
import threading, random, collections
from typing import Deque, List, Tuple

Transition = collections.namedtuple(
    "Transition",
    ["img", "lidar", "act", "rew", "img2", "lidar2", "done"],
)


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self._buf: Deque[Transition] = collections.deque(maxlen=capacity)
        self._lock = threading.Lock()

    # ---------- 生产者写入 ----------
    def push(self, data: Tuple):
        with self._lock:
            self._buf.append(Transition(*data))

    # ---------- 消费者随机采样 ----------
    def sample(self, batch_size: int) -> List[Transition]:
        with self._lock:
            return random.sample(self._buf, batch_size)

    def __len__(self):
        return len(self._buf)
