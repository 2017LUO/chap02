# =============================================================
# skill_predictor.py  (Python 3.8+)
# =============================================================
"""
主进程轻量推断器：
   probs4 = predictor.predict_top4(image_tensor)  # image_tensor shape=(3,84,84)
"""
from __future__ import annotations
import os, time
from typing import Dict, Union, List

import torch

from CLC_Rl.selector.tiny_cnn_top4_classifier import Net


class Selector:
    def __init__(
        self,
        checkpoint_path: Union[str, os.PathLike],
        *,
        device: str = "cuda",
        n_cls: int = 6,
        check_interval: float = 2.0,
        use_compile: bool = False,
    ):
        self.path = os.fspath(checkpoint_path)
        self.device = torch.device(device)
        self.n_cls, self.check_interval = n_cls, check_interval
        self._last_mtime = 0.0; self._last_check = 0.0

        self.net = Net(n_cls).to(self.device).eval()
        if use_compile and hasattr(torch, "compile"):
            self.net = torch.compile(self.net, mode="max-autotune")
        self._maybe_reload(force=True)

    # ---------- public ----------
    def predict(self, x: torch.Tensor, candidate_labels: List[int]) -> Dict[int, float]:
        """返回给定标签的概率（无需 softmax 再做）"""
        self._maybe_reload()
        x = x.unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.net(x)[0], dim=1)[0].cpu()
        return {i: probs[i].item() for i in candidate_labels}

    def predict_top4(self, x: torch.Tensor) -> Dict[int, float]:
        """直接返回 top-4 {label: prob}"""
        self._maybe_reload()
        x = x.unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.net(x)[0], dim=1)[0]
            conf, idx = torch.topk(probs, k=4)
        return {int(i): float(c) for i, c in zip(idx, conf)}

    # ---------- internal ----------
    def _maybe_reload(self, force=False):
        now = time.time()
        if not force and now - self._last_check < self.check_interval:
            return
        self._last_check = now
        try:
            mtime = os.path.getmtime(self.path)
        except FileNotFoundError:
            return
        if mtime <= self._last_mtime:
            return
        try:
            state = torch.load(self.path, map_location=self.device)
            self.net.load_state_dict(state, strict=False)
            self.net.eval()
            self._last_mtime = mtime
            print("[Predictor] Hot-reloaded", self.path)
        except Exception as e:
            print("[Predictor] Failed to load ckpt:", e)
