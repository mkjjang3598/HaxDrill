"""Dataset helpers: load npz files and build k-step windows for rollout training."""
from __future__ import annotations

import numpy as np
import torch


def load_npz(paths):
    obs, act, ep = [], [], []
    off = 0
    for p in paths:
        d = np.load(p)
        obs.append(d["obs"].astype(np.float32))
        act.append(d["act"].astype(np.int64))
        ep.append(d["ep"].astype(np.int64) + off)
        off = int(ep[-1].max()) + 1
    return np.concatenate(obs), np.concatenate(act), np.concatenate(ep)


def window_starts(ep: np.ndarray, k: int) -> np.ndarray:
    """Indices i such that obs[i..i+k] all lie in the same episode."""
    same = ep[k:] == ep[:-k]
    return np.nonzero(same)[0]


class WindowSampler:
    """Samples (s_0, a_{0..k-1}, s_{1..k}) windows on the GPU."""

    def __init__(self, obs, act, ep, k: int, device):
        self.obs = torch.as_tensor(obs, device=device)
        self.act = torch.as_tensor(act, device=device)
        self.k = k
        self.starts = torch.as_tensor(window_starts(ep, k), device=device)
        self.device = device

    def __len__(self):
        return len(self.starts)

    def sample(self, batch: int):
        i = self.starts[torch.randint(len(self.starts), (batch,), device=self.device)]
        idx = i[:, None] + torch.arange(self.k + 1, device=self.device)[None]
        s = self.obs[idx]  # (B,k+1,12)
        a = self.act[idx[:, :-1]]  # (B,k)
        return s[:, 0], a, s[:, 1:]
