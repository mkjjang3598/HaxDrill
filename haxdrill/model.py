"""World model: f(state_t, action_t) -> state_{t+1}.

An MLP that predicts the normalised *delta* of the full 12-d state. The model is the
whole environment: ball physics, the user's own disc dynamics AND the opponent's
behaviour are all learned from data. Normalisation statistics are stored in the module
so a checkpoint is self-contained.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


class WorldModel(nn.Module):
    def __init__(self, hidden: int = 512, layers: int = 4, state_dim: int = C.STATE_DIM, n_actions: int = C.N_ACTIONS):
        super().__init__()
        self.state_dim, self.n_actions = state_dim, n_actions
        self.register_buffer("s_mean", torch.zeros(state_dim))
        self.register_buffer("s_std", torch.ones(state_dim))
        self.register_buffer("d_mean", torch.zeros(state_dim))
        self.register_buffer("d_std", torch.ones(state_dim))
        self.inp = nn.Linear(state_dim + n_actions, hidden)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)) for _ in range(layers)])
        self.out = nn.Linear(hidden, state_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @torch.no_grad()
    def set_normalization(self, obs: torch.Tensor, delta: torch.Tensor):
        self.s_mean.copy_(obs.mean(0))
        self.s_std.copy_(obs.std(0).clamp_min(1e-3))
        self.d_mean.copy_(delta.mean(0))
        self.d_std.copy_(delta.std(0).clamp_min(1e-4))

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """s: (B,12) raw units, a: (B,) long. Returns next state in raw units."""
        x = torch.cat([(s - self.s_mean) / self.s_std, F.one_hot(a, self.n_actions).to(s.dtype)], -1)
        h = F.silu(self.inp(x))
        for blk in self.blocks:
            h = h + blk(h)
        d = self.out(h) * self.d_std + self.d_mean
        return s + d

    @torch.no_grad()
    def rollout(self, s0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """s0 (B,12), actions (B,T) -> (B,T,12) predicted states after each action."""
        out, s = [], s0
        for t in range(actions.shape[1]):
            s = self(s, actions[:, t])
            out.append(s)
        return torch.stack(out, 1)


def save(model: WorldModel, path: str, **extra):
    torch.save({"state_dict": model.state_dict(), "hidden": model.inp.out_features, "layers": len(model.blocks), **extra}, path)


def load(path: str, device="cpu") -> WorldModel:
    ck = torch.load(path, map_location=device)
    m = WorldModel(hidden=ck["hidden"], layers=ck["layers"])
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()
