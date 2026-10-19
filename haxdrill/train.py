"""Train the world model.

    python -m haxdrill.train data/sim.npz --out runs/wm.pt --steps 30000

Uses a 1-step loss plus a multi-step rollout loss (horizon grows over training) so the
model stays stable when it is run closed-loop in play mode.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .data import WindowSampler, load_npz
from .model import WorldModel, save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--out", default="runs/wm.pt")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=8, help="max rollout horizon for the multi-step loss")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    obs, act, ep = load_npz(args.data)
    # split by episode so validation windows are unseen trajectories
    n_ep = ep.max() + 1
    val_eps = np.zeros(n_ep, bool)
    val_eps[np.random.default_rng(args.seed).random(n_ep) < args.val_frac] = True
    is_val = val_eps[ep]
    tr = WindowSampler(obs[~is_val], act[~is_val], ep[~is_val], args.horizon, dev)
    va = WindowSampler(obs[is_val], act[is_val], ep[is_val], args.horizon, dev)
    print(f"device={dev} train windows={len(tr):,} val windows={len(va):,} episodes={n_ep}")

    model = WorldModel(args.hidden, args.layers).to(dev)
    same = ep[1:] == ep[:-1]
    model.set_normalization(torch.as_tensor(obs[:-1][same], device=dev), torch.as_tensor((obs[1:] - obs[:-1])[same], device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    d_std = model.d_std

    def loss_fn(s0, a, target, k):
        """Normalised MSE over a k-step open-loop rollout."""
        s, tot = s0, 0.0
        for t in range(k):
            s = model(s, a[:, t])
            tot = tot + F.mse_loss((s - target[:, t]) / d_std, torch.zeros_like(s))
        return tot / k

    t0 = time.time()
    for step in range(1, args.steps + 1):
        # curriculum: 1-step for the first 20%, then grow the horizon
        frac = step / args.steps
        k = 1 if frac < 0.2 else min(args.horizon, 1 + int((frac - 0.2) / 0.8 * args.horizon))
        s0, a, tgt = tr.sample(args.batch)
        loss = loss_fn(s0, a, tgt, k)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                s0, a, tgt = va.sample(8192)
                l1 = loss_fn(s0, a, tgt, 1).item()
                lk = loss_fn(s0, a, tgt, args.horizon).item()
                pred = model.rollout(s0, a)
                pos_err = (pred[..., [4, 5]] - tgt[..., [4, 5]]).norm(dim=-1)  # ball position error (units)
                opp_err = (pred[..., [8, 9]] - tgt[..., [8, 9]]).norm(dim=-1)
            model.train()
            print(
                f"step {step:6d} k={k} train {loss.item():.4f} | val 1-step {l1:.4f} {args.horizon}-step {lk:.4f} "
                f"| ball err @1 {pos_err[:,0].mean():.2f} @{args.horizon} {pos_err[:,-1].mean():.2f} "
                f"| opp err @{args.horizon} {opp_err[:,-1].mean():.2f} | {time.time()-t0:.0f}s"
            )
    save(model, args.out, args=vars(args))
    print("saved", args.out)


if __name__ == "__main__":
    main()
