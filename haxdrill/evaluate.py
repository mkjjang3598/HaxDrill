"""Closed-loop evaluation: roll the model and the simulator from the same start with the
same action sequence and report how fast they diverge.

    python -m haxdrill.evaluate runs/wm.pt --episodes 50 --horizon 300
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .model import load
from .policies import OpponentPolicy, RedExplorer
from .sim import Sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--seed", type=int, default=123)
    from . import config as C

    C.add_stadium_arg(ap)
    args = ap.parse_args()
    C.handle_stadium_arg(args)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load(args.ckpt, dev)
    rng = np.random.default_rng(args.seed)
    sim = Sim(args.seed)
    red, blue = RedExplorer(rng), OpponentPolicy(rng)
    errs = np.zeros((args.episodes, args.horizon, 3))  # player, ball, opponent position error
    for e in range(args.episodes):
        sim.reset(randomize=rng.random() < 0.5)
        red.new_episode()
        blue.new_personality()
        s_sim = sim.get_state()
        s_mod = torch.as_tensor(s_sim, device=dev)[None]
        for t in range(args.horizon):
            a_r = red.act(s_sim, sim.players[0].kick_armed)
            a_b = blue.act(s_sim, sim.players[1].kick_armed)
            goal = sim.step(a_r, a_b)
            s_sim = sim.get_state()
            with torch.no_grad():
                s_mod = model(s_mod, torch.tensor([a_r], device=dev))
            m = s_mod[0].cpu().numpy()
            for j, (i0, i1) in enumerate(((0, 1), (4, 5), (8, 9))):
                errs[e, t, j] = np.hypot(m[i0] - s_sim[i0], m[i1] - s_sim[i1])
            if goal is not None:
                errs[e, t + 1 :] = np.nan
                break
    mean = np.nanmean(errs, 0)
    print("closed-loop position error (HaxBall units; ball radius = 10, player radius = 15)")
    print(f"{'tick':>6} {'player':>8} {'ball':>8} {'opponent':>9}")
    for t in [0, 9, 29, 59, 119, 179, 239, 299]:
        if t < args.horizon:
            print(f"{t+1:>6} {mean[t,0]:8.2f} {mean[t,1]:8.2f} {mean[t,2]:9.2f}")


if __name__ == "__main__":
    main()
