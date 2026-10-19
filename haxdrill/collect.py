"""Generate a world-model dataset from the simulator.

    python -m haxdrill.collect --ticks 2000000 --out data/sim.npz

Output npz: obs (T,12) float32, act (T,) int8, ep (T,) int32, kicked (T,) bool.
Transition i -> i+1 is valid iff ep[i] == ep[i+1].
"""
from __future__ import annotations

import argparse

import numpy as np
from tqdm import tqdm

from . import config as C
from .policies import OpponentPolicy, RedExplorer
from .sim import Sim


def collect(ticks: int, seed: int = 0, max_ep_ticks: int = 1200, p_random_reset: float = 0.6, policy: str | None = None, p_rl_red: float = 0.5):
    """Blue = scripted opponent, or the RL policy if `policy` is given (then red is also driven by
    the RL policy for a fraction p_rl_red of episodes so the data contains competent play)."""
    rng = np.random.default_rng(seed)
    sim = Sim(seed)
    red, blue = RedExplorer(rng), OpponentPolicy(rng)
    ctrl = None
    if policy:
        from .rl import PolicyController

        ctrl = PolicyController(policy, 1, 1)
    red_is_rl = False
    obs = np.zeros((ticks, C.STATE_DIM), np.float32)
    act = np.zeros(ticks, np.int8)
    ep = np.zeros(ticks, np.int32)
    kicked = np.zeros(ticks, bool)
    ep_id, ep_t = 0, 0
    sim.reset(randomize=rng.random() < p_random_reset)
    for t in tqdm(range(ticks), desc="collect", unit="tick"):
        s = sim.get_state()
        if ctrl is not None:
            acts = ctrl.act(s, [p.kick_armed for p in sim.players], [0, 1])
            a_b = acts[1]
            a_r = acts[0] if red_is_rl else red.act(s, sim.players[0].kick_armed)
        else:
            a_r = red.act(s, sim.players[0].kick_armed)
            a_b = blue.act(s, sim.players[1].kick_armed)
        obs[t], act[t], ep[t] = s, a_r, ep_id
        goal = sim.step(a_r, a_b)
        kicked[t] = sim.players[0].kicked
        ep_t += 1
        if goal is not None or ep_t >= max_ep_ticks:
            ep_id += 1
            ep_t = 0
            sim.reset(randomize=rng.random() < p_random_reset, kickoff_team=(1 - goal) if goal is not None else int(rng.integers(0, 2)))
            red.new_episode()
            blue.new_personality()
            if ctrl is not None:
                ctrl.reset()
            red_is_rl = rng.random() < p_rl_red
    return dict(obs=obs, act=act, ep=ep, kicked=kicked, n_red=1, n_blue=1, stadium=C.STADIUM_NAME)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/sim.npz")
    ap.add_argument("--policy", default=None, help="PPO checkpoint: use the RL policy as the opponent (and partly as red)")
    C.add_stadium_arg(ap)
    args = ap.parse_args()
    C.handle_stadium_arg(args)
    d = collect(args.ticks, args.seed, policy=args.policy)
    np.savez_compressed(args.out, **d)
    n_ep = int(d["ep"].max()) + 1
    print(f"saved {args.out}: {args.ticks} ticks, {n_ep} episodes, {int(d['kicked'].sum())} red kicks")


if __name__ == "__main__":
    main()
