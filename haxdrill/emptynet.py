"""Empty-net finishing test: how often does the policy score with no opponent in the way?

    python -m haxdrill.emptynet runs/ppo_n1.pt
"""
from __future__ import annotations

import sys

import numpy as np

from . import config as C
from .rl import PolicyController, shot_on_target
from .sim import Sim


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/ppo_n1.pt"
    rng = np.random.default_rng(0)
    ctrl = PolicyController(ckpt, 1, 1)
    sim = Sim(0)

    def trial(bx, by, px, py, seconds=10):
        sim.reset(kickoff_team=None)
        sim.ball.x, sim.ball.y = bx, by
        sim.players[0].disc.x, sim.players[0].disc.y = px, py
        sim.players[1].disc.x, sim.players[1].disc.y = -400, -190
        ctrl.reset()
        kicks = on = 0
        for t in range(seconds * C.TICK_HZ):
            s = sim.get_state()
            a = ctrl.act(s, [p.kick_armed for p in sim.players], [0])[0]
            g = sim.step(a, 4)
            if sim.players[0].kicked:
                kicks += 1
                b = sim.ball
                on += int(shot_on_target(np.array([b.x]), np.array([b.y]), np.array([b.vx]), np.array([b.vy]), 1.0)[0])
            if g is not None:
                return (1 if g == 0 else -1), kicks, on, t / C.TICK_HZ
        return 0, kicks, on, seconds

    print(f"{'ball x':>8} {'goals/40':>9} {'kicks':>6} {'on-target':>10} {'avg s':>6}")
    for bx in (150, 250, 320):
        res = [trial(bx, float(rng.uniform(-120, 120)), bx - 60 + float(rng.uniform(-20, 20)), float(rng.uniform(-150, 150))) for _ in range(40)]
        goals = sum(r[0] == 1 for r in res)
        tt = np.mean([r[3] for r in res if r[0] == 1]) if goals else float("nan")
        print(f"{bx:8d} {goals:9d} {sum(r[1] for r in res):6d} {sum(r[2] for r in res):10d} {tt:6.1f}")


if __name__ == "__main__":
    main()
