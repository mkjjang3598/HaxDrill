"""Human-vs-bot record from play logs: the metric that matters for "is the bot getting better?".

    python -m haxdrill.stats                      # data/play/*.npz + data/web/*.npz
    python -m haxdrill.stats 'data/play/*.npz'

Goal events come from the `goal` column (logs written after 2026-10-19); older logs fall back to a
heuristic on the last state of each episode (ball at the goal line, moving in).
"""
from __future__ import annotations

import glob
import sys
from collections import defaultdict

import numpy as np

from . import config as C


def goals_in_file(path: str):
    d = np.load(path)
    n = int(d["n_red"]) if "n_red" in d.files else 1
    obs, ep = d["obs"], d["ep"]
    red = blue = 0
    if "goal" in d.files:
        g = d["goal"]
        red, blue = int((g == 0).sum()), int((g == 1).sum())
    else:  # heuristic: last tick of an episode with the ball about to cross a goal line
        bi = 4 * n
        for e in np.unique(ep):
            last = obs[ep == e][-1]
            bx, by, bvx = last[bi], last[bi + 1], last[bi + 2]
            if abs(by) < C.GOAL_HALF + 6:
                if bx > C.GOAL_X - 12 and bvx > 0:
                    red += 1
                elif bx < -C.GOAL_X + 12 and bvx < 0:
                    blue += 1
    name = str(d["name"]) if "name" in d.files else ""
    return n, len(obs), red, blue, int(ep.max() - ep.min()) + 1, name


def main():
    pats = sys.argv[1:] or ["data/play/*.npz", "data/web/*.npz"]
    files = sorted(sum((glob.glob(p) for p in pats), []))
    if not files:
        print("no logs found")
        return
    agg = defaultdict(lambda: [0, 0, 0, 0])  # n -> ticks, red(human) goals, blue(bot) goals, episodes
    print(f"{'file':52s} {'mode':5s} {'min':>5s} {'you':>4s} {'bot':>4s}")
    for f in files:
        n, ticks, r, b, eps, name = goals_in_file(f)
        a = agg[n]
        a[0] += ticks; a[1] += r; a[2] += b; a[3] += eps
        print(f"{f[-52:]:52s} {n}v{n}   {ticks/3600:5.1f} {r:4d} {b:4d}  {name}")
    print()
    for n, (ticks, r, b, eps) in sorted(agg.items()):
        mins = ticks / 3600
        print(f"{n}v{n}: {mins:.1f} min, {eps} episodes | you {r} - {b} bot | you {r/max(mins,1e-9):.2f}/min, bot {b/max(mins,1e-9):.2f}/min")


if __name__ == "__main__":
    main()
