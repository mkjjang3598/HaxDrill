"""The improvement loop: fold collected human logs back into the opponent policy and the world
model, then re-export for the web build.

    python -m haxdrill.retrain --n 1 --iters 300          # 1v1 bots from data/play + data/web logs
    python -m haxdrill.retrain --n 2 --iters 200          # 2v2

Steps (each is a normal CLI you can also run by hand):
  1. rl.py  --bc <logs> --iters 0 -> runs/human_bc.pt         (pure imitation of the players' logs)
  2. rl.py  --resume runs/ppo_n1.pt --pool-extra runs/human_bc.pt --iters N
            (self-play continues, but human-like opponents stay in the pool -> robust to human play)
  3. export.py  runs/ppo_n1.pt -> web/policy_n1.json
  4. collect.py --policy runs/ppo_n1.pt (fresh sim data vs the new opponent)
  5. train.py   data/sim.npz + <logs> -> runs/wm.pt ; export -> web/wm.json
"""
from __future__ import annotations

import argparse
import glob
import subprocess
import sys


def run(*cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, "-m", *cmd], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", default=["data/play/*.npz", "data/web/*.npz"])
    ap.add_argument("--n", type=int, default=1, help="team size (uses runs/ppo_n{n}.pt and logs of that size)")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--envs", type=int, default=None)
    ap.add_argument("--wm", action="store_true", help="also retrain the 1v1 world model (off by default)")
    ap.add_argument("--wm-ticks", type=int, default=2_000_000)
    args = ap.parse_args()
    n = args.n
    policy = args.policy or f"runs/ppo_n{n}.pt"
    envs = str(args.envs or (2048 if n == 1 else 1024))
    logs = sorted(sum((glob.glob(p) for p in args.logs), []))
    print(f"{len(logs)} log files")
    extra = []
    if logs:
        bc_path = f"runs/human_bc_n{n}.pt"
        run("haxdrill.rl", "--n", str(n), "--bc", *logs, "--bc-epochs", "20", "--iters", "0", "--out", bc_path)
        extra = ["--pool-extra", bc_path]
    if logs:
        extra += ["--bc-aux", *logs]  # style retention: keep imitating the humans while self-playing
    run("haxdrill.rl", "--n", str(n), "--resume", policy, "--iters", str(args.iters), "--envs", envs, "--out", policy, *extra)
    run("haxdrill.export", policy, f"web/policy_n{n}.json")
    if args.wm and n == 1:
        run("haxdrill.collect", "--ticks", str(args.wm_ticks), "--out", "data/sim.npz", "--policy", policy)
        run("haxdrill.train", "data/sim.npz", *logs, "--out", "runs/wm.pt", "--steps", "30000", "--hidden", "256", "--layers", "3")
        run("haxdrill.export", "runs/wm.pt", "web/wm.json")


if __name__ == "__main__":
    main()
