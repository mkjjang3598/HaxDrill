"""PPO self-play for the opponent (and teammates) in the exact simulator.

    python -m haxdrill.rl --n 1 --iters 1000 --out runs/ppo_n1.pt        # 1v1
    python -m haxdrill.rl --n 2 --iters 2000 --out runs/ppo_n2.pt        # 2v2
    python -m haxdrill.rl --n 1 --bc data/me_*.npz ...                    # behaviour-clone from human logs first

AlphaGo-style loop: (optional) imitation init from human logs -> self-play against a pool of
past snapshots (so the policy keeps improving instead of cycling) -> checkpoints that play.py,
collect.py and the web build load as the opponent. One network controls every player; blue's
observations are mirrored so everybody "attacks +x".
"""
from __future__ import annotations

import argparse
import copy
import glob
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C
from .vecsim import VecSim

POS_SCALE, VEL_SCALE = 400.0, 5.0


OBS_VERSION = 2  # 1 = raw ego-centric features; 2 = + aiming geometry (see build_obs)
GEO_DIM = 9


def obs_dim(n: int, version: int = OBS_VERSION) -> int:
    base = 10 + 4 * (n - 1) + 4 * n + 1
    return base + (GEO_DIM if version >= 2 else 0)


def build_obs(pos: np.ndarray, vel: np.ndarray, kick_armed: np.ndarray, n_red: int, n_blue: int, version: int = OBS_VERSION) -> np.ndarray:
    """pos, vel: (E, D, 2) with disc 0 = ball, then reds, then blues. Returns (E, P, obs_dim).
    Ego-centric; blue players see an x-mirrored world so the same policy plays both sides.
    version 2 appends aiming geometry: ball->goal vector & distance, me->ball unit vector & distance,
    cos/sin of the angle between (me->ball) and (ball->goal) ("am I behind the ball"), kick-in-range flag."""
    E, D, _ = pos.shape
    P = D - 1
    out = []
    for p in range(P):
        team = 0 if p < n_red else 1
        sgn = 1.0 if team == 0 else -1.0
        flip = np.array([sgn, 1.0])
        me = pos[:, p + 1] * flip
        mv = vel[:, p + 1] * flip
        ball = pos[:, 0] * flip
        bv = vel[:, 0] * flip
        feats = [me / POS_SCALE, mv / VEL_SCALE, ball / POS_SCALE, bv / VEL_SCALE, (ball - me) / POS_SCALE]
        mates = [q for q in range(P) if q != p and (q < n_red) == (team == 0)]
        opps = [q for q in range(P) if (q < n_red) != (team == 0)]
        for group in (mates, opps):
            if not group:
                continue
            gp = pos[:, [q + 1 for q in group]] * flip  # (E, k, 2)
            gv = vel[:, [q + 1 for q in group]] * flip
            rel = gp - me[:, None]
            order = np.argsort(np.linalg.norm(rel, axis=-1), axis=1)
            rel = np.take_along_axis(rel, order[..., None], 1)
            gv = np.take_along_axis(gv, order[..., None], 1)
            feats += [rel.reshape(E, -1) / POS_SCALE, gv.reshape(E, -1) / VEL_SCALE]
        feats.append(kick_armed[:, p : p + 1].astype(float))
        if version >= 2:
            g = np.array([C.GOAL_X, 0.0])
            bg = g[None] - ball  # ball -> enemy goal (mirrored frame: always +x)
            bg_d = np.linalg.norm(bg, axis=-1, keepdims=True)
            mb = ball - me  # me -> ball
            mb_d = np.linalg.norm(mb, axis=-1, keepdims=True)
            u_mb = mb / np.maximum(mb_d, 1e-6)
            u_bg = bg / np.maximum(bg_d, 1e-6)
            cos_a = (u_mb * u_bg).sum(-1, keepdims=True)
            sin_a = (u_mb[:, 0:1] * u_bg[:, 1:2] - u_mb[:, 1:2] * u_bg[:, 0:1])
            in_range = (mb_d <= C.PLAYER["radius"] + C.BALL["radius"] + C.KICK_RANGE_EXTRA).astype(float)
            feats += [bg / POS_SCALE, bg_d / POS_SCALE, u_mb, mb_d / POS_SCALE, cos_a, sin_a, in_range]
        out.append(np.concatenate(feats, -1))
    return np.stack(out, 1).astype(np.float32)


def mirror_action(a: np.ndarray) -> np.ndarray:
    """Blue acts in mirrored coordinates: flip dx."""
    kick, m = a // 9, a % 9
    dx, dy = m // 3, m % 3
    return (2 - dx) * 3 + dy + 9 * kick


class Policy(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256, n_actions: int = C.N_ACTIONS):
        super().__init__()
        self.obs_dim, self.hidden = obs_dim, hidden
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, obs):
        h = self.trunk(obs)
        return self.pi(h), self.v(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        logits, v = self(obs)
        if deterministic:
            a = logits.argmax(-1)
            return a, None, v
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), v


def save_policy(policy: Policy, path: str, **extra):
    torch.save({"state_dict": policy.state_dict(), "obs_dim": policy.obs_dim, "hidden": policy.hidden, **extra}, path)


def load_policy(path: str, device="cpu") -> tuple[Policy, dict]:
    ck = torch.load(path, map_location=device)
    ck.setdefault("obs_version", 1)
    pol = Policy(ck["obs_dim"], ck["hidden"]).to(device)
    pol.load_state_dict(ck["state_dict"])
    pol.eval()
    return pol, ck


# --------------------------------------------------------------------------- scalar helpers
def state_to_posvel(state: np.ndarray, n_red: int, n_blue: int):
    """sim.Sim state [reds, ball, blues] -> (pos (1,D,2), vel (1,D,2)) in VecSim order [ball, reds, blues]."""
    s = np.asarray(state, float).reshape(n_red + n_blue + 1, 4)
    order = [n_red] + list(range(n_red)) + list(range(n_red + 1, n_red + n_blue + 1))
    s = s[order]
    return s[None, :, :2], s[None, :, 2:]


class PolicyController:
    """Drives a set of players in a scalar sim.Sim with a trained policy (used by play/collect/evaluate)."""

    def __init__(self, path: str, n_red: int, n_blue: int, device="cpu", repeat: int | None = None, deterministic: bool = False):
        self.policy, ck = load_policy(path, device)
        self.device = device
        self.version = ck["obs_version"]
        self.K = ck.get("stack", 1)
        self.hist = None
        hc = ck.get("handicap") or {}
        self.temp, self.kick_keep = float(hc.get("temperature", 1.0)), float(hc.get("kick_keep", 1.0))
        if "repeat" in hc and repeat is None:
            self.repeat = int(hc["repeat"])
        self._rng = np.random.default_rng()
        self.n_red, self.n_blue = n_red, n_blue
        self.repeat = repeat or ck.get("repeat", 2)
        self.deterministic = deterministic
        self._last = None
        self._t = 0

    def reset(self):
        """Call on episode reset so the short memory does not carry over."""
        self._last, self._t, self.hist = None, 0, None

    def act(self, state: np.ndarray, kick_armed: list[bool], players: list[int]) -> dict[int, int]:
        """Returns {player_index: action} for the requested player indices (sim.Sim order)."""
        if self._last is not None and self._t % self.repeat != 0:
            self._t += 1
            return {p: self._last[p] for p in players}
        pos, vel = state_to_posvel(state, self.n_red, self.n_blue)
        obs = build_obs(pos, vel, np.array(kick_armed)[None], self.n_red, self.n_blue, self.version)[0]
        if self.K > 1:
            if self.hist is None:
                self.hist = np.repeat(obs[:, None, :], self.K, 1)
            else:
                self.hist = np.roll(self.hist, 1, 1)
                self.hist[:, 0] = obs
            obs = self.hist.reshape(obs.shape[0], -1)
        if self.temp != 1.0 and not self.deterministic:
            with torch.no_grad():
                logits, _ = self.policy(torch.as_tensor(obs, device=self.device))
            a = torch.distributions.Categorical(logits=logits / self.temp).sample()
        else:
            a, _, _ = self.policy.act(torch.as_tensor(obs, device=self.device), self.deterministic)
        a = a.cpu().numpy().copy()
        if self.kick_keep < 1.0:
            drop = (a >= 9) & (self._rng.random(a.shape) > self.kick_keep)
            a[drop] -= 9
        out = {}
        for p in range(self.n_red + self.n_blue):
            out[p] = int(a[p]) if p < self.n_red else int(mirror_action(a[p : p + 1])[0])
        self._last = out
        self._t = 1
        return {p: out[p] for p in players}


# --------------------------------------------------------------------------- shot geometry
def shot_on_target(bx, by, bvx, bvy, att_dir, max_bounce: int = 2):
    """Vectorised: would the ball's current velocity carry it into the enemy goal mouth, allowing up to
    `max_bounce` reflections off the top/bottom sidelines? Ignores damping (direction only).
    bx, by, bvx, bvy: arrays; att_dir: +1 (attacking +x) or -1, scalar or array."""
    bx, by, bvx, bvy = (np.asarray(a, float).copy() for a in (bx, by, bvx, bvy))
    att = np.broadcast_to(np.asarray(att_dir, float), bx.shape).copy()
    wall = C.FIELD_HALF_H - C.BALL["radius"]
    gx = C.GOAL_X * att
    alive = bvx * att > 0.5  # must be moving towards that goal with some speed
    on = np.zeros(bx.shape, bool)
    for _ in range(max_bounce + 1):
        if not alive.any():
            break
        t_goal = np.where(alive, (gx - bx) / np.where(bvx == 0, 1e-9, bvx), 0.0)
        y_goal = by + bvy * t_goal
        reach = alive & (np.abs(y_goal) <= wall)  # reaches the goal line without touching a sideline
        on |= reach & (np.abs(y_goal) < C.GOAL_HALF - 4.0)
        alive &= ~reach
        sgn = np.sign(bvy)
        t_wall = np.where(alive & (bvy != 0), (sgn * wall - by) / np.where(bvy == 0, 1e-9, bvy), 0.0)
        bx = np.where(alive, bx + bvx * t_wall, bx)
        by = np.where(alive, sgn * wall, by)
        bvy = np.where(alive, -bvy, bvy)
    return on


# --------------------------------------------------------------------------- evaluation
def eval_vs_scripted(policy: Policy, device, games: int = 3, seconds: int = 60, repeat: int = 2, seed: int = 0, version: int = OBS_VERSION):
    """1v1: our policy as RED vs the scripted OpponentPolicy as BLUE. Returns (goals_for, goals_against)."""
    from .policies import OpponentPolicy
    from .sim import Sim

    rng = np.random.default_rng(seed)
    sim = Sim(seed)
    opp = OpponentPolicy(rng)
    gf = ga = 0
    for g in range(games):
        sim.reset()
        opp.new_personality()
        a_red = 4
        for t in range(seconds * C.TICK_HZ):
            s = sim.get_state()
            if t % repeat == 0:
                pos, vel = state_to_posvel(s, 1, 1)
                obs = build_obs(pos, vel, np.array([[p.kick_armed for p in sim.players]]), 1, 1, version)[0, 0]
                a, _, _ = policy.act(torch.as_tensor(obs, device=device)[None])
                a_red = int(a.item())
            a_blue = opp.act(s, sim.players[1].kick_armed)
            goal = sim.step(a_red, a_blue)
            if goal is not None:
                gf += goal == 0
                ga += goal == 1
                sim.reset(kickoff_team=1 - goal)
    return gf, ga


# --------------------------------------------------------------------------- behaviour cloning
def load_human_data(paths: list[str], n: int, device, version: int = OBS_VERSION):
    """(obs, act) tensors for the human (red_0) from play logs whose team size is n. None if empty."""
    obs_l, act_l = [], []
    for p in paths:
        d = np.load(p)
        n_file = int(d["n_red"]) if "n_red" in d.files else 1
        if n_file != n:
            continue
        st, act = d["obs"].astype(float), d["act"].astype(np.int64)
        E, D = len(st), 2 * n + 1
        sv = st.reshape(E, D, 4)
        order = [n] + list(range(n)) + list(range(n + 1, D))
        obs_l.append(build_obs(sv[:, order, :2], sv[:, order, 2:], np.ones((E, 2 * n), bool), n, n, version)[:, 0])
        act_l.append(act)
    if not obs_l:
        return None
    return torch.as_tensor(np.concatenate(obs_l), device=device), torch.as_tensor(np.concatenate(act_l), device=device)


def behaviour_clone(policy: Policy, paths: list[str], device, epochs: int = 5, lr: float = 1e-3, n: int = 1, version: int = OBS_VERSION):
    """Supervised init from play logs (obs/act npz as written by play --record / the web server).
    Only files whose team size matches `n` are used; the human is always red_0."""
    obs_l, act_l = [], []
    for p in paths:
        d = np.load(p)
        n_file = int(d["n_red"]) if "n_red" in d.files else 1
        if n_file != n:
            continue
        st, act = d["obs"].astype(float), d["act"].astype(np.int64)
        E = len(st)
        D = 2 * n + 1
        sv = st.reshape(E, D, 4)
        order = [n] + list(range(n)) + list(range(n + 1, D))  # [reds, ball, blues] -> [ball, reds, blues]
        pos, vel = sv[:, order, :2], sv[:, order, 2:]
        armed = np.ones((E, 2 * n), bool)
        obs_l.append(build_obs(pos, vel, armed, n, n, version)[:, 0])
        act_l.append(act)
    if not obs_l:
        print(f"  BC: no {n}v{n} logs found, skipping")
        return 0
    X = torch.as_tensor(np.concatenate(obs_l), device=device)
    Y = torch.as_tensor(np.concatenate(act_l), device=device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(len(X), device=device)
        tot = 0.0
        for i in range(0, len(X), 4096):
            idx = perm[i : i + 4096]
            logits, _ = policy(X[idx])
            loss = F.cross_entropy(logits, Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        print(f"  BC epoch {ep+1}: loss {tot/len(X):.4f}  ({len(X):,} samples, {n}v{n})")
    return len(X)


# --------------------------------------------------------------------------- PPO
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1, help="players per team")
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=64, help="decision steps per rollout")
    ap.add_argument("--repeat", type=int, default=2, help="action repeat (ticks per decision)")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--obs-version", type=int, default=OBS_VERSION, help="1 = raw features (old checkpoints), 2 = + aiming geometry")
    ap.add_argument("--max-seconds", type=int, default=60, help="episode time limit")
    ap.add_argument("--p-random", type=float, default=0.3, help="fraction of resets with scattered positions")
    ap.add_argument("--pass-bonus", type=float, default=None, help="team reward when a kick is received (kicked again) by a teammate within 3 s (default 0.1 if n>1 else 0)")
    ap.add_argument("--spread", type=float, default=None, help="per-tick penalty for teammates closer than 60 units (default 0.0005 if n>1 else 0)")
    ap.add_argument("--marking", type=float, default=None, help="man-marking: per-tick penalty per 100 units a defender is away from its assigned opponent while the ball is in its own half (default 0.0005 if n>1 else 0)")
    ap.add_argument("--bc-aux", nargs="*", default=None, help="human play logs mixed into every PPO update as an imitation loss (style retention, AlphaGo SL->RL style)")
    ap.add_argument("--bc-coef", type=float, default=0.05, help="weight of the --bc-aux imitation loss")
    ap.add_argument("--wall-bonus", type=float, default=0.1, help="reward for a wall-bounce dribble: kick -> ball bounces off a side/goal line -> same player re-collects it within 2.5 s having gained >= 40 units towards the enemy goal")
    ap.add_argument("--shot-bonus", type=float, default=0.15, help="reward for a kick whose ball trajectory (straight or banked off a sideline, <=2 bounces) would enter the enemy goal mouth")
    ap.add_argument("--beat-bonus", type=float, default=0.02, help="reward for dribbling past an opponent (opponent goes from in front of to behind the ball carrier); once per opponent per possession")
    ap.add_argument("--pool-every", type=int, default=25, help="snapshot the policy into the opponent pool every N iters")
    ap.add_argument("--pool-size", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--bc", nargs="*", default=None, help="npz play logs for behaviour-cloning init (files matching --n are used)")
    ap.add_argument("--bc-epochs", type=int, default=5)
    ap.add_argument("--pool-extra", nargs="*", default=None, help="extra opponent checkpoints kept permanently in the pool (e.g. a behaviour-cloned human policy)")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    C.add_stadium_arg(ap)
    args = ap.parse_args()
    C.handle_stadium_arg(args)
    if args.selftest:
        from .vecsim import selftest

        print("max |Sim - VecSim|:", selftest(n=args.n))
        return
    out = args.out or f"runs/ppo_n{args.n}.pt"
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n, E, T, R = args.n, args.envs, args.steps, args.repeat
    P = 2 * n
    V = args.obs_version
    pass_bonus = args.pass_bonus if args.pass_bonus is not None else (0.1 if n > 1 else 0.0)
    spread = args.spread if args.spread is not None else (0.0005 if n > 1 else 0.0)
    marking = args.marking if args.marking is not None else (0.0005 if n > 1 else 0.0)
    beat_bonus = args.beat_bonus
    wall_bonus = args.wall_bonus
    shot_bonus = args.shot_bonus
    team_of = np.where(np.arange(P) < n, 0, 1)
    att_dir = np.where(team_of == 0, 1.0, -1.0)
    human = None
    if args.bc_aux:
        paths = sorted(sum((glob.glob(p) for p in args.bc_aux), []))
        human = load_human_data(paths, n, dev, V)
        print(f"style-retention imitation: {0 if human is None else len(human[0]):,} human samples, coef {args.bc_coef}")
    # wall-bounce dribble bookkeeping (per env)
    wb_kicker = np.full(E, -1, np.int16)  # who kicked
    wb_tick = np.zeros(E, np.int64)  # when
    wb_x = np.zeros(E)  # kicker's progress coordinate (towards its goal) at kick time
    wb_bounced = np.zeros(E, bool)  # ball has hit a wall since that kick
    # dribbling bookkeeping: for each (env, carrier p, opponent q) whether q is "in front" of p
    # (closer to p's attacking goal along x than p is) while p carries the ball
    in_front_prev = np.zeros((E, P, P), bool)
    beat_done = np.zeros((E, P, P), bool)  # (env, carrier, opponent) already rewarded this possession

    def carrier_mask():
        """(E,P) True if player p has the ball: last kicker within 2 s or nearest player within 30 units."""
        d = np.linalg.norm(vs.pos[:, 1:] - vs.pos[:, :1], axis=-1)  # (E,P)
        nearest = d.argmin(1)
        has = np.zeros((E, P), bool)
        has[np.arange(E), nearest] = d[np.arange(E), nearest] < 30
        recent = (vs.last_kicker >= 0) & (vs.tick - vs.last_kick_tick <= 120)
        has[recent, vs.last_kicker[recent]] = True
        return has

    def in_front():
        """(E,P,P): [e,p,q] = opponent q is between carrier p and p's attacking goal (x-wise)."""
        x = vs.pos[:, 1:, 0]  # (E,P)
        att = np.where(team_of == 0, 1.0, -1.0)  # attacking direction per player
        rel = (x[:, None, :] - x[:, :, None]) * att[None, :, None]  # (E,p,q): q ahead of p along p's attack axis
        opp = team_of[None, :, None] != team_of[None, None, :]
        return (rel > 0) & opp

    def marking_penalty():
        """Greedy nearest-unmarked assignment of defenders to attackers while the ball is in the defending half."""
        pen = np.zeros((E, P))
        bx = vs.pos[:, 0, 0]
        for t_ in (0, 1):
            defending = (bx < 0) if t_ == 0 else (bx > 0)
            if not defending.any():
                continue
            mine = np.nonzero(team_of == t_)[0]
            theirs = np.nonzero(team_of != t_)[0]
            dmat = np.linalg.norm(vs.pos[:, 1 + mine][:, :, None] - vs.pos[:, 1 + theirs][:, None, :], axis=-1)  # (E,n,n)
            dmat = dmat[defending]
            assigned = np.zeros(dmat.shape, bool)
            taken_d = np.zeros((dmat.shape[0], len(mine)), bool)
            taken_a = np.zeros((dmat.shape[0], len(theirs)), bool)
            for _ in range(len(mine)):
                dm = np.where(taken_d[:, :, None] | taken_a[:, None, :], np.inf, dmat)
                flat = dm.reshape(len(dm), -1).argmin(1)
                di, ai = np.divmod(flat, len(theirs))
                rows = np.arange(len(dm))
                assigned[rows, di, ai] = True
                taken_d[rows, di] = True
                taken_a[rows, ai] = True
            dist_assigned = (dmat * assigned).sum(2)  # (e,n)
            pen_rows = np.maximum(dist_assigned - 80.0, 0.0) / 100.0 * marking
            idx = np.nonzero(defending)[0]
            pen[idx[:, None], mine[None, :]] -= pen_rows
        return pen
    od = obs_dim(n, V)
    policy = Policy(od, args.hidden).to(dev)
    start_iter = 0
    if args.resume:
        policy, ck = load_policy(args.resume, dev)
        policy.train()
        start_iter = ck.get("iters", 0)
        V = ck["obs_version"]
        od = obs_dim(n, V)
        assert policy.obs_dim == od, f"checkpoint obs_dim {policy.obs_dim} != {od}"
    if args.bc:
        paths = sorted(sum((glob.glob(p) for p in args.bc), []))
        print(f"behaviour cloning from {len(paths)} files")
        behaviour_clone(policy, paths, dev, epochs=args.bc_epochs, n=n, version=V)
    if args.iters == 0:  # imitation only
        save_policy(policy, out, n=n, repeat=R, iters=0, stadium=C.STADIUM_NAME, obs_version=V)
        print("saved", out)
        return
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    pool: list[dict] = [copy.deepcopy(policy.state_dict())]
    fixed_pool: list[dict] = []
    for pth in args.pool_extra or []:
        extra, eck = load_policy(pth, dev)
        assert extra.obs_dim == od and eck["obs_version"] == V, f"{pth}: obs_dim/version mismatch (need v{V}, dim {od})"
        fixed_pool.append(copy.deepcopy(extra.state_dict()))
    pool_policy = Policy(od, args.hidden).to(dev)
    pool_policy.load_state_dict(pool[-1])
    pool_policy.eval()

    vs = VecSim(E, n, n, args.seed)
    vs.reset(p_random=args.p_random)
    half = E // 2  # envs [half:] have blue driven by a pool snapshot
    max_ticks = args.max_seconds * C.TICK_HZ
    # which (env, player) samples are on-policy and stored: all reds + blues in envs[:half]
    store_mask = np.zeros((E, P), bool)
    store_mask[:, :n] = True
    store_mask[:half, n:] = True
    N = int(store_mask.sum())
    sign = np.where(np.arange(P) < n, 1.0, -1.0)  # reward sign per player (red +, blue -)

    buf_obs = torch.zeros((T, N, od), device=dev)
    buf_act = torch.zeros((T, N), dtype=torch.long, device=dev)
    buf_logp = torch.zeros((T, N), device=dev)
    buf_val = torch.zeros((T, N), device=dev)
    buf_rew = torch.zeros((T, N), device=dev)
    buf_done = torch.zeros((T, N), device=dev)

    def potential():
        """Shaping potential per (E,P): ball progress towards enemy goal + closeness to the ball."""
        bx = vs.pos[:, 0, 0]
        phi_team = 0.001 * bx[:, None] * sign[None]  # (E,P)
        d = np.linalg.norm(vs.pos[:, 1:] - vs.pos[:, :1], axis=-1)  # (E,P) player-ball distance
        return phi_team - 0.0005 * d

    obs = build_obs(vs.pos, vs.vel, vs.kick_armed, n, n, V)
    t0 = time.time()
    ep_goals = np.zeros(2)  # red, blue goals in on-policy-vs-pool envs for logging
    ep_count = 0
    for it in range(start_iter + 1, start_iter + args.iters + 1):
        if it % args.pool_every == 0:
            pool.append(copy.deepcopy(policy.state_dict()))
            pool = pool[-args.pool_size :]
        # pick one opponent snapshot for this rollout (recent ones more likely); fixed extras
        # (e.g. human-imitation policies) are drawn with the same weight as the newest snapshot
        cands = pool + fixed_pool
        w = np.concatenate([np.arange(1, len(pool) + 1, dtype=float), np.full(len(fixed_pool), 0.5 * len(pool))])
        pool_policy.load_state_dict(cands[np.random.default_rng().choice(len(cands), p=w / w.sum())])
        policy.eval()
        for t in range(T):
            obs_t = torch.as_tensor(obs, device=dev)  # (E,P,od)
            a_cur, logp_cur, v_cur = policy.act(obs_t.reshape(E * P, od))
            a_cur = a_cur.reshape(E, P)
            a_pool, _, _ = pool_policy.act(obs_t[half:, n:].reshape(-1, od))
            actions = a_cur.cpu().numpy()
            actions[half:, n:] = a_pool.cpu().numpy().reshape(-1, n)
            # mirror blue's actions back to world coordinates
            sim_actions = actions.copy()
            sim_actions[:, n:] = mirror_action(actions[:, n:])
            phi0 = potential()
            goal = np.full(E, -1, np.int8)
            rew = np.zeros((E, P))
            for _ in range(R):
                prev_k, prev_t = vs.last_kicker.copy(), vs.last_kick_tick.copy()
                g = vs.step(sim_actions)
                newg = (goal < 0) & (g >= 0)
                goal[newg] = g[newg]
                if pass_bonus > 0:
                    # a completed pass: player p kicks a ball last kicked by a teammate (not p) within 3 s
                    for p_ in range(P):
                        k = vs.kicked[:, p_] & (prev_k >= 0) & (prev_k != p_) & (team_of[np.maximum(prev_k, 0)] == team_of[p_]) & (vs.tick - prev_t <= 180)
                        if k.any():
                            rew[k, p_] += pass_bonus
                            rew[k, prev_k[k]] += pass_bonus
                if spread > 0:
                    for a_ in range(P):
                        for b_ in range(a_ + 1, P):
                            if team_of[a_] != team_of[b_]:
                                continue
                            close = np.linalg.norm(vs.pos[:, a_ + 1] - vs.pos[:, b_ + 1], axis=-1) < 60
                            rew[close, a_] -= spread
                            rew[close, b_] -= spread
                if shot_bonus > 0:
                    # on-target shot right after a kick: straight, or banked off a sideline (up to 2 bounces)
                    for p_ in range(P):
                        k = vs.kicked[:, p_]
                        if not k.any():
                            continue
                        idx = np.nonzero(k)[0]
                        on = shot_on_target(vs.pos[idx, 0, 0], vs.pos[idx, 0, 1], vs.vel[idx, 0, 0], vs.vel[idx, 0, 1], att_dir[p_])
                        rew[idx[on], p_] += shot_bonus
                if wall_bonus > 0:
                    # 1) new kicks start a possible wall-bounce sequence
                    for p_ in range(P):
                        k = vs.kicked[:, p_]
                        if k.any():
                            wb_kicker[k] = p_
                            wb_tick[k] = vs.tick[k]
                            wb_x[k] = vs.pos[k, p_ + 1, 0] * att_dir[p_]
                            wb_bounced[k] = False
                    # 2) wall contact while the sequence is live
                    live = (wb_kicker >= 0) & (vs.tick - wb_tick <= 150)
                    wb_bounced |= live & vs.ball_wall_hit
                    # 3) the kicker re-collects the ball after the bounce, further up the pitch -> bonus
                    d_pb = np.linalg.norm(vs.pos[:, 1:] - vs.pos[:, :1], axis=-1)
                    for p_ in range(P):
                        got = live & wb_bounced & (wb_kicker == p_) & (d_pb[:, p_] < 30) & (vs.tick - wb_tick >= 10)
                        if got.any():
                            prog = vs.pos[got, p_ + 1, 0] * att_dir[p_] - wb_x[got]
                            ok = prog >= 40.0
                            idx = np.nonzero(got)[0]
                            rew[idx[ok], p_] += wall_bonus
                            wb_kicker[idx] = -1  # sequence consumed either way
                    wb_kicker[~live] = -1
                if marking > 0:
                    rew += marking_penalty()
                if beat_bonus > 0:
                    # dribble past: carrier p keeps the ball while opponent q flips from in front to behind
                    carry = carrier_mask()
                    front_now = in_front()
                    beaten = in_front_prev & ~front_now & carry[:, :, None] & ~beat_done
                    rew += beaten.sum(2) * beat_bonus
                    beat_done |= beaten
                    beat_done[~carry] = False  # new possession -> can be rewarded again
                    in_front_prev = front_now
            scored = goal >= 0
            timeout = (~scored) & (vs.tick >= max_ticks)
            rew += (potential() - phi0)  # potential-based shaping
            team_goal = np.where(goal[:, None] == 0, 1.0, np.where(goal[:, None] == 1, -1.0, 0.0)) * sign[None]
            rew += team_goal
            # bootstrap truncated episodes with V(s_T)
            if timeout.any():
                obs_to = build_obs(vs.pos[timeout], vs.vel[timeout], vs.kick_armed[timeout], n, n, V)
                with torch.no_grad():
                    _, v_to = policy(torch.as_tensor(obs_to, device=dev).reshape(-1, od))
                rew[timeout] += args.gamma * v_to.reshape(-1, P).cpu().numpy()
            done = scored | timeout
            if (scored[half:]).any():
                ep_goals[0] += (goal[half:] == 0).sum()
                ep_goals[1] += (goal[half:] == 1).sum()
            ep_count += done.sum()
            ko = np.where(goal >= 0, 1 - goal, 0).astype(np.int8)  # conceding team kicks off; red after a timeout
            vs.reset(done, p_random=args.p_random, kickoff_team=ko)
            in_front_prev[done] = False
            beat_done[done] = False
            wb_kicker[done] = -1
            # store on-policy samples
            m = torch.as_tensor(store_mask, device=dev)
            buf_obs[t] = obs_t[m]
            buf_act[t] = a_cur.reshape(E, P)[m]
            buf_logp[t] = logp_cur.reshape(E, P)[m]
            buf_val[t] = v_cur.reshape(E, P)[m]
            buf_rew[t] = torch.as_tensor(rew, device=dev, dtype=torch.float32)[m]
            buf_done[t] = torch.as_tensor(np.repeat(done[:, None], P, 1), device=dev, dtype=torch.float32)[m]
            obs = build_obs(vs.pos, vs.vel, vs.kick_armed, n, n, V)
        # GAE
        with torch.no_grad():
            _, last_v = policy(torch.as_tensor(obs, device=dev).reshape(E * P, od))
            last_v = last_v.reshape(E, P)[torch.as_tensor(store_mask, device=dev)]
            adv = torch.zeros_like(buf_rew)
            gae = torch.zeros(N, device=dev)
            for t in reversed(range(T)):
                nv = last_v if t == T - 1 else buf_val[t + 1]
                nonterm = 1.0 - buf_done[t]
                delta = buf_rew[t] + args.gamma * nv * nonterm - buf_val[t]
                gae = delta + args.gamma * args.lam * nonterm * gae
                adv[t] = gae
            ret = adv + buf_val
        # PPO update
        policy.train()
        b_obs = buf_obs.reshape(-1, od)
        b_act, b_logp, b_adv, b_ret = buf_act.reshape(-1), buf_logp.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        tot = len(b_obs)
        stats = []
        for _ in range(args.epochs):
            perm = torch.randperm(tot, device=dev)
            for i in range(0, tot, args.minibatch):
                idx = perm[i : i + args.minibatch]
                logits, v = policy(b_obs[idx])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(b_act[idx])
                ratio = torch.exp(logp - b_logp[idx])
                pg = -torch.min(ratio * b_adv[idx], torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * b_adv[idx]).mean()
                vl = F.mse_loss(v, b_ret[idx])
                ent = dist.entropy().mean()
                loss = pg + 0.5 * vl - args.ent * ent
                if human is not None and args.bc_coef > 0:
                    hx, hy = human
                    hi = torch.randint(len(hx), (min(4096, len(hx)),), device=dev)
                    h_logits, _ = policy(hx[hi])
                    loss = loss + args.bc_coef * F.cross_entropy(h_logits, hy[hi])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()
                stats.append((pg.item(), vl.item(), ent.item(), ((ratio - 1).abs() > args.clip).float().mean().item()))
        pg, vl, ent, clipfrac = np.mean(stats, 0)
        if it % 5 == 0 or it == start_iter + 1:
            gpm = ep_goals / max(1e-9, (5 * T * R * half) / (C.TICK_HZ * 60))  # goals per env-minute vs pool
            print(
                f"it {it:5d} | {tot:,} samples | pg {pg:+.4f} v {vl:.4f} ent {ent:.3f} clip {clipfrac:.2f} "
                f"| vs pool: red {gpm[0]:.2f} blue {gpm[1]:.2f} goals/min | pool {len(pool)} | {time.time()-t0:.0f}s",
                flush=True,
            )
            ep_goals[:] = 0
        if n == 1 and it % args.eval_every == 0:
            policy.eval()
            gf, ga = eval_vs_scripted(policy, dev, games=3, seconds=60, repeat=R, version=V)
            print(f"      eval vs scripted opponent (3 x 60s): {gf} - {ga}", flush=True)
        if it % args.save_every == 0 or it == start_iter + args.iters:
            save_policy(policy, out, n=n, repeat=R, iters=it, stadium=C.STADIUM_NAME, obs_version=V)
    print("saved", out)


if __name__ == "__main__":
    main()
