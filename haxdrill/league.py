"""League training (AlphaStar-lite) for the "medium" bot.

    nohup .venv/bin/python -m haxdrill.league --n 1 --hours 24 --out-dir runs/league > runs/league.log 2>&1 &
    .venv/bin/python -m haxdrill.league --resume-dir runs/league --hours 12     # continue later

Each cycle:
  1. MAIN agent trains `--main-iters` PPO iterations against a mix of its past snapshots, the
     exploiters found so far and (optionally) a human-imitation policy. Its shaping rewards decay
     linearly to zero over `--shaping-decay-iters`, leaving only goals.
  2. An EXPLOITER (initialised from main) trains `--exploiter-iters` against the frozen main only,
     with goals as reward. If it out-scores main it is frozen into main's opponent pool
     (max --max-exploiters, oldest dropped).
  3. main is saved to <out-dir>/main_latest.pt and exported to web/policy_n{n}_medium.json.

Observations: rl.build_obs v2 (aiming geometry) stacked over the last `--stack` decisions (short memory).
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C
from .rl import OBS_VERSION, Policy, build_obs, load_human_data, load_policy, mirror_action, obs_dim, shot_on_target, state_to_posvel
from .vecsim import VecSim


def save_policy(policy: Policy, path: str, **extra):
    torch.save({"state_dict": policy.state_dict(), "obs_dim": policy.obs_dim, "hidden": policy.hidden, **extra}, path)


class FrameStack:
    """Keeps the last K base observations per (env, player) and returns their concatenation."""

    def __init__(self, E: int, P: int, od: int, K: int):
        self.K = K
        self.buf = np.zeros((E, P, K, od), np.float32)
        self.fresh = np.ones(E, bool)

    def reset(self, mask):
        self.fresh |= mask

    def push(self, obs: np.ndarray) -> np.ndarray:
        """obs (E,P,od) -> stacked (E,P,K*od). Envs flagged fresh get the frame repeated."""
        if self.K == 1:
            return obs
        self.buf = np.roll(self.buf, 1, axis=2)
        self.buf[:, :, 0] = obs
        if self.fresh.any():
            self.buf[self.fresh] = obs[self.fresh][:, :, None, :]
            self.fresh[:] = False
        E, P = obs.shape[:2]
        return self.buf.reshape(E, P, -1)


class PPOTrainer:
    """One policy learning by PPO in its own VecSim. Blue in the second half of the envs is driven
    by `opponents` (frozen state dicts with sampling weights); blue in the first half mirrors the
    learner (plain self-play). Red always = learner."""

    def __init__(self, n: int, envs: int, dev: str, a: argparse.Namespace, policy: Policy | None = None, seed: int = 0, shaping: bool = True):
        self.n, self.E, self.dev, self.a = n, envs, dev, a
        self.P = 2 * n
        self.V = a.obs_version
        self.K = a.stack
        self.od_base = obs_dim(n, self.V)
        self.od = self.od_base * self.K
        self.policy = policy or Policy(self.od, a.hidden).to(dev)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=a.lr, eps=1e-5)
        self.opp_policy = Policy(self.od, a.hidden).to(dev).eval()
        self.opponents: list[dict] = []
        self.opp_weights = np.ones(0)
        self.vs = VecSim(envs, n, n, seed)
        self.vs.reset(p_random=a.p_random)
        self.stack = FrameStack(envs, self.P, self.od_base, self.K)
        self.half = envs // 2
        self.store_mask = np.zeros((envs, self.P), bool)
        self.store_mask[:, :n] = True
        self.store_mask[: self.half, n:] = True
        self.N = int(self.store_mask.sum())
        self.team_of = np.where(np.arange(self.P) < n, 0, 1)
        self.sign = np.where(self.team_of == 0, 1.0, -1.0)
        self.att_dir = self.sign
        self.shaping = shaping
        self.in_front_prev = np.zeros((envs, self.P, self.P), bool)
        self.beat_done = np.zeros((envs, self.P, self.P), bool)
        self.wb_kicker = np.full(envs, -1, np.int16)
        self.wb_tick = np.zeros(envs, np.int64)
        self.wb_x = np.zeros(envs)
        self.wb_bounced = np.zeros(envs, bool)
        self.human = None
        self.obs = self.stack.push(self._obs_base())
        self.iters = 0
        self.ep_goals = np.zeros(2)
        T = a.steps
        self.buf_obs = torch.zeros((T, self.N, self.od), device=dev)
        self.buf_act = torch.zeros((T, self.N), dtype=torch.long, device=dev)
        self.buf_logp = torch.zeros((T, self.N), device=dev)
        self.buf_val = torch.zeros((T, self.N), device=dev)
        self.buf_rew = torch.zeros((T, self.N), device=dev)
        self.buf_done = torch.zeros((T, self.N), device=dev)

    # ------------------------------------------------------------------ helpers
    def _obs_base(self):
        return build_obs(self.vs.pos, self.vs.vel, self.vs.kick_armed, self.n, self.n, self.V)

    def set_opponents(self, state_dicts: list[dict], weights=None):
        self.opponents = state_dicts
        w = np.ones(len(state_dicts)) if weights is None else np.asarray(weights, float)
        self.opp_weights = w / w.sum() if len(w) else w

    def set_human(self, paths: list[str], coef: float):
        data = load_human_data(paths, self.n, self.dev, self.V) if paths else None
        if data is None:
            self.human = None
            return 0
        X, Y = data
        if self.K > 1:  # stack consecutive human frames (logs are per tick; decisions every `repeat` ticks)
            R = self.a.repeat
            idx = np.arange(len(X))
            frames = [X[np.clip(idx - k * R, 0, len(X) - 1)] for k in range(self.K)]
            X = torch.cat(frames, -1)
        self.human = (X, Y, coef)
        return len(X)

    def _potential(self):
        vs = self.vs
        bx = vs.pos[:, 0, 0]
        phi_team = 0.001 * bx[:, None] * self.sign[None]
        d = np.linalg.norm(vs.pos[:, 1:] - vs.pos[:, :1], axis=-1)
        return phi_team - 0.0005 * d

    def _carrier_mask(self):
        vs, E, P = self.vs, self.E, self.P
        d = np.linalg.norm(vs.pos[:, 1:] - vs.pos[:, :1], axis=-1)
        nearest = d.argmin(1)
        has = np.zeros((E, P), bool)
        has[np.arange(E), nearest] = d[np.arange(E), nearest] < 30
        recent = (vs.last_kicker >= 0) & (vs.tick - vs.last_kick_tick <= 120)
        has[recent, vs.last_kicker[recent]] = True
        return has

    def _in_front(self):
        x = self.vs.pos[:, 1:, 0]
        rel = (x[:, None, :] - x[:, :, None]) * self.att_dir[None, :, None]
        opp = self.team_of[None, :, None] != self.team_of[None, None, :]
        return (rel > 0) & opp

    def _shaping_subtick(self, rew: np.ndarray, prev_k, prev_t):
        """Event rewards computed after each sim tick (pass / beat / wall-bounce / shot)."""
        a, vs, P = self.a, self.vs, self.P
        if self.n > 1 and a.pass_bonus > 0:
            for p_ in range(P):
                k = vs.kicked[:, p_] & (prev_k >= 0) & (prev_k != p_) & (self.team_of[np.maximum(prev_k, 0)] == self.team_of[p_]) & (vs.tick - prev_t <= 180)
                if k.any():
                    rew[k, p_] += a.pass_bonus
                    rew[k, prev_k[k]] += a.pass_bonus
        if a.shot_bonus > 0:
            for p_ in range(P):
                k = vs.kicked[:, p_]
                if k.any():
                    idx = np.nonzero(k)[0]
                    on = shot_on_target(vs.pos[idx, 0, 0], vs.pos[idx, 0, 1], vs.vel[idx, 0, 0], vs.vel[idx, 0, 1], self.att_dir[p_])
                    rew[idx[on], p_] += a.shot_bonus
        if a.wall_bonus > 0:
            for p_ in range(P):
                k = vs.kicked[:, p_]
                if k.any():
                    self.wb_kicker[k] = p_
                    self.wb_tick[k] = vs.tick[k]
                    self.wb_x[k] = vs.pos[k, p_ + 1, 0] * self.att_dir[p_]
                    self.wb_bounced[k] = False
            live = (self.wb_kicker >= 0) & (vs.tick - self.wb_tick <= 150)
            self.wb_bounced |= live & vs.ball_wall_hit
            d_pb = np.linalg.norm(vs.pos[:, 1:] - vs.pos[:, :1], axis=-1)
            for p_ in range(P):
                got = live & self.wb_bounced & (self.wb_kicker == p_) & (d_pb[:, p_] < 30) & (vs.tick - self.wb_tick >= 10)
                if got.any():
                    prog = vs.pos[got, p_ + 1, 0] * self.att_dir[p_] - self.wb_x[got]
                    idx = np.nonzero(got)[0]
                    rew[idx[prog >= 40.0], p_] += a.wall_bonus
                    self.wb_kicker[idx] = -1
            self.wb_kicker[~live] = -1
        if a.beat_bonus > 0:
            carry = self._carrier_mask()
            front_now = self._in_front()
            beaten = self.in_front_prev & ~front_now & carry[:, :, None] & ~self.beat_done
            rew += beaten.sum(2) * a.beat_bonus
            self.beat_done |= beaten
            self.beat_done[~carry] = False
            self.in_front_prev = front_now

    # ------------------------------------------------------------------ one PPO iteration
    def iterate(self, shape_scale: float = 1.0) -> dict:
        a, vs, E, P, n, dev = self.a, self.vs, self.E, self.P, self.n, self.dev
        T, R = a.steps, a.repeat
        half, od = self.half, self.od
        max_ticks = a.max_seconds * C.TICK_HZ
        if len(self.opponents):
            self.opp_policy.load_state_dict(self.opponents[np.random.default_rng().choice(len(self.opponents), p=self.opp_weights)])
            use_opp = True
        else:
            use_opp = False
        self.policy.eval()
        m = torch.as_tensor(self.store_mask, device=dev)
        for t in range(T):
            obs_t = torch.as_tensor(self.obs, device=dev)
            a_cur, logp_cur, v_cur = self.policy.act(obs_t.reshape(E * P, od))
            a_cur = a_cur.reshape(E, P)
            actions = a_cur.cpu().numpy()
            if use_opp:
                a_opp, _, _ = self.opp_policy.act(obs_t[half:, n:].reshape(-1, od))
                actions[half:, n:] = a_opp.cpu().numpy().reshape(-1, n)
            sim_actions = actions.copy()
            sim_actions[:, n:] = mirror_action(actions[:, n:])
            phi0 = self._potential()
            goal = np.full(E, -1, np.int8)
            rew = np.zeros((E, P))
            for _ in range(R):
                prev_k, prev_t = vs.last_kicker.copy(), vs.last_kick_tick.copy()
                g = vs.step(sim_actions)
                newg = (goal < 0) & (g >= 0)
                goal[newg] = g[newg]
                if self.shaping and shape_scale > 0:
                    self._shaping_subtick(rew, prev_k, prev_t)
            scored = goal >= 0
            timeout = (~scored) & (vs.tick >= max_ticks)
            if self.shaping and shape_scale > 0:
                rew += self._potential() - phi0
                rew *= shape_scale
            team_goal = np.where(goal[:, None] == 0, 1.0, np.where(goal[:, None] == 1, -1.0, 0.0)) * self.sign[None]
            rew += team_goal
            if timeout.any():
                obs_to = self.stack.buf[timeout].reshape(int(timeout.sum()), P, -1) if self.K > 1 else self._obs_base()[timeout]
                # value of the state we are truncating at: use the current stacked obs for those envs
                with torch.no_grad():
                    _, v_to = self.policy(torch.as_tensor(self.obs[timeout], device=dev).reshape(-1, od))
                rew[timeout] += a.gamma * v_to.reshape(-1, P).cpu().numpy()
            done = scored | timeout
            if scored[half:].any():
                self.ep_goals[0] += (goal[half:] == 0).sum()
                self.ep_goals[1] += (goal[half:] == 1).sum()
            ko = np.where(goal >= 0, 1 - goal, 0).astype(np.int8)
            vs.reset(done, p_random=a.p_random, kickoff_team=ko)
            self.in_front_prev[done] = False
            self.beat_done[done] = False
            self.wb_kicker[done] = -1
            self.stack.reset(done)
            self.buf_obs[t] = obs_t[m]
            self.buf_act[t] = a_cur[m]
            self.buf_logp[t] = logp_cur.reshape(E, P)[m]
            self.buf_val[t] = v_cur.reshape(E, P)[m]
            self.buf_rew[t] = torch.as_tensor(rew, device=dev, dtype=torch.float32)[m]
            self.buf_done[t] = torch.as_tensor(np.repeat(done[:, None], P, 1), device=dev, dtype=torch.float32)[m]
            self.obs = self.stack.push(self._obs_base())
        # GAE
        with torch.no_grad():
            _, last_v = self.policy(torch.as_tensor(self.obs, device=dev).reshape(E * P, od))
            last_v = last_v.reshape(E, P)[m]
            adv = torch.zeros_like(self.buf_rew)
            gae = torch.zeros(self.N, device=dev)
            for t in reversed(range(T)):
                nv = last_v if t == T - 1 else self.buf_val[t + 1]
                nonterm = 1.0 - self.buf_done[t]
                delta = self.buf_rew[t] + a.gamma * nv * nonterm - self.buf_val[t]
                gae = delta + a.gamma * a.lam * nonterm * gae
                adv[t] = gae
            ret = adv + self.buf_val
        # PPO update
        self.policy.train()
        b_obs = self.buf_obs.reshape(-1, od)
        b_act, b_logp, b_adv, b_ret = self.buf_act.reshape(-1), self.buf_logp.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        tot = len(b_obs)
        stats = []
        for _ in range(a.epochs):
            perm = torch.randperm(tot, device=dev)
            for i in range(0, tot, a.minibatch):
                idx = perm[i : i + a.minibatch]
                logits, v = self.policy(b_obs[idx])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(b_act[idx])
                ratio = torch.exp(logp - b_logp[idx])
                pg = -torch.min(ratio * b_adv[idx], torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * b_adv[idx]).mean()
                vl = F.mse_loss(v, b_ret[idx])
                ent = dist.entropy().mean()
                loss = pg + 0.5 * vl - a.ent * ent
                if self.human is not None:
                    hx, hy, coef = self.human
                    hi = torch.randint(len(hx), (min(4096, len(hx)),), device=dev)
                    h_logits, _ = self.policy(hx[hi])
                    loss = loss + coef * F.cross_entropy(h_logits, hy[hi])
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.opt.step()
                stats.append((pg.item(), vl.item(), ent.item()))
        self.iters += 1
        pg, vl, ent = np.mean(stats, 0)
        return dict(pg=pg, vl=vl, ent=ent)

    def pop_goal_rate(self, iters: int):
        """(red, blue) goals/min in the opponent half over the last `iters` iterations; resets."""
        mins = iters * self.a.steps * self.a.repeat * self.half / (C.TICK_HZ * 60)
        r = self.ep_goals / max(mins, 1e-9)
        self.ep_goals[:] = 0
        return float(r[0]), float(r[1])


# --------------------------------------------------------------------------- evaluation helpers
class StackedActor:
    """Drives players of a scalar sim.Sim with a (possibly frame-stacked) policy."""

    def __init__(self, policy: Policy, n: int, K: int, V: int, repeat: int, dev: str, deterministic=False):
        self.policy, self.n, self.K, self.V, self.R, self.dev, self.det = policy, n, K, V, repeat, dev, deterministic
        self.hist = None
        self.t = 0
        self.last = None

    def reset(self):
        self.hist, self.t, self.last = None, 0, None

    def act(self, state, kick_armed, players):
        if self.last is not None and self.t % self.R != 0:
            self.t += 1
            return {p: self.last[p] for p in players}
        pos, vel = state_to_posvel(state, self.n, self.n)
        ob = build_obs(pos, vel, np.array(kick_armed)[None], self.n, self.n, self.V)[0]  # (P, od)
        if self.K > 1:
            if self.hist is None:
                self.hist = np.repeat(ob[:, None, :], self.K, 1)
            else:
                self.hist = np.roll(self.hist, 1, 1)
                self.hist[:, 0] = ob
            ob = self.hist.reshape(ob.shape[0], -1)
        a, _, _ = self.policy.act(torch.as_tensor(ob, device=self.dev), self.det)
        a = a.cpu().numpy()
        self.last = {p: (int(a[p]) if p < self.n else int(mirror_action(a[p : p + 1])[0])) for p in range(2 * self.n)}
        self.t = 1
        return {p: self.last[p] for p in players}


def head_to_head(red: StackedActor, blue: StackedActor, seconds: int, seed: int = 0, n: int = 1):
    """red vs blue in the scalar sim; returns (red goals, blue goals)."""
    from .sim import Sim

    sim = Sim(seed, n_red=n, n_blue=n)
    sim.reset()
    red.reset(), blue.reset()
    gr = gb = 0
    reds, blues = list(range(n)), list(range(n, 2 * n))
    for t in range(seconds * C.TICK_HZ):
        s = sim.get_state()
        armed = [p.kick_armed for p in sim.players]
        ar = red.act(s, armed, reds)
        ab = blue.act(s, armed, blues)
        g = sim.step([ar[i] for i in reds] + [ab[i] for i in blues])
        if g is not None:
            gr += g == 0
            gb += g == 1
            sim.reset(kickoff_team=1 - g)
            red.reset(), blue.reset()
    return gr, gb


def vs_scripted(actor: StackedActor, seconds: int = 180, seed: int = 0):
    from .policies import OpponentPolicy
    from .sim import Sim

    rng = np.random.default_rng(seed)
    sim = Sim(seed)
    opp = OpponentPolicy(rng)
    actor.reset()
    gf = ga = 0
    for t in range(seconds * C.TICK_HZ):
        s = sim.get_state()
        a_red = actor.act(s, [p.kick_armed for p in sim.players], [0])[0]
        a_blue = opp.act(s, sim.players[1].kick_armed)
        g = sim.step(a_red, a_blue)
        if g is not None:
            gf += g == 0
            ga += g == 1
            sim.reset(kickoff_team=1 - g)
            actor.reset()
            opp.new_personality()
    return gf, ga


# --------------------------------------------------------------------------- league loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--stack", type=int, default=4, help="frames of short memory")
    ap.add_argument("--obs-version", type=int, default=OBS_VERSION)
    ap.add_argument("--max-seconds", type=int, default=60)
    ap.add_argument("--p-random", type=float, default=0.3)
    ap.add_argument("--pass-bonus", type=float, default=0.1)
    ap.add_argument("--shot-bonus", type=float, default=0.15)
    ap.add_argument("--wall-bonus", type=float, default=0.1)
    ap.add_argument("--beat-bonus", type=float, default=0.02)
    ap.add_argument("--shaping-decay-iters", type=int, default=3000, help="main's shaping scale goes 1 -> 0 linearly over this many main iterations")
    ap.add_argument("--main-iters", type=int, default=100, help="main PPO iterations per cycle")
    ap.add_argument("--exploiter-iters", type=int, default=60, help="exploiter PPO iterations per cycle")
    ap.add_argument("--snapshot-every", type=int, default=25)
    ap.add_argument("--pool-size", type=int, default=20)
    ap.add_argument("--max-exploiters", type=int, default=8)
    ap.add_argument("--human", nargs="*", default=["data/play/*.npz", "data/web/*.npz"], help="human logs: imitation loss for main (+ a BC policy in the pool)")
    ap.add_argument("--human-coef", type=float, default=0.01)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--out-dir", default="runs/league")
    ap.add_argument("--resume-dir", default=None)
    ap.add_argument("--export", default=None, help="JSON path for the main policy (default web/policy_n{n}_medium.json)")
    ap.add_argument("--seed", type=int, default=0)
    C.add_stadium_arg(ap)
    a = ap.parse_args()
    C.handle_stadium_arg(a)
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = a.n
    out_dir = a.resume_dir or a.out_dir
    os.makedirs(out_dir, exist_ok=True)
    export_path = a.export or f"web/policy_n{n}_medium.json"

    main_tr = PPOTrainer(n, a.envs, dev, a, seed=a.seed)
    expl_tr = PPOTrainer(n, a.envs, dev, a, seed=a.seed + 1, shaping=True)
    state = dict(cycle=0, main_iters=0, exploiters=[], snapshots=[])
    if a.resume_dir:
        ck = torch.load(os.path.join(a.resume_dir, "main_latest.pt"), map_location=dev)
        main_tr.policy.load_state_dict(ck["state_dict"])
        with open(os.path.join(a.resume_dir, "state.json")) as fh:
            state = json.load(fh)
        print(f"resumed cycle {state['cycle']}, main iters {state['main_iters']}, exploiters {len(state['exploiters'])}")
    snapshots: list[dict] = [copy.deepcopy(main_tr.policy.state_dict())]
    for p in state["snapshots"]:
        snapshots.append(torch.load(p, map_location=dev)["state_dict"])
    exploiters: list[dict] = [torch.load(p, map_location=dev)["state_dict"] for p in state["exploiters"]]
    human_paths = sorted(sum((glob.glob(p) for p in a.human), [])) if a.human else []
    n_h = main_tr.set_human(human_paths, a.human_coef)
    human_bc = None
    if n_h:
        print(f"human logs: {n_h:,} samples ({len(human_paths)} files)")
        # a quick behaviour-cloned human policy as a permanent pool member
        human_bc = Policy(main_tr.od, a.hidden).to(dev)
        hx, hy, _ = main_tr.human
        opt = torch.optim.Adam(human_bc.parameters(), lr=1e-3)
        for ep in range(15):
            perm = torch.randperm(len(hx), device=dev)
            for i in range(0, len(hx), 4096):
                idx = perm[i : i + 4096]
                loss = F.cross_entropy(human_bc(hx[idx])[0], hy[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        print(f"human BC loss {loss.item():.3f}")
        human_bc = copy.deepcopy(human_bc.state_dict())

    def main_opponents():
        sd = list(snapshots) + list(exploiters) + ([human_bc] if human_bc is not None else [])
        w = list(np.arange(1, len(snapshots) + 1, dtype=float)) + [float(len(snapshots))] * len(exploiters) + ([0.5 * len(snapshots)] if human_bc is not None else [])
        return sd, w

    t0 = time.time()
    log = open(os.path.join(out_dir, "league.log"), "a")

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')} +{(time.time()-t0)/60:6.1f}m] {msg}"
        print(line, flush=True)
        log.write(line + "\n"); log.flush()

    say(f"league start: n={n} envs={a.envs} stack={a.stack} obs v{a.obs_version} dev={dev} hours={a.hours}")
    while (time.time() - t0) / 3600 < a.hours:
        state["cycle"] += 1
        # ---- 1. main vs pool ----
        sd, w = main_opponents()
        main_tr.set_opponents(sd, w)
        for k in range(a.main_iters):
            scale = max(0.0, 1.0 - state["main_iters"] / max(1, a.shaping_decay_iters))
            st = main_tr.iterate(shape_scale=scale)
            state["main_iters"] += 1
            if state["main_iters"] % a.snapshot_every == 0:
                snapshots.append(copy.deepcopy(main_tr.policy.state_dict()))
                snapshots[:] = snapshots[-a.pool_size :]
                sd, w = main_opponents()
                main_tr.set_opponents(sd, w)
        r_red, r_blue = main_tr.pop_goal_rate(a.main_iters)
        say(f"cycle {state['cycle']} main: iters {state['main_iters']} shaping x{scale:.2f} pg {st['pg']:+.4f} v {st['vl']:.4f} ent {st['ent']:.3f} | vs pool {r_red:.2f}-{r_blue:.2f} goals/min | pool {len(snapshots)} snaps + {len(exploiters)} exploiters")
        # ---- 2. exploiter vs frozen main ----
        frozen = copy.deepcopy(main_tr.policy.state_dict())
        expl_tr.policy.load_state_dict(frozen)  # start from main (knows how to play), then specialise
        expl_tr.opt = torch.optim.Adam(expl_tr.policy.parameters(), lr=a.lr, eps=1e-5)
        expl_tr.set_opponents([frozen], [1.0])
        expl_tr.pop_goal_rate(1)
        for k in range(a.exploiter_iters):
            expl_tr.iterate(shape_scale=0.3)
        e_red, e_blue = expl_tr.pop_goal_rate(a.exploiter_iters)
        # verify head-to-head in the scalar sim (exploiter red vs main blue, 3 minutes)
        ex_actor = StackedActor(expl_tr.policy, n, a.stack, a.obs_version, a.repeat, dev)
        main_actor = StackedActor(main_tr.policy, n, a.stack, a.obs_version, a.repeat, dev)
        h_red, h_blue = head_to_head(ex_actor, main_actor, seconds=180, seed=state["cycle"], n=n)
        accepted = h_red > h_blue
        if accepted:
            path = os.path.join(out_dir, f"exploiter_{state['cycle']:03d}.pt")
            save_policy(expl_tr.policy, path, n=n, repeat=a.repeat, obs_version=a.obs_version, stack=a.stack, stadium=C.STADIUM_NAME)
            exploiters.append(copy.deepcopy(expl_tr.policy.state_dict()))
            state["exploiters"].append(path)
            exploiters[:] = exploiters[-a.max_exploiters :]
            state["exploiters"] = state["exploiters"][-a.max_exploiters :]
        say(f"cycle {state['cycle']} exploiter: train {e_red:.2f}-{e_blue:.2f} g/min | h2h 3min {h_red}-{h_blue} -> {'ACCEPTED' if accepted else 'rejected'}")
        # ---- 3. save / export / eval ----
        save_policy(main_tr.policy, os.path.join(out_dir, "main_latest.pt"), n=n, repeat=a.repeat, iters=state["main_iters"], obs_version=a.obs_version, stack=a.stack, stadium=C.STADIUM_NAME)
        snap_paths = []
        for i, sdict in enumerate(snapshots[-5:]):
            p = os.path.join(out_dir, f"snap_{i}.pt")
            torch.save({"state_dict": sdict}, p)
            snap_paths.append(p)
        state["snapshots"] = snap_paths
        with open(os.path.join(out_dir, "state.json"), "w") as fh:
            json.dump(state, fh)
        from .export import export_policy

        export_policy(os.path.join(out_dir, "main_latest.pt"), export_path)
        if n == 1 and state["cycle"] % 2 == 1:
            gf, ga = vs_scripted(main_actor, seconds=120, seed=state["cycle"])
            hb = ""
            if human_bc is not None:
                hp = Policy(main_tr.od, a.hidden).to(dev).eval(); hp.load_state_dict(human_bc)
                hr, hbb = head_to_head(main_actor, StackedActor(hp, n, a.stack, a.obs_version, a.repeat, dev), seconds=120, seed=7, n=n)
                hb = f" | vs human-BC 2min {hr}-{hbb}"
            say(f"cycle {state['cycle']} eval: vs scripted 2min {gf}-{ga}{hb}")
    say("league finished")


if __name__ == "__main__":
    main()
