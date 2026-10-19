"""Vectorised NvN simulator (numpy, E environments at once) with physics identical to sim.Sim.

Disc order per env: [ball, red_0..red_{n-1}, blue_0..blue_{m-1}]. Used for RL self-play where
we need ~1M ticks/s. Validated against sim.Sim to float precision (see tests in rl.py --selftest).
"""
from __future__ import annotations

import math

import numpy as np

from . import config as C
from .sim import build_geometry, kickoff_barrier


class VecSim:
    def __init__(self, n_envs: int, n_red: int = 1, n_blue: int = 1, seed: int = 0):
        self.E, self.n_red, self.n_blue = n_envs, n_red, n_blue
        self.P = n_red + n_blue
        self.D = self.P + 1
        self.rng = np.random.default_rng(seed)
        B, Pl = C.BALL, C.PLAYER
        self.radius = np.array([B["radius"]] + [Pl["radius"]] * self.P)
        self.inv_mass = np.array([B["inv_mass"]] + [Pl["inv_mass"]] * self.P)
        self.bcoef = np.array([B["bcoef"]] + [Pl["bcoef"]] * self.P)
        self.pos = np.zeros((self.E, self.D, 2))
        self.vel = np.zeros((self.E, self.D, 2))
        self.damping = np.zeros((self.E, self.D))
        self.damping[:, 0] = B["damping"]
        self.damping[:, 1:] = Pl["damping"]
        self.kick_armed = np.ones((self.E, self.P), bool)
        self.kicked = np.zeros((self.E, self.P), bool)
        self.team = np.array([0] * n_red + [1] * n_blue)
        planes, segments, posts = build_geometry()
        self.planes = planes
        self.seg_p0 = np.array([s[0] for s in segments])
        self.seg_p1 = np.array([s[1] for s in segments])
        self.seg_bc = np.array([s[2] for s in segments])
        self.post_pos = np.array([[d.x, d.y] for d in posts])
        self.post_r, self.post_bc = posts[0].radius, posts[0].bcoef
        self.tick = np.zeros(self.E, np.int64)
        self.kickoff = np.full(self.E, -1, np.int8)  # -1 none, 0 red kicks off, 1 blue kicks off
        self.last_kicker = np.full(self.E, -1, np.int16)  # player index of the last kick, -1 none
        self.ball_wall_hit = np.zeros(self.E, bool)  # ball bounced off a sideline / goal line this tick
        self.last_kick_tick = np.zeros(self.E, np.int64)
        ko = {t: kickoff_barrier(t) for t in (0, 1)}
        self.ko_p0 = {t: np.array([g[0] for g in ko[t]]) for t in (0, 1)}
        self.ko_p1 = {t: np.array([g[1] for g in ko[t]]) for t in (0, 1)}
        self.ko_bc = ko[0][0][2]
        self.reset()

    # ------------------------------------------------------------------ state
    def spawn_positions(self):
        out = []
        for team, n in ((0, self.n_red), (1, self.n_blue)):
            x = -C.SPAWN_DISTANCE if team == 0 else C.SPAWN_DISTANCE
            out += [(x, (i - (n - 1) / 2) * 55.0) for i in range(n)]
        return np.array(out)

    def reset(self, mask=None, p_random: float = 0.0, kickoff_team=0):
        """Reset envs where mask is True (all if None) into a kick-off for `kickoff_team`
        (int, or per-env array aligned with the full E axis). A fraction p_random get scattered
        starts instead (no kick-off barrier)."""
        idx = np.arange(self.E) if mask is None else np.nonzero(mask)[0]
        if len(idx) == 0:
            return
        n = len(idx)
        self.kickoff[idx] = np.asarray(kickoff_team, np.int8)[idx] if np.ndim(kickoff_team) else int(kickoff_team)
        self.vel[idx] = 0.0
        self.pos[idx, 0] = 0.0
        self.pos[idx, 1:] = self.spawn_positions()[None]
        self.kick_armed[idx] = True
        self.kicked[idx] = False
        self.damping[idx, 1:] = C.PLAYER["damping"]
        self.tick[idx] = 0
        self.last_kicker[idx] = -1
        if p_random > 0:
            rnd = idx[self.rng.random(n) < p_random]
            if len(rnd):
                self._randomize(rnd)
                self.kickoff[rnd] = -1

    def _randomize(self, idx):
        rng, m = self.rng, len(idx)
        self.pos[idx, 0, 0] = rng.uniform(-C.FIELD_HALF_W + 20, C.FIELD_HALF_W - 20, m)
        self.pos[idx, 0, 1] = rng.uniform(-C.FIELD_HALF_H + 20, C.FIELD_HALF_H - 20, m)
        sp = rng.uniform(0, 6, m) * (rng.random(m) < 0.6)
        ang = rng.uniform(0, 2 * math.pi, m)
        self.vel[idx, 0] = np.stack([sp * np.cos(ang), sp * np.sin(ang)], -1)
        for p in range(1, self.D):
            self.pos[idx, p, 0] = rng.uniform(-C.HALF_W + 30, C.HALF_W - 30, m)
            self.pos[idx, p, 1] = rng.uniform(-C.HALF_H + 30, C.HALF_H - 30, m)
            sp = rng.uniform(0, 2.5, m)
            ang = rng.uniform(0, 2 * math.pi, m)
            self.vel[idx, p] = np.stack([sp * np.cos(ang), sp * np.sin(ang)], -1)
        # push apart any overlapping discs (cheap rejection-free fix: a few separation passes)
        for _ in range(5):
            for i in range(self.D):
                for j in range(i + 1, self.D):
                    d = self.pos[idx, j] - self.pos[idx, i]
                    dist = np.linalg.norm(d, axis=-1)
                    bad = dist < self.radius[i] + self.radius[j] + 5
                    if bad.any():
                        self.pos[idx[bad], j] += np.array([40.0, 0.0])

    def get_state(self) -> np.ndarray:
        """(E, 4*D) in the sim.Sim layout [reds, ball, blues]."""
        order = list(range(1, 1 + self.n_red)) + [0] + list(range(1 + self.n_red, self.D))
        return np.concatenate([self.pos[:, order], self.vel[:, order]], -1).reshape(self.E, -1) if False else \
            np.stack([np.concatenate([self.pos[:, i], self.vel[:, i]], -1) for i in order], 1).reshape(self.E, -1)

    def set_state(self, s: np.ndarray, env: int = 0):
        order = list(range(1, 1 + self.n_red)) + [0] + list(range(1 + self.n_red, self.D))
        s = np.asarray(s, float).reshape(self.D, 4)
        for k, i in enumerate(order):
            self.pos[env, i] = s[k, :2]
            self.vel[env, i] = s[k, 2:]

    # ---------------------------------------------------------------- physics
    def step(self, actions: np.ndarray) -> np.ndarray:
        """actions: (E, P) ints. Returns goal (E,) int8: -1 none, 0 red scored, 1 blue scored."""
        Pl = C.PLAYER
        pos, vel = self.pos, self.vel
        actions = np.asarray(actions)
        self.kicked[:] = False
        self.ball_wall_hit[:] = False
        for p in range(self.P):
            a = actions[:, p]
            kick = a // 9
            m = a % 9
            dx = (m // 3 - 1).astype(float)
            dy = (m % 3 - 1).astype(float)
            self.kick_armed[kick == 0, p] = True
            kicking = (kick == 1) & self.kick_armed[:, p]
            n = np.hypot(dx, dy)
            mv = n > 0
            acc = np.where(kicking, Pl["kick_accel"], Pl["accel"])
            vel[mv, p + 1, 0] += dx[mv] / n[mv] * acc[mv]
            vel[mv, p + 1, 1] += dy[mv] / n[mv] * acc[mv]
            self.damping[:, p + 1] = np.where(kicking, Pl["kick_damping"], Pl["damping"])
            d = pos[:, 0] - pos[:, p + 1]
            dist = np.hypot(d[:, 0], d[:, 1])
            fire = kicking & (dist <= self.radius[p + 1] + self.radius[0] + C.KICK_RANGE_EXTRA) & (dist > 0)
            if fire.any():
                u = d[fire] / dist[fire, None]
                vel[fire, 0] += u * Pl["kick_strength"]
                self.kick_armed[fire, p] = False
                self.kicked[fire, p] = True
                self.last_kicker[fire] = p
                self.last_kick_tick[fire] = self.tick[fire]
        # disc-disc
        for i in range(self.D):
            for j in range(i + 1, self.D):
                self._collide(i, j)
        # posts (static discs)
        for i in range(self.D):
            for k in range(len(self.post_pos)):
                self._collide_static(i, self.post_pos[k], self.post_r, self.post_bc)
        # planes / segments
        for i in range(self.D):
            is_ball = i == 0
            for nx, ny, dist, bc, mask in self.planes:
                if mask == "all" or is_ball:
                    pen = self.radius[i] - (pos[:, i, 0] * nx + pos[:, i, 1] * ny - dist)
                    hit = pen > 0
                    if is_ball and mask == "ball":
                        self.ball_wall_hit |= hit
                    if hit.any():
                        pos[hit, i, 0] += nx * pen[hit]
                        pos[hit, i, 1] += ny * pen[hit]
                        vn = vel[hit, i, 0] * nx + vel[hit, i, 1] * ny
                        k = np.where(vn < 0, vn * (1.0 + bc * self.bcoef[i]), 0.0)
                        vel[hit, i, 0] -= nx * k
                        vel[hit, i, 1] -= ny * k
            if is_ball:
                for s in range(len(self.seg_p0)):
                    h = self._collide_segment(i, self.seg_p0[s], self.seg_p1[s], self.seg_bc[s])
                    if s < 4 and h is not None:
                        self.ball_wall_hit |= h
            else:
                for t in (0, 1):
                    m = self.kickoff == t
                    if m.any():
                        for s in range(len(self.ko_p0[t])):
                            self._collide_segment(i, self.ko_p0[t][s], self.ko_p1[t][s], self.ko_bc, m)
        # integrate (ball: swept test against segments when fast enough to tunnel)
        bv = vel[:, 0]
        fast = (bv * bv).sum(-1) > self.radius[0] ** 2
        handled = np.zeros(self.E, bool)
        if fast.any():
            for s in range(len(self.seg_p0)):
                todo = fast & ~handled
                if not todo.any():
                    break
                handled |= self._sweep_segment(todo, self.seg_p0[s], self.seg_p1[s], self.seg_bc[s])
        pos[:, 1:] += vel[:, 1:]
        pos[~handled, 0] += vel[~handled, 0]
        vel *= self.damping[..., None]
        touched = (vel[:, 0, 0] != 0.0) | (vel[:, 0, 1] != 0.0)
        self.kickoff[(self.kickoff >= 0) & touched] = -1
        self.tick += 1
        bx, by = pos[:, 0, 0], pos[:, 0, 1]
        goal = np.full(self.E, -1, np.int8)
        inmouth = np.abs(by) < C.GOAL_HALF
        goal[inmouth & (bx > C.GOAL_X)] = 0
        goal[inmouth & (bx < -C.GOAL_X)] = 1
        return goal

    def _collide(self, i, j):
        pos, vel = self.pos, self.vel
        d = pos[:, j] - pos[:, i]
        d2 = (d * d).sum(-1)
        rsum = self.radius[i] + self.radius[j]
        hit = (d2 < rsum * rsum) & (d2 > 0)
        if not hit.any():
            return
        dist = np.sqrt(d2[hit])
        n = d[hit] / dist[:, None]
        imi, imj = self.inv_mass[i], self.inv_mass[j]
        tot = imi + imj
        pen = rsum - dist
        pos[hit, i] -= n * (pen * (imi / tot))[:, None]
        pos[hit, j] += n * (pen * (imj / tot))[:, None]
        vrel = ((vel[hit, i] - vel[hit, j]) * n).sum(-1)
        jimp = np.where(vrel > 0, vrel * (1.0 + self.bcoef[i] * self.bcoef[j]) / tot, 0.0)
        vel[hit, i] -= n * (jimp * imi)[:, None]
        vel[hit, j] += n * (jimp * imj)[:, None]

    def _collide_static(self, i, spos, sr, sbc):
        pos, vel = self.pos, self.vel
        d = spos[None] - pos[:, i]
        d2 = (d * d).sum(-1)
        rsum = self.radius[i] + sr
        hit = (d2 < rsum * rsum) & (d2 > 0)
        if not hit.any():
            return
        dist = np.sqrt(d2[hit])
        n = d[hit] / dist[:, None]
        pen = rsum - dist
        pos[hit, i] -= n * pen[:, None]
        vrel = (vel[hit, i] * n).sum(-1)
        jimp = np.where(vrel > 0, vrel * (1.0 + self.bcoef[i] * sbc), 0.0)
        vel[hit, i] -= n * jimp[:, None]

    def _sweep_segment(self, mask, p0, p1, bc):
        """Vectorised twin of sim._sweep_segment for the ball (disc 0) on envs in `mask`.
        Returns the env mask that was handled."""
        pos, vel = self.pos, self.vel
        ax, ay = pos[:, 0, 0], pos[:, 0, 1]
        dx, dy = vel[:, 0, 0], vel[:, 0, 1]
        (x0, y0), (x1, y1) = p0, p1
        ex, ey = x1 - x0, y1 - y0
        denom = dx * ey - dy * ex
        ok = mask & (np.abs(denom) >= 1e-12)
        safe = np.where(ok, denom, 1.0)
        t = ((x0 - ax) * ey - (y0 - ay) * ex) / safe
        u = ((x0 - ax) * dy - (y0 - ay) * dx) / safe
        hit = ok & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
        if not hit.any():
            return hit
        L = math.hypot(ex, ey)
        nx, ny = ey / L, -ex / L
        side = (ax[hit] - x0) * nx + (ay[hit] - y0) * ny
        sgn = np.where(side < 0.0, -1.0, 1.0)
        nxs, nys = nx * sgn, ny * sgn
        cx, cy = ax[hit] + dx[hit] * t[hit], ay[hit] + dy[hit] * t[hit]
        r = self.radius[0]
        pos[hit, 0, 0] = cx + nxs * r
        pos[hit, 0, 1] = cy + nys * r
        vn = vel[hit, 0, 0] * nxs + vel[hit, 0, 1] * nys
        k = np.where(vn < 0.0, vn * (1.0 + bc * self.bcoef[0]), 0.0)
        vel[hit, 0, 0] -= nxs * k
        vel[hit, 0, 1] -= nys * k
        return hit

    def _collide_segment(self, i, p0, p1, bc, env_mask=None):
        pos, vel = self.pos, self.vel
        e = p1 - p0
        L2 = float(e @ e)
        t = ((pos[:, i] - p0) @ e) / L2
        t = np.clip(t, 0.0, 1.0)
        c = p0[None] + t[:, None] * e[None]
        d = pos[:, i] - c
        d2 = (d * d).sum(-1)
        r = self.radius[i]
        hit = (d2 < r * r) & (d2 > 0)
        if env_mask is not None:
            hit &= env_mask
        if not hit.any():
            return hit
        dist = np.sqrt(d2[hit])
        n = d[hit] / dist[:, None]
        pen = r - dist
        pos[hit, i] += n * pen[:, None]
        vn = (vel[hit, i] * n).sum(-1)
        k = np.where(vn < 0, vn * (1.0 + bc * self.bcoef[i]), 0.0)
        vel[hit, i] -= n * k[:, None]
        return hit


def selftest(ticks: int = 3000, n: int = 2, seed: int = 0):
    """Drive sim.Sim and VecSim with identical random actions and compare trajectories."""
    from .sim import Sim

    rng = np.random.default_rng(seed)
    sim = Sim(seed, n_red=n, n_blue=n)
    sim.reset(randomize=False, kickoff_team=0)
    vs = VecSim(1, n, n, seed)
    vs.kickoff[:] = 0
    # copy exact float64 state
    discs = [p.disc for p in sim.players[:n]] + [sim.ball] + [p.disc for p in sim.players[n:]]
    vs.set_state(np.array([[d.x, d.y, d.vx, d.vy] for d in discs]).reshape(-1))
    worst = 0.0
    for t in range(ticks):
        a = rng.integers(0, 18, 2 * n)
        g1 = sim.step(*a)
        g2 = vs.step(a[None])[0]
        s1 = np.array([[d.x, d.y, d.vx, d.vy] for d in discs]).reshape(-1)
        s2 = vs.get_state()[0]
        worst = max(worst, float(np.abs(s1 - s2).max()))
        assert (g1 is None and g2 == -1) or g1 == g2, f"goal mismatch at tick {t}: {g1} vs {g2}"
        assert (sim.kickoff_team if sim.kickoff_team is not None else -1) == vs.kickoff[0], f"kickoff state mismatch at tick {t}"
        if g1 is not None or t % 700 == 699:
            ko = int(rng.integers(0, 2))
            sim.reset(randomize=(t % 1400 == 1399), kickoff_team=ko)
            vs.set_state(np.array([[d.x, d.y, d.vx, d.vy] for d in discs]).reshape(-1))
            vs.kick_armed[:] = [p.kick_armed for p in sim.players]
            vs.kickoff[:] = -1 if sim.kickoff_team is None else sim.kickoff_team
    return worst


if __name__ == "__main__":
    import time

    print("max |Sim - VecSim| over 3000 random ticks (2v2):", selftest())
    vs = VecSim(2048, 1, 1)
    acts = np.random.default_rng(0).integers(0, 18, (200, 2048, 2))
    t0 = time.time()
    for k in range(200):
        g = vs.step(acts[k])
        vs.reset(g >= 0)
    dt = time.time() - t0
    print(f"VecSim 1v1 x2048 envs: {200*2048/dt:,.0f} ticks/s")
    vs = VecSim(1024, 3, 3)
    acts = np.random.default_rng(0).integers(0, 18, (100, 1024, 6))
    t0 = time.time()
    for k in range(100):
        g = vs.step(acts[k])
        vs.reset(g >= 0)
    print(f"VecSim 3v3 x1024 envs: {100*1024/(time.time()-t0):,.0f} ticks/s")
