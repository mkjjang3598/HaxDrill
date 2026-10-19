"""Deterministic HaxBall-Classic physics for 1 red player, 1 blue player and a ball.

This is the *ground truth* simulator: it generates training data for the world model
and can be run side-by-side with the model (ghost mode) to measure drift.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import config as C


@dataclass
class Disc:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 10.0
    bcoef: float = 0.5
    inv_mass: float = 1.0
    damping: float = 0.99

    @property
    def pos(self):
        return np.array([self.x, self.y])

    @property
    def speed(self):
        return math.hypot(self.vx, self.vy)


@dataclass
class Player:
    disc: Disc
    team: int  # 0 = red (left, attacks +x), 1 = blue (right, attacks -x)
    kick_armed: bool = True  # must release kick key between kicks (HaxBall behaviour)
    kicked: bool = False  # True on the tick a kick actually fired


# Static geometry -----------------------------------------------------------------
NET_ARC_STEPS = 6


def net_polyline(sgn: int):
    """Points of one goal net (sgn=+1 right goal): post -> quarter arc -> back line -> arc -> post.
    Matches classic.hbs: (380,-64) curve -90 -> (400,-44) -> (400,44) curve -90 -> (380,64)."""
    bx, dx, ay, gh = C.NET_BACK_X, C.NET_DEPTH_X, C.NET_ARC_Y, C.GOAL_HALF
    r = dx - bx  # 20 = arc radius; centres at (380, -+44)
    pts = [(sgn * C.GOAL_X, -gh), (sgn * bx, -gh)]
    for k in range(1, NET_ARC_STEPS + 1):
        a = -math.pi / 2 + (math.pi / 2) * k / NET_ARC_STEPS  # from top (-90deg) sweeping to 0deg
        pts.append((sgn * (bx + r * math.cos(a)), -ay + r * math.sin(a)))
    for k in range(1, NET_ARC_STEPS + 1):
        a = (math.pi / 2) * k / NET_ARC_STEPS  # 0 -> 90deg
        pts.append((sgn * (bx + r * math.cos(a)), ay + r * math.sin(a)))
    pts.append((sgn * C.GOAL_X, gh))
    return pts


def build_geometry():
    """(planes, segments, posts) from the current config. planes: (nx, ny, dist, bcoef, mask);
    segments: (p0, p1, bcoef) ball-only; posts: Disc list."""
    planes = [
        (0.0, 1.0, -C.FIELD_HALF_H, C.LINE_BCOEF, "ball"),
        (0.0, -1.0, -C.FIELD_HALF_H, C.LINE_BCOEF, "ball"),
        (0.0, 1.0, -C.HALF_H, C.WALL_BCOEF, "all"),
        (0.0, -1.0, -C.HALF_H, C.WALL_BCOEF, "all"),
        (1.0, 0.0, -C.HALF_W, C.WALL_BCOEF, "all"),
        (-1.0, 0.0, -C.HALF_W, C.WALL_BCOEF, "all"),
    ]
    segments = [
        ((-C.GOAL_X, -C.FIELD_HALF_H), (-C.GOAL_X, -C.GOAL_HALF), C.LINE_BCOEF),
        ((-C.GOAL_X, C.GOAL_HALF), (-C.GOAL_X, C.FIELD_HALF_H), C.LINE_BCOEF),
        ((C.GOAL_X, -C.FIELD_HALF_H), (C.GOAL_X, -C.GOAL_HALF), C.LINE_BCOEF),
        ((C.GOAL_X, C.GOAL_HALF), (C.GOAL_X, C.FIELD_HALF_H), C.LINE_BCOEF),
    ]
    for sgn in (-1, 1):
        pts = net_polyline(sgn)
        for a, b in zip(pts[1:-1], pts[2:]):  # skip the post->380 stub (behind the post)
            segments.append((a, b, C.NET_BCOEF))
    posts = [Disc(sx * C.GOAL_X, sy * C.GOAL_HALF, radius=C.POST_RADIUS, bcoef=C.POST_BCOEF, inv_mass=0.0) for sx in (-1, 1) for sy in (-1, 1)]
    return planes, segments, posts


PLANES, SEGMENTS, POSTS = build_geometry()

KO_ARC_STEPS = 16
KO_BCOEF = 0.1


def kickoff_barrier(kickoff_team: int):
    """Player-only segments active while `kickoff_team` is taking the kick-off (classic.hbs
    kickOffBarrier): the centre line outside the circle, plus the half of the kick-off circle
    that bulges into the *defending* team's half. Both teams stay in their own half; the
    defending team also stays out of the circle."""
    r, hh = C.KICKOFF_RADIUS, C.HALF_H
    segs = [((0.0, -hh), (0.0, -r), KO_BCOEF), ((0.0, r), (0.0, hh), KO_BCOEF)]
    side = 1.0 if kickoff_team == 0 else -1.0  # defending team's half
    pts = []
    for k in range(KO_ARC_STEPS + 1):
        a = -math.pi / 2 + math.pi * k / KO_ARC_STEPS  # from (0,-r) through (side*r, 0) to (0, r)
        pts.append((side * r * math.cos(a), r * math.sin(a)))
    for a_, b_ in zip(pts[:-1], pts[1:]):
        segs.append((a_, b_, KO_BCOEF))
    return segs


KICKOFF_SEGMENTS = {0: kickoff_barrier(0), 1: kickoff_barrier(1)}


def _collide_discs(a: Disc, b: Disc):
    dx, dy = b.x - a.x, b.y - a.y
    d2 = dx * dx + dy * dy
    rsum = a.radius + b.radius
    if d2 >= rsum * rsum or d2 == 0.0:
        return
    d = math.sqrt(d2)
    nx, ny = dx / d, dy / d
    tot = a.inv_mass + b.inv_mass
    if tot <= 0.0:
        return
    pen = rsum - d
    a.x -= nx * pen * (a.inv_mass / tot)
    a.y -= ny * pen * (a.inv_mass / tot)
    b.x += nx * pen * (b.inv_mass / tot)
    b.y += ny * pen * (b.inv_mass / tot)
    vrel = (a.vx - b.vx) * nx + (a.vy - b.vy) * ny
    if vrel > 0.0:  # approaching
        j = vrel * (1.0 + a.bcoef * b.bcoef) / tot
        a.vx -= nx * j * a.inv_mass
        a.vy -= ny * j * a.inv_mass
        b.vx += nx * j * b.inv_mass
        b.vy += ny * j * b.inv_mass


def _collide_plane(d: Disc, nx, ny, dist, bcoef):
    pen = d.radius - (d.x * nx + d.y * ny - dist)
    if pen <= 0.0:
        return
    d.x += nx * pen
    d.y += ny * pen
    vn = d.vx * nx + d.vy * ny
    if vn < 0.0:
        k = vn * (1.0 + bcoef * d.bcoef)
        d.vx -= nx * k
        d.vy -= ny * k


def _collide_segment(d: Disc, p0, p1, bcoef):
    (x0, y0), (x1, y1) = p0, p1
    ex, ey = x1 - x0, y1 - y0
    L2 = ex * ex + ey * ey
    t = ((d.x - x0) * ex + (d.y - y0) * ey) / L2
    t = min(1.0, max(0.0, t))
    cx, cy = x0 + t * ex, y0 + t * ey
    dx, dy = d.x - cx, d.y - cy
    dist2 = dx * dx + dy * dy
    if dist2 >= d.radius * d.radius or dist2 == 0.0:
        return
    dist = math.sqrt(dist2)
    nx, ny = dx / dist, dy / dist
    pen = d.radius - dist
    d.x += nx * pen
    d.y += ny * pen
    vn = d.vx * nx + d.vy * ny
    if vn < 0.0:
        k = vn * (1.0 + bcoef * d.bcoef)
        d.vx -= nx * k
        d.vy -= ny * k


def _sweep_segment(d: Disc, p0, p1, bcoef) -> bool:
    """Continuous check for a fast disc: if the centre's path over this tick crosses the segment,
    place the disc at the contact point (radius away, on the side it came from) and reflect the
    normal velocity. Returns True if handled. Only used when |v| > radius (tunnelling possible)."""
    ax, ay = d.x, d.y
    dx, dy = d.vx, d.vy
    (x0, y0), (x1, y1) = p0, p1
    ex, ey = x1 - x0, y1 - y0
    denom = dx * ey - dy * ex
    if abs(denom) < 1e-12:
        return False
    t = ((x0 - ax) * ey - (y0 - ay) * ex) / denom
    u = ((x0 - ax) * dy - (y0 - ay) * dx) / denom
    if t < 0.0 or t > 1.0 or u < 0.0 or u > 1.0:
        return False
    L = math.hypot(ex, ey)
    nx, ny = ey / L, -ex / L
    if (ax - x0) * nx + (ay - y0) * ny < 0.0:
        nx, ny = -nx, -ny
    cx, cy = ax + dx * t, ay + dy * t
    d.x, d.y = cx + nx * d.radius, cy + ny * d.radius
    vn = d.vx * nx + d.vy * ny
    if vn < 0.0:
        k = vn * (1.0 + bcoef * d.bcoef)
        d.vx -= nx * k
        d.vy -= ny * k
    return True


class Sim:
    """N v N HaxBall-Classic simulator.

    players[0:n_red] are red (user side, attacks +x), players[n_red:] are blue.
    State layout: [red_0(x,y,vx,vy), ..., red_{n-1}, ball(x,y,vx,vy), blue_0, ..., blue_{m-1}]
    so for 1v1 it is the 12-d vector described in config.STATE_NAMES.
    """

    def __init__(self, seed: int | None = None, n_red: int = 1, n_blue: int = 1):
        self.rng = np.random.default_rng(seed)
        self.ball = Disc(0.0, 0.0, **C.BALL)
        self.planes, self.segments, self.posts = build_geometry()
        P = C.PLAYER
        self.n_red, self.n_blue = n_red, n_blue
        mk = lambda team: Player(Disc(0.0, 0.0, radius=P["radius"], bcoef=P["bcoef"], inv_mass=P["inv_mass"], damping=P["damping"]), team)
        self.players = [mk(0) for _ in range(n_red)] + [mk(1) for _ in range(n_blue)]
        self.score = [0, 0]
        self.tick = 0
        self.kickoff_team: int | None = None  # team currently taking the kick-off (barrier active) or None
        self.reset()

    @property
    def state_dim(self):
        return 4 * (len(self.players) + 1)

    def spawn_positions(self):
        """HaxBall-style kick-off: teammates stacked vertically at x = -+spawnDistance."""
        out = []
        for team, n in ((0, self.n_red), (1, self.n_blue)):
            x = -C.SPAWN_DISTANCE if team == 0 else C.SPAWN_DISTANCE
            ys = [(i - (n - 1) / 2) * 55.0 for i in range(n)]
            out += [(x, y) for y in ys]
        return out

    # -- state io ---------------------------------------------------------------
    def reset(self, randomize: bool = False, kickoff_team: int | None = 0):
        """Kick-off positions; `kickoff_team` (0 red / 1 blue / None) takes the kick-off with the
        barrier active until the ball is touched. With randomize=True, scatter everything (no kick-off)."""
        b = self.ball
        for p in self.players:
            p.kick_armed = True
            p.kicked = False
        self.kickoff_team = None if randomize else kickoff_team
        if not randomize:
            b.x, b.y, b.vx, b.vy = 0.0, 0.0, 0.0, 0.0
            for p, (x, y) in zip(self.players, self.spawn_positions()):
                p.disc.x, p.disc.y, p.disc.vx, p.disc.vy = x, y, 0.0, 0.0
            return
        rng = self.rng
        b.x = rng.uniform(-C.FIELD_HALF_W + 20, C.FIELD_HALF_W - 20)
        b.y = rng.uniform(-C.FIELD_HALF_H + 20, C.FIELD_HALF_H - 20)
        sp = rng.uniform(0, 6) * (rng.random() < 0.6)
        ang = rng.uniform(0, 2 * math.pi)
        b.vx, b.vy = sp * math.cos(ang), sp * math.sin(ang)
        placed = []
        for p in self.players:
            d = p.disc
            while True:
                d.x = rng.uniform(-C.HALF_W + 30, C.HALF_W - 30)
                d.y = rng.uniform(-C.HALF_H + 30, C.HALF_H - 30)
                if math.hypot(d.x - b.x, d.y - b.y) > 30 and all(math.hypot(d.x - q.x, d.y - q.y) > 35 for q in placed):
                    break
            placed.append(d)
            sp = rng.uniform(0, 2.5)
            ang = rng.uniform(0, 2 * math.pi)
            d.vx, d.vy = sp * math.cos(ang), sp * math.sin(ang)

    def _discs_in_state_order(self):
        return [p.disc for p in self.players[: self.n_red]] + [self.ball] + [p.disc for p in self.players[self.n_red :]]

    def get_state(self) -> np.ndarray:
        out = []
        for d in self._discs_in_state_order():
            out += [d.x, d.y, d.vx, d.vy]
        return np.array(out, dtype=np.float32)

    def set_state(self, s):
        s = [float(v) for v in s]
        for i, d in enumerate(self._discs_in_state_order()):
            d.x, d.y, d.vx, d.vy = s[4 * i : 4 * i + 4]

    # -- dynamics ---------------------------------------------------------------
    def kick_in_range(self, idx: int) -> bool:
        p, b = self.players[idx].disc, self.ball
        return math.hypot(b.x - p.x, b.y - p.y) <= p.radius + b.radius + C.KICK_RANGE_EXTRA

    def step(self, *actions) -> int | None:
        """Advance one tick. `actions` are encoded ints (see config.encode_action), one per
        player in self.players order (reds first). A single list/tuple is also accepted.

        The kick bit is the *effective* kick input (key held AND armed); the sim re-arms
        the latch whenever kick==0 is given. Returns the scoring team (0/1) or None.
        """
        if len(actions) == 1 and not isinstance(actions[0], (int, np.integer)):
            actions = tuple(actions[0])
        assert len(actions) == len(self.players), f"need {len(self.players)} actions, got {len(actions)}"
        P = C.PLAYER
        for p, a in zip(self.players, actions):
            dx, dy, kick = C.decode_action(a)
            d = p.disc
            p.kicked = False
            if not kick:
                p.kick_armed = True
            kicking = bool(kick) and p.kick_armed
            if dx or dy:
                n = math.hypot(dx, dy)
                acc = P["kick_accel"] if kicking else P["accel"]
                d.vx += dx / n * acc
                d.vy += dy / n * acc
            d.damping = P["kick_damping"] if kicking else P["damping"]
            if kicking:
                if self.kick_in_range(self.players.index(p)):
                    b = self.ball
                    kx, ky = b.x - d.x, b.y - d.y
                    L = math.hypot(kx, ky)
                    if L > 0:
                        b.vx += kx / L * P["kick_strength"]
                        b.vy += ky / L * P["kick_strength"]
                    p.kick_armed = False
                    p.kicked = True

        discs = [self.ball] + [p.disc for p in self.players]
        # collisions (all pairs; a handful of discs so O(n^2) is fine)
        for i in range(len(discs)):
            for j in range(i + 1, len(discs)):
                _collide_discs(discs[i], discs[j])
        for d in discs:
            for post in self.posts:
                _collide_discs(d, post)
        for i, d in enumerate(discs):
            is_ball = i == 0
            for nx, ny, dist, bc, mask in self.planes:
                if mask == "all" or is_ball:
                    _collide_plane(d, nx, ny, dist, bc)
            if is_ball:
                for p0, p1, bc in self.segments:
                    _collide_segment(d, p0, p1, bc)
            elif self.kickoff_team is not None:
                for p0, p1, bc in KICKOFF_SEGMENTS[self.kickoff_team]:
                    _collide_segment(d, p0, p1, bc)
        # integrate (ball: swept test against segments when fast enough to tunnel)
        for i, d in enumerate(discs):
            swept = False
            if i == 0 and d.vx * d.vx + d.vy * d.vy > d.radius * d.radius:
                for p0, p1, bc in self.segments:
                    if _sweep_segment(d, p0, p1, bc):
                        swept = True
                        break
            if not swept:
                d.x += d.vx
                d.y += d.vy
            d.vx *= d.damping
            d.vy *= d.damping
        if self.kickoff_team is not None and (self.ball.vx != 0.0 or self.ball.vy != 0.0):
            self.kickoff_team = None  # ball touched -> kick-off over
        self.tick += 1
        return self.check_goal()

    def check_goal(self) -> int | None:
        b = self.ball
        if abs(b.y) < C.GOAL_HALF:
            if b.x > C.GOAL_X:
                return 0  # red scored on the right goal
            if b.x < -C.GOAL_X:
                return 1
        return None


def goal_from_state(s) -> int | None:
    """Same rule as Sim.check_goal, on a raw state vector (used in model-play mode)."""
    bx, by = float(s[4]), float(s[5])
    if abs(by) < C.GOAL_HALF:
        if bx > C.GOAL_X:
            return 0
        if bx < -C.GOAL_X:
            return 1
    return None
