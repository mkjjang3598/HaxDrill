"""Scripted behaviours used to generate training data.

- OpponentPolicy: the blue opponent (defends its goal AND attacks yours). This is what
  the world model learns to imitate as part of the environment.
- Red-side exploration policies: random sticky keys, ball chasing, shooting.
"""
from __future__ import annotations

import math

import numpy as np

from . import config as C


def _quantize_dir(tx: float, ty: float, deadband: float = 6.0):
    """Map a desired displacement to an 8-direction key input."""
    dx = 0 if abs(tx) < deadband else (1 if tx > 0 else -1)
    dy = 0 if abs(ty) < deadband else (1 if ty > 0 else -1)
    return dx, dy


class OpponentPolicy:
    """Blue: a 1v1 opponent that both defends the right goal (x=+370) and attacks the
    left goal (x=-370).

    Each decision tick it picks a mode:
      - DEFEND: ball is heading to / sitting near its goal, or red is closer to the ball
        -> stand between ball and own goal, clear when in range.
      - ATTACK: blue can reach the ball first (closer, or ball in red's half)
        -> move to the spot behind the ball (away from the left goal), then dribble/shoot.
    Reaction delay, positioning depth, aggression and noise are randomised per episode so
    the dataset covers a spread of opponent 'personalities'.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.new_personality()
        self._held = (0, 0, 0)
        self._hold_left = 0
        self.mode = "defend"

    def new_personality(self):
        r = self.rng
        self.reaction = int(r.integers(2, 12))  # ticks between re-deciding
        self.depth = float(r.uniform(35, 90))  # defensive distance in front of the ball
        self.line_x = float(r.uniform(150, 300))  # resting x when ball is far
        self.aggr = float(r.uniform(0.2, 1.0))  # bias towards attacking
        self.noise = float(r.uniform(0, 15))
        self.kick_prob = float(r.uniform(0.6, 1.0))
        self.shoot_dist = float(r.uniform(150, 420))  # shoot when closer than this to red goal

    def _choose_mode(self, s):
        px, py, bx, by, bvx, ox, oy = (float(s[i]) for i in (0, 1, 4, 5, 6, 8, 9))
        d_red = math.hypot(bx - px, by - py)
        d_blue = math.hypot(bx - ox, by - oy)
        threat = bx > 120 and bvx > 0.5  # ball travelling towards blue's goal
        if threat or bx > 250:
            return "defend"
        if d_blue < d_red * (0.7 + 0.6 * self.aggr) or bx < -50:
            return "attack"
        return "defend"

    def act(self, s: np.ndarray, kick_armed: bool) -> int:
        if self._hold_left > 0:
            self._hold_left -= 1
            dx, dy, k = self._held
            return C.encode_action(dx, dy, k)
        self._hold_left = self.reaction
        ox, oy = float(s[8]), float(s[9])
        bx, by, bvx, bvy = float(s[4]), float(s[5]), float(s[6]), float(s[7])
        pbx, pby = bx + bvx * 8, by + bvy * 8  # ball a little ahead
        self.mode = self._choose_mode(s)
        dist = math.hypot(bx - ox, by - oy)
        if self.mode == "attack":
            # target goal = left goal (-370, 0). Get behind the ball on the line
            # red_goal -> ball, then push/kick it towards the goal.
            gx, gy = -C.GOAL_X, 0.0
            vx, vy = bx - gx, by - gy
            L = math.hypot(vx, vy) or 1.0
            behind = 22.0
            # if already roughly behind the ball, aim straight at the ball to dribble
            along = ((ox - bx) * vx + (oy - by) * vy) / L
            if along > 10:
                tx, ty = bx, by
            else:
                tx, ty = bx + vx / L * behind, by + vy / L * behind
                # swing around rather than through the ball
                side = 1.0 if (oy - by) * vx - (ox - bx) * vy > 0 else -1.0
                tx += -vy / L * side * 20
                ty += vx / L * side * 20
            to_goal = math.hypot(ox - gx, oy - gy)
            kick = int(dist < 45 and kick_armed and along > -5 and to_goal < self.shoot_dist and self.rng.random() < self.kick_prob)
        else:
            gx, gy = C.GOAL_X, 0.0
            if pbx > -50:
                vx, vy = gx - pbx, gy - pby
                L = math.hypot(vx, vy) or 1.0
                tx = min(pbx + vx / L * self.depth, C.GOAL_X - 20)
                ty = pby + vy / L * self.depth
            else:
                tx, ty = self.line_x, pby * 0.5
            kick = int(dist < 45 and kick_armed and self.rng.random() < self.kick_prob)
        tx += self.rng.normal(0, self.noise)
        ty += self.rng.normal(0, self.noise)
        ty = max(-C.HALF_H + 20, min(C.HALF_H - 20, ty))
        tx = max(-C.HALF_W + 20, min(C.HALF_W - 20, tx))
        dx, dy = _quantize_dir(tx - ox, ty - oy)
        self._held = (dx, dy, kick)
        return C.encode_action(dx, dy, kick)


class RedExplorer:
    """Mixture of behaviours for the user side during data collection."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.mode = "random"
        self._held = (0, 0, 0)
        self._hold_left = 0
        self.new_episode()

    def new_episode(self):
        self.mode = self.rng.choice(["random", "chase", "shoot", "idle"], p=[0.25, 0.3, 0.4, 0.05])
        self.sticky = int(self.rng.integers(5, 40))
        self._hold_left = 0

    def act(self, s: np.ndarray, kick_armed: bool) -> int:
        if self._hold_left > 0:
            self._hold_left -= 1
            dx, dy, k = self._held
            return C.encode_action(dx, dy, k)
        r = self.rng
        px, py, bx, by = float(s[0]), float(s[1]), float(s[4]), float(s[5])
        if self.mode == "idle":
            dx, dy, k = 0, 0, 0
            self._hold_left = self.sticky
        elif self.mode == "random":
            dx, dy = int(r.integers(-1, 2)), int(r.integers(-1, 2))
            k = int(r.random() < 0.3)
            self._hold_left = int(r.integers(1, self.sticky))
        else:
            if self.mode == "chase":
                tx, ty = bx, by
            else:  # shoot: get behind the ball relative to the right goal, then kick
                vx, vy = C.GOAL_X - bx, 0.0 - by
                L = math.hypot(vx, vy) or 1.0
                tx, ty = bx - vx / L * 22, by - vy / L * 22
            tx += r.normal(0, 8)
            ty += r.normal(0, 8)
            dx, dy = _quantize_dir(tx - px, ty - py, deadband=4)
            dist = math.hypot(bx - px, by - py)
            k = int(dist < 70 and r.random() < 0.9)
            self._hold_left = int(r.integers(2, 10))
        self._held = (dx, dy, k)
        return C.encode_action(dx, dy, k)
