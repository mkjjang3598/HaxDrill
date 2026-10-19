"""Shared geometry/colours for drawing the Classic pitch, styled after the HaxBall client
(used by the pygame play loop and the mp4 renderer)."""
from __future__ import annotations

from . import config as C

MARGIN = 30  # px around the outer bounds (players can reach +-420 x +-200)
SCALE = 1.0
W = int((2 * C.HALF_W + 2 * MARGIN) * SCALE)
H = int((2 * C.HALF_H + 2 * MARGIN) * SCALE)
BAR = 36  # scoreboard bar height (pygame only)

# HaxBall palette
COL_CANVAS = (28, 45, 43)  # dark area outside the pitch where players can still walk
COL_GRASS = (113, 140, 90)  # #718C5A
COL_LINE = (199, 230, 189)  # #C7E6BD
COL_BALL = C.COL_BALL  # hot pink
COL_RED = (229, 110, 86)  # #E56E56
COL_BLUE = (86, 137, 229)  # #5689E5
COL_POST_RED = (255, 204, 204)  # FFCCCC
COL_POST_BLUE = (204, 204, 255)  # CCCCFF
COL_POST = COL_POST_RED
COL_OUT = (0, 0, 0)
COL_NET = (205, 205, 205)
COL_BAR = (31, 40, 44)
COL_KICK_OUTLINE = (255, 255, 255)
COL_BG = COL_CANVAS  # backwards compat


def to_px_f(x: float, y: float):
    return (x + C.HALF_W + MARGIN) * SCALE, (y + C.HALF_H + MARGIN) * SCALE


def to_px(x: float, y: float):
    fx, fy = to_px_f(x, y)
    return int(round(fx)), int(round(fy))


def grass_rect():
    """(x, y, w, h) in px of the green bg rectangle (bg.width/height = 370 x 170)."""
    x0, y0 = to_px(-C.FIELD_HALF_W, -C.FIELD_HALF_H)
    x1, y1 = to_px(C.FIELD_HALF_W, C.FIELD_HALF_H)
    return x0, y0, x1 - x0, y1 - y0


def pitch_lines():
    """('line', p0, p1) / ('circle', c, r) in field units for the white pitch markings."""
    fw, fh, gx, gh = C.FIELD_HALF_W, C.FIELD_HALF_H, C.GOAL_X, C.GOAL_HALF
    return [
        ("line", (-fw, -fh), (fw, -fh)),
        ("line", (-fw, fh), (fw, fh)),
        ("line", (-gx, -fh), (-gx, -gh)),
        ("line", (-gx, gh), (-gx, fh)),
        ("line", (gx, -fh), (gx, -gh)),
        ("line", (gx, gh), (gx, fh)),
        ("line", (0, -fh), (0, fh)),
        ("circle", (0, 0), C.KICKOFF_RADIUS),
    ]


def net_lines():
    """Goal nets as in classic.hbs: post -> (380,-64) quarter arc -> (400,-44) -> (400,44) arc -> (380,64)."""
    from .sim import net_polyline

    out = []
    for sgn in (-1, 1):
        pts = net_polyline(sgn)
        out += list(zip(pts[:-1], pts[1:]))
    return out


def pitch_primitives():  # backwards compat
    return pitch_lines() + [("line", a, b) for a, b in net_lines()]


def post_colours():
    st = C.POST_STYLE
    if st == "classic":
        return COL_POST_RED, COL_POST_BLUE
    if st == "white":
        return (255, 255, 255), (255, 255, 255)
    c = tuple(int(st[i : i + 2], 16) for i in (0, 2, 4))
    return c, c


def posts():
    """(x, y, colour) of the four goal posts."""
    cr, cb = post_colours()
    return [(-C.GOAL_X, -C.GOAL_HALF, cr), (-C.GOAL_X, C.GOAL_HALF, cr), (C.GOAL_X, -C.GOAL_HALF, cb), (C.GOAL_X, C.GOAL_HALF, cb)]
