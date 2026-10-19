"""Stadium constants (HaxBall units = px at 1x zoom, 60 ticks/s) + state/action encoding.

Default preset is HOT_CLASSIC (Classic geometry, smaller & livelier ball). Any .hbs file can
override the physics/geometry via `apply_stadium(path_or_preset)` or the HAXDRILL_STADIUM env var.
Coordinates: origin at pitch centre, x right, y down. Red (user) attacks +x.
"""
from __future__ import annotations

import json
import os
import re

TICK_HZ = 60

# ---- geometry (Classic; verified against haxball-issues/stadiums/classic.hbs) ----
HALF_W = 420.0  # outer bounds (players), bCoef 0.1
HALF_H = 200.0
WALL_BCOEF = 0.1
FIELD_HALF_W = 370.0  # ballArea lines, bCoef 1, ball only
FIELD_HALF_H = 170.0
LINE_BCOEF = 1.0
GOAL_X = 370.0
GOAL_HALF = 64.0
POST_RADIUS = 8.0
POST_BCOEF = 0.5
NET_BACK_X = 380.0  # net: post -> (380,-64) quarter arc -> (400,-44) straight -> (400,44) arc -> (380,64)
NET_DEPTH_X = 400.0
NET_ARC_Y = 44.0
NET_BCOEF = 0.1
KICKOFF_RADIUS = 75.0
SPAWN_DISTANCE = 170.0
KICK_RANGE_EXTRA = 4.0  # kick fires when dist <= r_player + r_ball + 4

# ---- physics presets ----
PLAYER_CLASSIC = dict(radius=15.0, bcoef=0.5, inv_mass=0.5, damping=0.96, accel=0.1, kick_accel=0.07, kick_damping=0.96, kick_strength=5.0)
BALL_CLASSIC = dict(radius=10.0, bcoef=0.5, inv_mass=1.0, damping=0.99)

# "G.Buffon Hot Classic" (stadiums/hot_classic.hbs): same pitch as Classic, smaller ball,
# heavier/quicker players, harder kick, goal posts r=5.5 (white). Values copied from the .hbs.
PLAYER_HOT = dict(radius=15.0, bcoef=0.5, inv_mass=0.3, damping=0.96, accel=0.105, kick_accel=0.073, kick_damping=0.97, kick_strength=5.7)
BALL_HOT = dict(radius=6.4, bcoef=0.5, inv_mass=1.0, damping=0.99)
POST_RADIUS_HOT = 5.5

PRESETS = {"classic": (PLAYER_CLASSIC, BALL_CLASSIC, 8.0, "classic"), "hot_classic": (PLAYER_HOT, BALL_HOT, POST_RADIUS_HOT, "white")}

STADIUM_NAME = "hot_classic"
PLAYER = dict(PLAYER_HOT)
BALL = dict(BALL_HOT)
POST_RADIUS = POST_RADIUS_HOT
POST_STYLE = "white"  # "classic" = pink/blue posts (FFCCCC/CCCCFF), "white" = ffffff, or a hex string

# ---- colours (HaxBall client palette; ball recoloured hot pink for visibility) ----
COL_BALL = (255, 20, 147)
COL_BALL_HEX = "ff1493"


def _parse_hbs(text: str) -> dict:
    """HaxBall .hbs files are JSON5-ish: allow // and /* */ comments and trailing commas."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return json.loads(text)


def apply_stadium(name_or_path: str) -> str:
    """Switch physics (and basic geometry) in place. Returns the stadium name."""
    global STADIUM_NAME, HALF_W, HALF_H, FIELD_HALF_W, FIELD_HALF_H, SPAWN_DISTANCE, KICKOFF_RADIUS, POST_RADIUS, POST_STYLE
    if name_or_path in PRESETS:
        p, b, pr, ps = PRESETS[name_or_path]
        PLAYER.clear(); PLAYER.update(p)
        BALL.clear(); BALL.update(b)
        POST_RADIUS, POST_STYLE = pr, ps
        STADIUM_NAME = name_or_path
        return STADIUM_NAME
    with open(name_or_path, encoding="utf-8") as fh:
        st = _parse_hbs(fh.read())
    PLAYER.clear(); PLAYER.update(PLAYER_CLASSIC)
    BALL.clear(); BALL.update(BALL_CLASSIC)
    pp = st.get("playerPhysics", {})
    for k_hbs, k in (("radius", "radius"), ("bCoef", "bcoef"), ("invMass", "inv_mass"), ("damping", "damping"), ("acceleration", "accel"),
                     ("kickingAcceleration", "kick_accel"), ("kickingDamping", "kick_damping"), ("kickStrength", "kick_strength")):
        if k_hbs in pp:
            PLAYER[k] = float(pp[k_hbs])
    bp = st.get("ballPhysics", {})
    for k_hbs, k in (("radius", "radius"), ("bCoef", "bcoef"), ("invMass", "inv_mass"), ("damping", "damping")):
        if k_hbs in bp:
            BALL[k] = float(bp[k_hbs])
    if "width" in st:
        HALF_W = float(st["width"])
    if "height" in st:
        HALF_H = float(st["height"])
    if "spawnDistance" in st:
        SPAWN_DISTANCE = float(st["spawnDistance"])
    bg = st.get("bg", {})
    if "width" in bg:
        FIELD_HALF_W = float(bg["width"])
    if "height" in bg:
        FIELD_HALF_H = float(bg["height"])
    if "kickOffRadius" in bg:
        KICKOFF_RADIUS = float(bg["kickOffRadius"])
    # goal posts: trait default radius 8 unless the disc overrides it; colour from the disc
    traits = st.get("traits", {})
    POST_RADIUS = float(traits.get("goalPost", {}).get("radius", 8.0))
    POST_STYLE = "classic"
    for d in st.get("discs", []):
        if d.get("trait") == "goalPost":
            if "radius" in d:
                POST_RADIUS = float(d["radius"])
            if "color" in d:
                POST_STYLE = "white" if str(d["color"]).lower() == "ffffff" else str(d["color"])
            break
    STADIUM_NAME = st.get("name", os.path.basename(name_or_path))
    return STADIUM_NAME


if os.environ.get("HAXDRILL_STADIUM"):
    apply_stadium(os.environ["HAXDRILL_STADIUM"])

# ---- state / action encoding shared by sim, dataset, model and play loop ----
# 1v1 state = [px, py, pvx, pvy, bx, by, bvx, bvy, ox, oy, ovx, ovy]
# NvN state = [red_0 .. red_{n-1}, ball, blue_0 .. blue_{m-1}] (4 each)
STATE_DIM = 12
STATE_NAMES = ["px", "py", "pvx", "pvy", "bx", "by", "bvx", "bvy", "ox", "oy", "ovx", "ovy"]
N_ACTIONS = 18  # 9 move directions x kick bit


def encode_action(dx: int, dy: int, kick: int) -> int:
    """dx, dy in {-1, 0, 1}; kick in {0, 1}."""
    return (dx + 1) * 3 + (dy + 1) + 9 * int(kick)


def decode_action(a: int):
    kick, m = divmod(int(a), 9)
    dx, dy = divmod(m, 3)
    return dx - 1, dy - 1, kick


def add_stadium_arg(ap):
    ap.add_argument("--stadium", default=None, help="preset name (classic, hot_classic) or path to a .hbs file")


def handle_stadium_arg(args):
    if getattr(args, "stadium", None):
        apply_stadium(args.stadium)
    return STADIUM_NAME
