"""Extract (state, action) trajectories from gameplay footage of the Classic map.

    python -m haxdrill.video clip.mp4 data/clip.npz --field X0 Y0 X1 Y1 [--red-team left]

--field gives the pixel coordinates of the pitch rectangle: top-left corner of the
white line box (x=-370,y=-170) and bottom-right corner (x=+370,y=+170). Everything else
is scaled from that. Use --probe to dump a frame with a grid so you can read them off.

Per frame we find the ball (white disc), red disc and blue disc by HSV thresholding and
take blob centroids. The video is resampled to 60 Hz. Red's keyboard input is recovered
by inverse dynamics (acceleration = v' - v*damping, quantised to 8 directions) and kicks by
a sudden ball-velocity jump while in kick range. If red is on the right in the footage,
pass --red-team right and the field is mirrored so red always attacks +x.

The output has the same schema as collect.py so it can be mixed into training.
"""
from __future__ import annotations

import argparse
import math

import cv2
import numpy as np

from . import config as C

# HSV ranges (OpenCV H in 0..179). Tune with --probe if your recording differs.
HSV = {
    "ball": [((150, 120, 150), (175, 255, 255))],  # hot pink (use ((0,0,200),(179,40,255)) for a white ball)
    "red": [((0, 120, 120), (12, 255, 255)), ((170, 120, 120), (179, 255, 255))],
    "blue": [((100, 120, 120), (130, 255, 255))],
}


def centroid(mask: np.ndarray, min_area: float):
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    j = int(np.argmax(areas))
    if areas[j] < min_area:
        return None
    return cents[j + 1]


def detect(frame_bgr: np.ndarray, roi, px_per_unit: float):
    x0, y0, x1, y1 = roi
    sub = frame_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    out = {}
    for name, ranges in HSV.items():
        m = None
        for lo, hi in ranges:
            mm = cv2.inRange(hsv, np.array(lo), np.array(hi))
            m = mm if m is None else (m | mm)
        r = C.BALL["radius"] if name == "ball" else C.PLAYER["radius"]
        c = centroid(m, 0.3 * np.pi * (r * px_per_unit) ** 2)
        out[name] = None if c is None else (c[0] + x0, c[1] + y0)
    return out


def px_to_field(p, field, red_right: bool):
    X0, Y0, X1, Y1 = field
    x = (p[0] - X0) / (X1 - X0) * 2 * C.FIELD_HALF_W - C.FIELD_HALF_W
    y = (p[1] - Y0) / (Y1 - Y0) * 2 * C.FIELD_HALF_H - C.FIELD_HALF_H
    if red_right:
        x = -x
    return x, y


def _savgol_coeffs(window: int, order: int, deriv: int) -> np.ndarray:
    half = window // 2
    t = np.arange(-half, half + 1, dtype=float)
    A = np.vander(t, order + 1, increasing=True)
    pinv = np.linalg.pinv(A)
    return pinv[deriv] * math.factorial(deriv)


def smooth_velocity(pos: np.ndarray, window: int = 9) -> np.ndarray:
    """Per-tick velocity from positions via a Savitzky-Golay derivative filter.
    Pixel-quantisation noise (~0.3 units) would otherwise swamp the 0.1/tick input signal."""
    c = _savgol_coeffs(window, 2, 1)
    pad = window // 2
    out = np.zeros_like(pos)
    for j in range(pos.shape[1]):
        x = np.pad(pos[:, j], pad, mode="edge")
        out[:, j] = np.convolve(x, c[::-1], mode="valid")
    return out


_DIRS = np.array([[dx, dy] for dx in (-1, 0, 1) for dy in (-1, 0, 1)], float)
_DIRS_N = _DIRS / np.maximum(np.linalg.norm(_DIRS, axis=1, keepdims=True), 1e-9)


def decode_moves_viterbi(vel: np.ndarray, p_stay: float = 0.95) -> np.ndarray:
    """Most likely 8-direction key sequence given noisy per-tick velocities.

    Model: v_{t+1} = (v_t + dir*accel) * damping  ->  acc_t = v_{t+1}/damping - v_t ~ N(dir*accel, sigma^2).
    A sticky transition prior (keys are held for many ticks) lets evidence accumulate over a run.
    Returns an int array of indices into _DIRS (0..8).
    """
    P = C.PLAYER
    acc = vel[1:] / P["damping"] - vel[:-1]  # (T-1, 2)
    mu = _DIRS_N * P["accel"]  # (9, 2)
    resid = acc[:, None, :] - mu[None]  # (T-1, 9, 2)
    # robust sigma estimate from the best-fitting direction per tick
    best = np.min((resid**2).sum(-1), axis=1)
    sigma = max(float(np.sqrt(np.median(best))) * 1.4826, 0.05)
    emis = -(resid**2).sum(-1) / (2 * sigma**2)  # (T-1, 9)
    T = len(emis)
    log_stay, log_switch = math.log(p_stay), math.log((1 - p_stay) / 8)
    trans = np.full((9, 9), log_switch)
    np.fill_diagonal(trans, log_stay)
    score = emis[0].copy()
    back = np.zeros((T, 9), np.int8)
    for t in range(1, T):
        cand = score[:, None] + trans  # (prev, cur)
        back[t] = np.argmax(cand, 0)
        score = cand[back[t], np.arange(9)] + emis[t]
    path = np.zeros(T + 1, np.int64)
    path[T - 1] = int(np.argmax(score))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    path[T] = path[T - 1]
    return path


def infer_actions(obs: np.ndarray, smooth_window: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """Inverse dynamics for red: key direction via smoothed-velocity Viterbi decoding,
    kick via a sudden ball-velocity jump while in kick range. Also overwrites the velocity
    columns of `obs` with the smoothed estimates."""
    P = C.PLAYER
    T = len(obs)
    for i0 in (0, 4, 8):
        obs[:, i0 + 2 : i0 + 4] = smooth_velocity(obs[:, i0 : i0 + 2], smooth_window)
    dirs = decode_moves_viterbi(obs[:, 2:4].astype(np.float64))
    act = np.zeros(T, np.int8)
    kicked = np.zeros(T, bool)
    bv_raw = np.vstack([np.diff(obs[:, 4:6], axis=0), np.zeros((1, 2))])  # raw (unsmoothed) ball velocity for impulse detection
    for t in range(T - 1):
        dx, dy = int(_DIRS[dirs[t], 0]), int(_DIRS[dirs[t], 1])
        dist = np.hypot(obs[t, 4] - obs[t, 0], obs[t, 5] - obs[t, 1])
        dv = bv_raw[t + 1] / C.BALL["damping"] - bv_raw[t]
        k = int(dist < P["radius"] + C.BALL["radius"] + 12 and np.linalg.norm(dv) > 0.5 * P["kick_strength"])
        kicked[t] = bool(k)
        act[t] = C.encode_action(dx, dy, k)
    act[T - 1] = act[T - 2] if T > 1 else 0
    return act, kicked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--field", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), help="pixel coords of pitch corners (-370,-170) and (370,170)")
    ap.add_argument("--red-team", choices=["left", "right"], default="left")
    ap.add_argument("--probe", default=None, help="write the first frame with a 50px grid to this path and exit")
    ap.add_argument("--max-gap", type=int, default=6, help="max consecutive undetected frames to interpolate over")
    ap.add_argument("--debug", default=None, help="write an overlay video of detections")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    if args.probe:
        ok, f = cap.read()
        assert ok, "could not read video"
        for x in range(0, f.shape[1], 50):
            cv2.line(f, (x, 0), (x, f.shape[0]), (0, 255, 255) if x % 100 == 0 else (0, 128, 128), 1)
            if x % 100 == 0:
                cv2.putText(f, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        for y in range(0, f.shape[0], 50):
            cv2.line(f, (0, y), (f.shape[1], y), (0, 255, 255) if y % 100 == 0 else (0, 128, 128), 1)
            if y % 100 == 0:
                cv2.putText(f, str(y), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.imwrite(args.probe, f)
        print("wrote", args.probe, "- read off the pitch corner pixel coords and pass them as --field")
        return
    assert args.field, "--field is required (use --probe first)"
    X0, Y0, X1, Y1 = args.field
    px_per_unit = (X1 - X0) / (2 * C.FIELD_HALF_W)
    # search region: outer bounds (players can leave the pitch lines) + a little slack
    pad = (C.HALF_W - C.FIELD_HALF_W + 20) * px_per_unit
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi = (max(0, int(X0 - pad)), max(0, int(Y0 - pad)), min(W, int(X1 + pad)), min(Hh, int(Y1 + pad)))

    red_right = args.red_team == "right"
    pos = []  # per frame: [px,py,bx,by,ox,oy] with nan if missing
    dbg = None
    if args.debug:
        dbg = cv2.VideoWriter(args.debug, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, Hh))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        det = detect(f, roi, px_per_unit)
        row = []
        for name in ("red", "ball", "blue"):
            p = det[name]
            row += [np.nan, np.nan] if p is None else list(px_to_field(p, args.field, red_right))
            if dbg is not None and p is not None:
                cv2.circle(f, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
        pos.append(row)
        if dbg is not None:
            cv2.rectangle(f, roi[:2], roi[2:], (0, 255, 255), 1)
            dbg.write(f)
    cap.release()
    if dbg is not None:
        dbg.release()
    pos = np.array(pos, np.float64)
    n_frames = len(pos)

    # fill short gaps by linear interpolation, split episodes at long gaps / teleports
    t_src = np.arange(n_frames) / fps
    for j in range(pos.shape[1]):
        col = pos[:, j]
        bad = np.isnan(col)
        if bad.all():
            raise SystemExit(f"column {j} never detected: check --field / HSV ranges (try --debug)")
        col[bad] = np.interp(t_src[bad], t_src[~bad], col[~bad])
    # resample to 60 Hz
    t_dst = np.arange(0, t_src[-1], 1.0 / C.TICK_HZ)
    P = np.stack([np.interp(t_dst, t_src, pos[:, j]) for j in range(pos.shape[1])], 1)
    V = np.vstack([np.diff(P, axis=0), np.zeros((1, P.shape[1]))])  # per-tick velocity
    obs = np.zeros((len(P), C.STATE_DIM), np.float32)
    obs[:, [0, 1, 4, 5, 8, 9]] = P
    obs[:, [2, 3, 6, 7, 10, 11]] = V
    # episode split: teleport (kick-off reset) when the ball jumps > 60 units in one tick
    jump = np.hypot(*V[:, 2:4].T) > 60
    ep = np.cumsum(np.r_[0, jump[:-1]]).astype(np.int32)
    act, kicked = infer_actions(obs)
    np.savez_compressed(args.out, obs=obs, act=act, ep=ep, kicked=kicked)
    print(f"saved {args.out}: {len(obs)} ticks from {n_frames} frames @ {fps:.1f}fps, {int(ep.max())+1} episodes, {int(kicked.sum())} kicks inferred")


if __name__ == "__main__":
    main()
