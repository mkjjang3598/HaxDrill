"""Render a dataset (or any state sequence) to an mp4 with OpenCV. Used to test the
video -> trajectory pipeline on footage with known ground truth.

    python -m haxdrill.render data/sim.npz out.mp4 --start 0 --ticks 3600 --fps 60
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from . import config as C
from . import draw as D


def draw_frame(s: np.ndarray, n_red: int = 1, n_blue: int = 1, names=None, jerseys=None) -> np.ndarray:
    """names / jerseys: optional lists in player order (reds then blues)."""
    k_of = lambda i0: i0 // 4 if i0 < 4 * n_red else i0 // 4 - 1
    img = np.zeros((D.H, D.W, 3), np.uint8)
    img[:] = D.COL_CANVAS[::-1]
    gx, gy, gw, gh = D.grass_rect()
    cv2.rectangle(img, (gx, gy), (gx + gw, gy + gh), D.COL_GRASS[::-1], -1)
    for a, b in D.net_lines():
        cv2.line(img, D.to_px(*a), D.to_px(*b), D.COL_NET[::-1], 2, cv2.LINE_AA)
    for prim in D.pitch_lines():
        if prim[0] == "line":
            cv2.line(img, D.to_px(*prim[1]), D.to_px(*prim[2]), D.COL_LINE[::-1], 3, cv2.LINE_AA)
        else:
            cv2.circle(img, D.to_px(*prim[1]), int(prim[2] * D.SCALE), D.COL_LINE[::-1], 3, cv2.LINE_AA)
    for px_, py_, col in D.posts():
        cv2.circle(img, D.to_px(px_, py_), int(C.POST_RADIUS * D.SCALE), col[::-1], -1, cv2.LINE_AA)
        cv2.circle(img, D.to_px(px_, py_), int(C.POST_RADIUS * D.SCALE), D.COL_OUT[::-1], 2, cv2.LINE_AA)
    SH = 4  # fixed-point shift -> sub-pixel circle centres like the HaxBall canvas
    discs = [(4 * (n_red + 1 + k), D.COL_BLUE, C.PLAYER["radius"], k + 1) for k in range(n_blue)]
    discs += [(4 * k, D.COL_RED, C.PLAYER["radius"], k + 1) for k in range(n_red)]
    discs.append((4 * n_red, D.COL_BALL, C.BALL["radius"], None))
    for i0, col, r, num in discs:
        fx, fy = D.to_px_f(float(s[i0]), float(s[i0 + 1]))
        c = (int(round(fx * 16)), int(round(fy * 16)))
        cv2.circle(img, c, int(r * D.SCALE * 16), col[::-1], -1, cv2.LINE_AA, shift=SH)
        cv2.circle(img, c, int(r * D.SCALE * 16), D.COL_OUT[::-1], 2, cv2.LINE_AA, shift=SH)
        if num is not None:
            txt = str(jerseys[k_of(i0)]) if jerseys else str(num)
            sc = 0.5 if len(txt) < 2 else 0.42
            (tw, th), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, sc, 1)
            cv2.putText(img, txt, (int(round(fx - tw / 2)), int(round(fy + th / 2))), cv2.FONT_HERSHEY_DUPLEX, sc, (255, 255, 255), 1, cv2.LINE_AA)
            if names:
                nm = names[k_of(i0)]
                (nw, nh), _ = cv2.getTextSize(nm, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                org = (int(round(fx - nw / 2)), int(round(fy + r * D.SCALE + nh + 4)))
                cv2.putText(img, nm, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(img, nm, org, cv2.FONT_HERSHEY_SIMPLEX, 0.38, (238, 244, 238), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("out")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--ticks", type=int, default=3600)
    ap.add_argument("--fps", type=int, default=60)
    args = ap.parse_args()
    d = np.load(args.npz)
    obs = d["obs"][args.start : args.start + args.ticks]
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (D.W, D.H))
    for s in obs:
        vw.write(draw_frame(s))
    vw.release()
    print(f"wrote {args.out}: {len(obs)} frames {D.W}x{D.H} @ {args.fps}fps")


if __name__ == "__main__":
    main()
