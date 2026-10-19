"""Export a trained world model to plain JSON so it can run in a browser (no torch needed).

    python -m haxdrill.export runs/wm.pt web/wm.json

The JS side only needs: x = concat((s - s_mean)/s_std, onehot(a)); h = silu(W_in x + b);
for each block: h += W2 silu(W1 LN(h) + b1) + b2;  next = s + (W_out h + b_out) * d_std + d_mean.
"""
from __future__ import annotations

import argparse
import json

import torch

from .model import load


def export_policy(ckpt: str, out: str):
    """PPO policy -> JSON: tanh MLP trunk + logits head (value head dropped)."""
    import torch as _t

    from .rl import load_policy

    pol, ck = load_policy(ckpt, "cpu")
    f = lambda t: t.detach().cpu().numpy().round(6).tolist()
    l1, _, l2, _ = pol.trunk
    d = dict(kind="policy", obs_dim=pol.obs_dim, obs_version=ck.get("obs_version", 1), stack=ck.get("stack", 1), n=ck.get("n", 1), repeat=ck.get("repeat", 2), stadium=ck.get("stadium", ""),
             w1=f(l1.weight), b1=f(l1.bias), w2=f(l2.weight), b2=f(l2.bias), w_pi=f(pol.pi.weight), b_pi=f(pol.pi.bias))
    with open(out, "w") as fh:
        json.dump(d, fh)
    write_js_wrapper(out, d)
    print(f"wrote {out} (policy n={d['n']}, obs_dim={d['obs_dim']})")


def write_js_wrapper(json_path: str, obj) -> str:
    """<name>.js next to <name>.json: registers the object on window.HAXDRILL so index.html can load
    it with a plain <script> tag (works when the page is opened as a local file, where fetch() is blocked)."""
    import os

    js_path = os.path.splitext(json_path)[0] + ".js"
    key = os.path.basename(json_path)
    with open(js_path, "w") as fh:
        fh.write("window.HAXDRILL=window.HAXDRILL||{files:{}};window.HAXDRILL.files[" + json.dumps(key) + "]=" + json.dumps(obj) + ";\n")
    return js_path


def export_config(out: str):
    """Stadium geometry + physics constants (+ net polyline) so web/physics.js stays in sync with sim.py."""
    from . import config as C
    from .sim import build_geometry, net_polyline

    planes, segments, posts = build_geometry()
    d = dict(
        stadium=C.STADIUM_NAME, tick_hz=C.TICK_HZ,
        half_w=C.HALF_W, half_h=C.HALF_H, field_half_w=C.FIELD_HALF_W, field_half_h=C.FIELD_HALF_H,
        goal_x=C.GOAL_X, goal_half=C.GOAL_HALF, post_radius=C.POST_RADIUS, post_bcoef=C.POST_BCOEF,
        kickoff_radius=C.KICKOFF_RADIUS, spawn_distance=C.SPAWN_DISTANCE, kick_range_extra=C.KICK_RANGE_EXTRA,
        ball=C.BALL, player=C.PLAYER, ball_color="#" + C.COL_BALL_HEX,
        planes=[dict(nx=a, ny=b, dist=c, bcoef=d_, mask=m) for a, b, c, d_, m in planes],
        segments=[dict(p0=list(p0), p1=list(p1), bcoef=bc) for p0, p1, bc in segments],
        posts=[dict(x=p.x, y=p.y, color="#%02x%02x%02x" % col) for p, (_, _, col) in zip(posts, __import__("haxdrill.draw", fromlist=["posts"]).posts())],
        nets=[net_polyline(-1), net_polyline(1)],
    )
    with open(out, "w") as fh:
        json.dump(d, fh)
    write_js_wrapper(out, d)
    print(f"wrote {out} (stadium {C.STADIUM_NAME})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("out")
    ap.add_argument("--config", action="store_true", help="export stadium config instead of a checkpoint")
    from . import config as C

    C.add_stadium_arg(ap)
    args = ap.parse_args()
    C.handle_stadium_arg(args)
    if args.config:
        export_config(args.out)
        return
    ck = torch.load(args.ckpt, map_location="cpu")
    if "obs_dim" in ck:  # PPO policy checkpoint
        export_policy(args.ckpt, args.out)
        return
    m = load(args.ckpt, "cpu")
    f = lambda t: t.detach().cpu().numpy().round(6).tolist()
    blocks = []
    for blk in m.blocks:
        ln, l1, _, l2 = blk
        blocks.append(dict(ln_w=f(ln.weight), ln_b=f(ln.bias), w1=f(l1.weight), b1=f(l1.bias), w2=f(l2.weight), b2=f(l2.bias)))
    out = dict(
        kind="world_model",
        state_dim=m.state_dim,
        n_actions=m.n_actions,
        s_mean=f(m.s_mean), s_std=f(m.s_std), d_mean=f(m.d_mean), d_std=f(m.d_std),
        w_in=f(m.inp.weight), b_in=f(m.inp.bias),
        blocks=blocks,
        w_out=f(m.out.weight), b_out=f(m.out.bias),
    )
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    n = sum(p.numel() for p in m.parameters())
    print(f"wrote {args.out} ({n:,} params)")


if __name__ == "__main__":
    main()
