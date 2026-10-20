"""Serve the web build and collect players' logs (stdlib only, no extra deps).

    python -m haxdrill.server --port 8000            # http://localhost:8000  (serves ./web, logs -> data/web/)

Endpoints:
    GET  /...          static files from the web directory
    POST /log          JSON {n, stadium, obs: [...flat floats...], act: [...], ep: [...], client}
                       -> saved as data/web/log_<utc>_<client>.npz in the standard dataset schema
    GET  /stats        number of saved logs / ticks
CORS is open so the page can be hosted elsewhere (e.g. a GitHub project page) and POST here.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

MAX_BODY = 8 * 1024 * 1024  # a 3-minute match is ~1 MB of JSON
RATE_PER_MIN = 30  # POSTs per client IP per minute


def make_handler(web_dir: str, out_dir: str, log_only: bool = False):
    import collections
    hits = collections.defaultdict(collections.deque)

    class H(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=web_dir, **kw)

        def log_message(self, fmt, *args):  # quieter
            if self.command == "POST":
                super().log_message(fmt, *args)

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, ngrok-skip-browser-warning")
            self.send_header("Access-Control-Max-Age", "86400")

        def end_headers(self):
            self._cors()
            super().end_headers()

        def do_OPTIONS(self):
            self.send_response(204)
            self.end_headers()

        def _json(self, code: int, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.split("?")[0] == "/health":
                return self._json(200, {"ok": True})
            if log_only and self.path.split("?")[0] != "/stats":
                return self._json(404, {"error": "log-only server"})
            if self.path.split("?")[0] == "/stats":
                files = [f for f in os.listdir(out_dir) if f.endswith(".npz")]
                ticks = 0
                for f in files:
                    try:
                        ticks += int(np.load(os.path.join(out_dir, f))["act"].shape[0])
                    except Exception:
                        pass
                return self._json(200, {"logs": len(files), "ticks": ticks})
            return super().do_GET()

        def do_POST(self):
            if self.path.split("?")[0] != "/log":
                return self._json(404, {"error": "not found"})
            ip = self.headers.get("X-Forwarded-For", self.client_address[0]).split(",")[0].strip()
            now = time.time(); q = hits[ip]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= RATE_PER_MIN:
                return self._json(429, {"error": "slow down"})
            q.append(now)
            n = int(self.headers.get("Content-Length", "0"))
            if n <= 0 or n > MAX_BODY:
                return self._json(413, {"error": "bad length"})
            try:
                d = json.loads(self.rfile.read(n))
                nr, nb = int(d.get("n", 1)), int(d.get("n", 1))
                dim = 4 * (nr + nb + 1)
                obs = np.asarray(d["obs"], np.float32).reshape(-1, dim)
                act = np.asarray(d["act"], np.int8)
                ep = np.asarray(d["ep"], np.int32)
                assert len(obs) == len(act) == len(ep) and len(obs) > 0
                assert np.isfinite(obs).all() and act.min() >= 0 and act.max() < 18
                client = re.sub(r"[^A-Za-z0-9_-]", "", str(d.get("client", "anon")))[:32] or "anon"
                nick = str(d.get("name", ""))[:16]
                policy = re.sub(r"[^A-Za-z0-9_.-]", "", str(d.get("policy", "")))[:64]
                goal = np.asarray(d.get("goal", [-1] * len(act)), np.int8)
                assert len(goal) == len(act)
                name = f"log_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{int(time.time()*1000)%1000:03d}_{client}.npz"
                np.savez_compressed(os.path.join(out_dir, name), obs=obs, act=act, ep=ep, kicked=np.zeros(len(obs), bool), goal=goal, n_red=nr, n_blue=nb, stadium=str(d.get("stadium", "")), client=client, name=nick, policy=policy)
                return self._json(200, {"ok": True, "ticks": int(len(obs)), "file": name})
            except Exception as e:  # noqa: BLE001
                return self._json(400, {"error": str(e)[:200]})

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--web", default="web")
    ap.add_argument("--out", default="data/web")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--log-only", action="store_true", help="only /log, /stats, /health (use this behind a public tunnel)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(os.path.abspath(args.web), os.path.abspath(args.out), args.log_only))
    print(f"{'log-only' if args.log_only else 'serving ' + args.web} on http://{args.host}:{args.port}  (logs -> {args.out})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
