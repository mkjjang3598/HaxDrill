// Port of haxdrill/sim.py (HaxBall disc physics) + rl.py observation builder + policy MLP.
// Works in the browser (global `HaxPhysics`) and in node (module.exports) for cross-checking.
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.HaxPhysics = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  function decodeAction(a) {
    const kick = Math.floor(a / 9), m = a % 9;
    return { dx: Math.floor(m / 3) - 1, dy: (m % 3) - 1, kick };
  }
  function encodeAction(dx, dy, kick) { return (dx + 1) * 3 + (dy + 1) + 9 * (kick ? 1 : 0); }
  function mirrorAction(a) {
    const kick = Math.floor(a / 9), m = a % 9, dx = Math.floor(m / 3), dy = m % 3;
    return (2 - dx) * 3 + dy + 9 * kick;
  }

  const KO_ARC_STEPS = 16, KO_BCOEF = 0.1;
  // Player-only barrier while `team` takes the kick-off (mirrors sim.kickoff_barrier):
  // centre line outside the circle + the half-circle bulging into the defending half.
  function kickoffBarrier(cfg, team) {
    const r = cfg.kickoff_radius, hh = cfg.half_h, side = team === 0 ? 1 : -1;
    const segs = [[[0, -hh], [0, -r]], [[0, r], [0, hh]]];
    const pts = [];
    for (let k = 0; k <= KO_ARC_STEPS; k++) { const a = -Math.PI / 2 + Math.PI * k / KO_ARC_STEPS; pts.push([side * r * Math.cos(a), r * Math.sin(a)]); }
    for (let k = 0; k + 1 < pts.length; k++) segs.push([pts[k], pts[k + 1]]);
    return segs;
  }

  class Sim {
    // cfg: contents of web/config.json. Disc order: [ball, red_0.., blue_0..] (VecSim order)
    constructor(cfg, nRed, nBlue) {
      this.cfg = cfg; this.nRed = nRed; this.nBlue = nBlue; this.P = nRed + nBlue;
      this.koSegs = { 0: kickoffBarrier(cfg, 0), 1: kickoffBarrier(cfg, 1) };
      this.kickoffTeam = -1; // -1 none, 0 red kicks off, 1 blue kicks off
      const B = cfg.ball, Pl = cfg.player;
      this.discs = [{ x: 0, y: 0, vx: 0, vy: 0, r: B.radius, im: B.inv_mass, b: B.bcoef, damp: B.damping }];
      for (let i = 0; i < this.P; i++) this.discs.push({ x: 0, y: 0, vx: 0, vy: 0, r: Pl.radius, im: Pl.inv_mass, b: Pl.bcoef, damp: Pl.damping });
      this.kickArmed = new Array(this.P).fill(true);
      this.kicked = new Array(this.P).fill(false);
      this.tick = 0;
      this.reset();
    }
    spawnPositions() {
      const out = [];
      for (const [team, n] of [[0, this.nRed], [1, this.nBlue]]) {
        const x = team === 0 ? -this.cfg.spawn_distance : this.cfg.spawn_distance;
        for (let i = 0; i < n; i++) out.push([x, (i - (n - 1) / 2) * 55.0]);
      }
      return out;
    }
    reset(kickoffTeam = 0) {
      const sp = this.spawnPositions();
      for (const d of this.discs) { d.vx = 0; d.vy = 0; }
      this.discs[0].x = 0; this.discs[0].y = 0;
      for (let p = 0; p < this.P; p++) { this.discs[p + 1].x = sp[p][0]; this.discs[p + 1].y = sp[p][1]; this.kickArmed[p] = true; this.kicked[p] = false; }
      this.tick = 0;
      this.kickoffTeam = kickoffTeam;
      this._hist = {};
    }
    // state in sim.Sim layout: [red_0..red_{n-1}, ball, blue_0..] x,y,vx,vy
    getState() {
      const order = [];
      for (let i = 1; i <= this.nRed; i++) order.push(i);
      order.push(0);
      for (let i = this.nRed + 1; i <= this.P; i++) order.push(i);
      const s = [];
      for (const i of order) { const d = this.discs[i]; s.push(d.x, d.y, d.vx, d.vy); }
      return s;
    }
    setState(s) {
      const order = [];
      for (let i = 1; i <= this.nRed; i++) order.push(i);
      order.push(0);
      for (let i = this.nRed + 1; i <= this.P; i++) order.push(i);
      order.forEach((i, k) => { const d = this.discs[i]; d.x = s[4 * k]; d.y = s[4 * k + 1]; d.vx = s[4 * k + 2]; d.vy = s[4 * k + 3]; });
    }
    kickInRange(p) {
      const d = this.discs[p + 1], b = this.discs[0];
      return Math.hypot(b.x - d.x, b.y - d.y) <= d.r + b.r + this.cfg.kick_range_extra;
    }
    // actions: array of P encoded ints (kick bit = effective kick: key held AND armed). Returns -1 | 0 (red scored) | 1
    step(actions) {
      const Pl = this.cfg.player, ball = this.discs[0];
      for (let p = 0; p < this.P; p++) {
        const { dx, dy, kick } = decodeAction(actions[p]);
        const d = this.discs[p + 1];
        this.kicked[p] = false;
        if (!kick) this.kickArmed[p] = true;
        const kicking = kick === 1 && this.kickArmed[p];
        if (dx || dy) {
          const n = Math.hypot(dx, dy), acc = kicking ? Pl.kick_accel : Pl.accel;
          d.vx += dx / n * acc; d.vy += dy / n * acc;
        }
        d.damp = kicking ? Pl.kick_damping : Pl.damping;
        if (kicking && this.kickInRange(p)) {
          const kx = ball.x - d.x, ky = ball.y - d.y, L = Math.hypot(kx, ky);
          if (L > 0) { ball.vx += kx / L * Pl.kick_strength; ball.vy += ky / L * Pl.kick_strength; }
          this.kickArmed[p] = false; this.kicked[p] = true;
        }
      }
      const D = this.discs;
      for (let i = 0; i < D.length; i++) for (let j = i + 1; j < D.length; j++) collideDiscs(D[i], D[j]);
      for (const d of D) for (const post of this.cfg.posts) collideStatic(d, post.x, post.y, this.cfg.post_radius, this.cfg.post_bcoef);
      for (let i = 0; i < D.length; i++) {
        const isBall = i === 0, d = D[i];
        for (const pl of this.cfg.planes) if (pl.mask === "all" || isBall) collidePlane(d, pl.nx, pl.ny, pl.dist, pl.bcoef);
        if (isBall) for (const sg of this.cfg.segments) collideSegment(d, sg.p0, sg.p1, sg.bcoef);
        else if (this.kickoffTeam >= 0) for (const sg of this.koSegs[this.kickoffTeam]) collideSegment(d, sg[0], sg[1], KO_BCOEF);
      }
      for (let i = 0; i < D.length; i++) {
        const d = D[i];
        let swept = false;
        if (i === 0 && d.vx * d.vx + d.vy * d.vy > d.r * d.r) {
          for (const sg of this.cfg.segments) { if (sweepSegment(d, sg.p0, sg.p1, sg.bcoef)) { swept = true; break; } }
        }
        if (!swept) { d.x += d.vx; d.y += d.vy; }
        d.vx *= d.damp; d.vy *= d.damp;
      }
      if (this.kickoffTeam >= 0 && (ball.vx !== 0 || ball.vy !== 0)) this.kickoffTeam = -1;
      this.tick++;
      if (Math.abs(ball.y) < this.cfg.goal_half) {
        if (ball.x > this.cfg.goal_x) return 0;
        if (ball.x < -this.cfg.goal_x) return 1;
      }
      return -1;
    }
  }

  function collideDiscs(a, b) {
    const dx = b.x - a.x, dy = b.y - a.y, d2 = dx * dx + dy * dy, rsum = a.r + b.r;
    if (d2 >= rsum * rsum || d2 === 0) return;
    const d = Math.sqrt(d2), nx = dx / d, ny = dy / d, tot = a.im + b.im;
    if (tot <= 0) return;
    const pen = rsum - d;
    a.x -= nx * pen * (a.im / tot); a.y -= ny * pen * (a.im / tot);
    b.x += nx * pen * (b.im / tot); b.y += ny * pen * (b.im / tot);
    const vrel = (a.vx - b.vx) * nx + (a.vy - b.vy) * ny;
    if (vrel > 0) {
      const j = vrel * (1 + a.b * b.b) / tot;
      a.vx -= nx * j * a.im; a.vy -= ny * j * a.im;
      b.vx += nx * j * b.im; b.vy += ny * j * b.im;
    }
  }
  function collideStatic(a, sx, sy, sr, sb) {
    collideDiscs(a, { x: sx, y: sy, vx: 0, vy: 0, r: sr, im: 0, b: sb });
  }
  function collidePlane(d, nx, ny, dist, bc) {
    const pen = d.r - (d.x * nx + d.y * ny - dist);
    if (pen <= 0) return;
    d.x += nx * pen; d.y += ny * pen;
    const vn = d.vx * nx + d.vy * ny;
    if (vn < 0) { const k = vn * (1 + bc * d.b); d.vx -= nx * k; d.vy -= ny * k; }
  }
  // continuous check for a fast ball (mirrors sim._sweep_segment): path crosses the segment -> contact + reflect
  function sweepSegment(d, p0, p1, bc) {
    const ax = d.x, ay = d.y, dx = d.vx, dy = d.vy;
    const ex = p1[0] - p0[0], ey = p1[1] - p0[1];
    const denom = dx * ey - dy * ex;
    if (Math.abs(denom) < 1e-12) return false;
    const t = ((p0[0] - ax) * ey - (p0[1] - ay) * ex) / denom;
    const u = ((p0[0] - ax) * dy - (p0[1] - ay) * dx) / denom;
    if (t < 0 || t > 1 || u < 0 || u > 1) return false;
    const L = Math.hypot(ex, ey);
    let nx = ey / L, ny = -ex / L;
    if ((ax - p0[0]) * nx + (ay - p0[1]) * ny < 0) { nx = -nx; ny = -ny; }
    const cx = ax + dx * t, cy = ay + dy * t;
    d.x = cx + nx * d.r; d.y = cy + ny * d.r;
    const vn = d.vx * nx + d.vy * ny;
    if (vn < 0) { const k = vn * (1 + bc * d.b); d.vx -= nx * k; d.vy -= ny * k; }
    return true;
  }
  function collideSegment(d, p0, p1, bc) {
    const ex = p1[0] - p0[0], ey = p1[1] - p0[1], L2 = ex * ex + ey * ey;
    let t = ((d.x - p0[0]) * ex + (d.y - p0[1]) * ey) / L2;
    t = Math.min(1, Math.max(0, t));
    const cx = p0[0] + t * ex, cy = p0[1] + t * ey, dx = d.x - cx, dy = d.y - cy, dist2 = dx * dx + dy * dy;
    if (dist2 >= d.r * d.r || dist2 === 0) return;
    const dist = Math.sqrt(dist2), nx = dx / dist, ny = dy / dist, pen = d.r - dist;
    d.x += nx * pen; d.y += ny * pen;
    const vn = d.vx * nx + d.vy * ny;
    if (vn < 0) { const k = vn * (1 + bc * d.b); d.vx -= nx * k; d.vy -= ny * k; }
  }

  // ---------------- policy (rl.py build_obs + Policy) ----------------
  const POS_SCALE = 400.0, VEL_SCALE = 5.0;
  function buildObs(sim, p, version = 1) {
    const D = sim.discs, nRed = sim.nRed, P = sim.P;
    const team = p < nRed ? 0 : 1, sgn = team === 0 ? 1 : -1;
    const me = D[p + 1], ball = D[0];
    const mx = me.x * sgn, my = me.y, mvx = me.vx * sgn, mvy = me.vy;
    const bx = ball.x * sgn, by = ball.y, bvx = ball.vx * sgn, bvy = ball.vy;
    const f = [mx / POS_SCALE, my / POS_SCALE, mvx / VEL_SCALE, mvy / VEL_SCALE, bx / POS_SCALE, by / POS_SCALE, bvx / VEL_SCALE, bvy / VEL_SCALE, (bx - mx) / POS_SCALE, (by - my) / POS_SCALE];
    const mates = [], opps = [];
    for (let q = 0; q < P; q++) { if (q === p) continue; ((q < nRed) === (team === 0) ? mates : opps).push(q); }
    for (const group of [mates, opps]) {
      if (!group.length) continue;
      const rows = group.map(q => { const d = D[q + 1]; const rx = d.x * sgn - mx, ry = d.y - my; return { dist: Math.hypot(rx, ry), rx, ry, vx: d.vx * sgn, vy: d.vy }; });
      rows.sort((a, b) => a.dist - b.dist);
      for (const r of rows) f.push(r.rx / POS_SCALE, r.ry / POS_SCALE);
      for (const r of rows) f.push(r.vx / VEL_SCALE, r.vy / VEL_SCALE);
    }
    f.push(sim.kickArmed[p] ? 1 : 0);
    if (version >= 2) {
      const bgx = sim.cfg.goal_x - bx, bgy = 0 - by, bgd = Math.hypot(bgx, bgy);
      const mbx = bx - mx, mby = by - my, mbd = Math.hypot(mbx, mby);
      const umx = mbx / Math.max(mbd, 1e-6), umy = mby / Math.max(mbd, 1e-6);
      const ugx = bgx / Math.max(bgd, 1e-6), ugy = bgy / Math.max(bgd, 1e-6);
      const inRange = mbd <= sim.cfg.player.radius + sim.cfg.ball.radius + sim.cfg.kick_range_extra ? 1 : 0;
      f.push(bgx / POS_SCALE, bgy / POS_SCALE, bgd / POS_SCALE, umx, umy, mbd / POS_SCALE, umx * ugx + umy * ugy, umx * ugy - umy * ugx, inRange);
    }
    return f;
  }
  function matvec(W, b, x) {
    const out = new Array(W.length);
    for (let i = 0; i < W.length; i++) { let s = b[i]; const row = W[i]; for (let j = 0; j < row.length; j++) s += row[j] * x[j]; out[i] = s; }
    return out;
  }
  function policyLogits(pol, obs) {
    const h1 = matvec(pol.w1, pol.b1, obs).map(Math.tanh);
    const h2 = matvec(pol.w2, pol.b2, h1).map(Math.tanh);
    return matvec(pol.w_pi, pol.b_pi, h2);
  }
  function sampleLogits(logits, deterministic, rng) {
    if (deterministic) { let bi = 0; for (let i = 1; i < logits.length; i++) if (logits[i] > logits[bi]) bi = i; return bi; }
    const m = Math.max(...logits); const ex = logits.map(v => Math.exp(v - m)); const Z = ex.reduce((a, b) => a + b, 0);
    let u = (rng ? rng() : Math.random()) * Z;
    for (let i = 0; i < ex.length; i++) { u -= ex[i]; if (u <= 0) return i; }
    return ex.length - 1;
  }
  // action in WORLD coordinates for player p
  function policyAct(pol, sim, p, deterministic) {
    let obs = buildObs(sim, p, pol.obs_version || 1);
    const K = pol.stack || 1;
    if (K > 1) {  // short memory: concat the last K decision-frames (newest first), repeated on reset
      sim._hist = sim._hist || {};
      let h = sim._hist[p];
      if (!h || h.length !== K) { h = Array.from({ length: K }, () => obs); }
      else { h.pop(); h.unshift(obs); }
      sim._hist[p] = h;
      obs = [].concat(...h);
    }
    let logits = policyLogits(pol, obs);
    const temp = pol.temperature || 1;
    if (temp !== 1) logits = logits.map(v => v / temp);
    let a = sampleLogits(logits, deterministic);
    if (a >= 9 && pol.kick_keep !== undefined && pol.kick_keep < 1 && Math.random() > pol.kick_keep) a -= 9;  // easy bot misses kicks
    return p < sim.nRed ? a : mirrorAction(a);
  }

  return { Sim, decodeAction, encodeAction, mirrorAction, buildObs, policyLogits, policyAct };
});
