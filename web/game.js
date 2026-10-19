// Browser game loop: human = red_0 (your nickname), every other player = exported PPO policy (bot1, bot2, ...).
// Logs (state, action) per tick in the standard dataset layout and POSTs them to LOG_URL at the end of each episode.
(async function () {
  "use strict";
  const qs = new URLSearchParams(location.search);
  const LOG_URL = window.HAXDRILL_LOG_URL || qs.get("log") || "/log";
  const el = id => document.getElementById(id);
  const isTouchDevice = () => (window.matchMedia && matchMedia("(pointer: coarse)").matches) || navigator.maxTouchPoints > 0;
  window.addEventListener("error", e => { try { el("label").textContent = "error: " + (e.message || e.error); } catch (_) {} });
  window.addEventListener("unhandledrejection", e => { try { el("label").textContent = "error: " + (e.reason && e.reason.message || e.reason); } catch (_) {} });
  // state that handlers reference (declared first so a failure further down can never leave it uninitialised)
  let log = { obs: [], act: [], ep: [], goal: [] }, epId = 0, sent = 0, pending = [];
  let pol = null, sim = null, n = 1, P = 2, REPEAT = 2, names = [], jerseys = [];
  const score = [0, 0]; let ticks = 0, pause = 0, lastActs = null, overtime = false, matchOver = false, nextKickoff = 0;
  const keys = new Set();
  const fail = msg => { el("label").textContent = msg; const sub = document.querySelector("#setup .sub"); if (sub) { sub.textContent = msg; sub.style.color = "#ff9c8a"; } el("start").disabled = true; };
  const FILES = (window.HAXDRILL && window.HAXDRILL.files) || {};
  const cfg = FILES["config.json"];
  if (!cfg) { fail("config.js failed to load. Make sure all game files are present."); return; }

  // ---------------- settings ----------------
  const stored = (() => { try { return JSON.parse(localStorage.getItem("hw_settings") || "{}"); } catch { return {}; } })();
  el("nick").value = qs.get("nick") ?? stored.nick ?? "";
  el("jersey").value = qs.get("jersey") ?? stored.jersey ?? "";
  el("jersey").addEventListener("input", () => { el("jersey").value = el("jersey").value.replace(/[^0-9A-Za-z\u00C0-\uFFFF]/g, "").slice(0, 2); });
  el("players").value = "1";  // public build: 1v1 only (team-play bots are not ready); the engine still supports n>1
  el("level").value = qs.get("level") ?? stored.level ?? "easy";
  el("goals").value = qs.get("goals") ?? stored.goals ?? "3";
  el("mins").value = qs.get("time") ?? stored.mins ?? "3";
  const saveSettings = () => { try { localStorage.setItem("hw_settings", JSON.stringify({ nick: el("nick").value, jersey: el("jersey").value, n: el("players").value, level: el("level").value, goals: el("goals").value, mins: el("mins").value })); } catch {} };
  const nickname = () => (el("nick").value.trim() || "player").slice(0, 16);
  const jersey = () => (el("jersey").value.trim() || "7").slice(0, 2);
  const scoreLimit = () => parseInt(el("goals").value) || 0;
  const timeLimitTicks = () => Math.round((parseFloat(el("mins").value) || 0) * 60 * 60);

  // ---------------- match state ----------------
  const policyCache = {};
  async function loadPolicy(k, level) {
    const key = `${k}:${level}`;
    if (!policyCache[key]) {
      // policies are registered by <script src="policy_*.js"> tags; ?policy= may name another file
      const want = qs.get("policy") ? [qs.get("policy")] : [`policy_n${k}_${level}.json`, `policy_n${k}.json`];
      let pol = null;
      for (const name of want) {
        if (FILES[name]) { pol = FILES[name]; pol._file = name; break; }
        try { const r = await fetch(name); if (r.ok) { pol = await r.json(); pol._file = name; break; } } catch {}
      }
      if (!pol) throw new Error(`policy_n${k}.json not found`);
      policyCache[key] = pol;
    }
    return policyCache[key];
  }
  async function newMatch() {
    const want = parseInt(el("players").value) || 1, level = el("level").value;
    try { pol = await loadPolicy(want, level); }
    catch (e) { alert(`No bot available (${e.message}).`); throw e; }
    if (pol._file && !pol._file.includes(`_${level}.`) && !qs.get("policy")) el("level").title = `no ${level} bot yet, using easy`;
    n = pol.n || 1; P = 2 * n; REPEAT = pol.repeat || 2;
    sim = new HaxPhysics.Sim(cfg, n, n);
    names = [nickname()];
    for (let p = 1; p < P; p++) names.push(`bot${p}`);
    // jerseys: you pick yours (default 7); bots count up from 1 within their team, skipping yours
    jerseys = [jersey()];
    let k = 1;
    for (let p = 1; p < n; p++) { if (String(k) === jerseys[0]) k++; jerseys.push(String(k++)); }
    for (let p = 0; p < n; p++) jerseys.push(String(p + 1));
    score[0] = score[1] = 0; el("sr").textContent = el("sb").textContent = "0";
    ticks = 0; pause = 0; overtime = false; matchOver = false; nextKickoff = 0; lastActs = null;
    el("goal").textContent = "";
    saveSettings();
    flushLog(); epId++;
    sim.reset(0);
  }
  // segmented buttons -> hidden selects
  function wireSeg(segId, selId, attr) {
    const seg = el(segId), sel = el(selId);
    const sync = () => { for (const b of seg.querySelectorAll("button")) b.classList.toggle("on", b.dataset[attr] === sel.value); };
    for (const b of seg.querySelectorAll("button")) b.addEventListener("click", () => { if (b.disabled) return; sel.value = b.dataset[attr]; sync(); });
    sync();
  }
  wireSeg("levelseg", "level", "level");
  let started = false, paused = true;
  const overlay = el("overlay");
  function openSettings() { paused = true; overlay.classList.remove("hidden"); el("start").textContent = started ? "New match" : "Play"; }
  el("start").onclick = async () => {
    try { await newMatch(); }
    catch (e) { alert(`Could not start: ${e.message}`); return; }
    started = true; paused = false; overlay.classList.add("hidden"); el("start").blur();
    if (isTouchDevice()) {
      touchHintUntil = performance.now() + 4000;  // shown in the top bar, never over the pitch
      // maximise: go fullscreen (hides the browser chrome). No orientation lock - rotating keeps working.
      try { const de = document.documentElement; if (!document.fullscreenElement && de.requestFullscreen) de.requestFullscreen({ navigationUI: "hide" }).catch(() => {}); } catch (_) {}
    }
  };
  el("settings").onclick = openSettings;

  // ---------------- drawing (mirrors haxdrill/draw.py) ----------------
  const MARGIN = 30, W = 2 * cfg.half_w + 2 * MARGIN, H = 2 * cfg.half_h + 2 * MARGIN;
  const cv = el("c");
  const ctx = cv.getContext("2d");
  let lastCss = 0;
  function fitCanvas() {
    let cssW;
    if (isTouchDevice()) {
      // size from the *visible* viewport (browser toolbars, notches) so nothing is cut off at the bottom
      const vv = window.visualViewport;
      const vw = Math.round(vv ? vv.width : window.innerWidth), vh = Math.round(vv ? vv.height : window.innerHeight);
      const portrait = vh > vw;
      const gut = portrait ? 0 : (vw < 600 ? 92 : 100);
      const barH = el("bar").getBoundingClientRect().height || 32;
      const reserve = portrait ? 150 : 0;  // portrait: controls live below the pitch
      cssW = Math.max(200, Math.min(vw - 2 * gut, (vh - barH - 10 - reserve) * W / H));
      document.documentElement.style.setProperty("--w", `${Math.round(cssW)}px`);
      document.documentElement.style.setProperty("--vh", `${vh}px`);
    } else {
      const avail = window.innerHeight - 140;
      cssW = Math.min(window.innerWidth * 0.96, 1500, Math.max(900, avail * W / H));
      document.documentElement.style.setProperty("--w", `${Math.round(cssW)}px`);
    }
    if (Math.abs(cssW - lastCss) < 0.5) return;
    lastCss = cssW;
    const scale = cssW / W, dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(W * scale * dpr); cv.height = Math.round(H * scale * dpr);
    ctx.setTransform(scale * dpr, 0, 0, scale * dpr, 0, 0);
  }
  fitCanvas();
  addEventListener("resize", fitCanvas);
  addEventListener("orientationchange", () => { setTimeout(fitCanvas, 200); setTimeout(fitCanvas, 800); });
  if (window.visualViewport) window.visualViewport.addEventListener("resize", fitCanvas);
  const px = (x, y) => [x + cfg.half_w + MARGIN, y + cfg.half_h + MARGIN];
  const COL = { canvas: "#1c2d2b", grass: "#718c5a", line: "#c7e6bd", red: "#e56e56", blue: "#5689e5", ball: cfg.ball_color || "#ff1493", postRed: "#ffcccc", postBlue: "#ccccff", net: "#cdcdcd" };
  function circle(x, y, r, fill, stroke, lw) { const [X, Y] = px(x, y); ctx.beginPath(); ctx.arc(X, Y, r, 0, Math.PI * 2); if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.lineWidth = lw || 2; ctx.strokeStyle = stroke; ctx.stroke(); } }
  function line(a, b, col, lw) { ctx.beginPath(); ctx.moveTo(...px(a[0], a[1])); ctx.lineTo(...px(b[0], b[1])); ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.stroke(); }
  function label(d, text, dy, font, col) {
    const [X, Y] = px(d.x, d.y);
    ctx.font = font; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    if (col !== "#fff") { ctx.fillStyle = "rgba(0,0,0,0.6)"; ctx.fillText(text, X + 1, Y + dy + 1); }
    ctx.fillStyle = col; ctx.fillText(text, X, Y + dy);
  }
  function drawPlayer(i, col, outline) {
    const d = sim.discs[i];
    circle(d.x, d.y, d.r, col, outline, 2);
    // jersey number: digits in "middle" baseline sit visually ~1px high, nudge down
    const j = jerseys[i - 1] || "";
    label(d, j, 0.5, j.length > 1 ? "bold 13px system-ui, sans-serif" : "bold 15px system-ui, sans-serif", "#fff");
    label(d, names[i - 1], d.r + 11, "12px system-ui, sans-serif", "#eef4ee");
  }
  function draw(kicking) {
    ctx.fillStyle = COL.canvas; ctx.fillRect(0, 0, W, H);
    const [gx, gy] = px(-cfg.field_half_w, -cfg.field_half_h);
    ctx.fillStyle = COL.grass; ctx.fillRect(gx, gy, 2 * cfg.field_half_w, 2 * cfg.field_half_h);
    for (const net of cfg.nets) for (let i = 0; i + 1 < net.length; i++) line(net[i], net[i + 1], COL.net, 2);
    const fw = cfg.field_half_w, fh = cfg.field_half_h, GX = cfg.goal_x, GH = cfg.goal_half;
    for (const [a, b] of [[[-fw, -fh], [fw, -fh]], [[-fw, fh], [fw, fh]], [[-GX, -fh], [-GX, -GH]], [[-GX, GH], [-GX, fh]], [[GX, -fh], [GX, -GH]], [[GX, GH], [GX, fh]], [[0, -fh], [0, fh]]]) line(a, b, COL.line, 3);
    circle(0, 0, cfg.kickoff_radius, null, COL.line, 3);
    for (const p of cfg.posts) circle(p.x, p.y, cfg.post_radius, p.color || (p.x < 0 ? COL.postRed : COL.postBlue), "#000", 2);
    if (!sim) return;
    const D = sim.discs;
    for (let i = n + 1; i <= P; i++) drawPlayer(i, COL.blue, "#000");
    for (let i = 1; i <= n; i++) drawPlayer(i, COL.red, i === 1 && kicking ? "#fff" : "#000");
    circle(D[0].x, D[0].y, D[0].r, COL.ball, "#000", 2);
  }

  // ---------------- input ----------------
  const KEYMAP = { ArrowLeft: "L", KeyA: "L", ArrowRight: "R", KeyD: "R", ArrowUp: "U", KeyW: "U", ArrowDown: "D", KeyS: "D", KeyX: "K", Space: "K" };
  const typing = () => { const a = document.activeElement; return !!a && (a.tagName === "INPUT" || a.tagName === "SELECT"); };
  addEventListener("keydown", e => {
    if (e.code === "Escape" && started) { if (paused) { paused = false; overlay.classList.add("hidden"); } else openSettings(); return; }
    if (typing() || paused) return;
    if (KEYMAP[e.code]) { keys.add(KEYMAP[e.code]); e.preventDefault(); }
    if (e.code === "KeyR") resetEpisode(0);
  });
  addEventListener("keyup", e => { if (KEYMAP[e.code]) keys.delete(KEYMAP[e.code]); });
  // ---- touch controls: fixed round joystick left of the pitch + KICK right of it. Any touch on the
  // left half of the play area moves the knob relative to the stick's centre (8 directions). ----
  const wrap = el("wrap"), stick = el("stick"), knob = el("knob"), kickBtn = el("kickbtn");
  const isTouch = isTouchDevice();
  if (isTouch) document.body.classList.add("touch");
  let padId = null, kickId = null, touchHintUntil = 0;
  function setDir(dx, dy) {
    keys.delete("L"); keys.delete("R"); keys.delete("U"); keys.delete("D");
    if (dx < 0) keys.add("L"); if (dx > 0) keys.add("R"); if (dy < 0) keys.add("U"); if (dy > 0) keys.add("D");
  }
  function stickFromPoint(clientX, clientY) {
    const r = stick.getBoundingClientRect();
    const R = r.width / 2 - 6;
    let vx = clientX - (r.left + r.width / 2), vy = clientY - (r.top + r.height / 2);
    const L = Math.hypot(vx, vy);
    if (L > R) { vx *= R / L; vy *= R / L; }
    knob.style.transform = `translate(${vx}px, ${vy}px)`;
    if (L < R * 0.22) { setDir(0, 0); return; }
    const sector = Math.round(Math.atan2(vy, vx) / (Math.PI / 4));
    const dirs = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];
    const d = dirs[((sector % 8) + 8) % 8];
    setDir(d[0], d[1]);
  }
  function placeTouchControls() {
    // centre of each control = midpoint between the lower goal post (that side) and the screen's bottom corner
    if (!isTouch) return;
    try { placeTouchControlsInner(); } catch (e) { el("label").textContent = "layout error: " + e.message; }
  }
  function placeTouchControlsInner() {
    const wr = wrap.getBoundingClientRect(), c = cv.getBoundingClientRect();
    const scale = c.width / W;
    const postY = c.top - wr.top + (cfg.goal_half + cfg.half_h + MARGIN) * scale;            // lower posts (y = +64)
    const postL = c.left - wr.left + (-cfg.goal_x + cfg.half_w + MARGIN) * scale;            // left post x
    const postR = c.left - wr.left + (cfg.goal_x + cfg.half_w + MARGIN) * scale;             // right post x
    const vv = window.visualViewport;
    const bottom = (vv ? vv.height : window.innerHeight) - wr.top - 6;                       // visible bottom in wrap coords
    const cy = Math.min(postY + 0.42 * (bottom - postY), bottom - 50);  // a little above the exact midpoint
    // keep the controls fully on screen (portrait: the posts sit near the screen edge, so the midpoint would be cut off)
    const rs = stick.getBoundingClientRect().width / 2 + 8, rk = kickBtn.getBoundingClientRect().width / 2 + 8;
    const sx = Math.max(postL / 2, rs), kx = Math.min((postR + wr.width) / 2, wr.width - rk);
    stick.style.left = `${sx}px`; stick.style.top = `${cy}px`; stick.style.marginTop = "0";
    kickBtn.style.left = `${kx}px`; kickBtn.style.top = `${cy}px`; kickBtn.style.marginTop = "0";
  }
  function stickRelease() { knob.style.transform = ""; stick.classList.remove("active"); setDir(0, 0); }
  wrap.addEventListener("pointerdown", e => {
    if (!isTouch || e.pointerType === "mouse") return;
    e.preventDefault();
    const rect = wrap.getBoundingClientRect();
    if (e.clientX - rect.left < rect.width / 2) {
      if (padId !== null) return;
      padId = e.pointerId; stick.classList.add("active"); stickFromPoint(e.clientX, e.clientY);
    } else {
      if (kickId !== null) return;
      kickId = e.pointerId; keys.add("K"); kickBtn.classList.add("on");
    }
    try { wrap.setPointerCapture(e.pointerId); } catch (_) {}
  });
  wrap.addEventListener("pointermove", e => { if (e.pointerId === padId) stickFromPoint(e.clientX, e.clientY); });
  ["pointerup", "pointercancel"].forEach(ev => wrap.addEventListener(ev, e => {
    if (e.pointerId === padId) { padId = null; stickRelease(); }
    if (e.pointerId === kickId) { kickId = null; keys.delete("K"); kickBtn.classList.remove("on"); }
  }));
  document.addEventListener("visibilitychange", () => { if (document.hidden) { padId = kickId = null; stickRelease(); keys.delete("K"); kickBtn.classList.remove("on"); } });
  const rotateHint = el("rotate");
  const checkOrientation = () => {
    const portrait = window.innerHeight > window.innerWidth;
    document.body.classList.toggle("portrait", isTouch && portrait);
    rotateHint.hidden = !(isTouch && portrait);
  };
  checkOrientation(); addEventListener("resize", checkOrientation);
  let kickArmedLatch = true;  // HaxBall semantics: release to kick again
  function humanAction() {
    const dx = (keys.has("R") ? 1 : 0) - (keys.has("L") ? 1 : 0), dy = (keys.has("D") ? 1 : 0) - (keys.has("U") ? 1 : 0);
    const held = keys.has("K");
    if (!held) kickArmedLatch = true;
    const kick = held && kickArmedLatch ? 1 : 0;
    return HaxPhysics.encodeAction(dx, dy, kick);
  }

  // ---------------- logging ----------------
  const client = (() => { try { let c = localStorage.getItem("hw_client"); if (!c) { c = Math.random().toString(36).slice(2, 10); localStorage.setItem("hw_client", c); } return c; } catch { return "anon"; } })();
  const logstat = el("logstat");
  const payloadOf = l => ({ n, stadium: cfg.stadium, client, name: names[0] || "", obs: l.obs, act: l.act, ep: l.ep, goal: l.goal });
  async function flushLog() {
    if (!log.act.length) return;
    const payload = payloadOf(log);
    log = { obs: [], act: [], ep: [], goal: [] };
    pending.push(payload);
    try {
      const r = await fetch(LOG_URL, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      if (r.ok) { sent += payload.act.length; pending.splice(pending.indexOf(payload), 1); logstat.textContent = `log uploaded: ${Math.round(sent / 60)}s`; }
      else logstat.textContent = `log kept locally (${pending.length})`;
    } catch { logstat.textContent = `log kept locally (${pending.length})`; }
  }
  el("dl").onclick = () => {
    const all = pending.concat(log.act.length ? [payloadOf(log)] : []);
    const blob = new Blob([JSON.stringify(all)], { type: "application/json" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `haxdrill_logs_${Date.now()}.json`; a.click();
  };

  // ---------------- game loop ----------------
  function resetEpisode(kickoff = nextKickoff) { if (!sim) return; sim.reset(kickoff); epId++; lastActs = null; nextKickoff = 0; }
  function endMatch(w) { el("goal").textContent = `${w === 0 ? "Red" : "Blue"} Wins!  ${score[0]} - ${score[1]}`; matchOver = true; pause = 180; }
  function tick() {
    if (!sim || paused) { if (sim) draw(false); return; }
    const aHuman = humanAction();
    const kicking = keys.has("K");
    const state = sim.getState();
    const acts = new Array(P).fill(4);
    acts[0] = aHuman;
    if (lastActs === null || sim.tick % REPEAT === 0) {
      lastActs = [];
      for (let p = 1; p < P; p++) lastActs[p] = HaxPhysics.policyAct(pol, sim, p, false);
    }
    for (let p = 1; p < P; p++) acts[p] = lastActs[p];
    if (pause > 0) {
      // celebration: physics keeps running (ball flies into the net, players keep moving), no new goals
      pause--;
      sim.step(acts);
      if (sim.kicked[0]) kickArmedLatch = false;
      if (pause === 0) {
        el("goal").textContent = "";
        if (matchOver) { score[0] = score[1] = 0; el("sr").textContent = el("sb").textContent = "0"; ticks = 0; overtime = false; matchOver = false; nextKickoff = 0; }
        resetEpisode();
      }
      draw(kicking); return;
    }
    const inKickoff = sim.kickoffTeam >= 0;
    const g = sim.step(acts);
    log.obs.push(...state.map(v => Math.round(v * 1000) / 1000)); log.act.push(aHuman); log.ep.push(epId); log.goal.push(g);
    if (sim.kicked[0]) kickArmedLatch = false;
    if (!inKickoff) ticks++;  // the match clock waits for the kick-off
    if (g >= 0) {
      score[g]++; el("sr").textContent = score[0]; el("sb").textContent = score[1];
      el("goal").textContent = g === 0 ? "Red Scores!" : "Blue Scores!";
      nextKickoff = 1 - g;  // the conceding team kicks off
      pause = 90; flushLog();
      if ((scoreLimit() && score[g] >= scoreLimit()) || overtime) endMatch(g);
    } else if (timeLimitTicks() && ticks >= timeLimitTicks() && !overtime) {
      if (score[0] !== score[1]) { endMatch(score[0] > score[1] ? 0 : 1); flushLog(); }
      else { overtime = true; el("goal").textContent = "Overtime!"; setTimeout(() => { if (overtime && !matchOver) el("goal").textContent = ""; }, 1500); }
    }
    const s = Math.floor(ticks / 60), tl = timeLimitTicks() / 60;
    const fmt = v => `${String(Math.floor(v / 60)).padStart(2, "0")}:${String(Math.floor(v % 60)).padStart(2, "0")}`;
    el("time").textContent = overtime ? `${fmt(s)}  OVERTIME` : tl ? `${fmt(s)} / ${fmt(tl)}` : fmt(s);
    const lvl = pol._file && pol._file.includes("_medium.") ? "medium" : "easy";
    const base = `${n}v${n} · ${names[0]} vs bot [${lvl}]`;
    el("label").textContent = performance.now() < touchHintUntil ? "stick: move · KICK: hold" : (sim.kickoffTeam >= 0 ? `${sim.kickoffTeam === 0 ? "Red" : "Blue"} kick-off · ${base}` : base);
    draw(kicking);
  }
  let acc = 0, last = performance.now();
  function frame(now) {
    fitCanvas(); placeTouchControls();
    acc += Math.min(100, now - last); last = now;
    while (acc >= 1000 / 60) { tick(); acc -= 1000 / 60; }
    requestAnimationFrame(frame);
  }
  addEventListener("beforeunload", () => { if (log.act.length) { try { navigator.sendBeacon(LOG_URL, new Blob([JSON.stringify(payloadOf(log))], { type: "application/json" })); } catch {} } });
  // level availability: grey out medium if its policy file is missing
  if (!FILES["policy_n1_medium.json"]) { const b = document.querySelector('#levelseg [data-level="medium"]'); b.disabled = true; b.title = "medium bot not available yet"; if (el("level").value === "medium") { el("level").value = "easy"; wireSeg("levelseg", "level", "level"); } }
  el("label").textContent = "choose your settings and press Play";
  openSettings();
  requestAnimationFrame(frame);
})();
