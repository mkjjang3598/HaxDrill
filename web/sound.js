// HaxDrill sound effects, synthesised with the Web Audio API (no audio files).
// HaxSound.unlock() must be called from a user gesture (the Play button); browsers block audio before that.
(function () {
  "use strict";
  let ctx = null, master = null, noise = null;
  let muted = false;
  try { muted = localStorage.getItem("hw_muted") === "1"; } catch (_) {}

  function ensure() {
    if (ctx) return ctx;
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return null;
    ctx = new AC();
    master = ctx.createGain(); master.gain.value = muted ? 0 : 0.6; master.connect(ctx.destination);
    // 1 s of white noise, reused by every sound
    noise = ctx.createBuffer(1, ctx.sampleRate, ctx.sampleRate);
    const d = noise.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    return ctx;
  }

  function noiseBurst(t, dur, freq, q, gain) {
    const src = ctx.createBufferSource(); src.buffer = noise;
    const f = ctx.createBiquadFilter(); f.type = "bandpass"; f.frequency.value = freq; f.Q.value = q;
    const g = ctx.createGain();
    g.gain.setValueAtTime(gain, t); g.gain.exponentialRampToValueAtTime(0.001, t + dur);
    src.connect(f); f.connect(g); g.connect(master);
    src.start(t); src.stop(t + dur + 0.02);
  }

  // kick: a short low "thump" (falling sine) plus a soft leather "tock" (filtered noise click)
  function kick(volume = 1) {
    if (!ensure() || muted) return;
    const t = ctx.currentTime;
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.type = "sine";
    o.frequency.setValueAtTime(190, t); o.frequency.exponentialRampToValueAtTime(70, t + 0.09);
    g.gain.setValueAtTime(0.9 * volume, t); g.gain.exponentialRampToValueAtTime(0.001, t + 0.12);
    o.connect(g); g.connect(master); o.start(t); o.stop(t + 0.14);
    noiseBurst(t, 0.035, 1800, 1.2, 0.35 * volume);
  }

  // goal: a crowd-like swell of filtered noise with a short whistle-ish tone on top
  function goal() {
    if (!ensure() || muted) return;
    const t = ctx.currentTime;
    const src = ctx.createBufferSource(); src.buffer = noise; src.loop = true;
    const f = ctx.createBiquadFilter(); f.type = "bandpass"; f.frequency.value = 900; f.Q.value = 0.6;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.001, t); g.gain.exponentialRampToValueAtTime(0.35, t + 0.25); g.gain.exponentialRampToValueAtTime(0.001, t + 1.6);
    src.connect(f); f.connect(g); g.connect(master); src.start(t); src.stop(t + 1.7);
    const o = ctx.createOscillator(), og = ctx.createGain();
    o.type = "triangle"; o.frequency.setValueAtTime(880, t); o.frequency.linearRampToValueAtTime(1320, t + 0.18);
    og.gain.setValueAtTime(0.0001, t); og.gain.exponentialRampToValueAtTime(0.15, t + 0.03); og.gain.exponentialRampToValueAtTime(0.001, t + 0.35);
    o.connect(og); og.connect(master); o.start(t); o.stop(t + 0.4);
  }

  function unlock() { if (ensure() && ctx.state === "suspended") ctx.resume(); }
  function setMuted(m) {
    muted = !!m;
    try { localStorage.setItem("hw_muted", muted ? "1" : "0"); } catch (_) {}
    if (master) master.gain.value = muted ? 0 : 0.6;
  }
  window.HaxSound = { kick, goal, unlock, setMuted, isMuted: () => muted };
})();
