// node web/test_physics.mjs ref.json  -- compares JS physics/obs against a Python-generated reference
import { createRequire } from "module";
import fs from "fs";
const require = createRequire(import.meta.url);
const HP = require("./physics.js");
const ref = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const cfg = JSON.parse(fs.readFileSync(new URL("./config.json", import.meta.url), "utf8"));
const sim = new HP.Sim(cfg, ref.n, ref.n);
sim.setState(ref.s0); sim.kickoffTeam = ref.ko0;
let worst = 0, worstObs = 0;
for (let t = 0; t < ref.actions.length; t++) {
  const obs = HP.buildObs(sim, 0, ref.obs_version || 1);
  for (let k = 0; k < obs.length; k++) worstObs = Math.max(worstObs, Math.abs(obs[k] - ref.obs[t][k]));
  const g = sim.step(ref.actions[t]);
  const s = sim.getState();
  if (!ref.resets[t]) for (let k = 0; k < s.length; k++) worst = Math.max(worst, Math.abs(s[k] - ref.states[t][k]));
  if (g !== ref.goals[t]) { console.log("goal mismatch at", t, g, ref.goals[t]); process.exit(1); }
  if (!ref.resets[t] && sim.kickoffTeam !== ref.kickoff[t]) { console.log("kick-off state mismatch at", t, sim.kickoffTeam, ref.kickoff[t]); process.exit(1); }
  if (ref.resets[t]) { sim.setState(ref.states[t]); sim.kickArmed = ref.armed[t].slice(); sim.kickoffTeam = ref.kickoff[t]; }
}
console.log(`JS vs Python: max state diff ${worst.toExponential(2)}, max obs diff ${worstObs.toExponential(2)} over ${ref.actions.length} ticks (${ref.n}v${ref.n})`);
