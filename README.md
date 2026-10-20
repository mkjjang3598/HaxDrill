# HaxDrill

A browser-playable, HaxBall-style top-down football game with bots trained by **self-play reinforcement learning**.
Everything runs client-side: the physics engine and the policy network are plain JavaScript.
Korean docs: [README.ko.md](README.ko.md). Play: https://mkjjang3598.github.io/haxdrill/

![teaser](https://mkjjang3598.github.io/images/haxdrill_thumbnail.gif)

## What is in here

| Part | Files |
|---|---|
| Disc physics (HaxBall "Hot Classic" rules: 840x400 pitch, ball r=6.4, kick-off barrier, goal nets) | `haxdrill/sim.py` (scalar), `haxdrill/vecsim.py` (numpy, thousands of envs), `web/physics.js` — all three agree to float precision |
| Bots | `haxdrill/rl.py` PPO self-play with an opponent pool; `haxdrill/league.py` league training (exploiters vs. frozen main, shaping decay, frame-stacked observations) |
| Game | `haxdrill/play.py` (pygame, records your play logs), `web/` (canvas game with touch controls, start screen, difficulty levels) |
| Data loop | `haxdrill/server.py` collects players' logs; `haxdrill/retrain.py` imitates them and continues self-play; `haxdrill/stats.py` human-vs-bot record |
| Diagnostics | `haxdrill/emptynet.py` finishing test, `haxdrill/evaluate.py`, `haxdrill/video.py` (extract trajectories from footage) |

State = positions/velocities of players and ball; action = 8 move directions x kick (18 discrete).
Bots see an ego-centric, mirrored observation so one network plays both sides and any team size.

## Quick start

```bash
uv venv .venv && uv pip install -p .venv/bin/python -e .
.venv/bin/python -m haxdrill.play --level easy                      # play vs the bot (arrows/WASD, X kick)
.venv/bin/python -m haxdrill.rl --n 1 --iters 1500                  # train a 1v1 bot (~20 min on one GPU)
.venv/bin/python -m haxdrill.league --n 1 --hours 24                # league training (medium/hard bots)
.venv/bin/python -m haxdrill.league --n 1 --resume-dir runs/league --hours 24 --export web/policy_n1_expert.json \
    --max-exploiters 20 --exploiter-weight 2 --exploiter-iters 120 --exploiter-fresh 0.3   # continue with stronger exploiters (-> expert)
.venv/bin/python -m haxdrill.export runs/ppo_n1.pt web/policy_n1.json   # -> also writes policy_n1.js for the web build
.venv/bin/python -m haxdrill.server --port 8000                     # serve web/ and collect play logs
```

The web build (`web/`) is static: open `index.html` directly or host it anywhere. Logs are POSTed to `/log`
(configurable with `?log=URL`); without a server they stay in the browser.

## Not affiliated

This is an unofficial fan project. HaxBall is a trademark of its owner; no HaxBall code or assets are used.
The stadium parameters follow the community "Hot Classic" map; the map file itself is not redistributed.

## License

MIT, see [LICENSE](LICENSE).
