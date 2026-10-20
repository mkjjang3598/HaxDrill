"""Play with the keyboard, either in the real simulator or inside the learned world model.

    python -m haxdrill.play --mode sim                      # ground-truth physics + scripted opponent
    python -m haxdrill.play --mode model --ckpt runs/wm.pt  # everything (ball, opponent) predicted by the model
    python -m haxdrill.play --mode model --ckpt runs/wm.pt --ghost   # also draw the sim as outlines for comparison
    python -m haxdrill.play --opponent rl --policy runs/ppo_n1.pt    # vs the self-play RL opponent
    python -m haxdrill.play --n 2 --policy runs/ppo_n2.pt            # 2v2: your teammate + both opponents are the RL policy

Keys: arrows / WASD move, X or Space kick, R reset, Esc quit.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pygame

from . import config as C
from . import draw as D
from .policies import OpponentPolicy
from .sim import Sim, goal_from_state


class KickLatch:
    """HaxBall semantics: holding kick fires once; you must release to kick again.
    The *effective* kick bit (held AND armed) is what goes into the action."""

    def __init__(self):
        self.armed = True

    def effective(self, held: bool) -> int:
        if not held:
            self.armed = True
            return 0
        return int(self.armed)

    def fired(self):
        self.armed = False


def kick_in_range(s) -> bool:
    return math.hypot(s[4] - s[0], s[5] - s[1]) <= C.PLAYER["radius"] + C.BALL["radius"] + C.KICK_RANGE_EXTRA


def read_keys() -> tuple[int, int, bool]:
    k = pygame.key.get_pressed()
    dx = int(k[pygame.K_RIGHT] or k[pygame.K_d]) - int(k[pygame.K_LEFT] or k[pygame.K_a])
    dy = int(k[pygame.K_DOWN] or k[pygame.K_s]) - int(k[pygame.K_UP] or k[pygame.K_w])
    return dx, dy, bool(k[pygame.K_x] or k[pygame.K_SPACE])


_FONTS = {}


def _font(key, size, bold):
    if key not in _FONTS:
        _FONTS[key] = pygame.font.SysFont("dejavusans,arial,freesans", size, bold=bold)
    return _FONTS[key]


def player_names(n_red: int, n_blue: int, human: str = "player"):
    """red_0 = the human; everyone else bot1, bot2, ... in player order."""
    return [human] + [f"bot{k}" for k in range(1, n_red + n_blue)]


def player_jerseys(n_red: int, n_blue: int, human: str = "7"):
    """Your pick (max 2 chars) for red_0; bots count up from 1 within their team, skipping yours."""
    human = (human or "7")[:2]
    out, k = [human], 1
    for _ in range(1, n_red):
        if str(k) == human:
            k += 1
        out.append(str(k))
        k += 1
    out += [str(i + 1) for i in range(n_blue)]
    return out


def draw_discs(surf, s, kicking: bool, outline_only: bool = False, n_red: int = 1, n_blue: int = 1, names=None, jerseys=None):
    """State layout [red_0..red_{n-1}, ball, blue_0..]; red_0 is the human. Players carry jersey
    numbers (centred on the disc) and names underneath."""
    names = names or player_names(n_red, n_blue)
    jerseys = jerseys or player_jerseys(n_red, n_blue)
    num_font, num_font2, name_font = _font("num", 15, True), _font("num2", 12, True), _font("name", 12, False)
    discs = []
    for k in range(n_blue):
        discs.append((4 * (n_red + 1 + k), D.COL_BLUE, C.PLAYER["radius"], False, jerseys[n_red + k], names[n_red + k]))
    for k in range(n_red):
        discs.append((4 * k, D.COL_RED, C.PLAYER["radius"], kicking and k == 0, jerseys[k], names[k]))
    discs.append((4 * n_red, D.COL_BALL, C.BALL["radius"], False, None, None))
    for i0, col, r, flash, num, name in discs:
        c = D.to_px(float(s[i0]), float(s[i0 + 1]))
        if outline_only:
            pygame.draw.circle(surf, col, c, int(r * D.SCALE), 2)
            continue
        pygame.draw.circle(surf, col, c, int(r * D.SCALE))
        pygame.draw.circle(surf, D.COL_KICK_OUTLINE if flash else D.COL_OUT, c, int(r * D.SCALE), 2)
        if num is not None:
            t = (num_font if len(str(num)) < 2 else num_font2).render(str(num), True, (255, 255, 255))
            # centre on the glyph's ink box, not the font's line box (digits have no descender)
            ink = t.get_bounding_rect()
            surf.blit(t, (c[0] - ink.centerx, c[1] - ink.centery))
            tn = name_font.render(name, True, (238, 244, 238))
            sh = name_font.render(name, True, (0, 0, 0))
            y = c[1] + int(r * D.SCALE) + 4
            surf.blit(sh, (c[0] - tn.get_width() // 2 + 1, y + 1))
            surf.blit(tn, (c[0] - tn.get_width() // 2, y))


def draw(screen, font, big, s, score, label, ticks, kicking=False, ghost=None, msg=None, n_red=1, n_blue=1, limits=None, names=None, jerseys=None):
    # scoreboard bar
    screen.fill(D.COL_BAR, (0, 0, D.W, D.BAR))
    pygame.draw.rect(screen, D.COL_RED, (12, 9, 18, 18))
    pygame.draw.rect(screen, D.COL_BLUE, (84, 9, 18, 18))
    screen.blit(font.render(str(score[0]), True, (255, 255, 255)), (38, 8))
    screen.blit(font.render(str(score[1]), True, (255, 255, 255)), (110, 8))
    secs = ticks // C.TICK_HZ
    clock_txt = f"{secs // 60:02d}:{secs % 60:02d}"
    if limits:
        sl, tl, ot = limits
        if ot:
            clock_txt += "  OVERTIME"
        elif tl:
            clock_txt += f" / {int(tl):02d}:{int((tl % 1) * 60):02d}"
        if sl:
            clock_txt = f"first to {sl}   " + clock_txt
    t = font.render(clock_txt, True, (255, 255, 255))
    screen.blit(t, (D.W // 2 - t.get_width() // 2, 8))
    lab = font.render(label, True, (170, 190, 170))
    screen.blit(lab, (D.W - lab.get_width() - 12, 8))
    # pitch
    surf = screen.subsurface((0, D.BAR, D.W, D.H))
    surf.fill(D.COL_CANVAS)
    pygame.draw.rect(surf, D.COL_GRASS, D.grass_rect())
    for a, b in D.net_lines():
        pygame.draw.line(surf, D.COL_NET, D.to_px(*a), D.to_px(*b), 2)
    for prim in D.pitch_lines():
        if prim[0] == "line":
            pygame.draw.line(surf, D.COL_LINE, D.to_px(*prim[1]), D.to_px(*prim[2]), 3)
        else:
            pygame.draw.circle(surf, D.COL_LINE, D.to_px(*prim[1]), int(prim[2] * D.SCALE), 3)
    for px_, py_, col in D.posts():
        pygame.draw.circle(surf, col, D.to_px(px_, py_), int(C.POST_RADIUS * D.SCALE))
        pygame.draw.circle(surf, D.COL_OUT, D.to_px(px_, py_), int(C.POST_RADIUS * D.SCALE), 2)
    if ghost is not None:
        draw_discs(surf, ghost, False, outline_only=True, n_red=n_red, n_blue=n_blue)
    draw_discs(surf, s, kicking, n_red=n_red, n_blue=n_blue, names=names, jerseys=jerseys)
    if msg:
        t2 = big.render(msg, True, (255, 255, 255))
        pygame.draw.rect(surf, (0, 0, 0), (D.W // 2 - t2.get_width() // 2 - 14, D.H // 2 - 78, t2.get_width() + 28, t2.get_height() + 16), border_radius=8)
        surf.blit(t2, (D.W // 2 - t2.get_width() // 2, D.H // 2 - 70))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sim", "model"], default="sim")
    ap.add_argument("--ckpt", default="runs/wm.pt")
    ap.add_argument("--ghost", action="store_true", help="in model mode, also run the sim and draw it as outlines")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record", default=None, help="npz path for your play log (sim mode). Default: data/play/<timestamp>.npz; --no-record to disable")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0, help="auto-quit after N frames (for testing)")
    ap.add_argument("--opponent", choices=["scripted", "rl"], default=None, help="blue controller in sim mode (default: rl if --policy given else scripted)")
    ap.add_argument("--policy", default=None, help="PPO checkpoint for RL opponents/teammates (runs/ppo_n1.pt)")
    ap.add_argument("--level", choices=["easy", "medium", "hard"], default=None, help="shortcut: easy = runs/ppo_n{n}.pt, medium = runs/medium.pt, hard = runs/league/main_latest.pt")
    ap.add_argument("--n", type=int, default=1, help="players per team (n>1 requires --policy)")
    ap.add_argument("--deterministic", action="store_true", help="RL policy takes argmax actions")
    ap.add_argument("--name", default="player", help="your nickname shown under your disc")
    ap.add_argument("--jersey", default="7", help="your jersey number/letters (max 2 chars)")
    ap.add_argument("--score-limit", type=int, default=3, help="goals to win a match (0 = unlimited)")
    ap.add_argument("--time-limit", type=float, default=3.0, help="match length in minutes (0 = unlimited); tied at full time -> golden-goal overtime")
    C.add_stadium_arg(ap)
    args = ap.parse_args()
    C.handle_stadium_arg(args)
    n = args.n
    if args.level and not args.policy:
        args.policy = {"easy": f"runs/ppo_n{n}.pt", "medium": "runs/medium.pt", "hard": "runs/league/main_latest.pt"}[args.level]
    if args.opponent is None:
        args.opponent = "rl" if args.policy else "scripted"
    if (n > 1 or args.opponent == "rl") and not args.policy:
        raise SystemExit("--policy runs/ppo_nN.pt is required for --n > 1 or --opponent rl")
    if args.mode == "model" and n > 1:
        raise SystemExit("world-model mode is 1v1 only")
    if args.no_record:
        args.record = None
    elif args.record is None and args.mode == "sim":
        import os, time

        os.makedirs("data/play", exist_ok=True)
        args.record = time.strftime("data/play/%Y%m%d_%H%M%S") + f"_{n}v{n}.npz"

    pygame.init()
    screen = pygame.display.set_mode((D.W, D.H + D.BAR))
    pygame.display.set_caption(f"HaxDrill - {C.STADIUM_NAME}")
    font = pygame.font.SysFont(None, 26)
    big = pygame.font.SysFont(None, 48)
    clock = pygame.time.Clock()

    rng = np.random.default_rng(args.seed)
    sim = Sim(args.seed, n_red=n, n_blue=n)
    opp = OpponentPolicy(rng)
    ctrl = None
    if args.policy:
        from .rl import PolicyController

        ctrl = PolicyController(args.policy, n, n, deterministic=args.deterministic)
    others = [i for i in range(2 * n) if i != 0]  # every player except the human (red_0)
    latch = KickLatch()
    model = None
    dev = "cpu"
    if args.mode == "model":
        import torch

        from .model import load

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = load(args.ckpt, dev)
        state = sim.get_state()
    score = [0, 0]
    pause = 0
    msg = None
    match_ticks = 0
    overtime = False
    match_over = False
    next_kickoff = 0  # red kicks off at the start; the conceding team after a goal
    if args.mode == "model":
        sim.reset(kickoff_team=None)  # the world model has no kick-off rule
    rec_obs, rec_act, rec_ep, rec_goal = [], [], [], []
    ep_id = 0
    frames = 0
    game_ticks = 0
    kicking = False
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_r:
                sim.reset(kickoff_team=None if args.mode == "model" else 0)
                if ctrl is not None:
                    ctrl.reset()
                state = sim.get_state()
                ep_id += 1
        dx, dy, held = read_keys()
        if pause > 0:
            pause -= 1
            # celebration: physics keeps running (ball flies into the net, players keep moving), no new goals
            if args.mode == "sim":
                s = sim.get_state()
                kick = latch.effective(held)
                kicking = bool(held)
                a_red = C.encode_action(dx, dy, kick)
                if ctrl is not None:
                    acts = ctrl.act(s, [p.kick_armed for p in sim.players], others)
                    all_actions = [a_red] + [acts[i] for i in others]
                else:
                    all_actions = [a_red, opp.act(s, sim.players[1].kick_armed)]
                sim.step(all_actions)
                if sim.players[0].kicked:
                    latch.fired()
                state = sim.get_state()
            if pause == 0:
                if match_over:
                    score[:] = [0, 0]
                    match_ticks, overtime, match_over = 0, False, False
                    next_kickoff = 0
                sim.reset(kickoff_team=None if args.mode == "model" else next_kickoff)
                if ctrl is not None:
                    ctrl.reset()
                state = sim.get_state()
                opp.new_personality()
                ep_id += 1
                msg = None
        else:
            s = sim.get_state() if args.mode == "sim" else state
            kick = latch.effective(held)
            kicking = bool(held)
            a_red = C.encode_action(dx, dy, kick)
            if ctrl is not None and (args.mode == "sim" or args.ghost):
                acts = ctrl.act(s, [p.kick_armed for p in sim.players], others)
                all_actions = [a_red] + [acts[i] for i in others]
            else:
                a_blue = opp.act(s, sim.players[1].kick_armed) if (args.mode == "sim" or args.ghost) else 4
                all_actions = [a_red, a_blue]
            game_ticks += 1
            if sim.kickoff_team is None or args.mode == "model":
                match_ticks += 1
            if args.mode == "sim":
                goal = sim.step(all_actions)
                if args.record:
                    rec_obs.append(s.copy()), rec_act.append(a_red), rec_ep.append(ep_id), rec_goal.append(-1 if goal is None else goal)
                if sim.players[0].kicked:
                    latch.fired()
                state = sim.get_state()
            else:
                import torch

                if kick and kick_in_range(s):
                    latch.fired()
                with torch.no_grad():
                    state = model(torch.as_tensor(s, device=dev)[None], torch.tensor([a_red], device=dev))[0].cpu().numpy()
                # keep the model's state inside the arena so it can't run away
                state[[0, 8]] = np.clip(state[[0, 8]], -C.HALF_W, C.HALF_W)
                state[[1, 9]] = np.clip(state[[1, 9]], -C.HALF_H, C.HALF_H)
                goal = goal_from_state(state)
                if args.ghost:
                    sim.step(all_actions)
            if goal is not None:
                score[goal] += 1
                next_kickoff = 1 - goal
                msg = "Red Scores!" if goal == 0 else "Blue Scores!"
                pause = 90
                if (args.score_limit and score[goal] >= args.score_limit) or overtime:
                    msg = ("Red" if goal == 0 else "Blue") + f" Wins!  {score[0]} - {score[1]}"
                    match_over, pause = True, 180
            elif args.time_limit and match_ticks >= args.time_limit * 60 * C.TICK_HZ and not overtime:
                if score[0] != score[1]:
                    w = 0 if score[0] > score[1] else 1
                    msg = ("Red" if w == 0 else "Blue") + f" Wins!  {score[0]} - {score[1]}"
                    match_over, pause = True, 180
                else:
                    overtime = True
                    msg = "Overtime!"
        if args.mode == "sim":
            label = f"{n}v{n}  vs {'RL' if ctrl is not None else 'scripted'}"
            if sim.kickoff_team is not None and pause == 0:
                label = ("Red" if sim.kickoff_team == 0 else "Blue") + " kick-off   " + label
        else:
            label = "WORLD MODEL + ghost" if args.ghost else "WORLD MODEL"
        draw(screen, font, big, state, score, label, match_ticks, kicking=kicking, ghost=sim.get_state() if (args.mode == "model" and args.ghost) else None, msg=msg, n_red=n, n_blue=n, limits=(args.score_limit, args.time_limit, overtime), names=player_names(n, n, args.name), jerseys=player_jerseys(n, n, args.jersey))
        pygame.display.flip()
        clock.tick(C.TICK_HZ)
        frames += 1
        if args.max_frames and frames >= args.max_frames:
            running = False
    pygame.quit()
    if args.record and rec_obs:
        np.savez_compressed(args.record, obs=np.array(rec_obs, np.float32), act=np.array(rec_act, np.int8), ep=np.array(rec_ep, np.int32), kicked=np.zeros(len(rec_obs), bool), goal=np.array(rec_goal, np.int8), n_red=n, n_blue=n, stadium=C.STADIUM_NAME, name=args.name, policy=str(args.policy or "scripted"))
        print(f"saved {args.record}: {len(rec_obs)} ticks")


if __name__ == "__main__":
    main()
