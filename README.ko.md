# HaxDrill (한국어)

HaxBall **Hot Classic** 맵(G.Buffon Hot Classic .hbs 그대로: 840×400 피치, 공 r=6.4, 선수 invMass 0.3·가속 0.105·킥 5.7, 포스트 r=5.5)을
물리까지 그대로 재현한 1v1 ~ NvN 게임 + 그 위에서 돌아가는 **self-play 강화학습 상대**와 **월드모델**.

```
action = 키보드 (8방향 + 킥) → 18가지 이산 액션
state  = [red_0..red_{n-1} | ball | blue_0..blue_{n-1}] 각각 (x, y, vx, vy)
```

| 구성요소 | 역할 |
|---|---|
| `haxdrill/sim.py` / `vecsim.py` / `web/physics.js` | HaxBall 디스크 물리. 셋이 소수점까지 일치(검증 스크립트 포함) |
| `haxdrill/rl.py` | PPO self-play (과거 스냅샷 풀과 대전, 블루는 좌우 반전 관측, 한 네트워크가 모든 선수 조종, `--n`으로 NvN) |
| `haxdrill/model.py`, `train.py`, `evaluate.py` | 월드모델(1v1): 상태+키 → 다음 상태 |
| `haxdrill/play.py` | pygame 게임 (`--opponent scripted|rl`, `--n 2`, `--mode model`, `--record`) |
| `haxdrill/collect.py`, `video.py`, `render.py` | 데이터 생성 / 영상→궤적 / npz→mp4 |
| `haxdrill/server.py`, `retrain.py`, `export.py` | 웹 로그 수집 서버, 재학습 루프, JSON 가중치 내보내기 |
| `web/` | 브라우저 버전 (물리 + RL 상대 JS 추론, 플레이 로그 전송) |
| `stadiums/hot_classic.hbs` | 원본 맵 파일. 다른 .hbs도 `--stadium 경로`로 바로 로드 |

## 설치

```bash
uv venv .venv && uv pip install -p .venv/bin/python -e .
# NVIDIA 드라이버가 CUDA 12.9 이하면:
uv pip install -p .venv/bin/python --reinstall torch --index-url https://download.pytorch.org/whl/cu128
```

## 플레이

```bash
.venv/bin/python -m haxdrill.play                                   # 스크립트 상대
.venv/bin/python -m haxdrill.play --policy runs/ppo_n1.pt           # self-play RL 상대 (1v1)
.venv/bin/python -m haxdrill.play --n 2 --policy runs/ppo_n2.pt     # 2v2: 팀원 1 + 상대 2 모두 RL
.venv/bin/python -m haxdrill.play --mode model --ckpt runs/wm.pt --ghost   # 월드모델 안에서 플레이(윤곽선 = 실제 물리)
```
내 플레이 로그는 sim 모드에서 **자동으로** `data/play/<날짜_시간>_<n>v<n>.npz`에 저장된다(`--record 경로`로 지정, `--no-record`로 끔).
방향키/WASD 이동, X 또는 Space 킥(HaxBall처럼 떼었다 눌러야 다시 킥), R 리셋, Esc 종료. `--stadium classic`으로 원조 Classic.

경기 규칙: 기본 **3골 선취 또는 3분**(`--score-limit 3 --time-limit 3`, 0이면 무제한). 시간 종료 시 동점이면 골든골 연장.
웹에서는 상단 입력칸 또는 `index.html?goals=3&time=3`.
킥오프: 시작은 빨강, 실점 팀이 킥오프. 공을 건드릴 때까지 센터라인 + 수비팀 쪽 반원 장벽(원작 kickOffBarrier)이 선수만 막는다.
선수 디스크에는 등번호, 아래에 이름. 내 닉네임/등번호는 `--name player --jersey 7`(등번호는 글자·숫자 2자까지), 봇은 bot1, bot2, ...
웹은 시작 화면(닉네임, 등번호, 난이도, 인원, 목표 골, 시간)에서 정하고 "게임 시작". 공개 빌드는 당분간 1v1만 활성(2v2·3v3 버튼은 TODO). URL로도 `?nick=..&jersey=7&n=2&goals=3&time=3`.

## 상대 학습 (self-play PPO)

```bash
.venv/bin/python -m haxdrill.rl --n 1 --iters 1500 --envs 2048     # 4090 + 16코어 기준 약 0.7초/iter
.venv/bin/python -m haxdrill.rl --n 2 --iters 1500 --envs 1024     # 2v2
.venv/bin/python -m haxdrill.rl --n 3 --iters 1500 --envs 1024     # 3v3 (웹 "3 vs 3"에 필요한 policy_n3.json)
```
로그의 `vs pool: red X blue Y goals/min`은 현재 정책(red)이 과거 스냅샷(blue)을 상대로 넣고/먹은 골 수.

보상 = 골 ±1 + 공을 상대 골대 쪽으로 보낸 진행(potential) + 공 접근 + 아래 shaping(기본값은 n>1일 때만 켜짐, 모두 플래그로 조절):
`--pass-bonus 0.1`(팀원이 찬 공을 3초 내 다시 차면 둘 다), `--spread 0.0005`(팀원끼리 60유닛 안에 뭉치면 틱당 페널티),
`--marking 0.0005`(공이 우리 진영일 때 각 수비수가 배정된 상대와 80유닛 이상 떨어진 만큼 페널티 → 대인수비),
`--beat-bonus 0.02`(공을 가진 채 앞에 있던 상대를 뒤로 넘기면, 소유당 1회 → 드리블; 1v1에도 적용),
`--wall-bonus 0.1`(내가 찬 공이 사이드/골라인에 튕긴 뒤 2.5초 안에 내가 다시 잡고 40유닛 이상 전진했으면 → 벽 바운드 제치기),
`--shot-bonus 0.15`(킥 직후 공 궤적이 골문 안을 향하면 → 밀고 가는 대신 조준해서 차기; 사이드라인에 1~2번 튕겨 들어가는 뱅크슛도 인정).

**빈 골대 테스트** `python -m haxdrill.emptynet runs/ppo_n1.pt`: 상대를 치운 상태에서 세 거리 × 40회 득점률·킥 수·유효슛 수.
"슛을 안 하고 밀고 간다" 같은 보상 설계 문제를 바로 드러낸다.

사람 로그를 봇 자신의 스타일에 넣기(AlphaGo의 SL→RL): `--bc 로그`로 초기화하거나, 이미 강한 정책을 이어서 학습할 때
`--bc-aux 'data/play/*.npz' --bc-coef 0.05`로 PPO 업데이트마다 모방 손실을 섞어 사람 기술이 self-play 중에 지워지지 않게 한다.
`retrain.py`는 이 둘(모방 손실 + 사람 모방 정책을 상대 풀에 고정)을 기본으로 켠다.

**실제 지표**: 스크립트 상대 점수가 아니라 사람 vs 봇 전적. 로그에 골 이벤트가 기록되므로
`.venv/bin/python -m haxdrill.stats` 로 모드별 "you X - Y bot", 분당 득실을 본다.
1v1은 50 iter마다 스크립트 상대와 3경기(각 60초) 평가 결과도 찍힌다. `--resume`로 이어서 학습, `--bc 로그.npz`로 사람 로그 모방 초기화.

## 난이도와 리그 학습 (medium 봇)

- **easy** = `runs/ppo_n{n}.pt` / `web/policy_n{n}.json` (단순 self-play 봇).
- **medium** = 리그 학습 결과 `runs/league/main_latest.pt` / `web/policy_n1_medium.json`. 웹 상단 "난이도", pygame `--level medium`.

```bash
nohup .venv/bin/python -m haxdrill.league --n 1 --hours 24 --out-dir runs/league > runs/league.out 2>&1 &
tail -f runs/league/league.log          # cycle마다 main/exploiter/eval 한 줄씩
.venv/bin/python -m haxdrill.league --resume-dir runs/league --hours 12      # 이어서
\cp -f web/policy_n1_medium.json ~/Desktop/3dvision/projects/mkjjang3598.github.io/haxdrill/   # 사이트 반영
```
리그 구조(AlphaStar-lite): 메인은 과거 스냅샷 + exploiter + 사람 모방봇과 학습, shaping 보상은 3000 iter에 걸쳐 0으로.
exploiter는 매 cycle 메인에서 복사해 **메인만 상대로** 학습하고, 3분 맞대결에서 메인을 이기면 풀에 고정된다(최대 8개).
관측은 v2(기하 피처) + 최근 4 결정 프레임 스택(짧은 기억). `web/policy_n1_medium.json`은 cycle마다 자동 갱신.

## 월드모델 (선택, 현재 비활성)

게임과 봇은 월드모델을 쓰지 않는다(물리 엔진 + RL 정책만). 연구용으로 남겨둔 것:

```bash
.venv/bin/python -m haxdrill.collect --ticks 4000000 --policy runs/ppo_n1.pt --out data/sim.npz   # RL 상대와의 경기 데이터
.venv/bin/python -m haxdrill.train data/sim.npz data/me_*.npz --out runs/wm.pt --steps 30000 --hidden 256 --layers 3
.venv/bin/python -m haxdrill.evaluate runs/wm.pt --episodes 100 --horizon 300
```

## 웹 배포 + 사용자 로그로 계속 개선 (AlphaGo 루프)

```bash
.venv/bin/python -m haxdrill.export --config web/config.json          # 맵 상수 (.hbs 바꾸면 재실행)
.venv/bin/python -m haxdrill.export runs/ppo_n1.pt web/policy_n1.json # RL 상대
.venv/bin/python -m haxdrill.export runs/wm.pt web/wm.json            # (선택) 월드모델
.venv/bin/python -m haxdrill.server --port 8000                        # http://localhost:8000 , 로그 → data/web/*.npz
```
`export`는 `.json`과 함께 같은 내용의 `.js`(window.HAXDRILL.files에 등록)를 만든다. 페이지는 `<script>` 태그로 이 `.js`들을 읽으므로
`index.html`을 더블클릭해서 파일로 열어도 동작한다(fetch 없음). 사이트에 올릴 때는 `.js`만 있으면 된다.
`web/`는 정적 파일이라 GitHub Pages 등 아무 데나 올려도 되고, 로그만 서버로 보낸다
(`index.html?log=https://내서버/log`, 또는 `window.HAXDRILL_LOG_URL`). 서버가 없으면 "로그 다운로드" 버튼으로 JSON 저장.
`index.html?policy=policy_n2.json`처럼 다른 정책 파일을 지정하면 2v2로 열린다.

로그가 쌓이면:
```bash
.venv/bin/python -m haxdrill.retrain --n 1 --iters 300     # data/play + data/web 로그 → 1v1 봇 갱신 (web/policy_n1.json)
.venv/bin/python -m haxdrill.retrain --n 2 --iters 200     # 2v2 (2v2 로그만 사용)
```
= 사람 로그를 모방한 정책 `runs/human_bc.pt` 생성 → 그 정책을 상대 풀에 **고정 멤버**로 넣고(`--pool-extra`) self-play 이어서 학습
→ `web/policy_n{n}.json` 갱신. (`--wm`을 주면 1v1 월드모델도 재학습.)
사람 로그의 역할: 강함 자체는 self-play에서 나오고, 로그는 "사람이 실제로 쓰는 전략"에 상대가 뚫리지 않게 만드는 데 쓰인다
(self-play만 하면 자기들끼리만 통하는 전략에 과적합될 수 있음). 결과물은 `~/Desktop/3dvision/projects/mkjjang3598.github.io/haxdrill/`로 복사해서 푸시.

## 플레이 영상에서 데이터 추출 (선택)

```bash
.venv/bin/python -m haxdrill.video clip.mp4 --probe probe.png          # 격자에서 피치 모서리 픽셀 좌표 읽기
.venv/bin/python -m haxdrill.video clip.mp4 data/clip.npz --field X0 Y0 X1 Y1 [--red-team right] [--debug dbg.mp4]
```
HSV 색 분리로 공(핫핑크)·선수 위치를 찾고 60Hz로 리샘플, 키 입력은 Savitzky-Golay + Viterbi 역동역학으로 복원.
합성 영상 기준 위치 오차 0.06유닛, 방향 복원 76%(축별 85%). 공 색이 다르면 `video.py`의 HSV 범위를 바꾼다.

## 아직 안 된 것 / 주의

- 월드모델은 1v1 전용. NvN은 RL 정책만 지원.
- Hot Classic .hbs의 골대 뒤 보이지 않는 보조 세그먼트(ballArea 클러터)는 생략. 네트·골라인·포스트 충돌은 그대로.
