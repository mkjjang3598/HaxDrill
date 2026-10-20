#!/usr/bin/env bash
# Collect HaxDrill play logs from the public site on this PC.
#   1) ngrok account: https://dashboard.ngrok.com  -> copy your authtoken
#      ngrok config add-authtoken <TOKEN>
#   2) claim your free static domain (dashboard > Domains), e.g. something.ngrok-free.app
#   3) bash scripts/setup_logging.sh something.ngrok-free.app
# Installs two user services (log server on 127.0.0.1:8090 + ngrok tunnel) that start at boot,
# and writes web/endpoint.js so the game posts logs there. Logs land in data/web/*.npz.
set -euo pipefail
DOMAIN="${1:?usage: setup_logging.sh <your-static-domain.ngrok-free.app>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT=8090
NGROK="$(command -v ngrok || echo "$HOME/.local/bin/ngrok")"
UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNIT_DIR" "$ROOT/data/web"

cat > "$UNIT_DIR/haxdrill-logs.service" <<UNIT
[Unit]
Description=HaxDrill play-log server
After=network-online.target
[Service]
WorkingDirectory=$ROOT
ExecStart=$ROOT/.venv/bin/python -m haxdrill.server --log-only --host 127.0.0.1 --port $PORT --out $ROOT/data/web
Restart=always
RestartSec=5
[Install]
WantedBy=default.target
UNIT

cat > "$UNIT_DIR/haxdrill-tunnel.service" <<UNIT
[Unit]
Description=HaxDrill ngrok tunnel ($DOMAIN)
After=network-online.target haxdrill-logs.service
[Service]
ExecStart=$NGROK http --url=https://$DOMAIN $PORT --log=stdout
Restart=always
RestartSec=10
[Install]
WantedBy=default.target
UNIT

systemctl --user daemon-reload
systemctl --user enable --now haxdrill-logs.service haxdrill-tunnel.service
loginctl enable-linger "$USER" 2>/dev/null || echo "note: run 'sudo loginctl enable-linger $USER' so the services also run when you are logged out"

printf '// Public URL of the log server (set by scripts/setup_logging.sh). Empty = logs stay in the browser.\nwindow.HAXDRILL_LOG_URL = "https://%s";\n' "$DOMAIN" > "$ROOT/web/endpoint.js"
"$ROOT/.venv/bin/python" -m haxdrill.webstamp "$ROOT/web/index.html" >/dev/null

echo "waiting for the tunnel..."
for i in $(seq 1 20); do
  if curl -fsS -H "ngrok-skip-browser-warning: 1" "https://$DOMAIN/health" >/dev/null 2>&1; then
    echo "OK: https://$DOMAIN/health reachable"
    echo "Now copy web/endpoint.js and web/index.html into the site folder and push:"
    echo "  cp -f $ROOT/web/endpoint.js $ROOT/web/index.html ~/Desktop/3dvision/projects/mkjjang3598.github.io/haxdrill/"
    exit 0
  fi
  sleep 2
done
echo "tunnel not reachable yet; check: systemctl --user status haxdrill-tunnel haxdrill-logs"
exit 1
