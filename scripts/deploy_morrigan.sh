#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${TRAININGHUB_MORRIGAN_REMOTE:-jhernandez@10.0.65.20}"
SSH_KEY="${TRAININGHUB_MORRIGAN_KEY:-$HOME/.ssh/id_ed25519_morrigan}"
REMOTE_APP="${TRAININGHUB_MORRIGAN_APP_ROOT:-/home/jhernandez/traininghub}"
REMOTE_DATA="${TRAININGHUB_MORRIGAN_DATA_ROOT:-/home/jhernandez/traininghub-data}"
INFERENCE_RUNTIME="${TRAININGHUB_INFERENCE_RUNTIME:-transformers}"
REAL_WORKERS="${TRAININGHUB_ENABLE_REAL_WORKERS:-1}"
INSTALL_LLAMA_CPP="${TRAININGHUB_INSTALL_LLAMA_CPP:-0}"
CLEANUP_LLAMA="${1:-}"
REMOTE_ADMIN_PASSWORD_EXPORT=""
if [[ -n "${TRAININGHUB_ADMIN_PASSWORD:-}" ]]; then
  REMOTE_ADMIN_PASSWORD_EXPORT="export TRAININGHUB_ADMIN_PASSWORD=$(printf '%q' "$TRAININGHUB_ADMIN_PASSWORD")"
fi
REMOTE_INFERENCE_RUNTIME_EXPORT="export TRAININGHUB_INFERENCE_RUNTIME=$(printf '%q' "$INFERENCE_RUNTIME")"
REMOTE_REAL_WORKERS_EXPORT="export TRAININGHUB_ENABLE_REAL_WORKERS=$(printf '%q' "$REAL_WORKERS")"

"$ROOT_DIR/scripts/build_frontend.sh"

rsync -az --delete \
  --exclude ".venv" \
  --exclude ".traininghub-data" \
  --exclude "frontend/node_modules" \
  -e "ssh -i $SSH_KEY" \
  "$ROOT_DIR/" "$REMOTE:$REMOTE_APP/"

ssh -i "$SSH_KEY" "$REMOTE" "set -euo pipefail
cd '$REMOTE_APP'
mkdir -p '$REMOTE_DATA'
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
if [[ '$INSTALL_LLAMA_CPP' == '1' ]]; then
  python -m pip install 'llama-cpp-python>=0.3.9'
fi
if [[ '$CLEANUP_LLAMA' == '--cleanup-llama' ]]; then
  TRAININGHUB_DATA_ROOT='$REMOTE_DATA' bash scripts/morrigan_stop_llama_server.sh
fi
if [[ -f traininghub.pid ]]; then
  kill \$(cat traininghub.pid) 2>/dev/null || true
  rm -f traininghub.pid
fi
for pid in \$(lsof -tiTCP:7860 -sTCP:LISTEN 2>/dev/null || true); do
  kill "\$pid" 2>/dev/null || true
done
sleep 1
export TRAININGHUB_APP_ROOT='$REMOTE_APP'
export TRAININGHUB_DATA_ROOT='$REMOTE_DATA'
export TRAININGHUB_DATABASE_PATH='$REMOTE_DATA/traininghub.sqlite3'
export TRAININGHUB_WORKER_PYTHON='$REMOTE_APP/.venv/bin/python'
export PYTHONPATH='$REMOTE_APP/backend'
$REMOTE_ADMIN_PASSWORD_EXPORT
$REMOTE_INFERENCE_RUNTIME_EXPORT
$REMOTE_REAL_WORKERS_EXPORT
nohup .venv/bin/uvicorn traininghub.main:app --host 0.0.0.0 --port 7860 > traininghub.log 2>&1 &
echo \$! > traininghub.pid
sleep 2
if ! kill -0 \$(cat traininghub.pid) 2>/dev/null; then
  tail -80 traininghub.log
  exit 1
fi
curl -fsS http://127.0.0.1:7860/api/health
"

echo "TrainingHub deployed to http://10.0.65.20:7860"
