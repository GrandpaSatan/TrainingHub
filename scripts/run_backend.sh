#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt

export TRAININGHUB_APP_ROOT="${TRAININGHUB_APP_ROOT:-$ROOT_DIR}"
export TRAININGHUB_DATA_ROOT="${TRAININGHUB_DATA_ROOT:-$ROOT_DIR/.traininghub-data}"
export PYTHONPATH="$ROOT_DIR/backend:${PYTHONPATH:-}"

exec uvicorn traininghub.main:app --host 0.0.0.0 --port "${TRAININGHUB_PORT:-7860}"

