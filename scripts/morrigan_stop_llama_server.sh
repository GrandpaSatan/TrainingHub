#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${TRAININGHUB_DATA_ROOT:-/home/jhernandez/traininghub-data}"
MANIFEST_DIR="$DATA_ROOT/cleanup/$(date -u +cl_%Y%m%d_%H%M%S_immediate-llama-server)"
QUARANTINE_DIR="$MANIFEST_DIR/quarantine"
MODEL_PATH="/home/jhernandez/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-MXFP4_MOE.gguf"
PORT="8080"

mkdir -p "$QUARANTINE_DIR"

{
  echo "{"
  echo "  \"created_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"port\": $PORT,"
  echo "  \"process_before\": $(python3 - <<'PY'
import json
import subprocess
commands = [
    ["bash", "-lc", "ss -ltnp 'sport = :8080' || true"],
    ["bash", "-lc", "lsof -nP -iTCP:8080 -sTCP:LISTEN || true"],
]
outputs = []
for command in commands:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5)
        outputs.append({"command": command, "stdout": result.stdout, "stderr": result.stderr})
    except Exception as exc:
        outputs.append({"command": command, "error": str(exc)})
print(json.dumps(outputs))
PY
),"
  echo "  \"actions\": []"
  echo "}"
} > "$MANIFEST_DIR/manifest.before.json"

PIDS="$(lsof -tiTCP:$PORT -sTCP:LISTEN 2>/dev/null || true)"
if [[ -n "$PIDS" ]]; then
  while read -r pid; do
    [[ -z "$pid" ]] && continue
    kill "$pid" 2>/dev/null || true
  done <<< "$PIDS"
  sleep 2
fi

if [[ -f "$MODEL_PATH" ]]; then
  mv "$MODEL_PATH" "$QUARANTINE_DIR/"
  PARENT_DIR="$(dirname "$MODEL_PATH")"
  rmdir "$PARENT_DIR" 2>/dev/null || true
fi

{
  echo "{"
  echo "  \"created_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"stopped_pids\": $(python3 - <<PY
import json
print(json.dumps("""$PIDS""".split()))
PY
),"
  echo "  \"quarantined_model\": \"$QUARANTINE_DIR/$(basename "$MODEL_PATH")\","
  echo "  \"port_after\": $(python3 - <<'PY'
import json
import subprocess
result = subprocess.run(["bash", "-lc", "ss -ltnp 'sport = :8080' || true"], capture_output=True, text=True)
print(json.dumps(result.stdout))
PY
)"
  echo "}"
} > "$MANIFEST_DIR/manifest.after.json"

echo "$MANIFEST_DIR"

