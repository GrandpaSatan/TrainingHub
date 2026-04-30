#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${TRAININGHUB_BASE_URL:-http://10.0.65.20:7860}"
USERNAME="${TRAININGHUB_ADMIN_USERNAME:-admin}"
PASSWORD="${TRAININGHUB_ADMIN_PASSWORD:-traininghub}"
COOKIE_JAR="$(mktemp)"
CSV_FILE="$(mktemp --suffix=.csv)"

cleanup() {
  rm -f "$COOKIE_JAR" "$CSV_FILE"
}
trap cleanup EXIT

curl -fsS -c "$COOKIE_JAR" -H "Content-Type: application/json" \
  -d "{\"username\":\"$USERNAME\",\"password\":\"$PASSWORD\"}" \
  "$BASE_URL/api/auth/login" >/dev/null

cat > "$CSV_FILE" <<'CSV'
id,system,prompt,response,final_answer,category,difficulty,source,split,tags,notes
row_001,You are a careful math tutor.,What is 1 + 1?,1 + 1 = 2. The final answer is 2.,2,arithmetic,easy,smoke,train,math,smoke
row_002,You are a careful math tutor.,What is 2 + 2?,2 + 2 = 4. The final answer is 4.,4,arithmetic,easy,smoke,train,math,smoke
row_003,You are a careful math tutor.,What is 3 + 3?,3 + 3 = 6. The final answer is 6.,6,arithmetic,easy,smoke,train,math,smoke
row_004,You are a careful math tutor.,What is 4 + 4?,4 + 4 = 8. The final answer is 8.,8,arithmetic,easy,smoke,train,math,smoke
row_005,You are a careful math tutor.,What is 5 + 5?,5 + 5 = 10. The final answer is 10.,10,arithmetic,easy,smoke,train,math,smoke
row_006,You are a careful math tutor.,What is 6 + 6?,6 + 6 = 12. The final answer is 12.,12,arithmetic,easy,smoke,train,math,smoke
row_007,You are a careful math tutor.,What is 7 + 7?,7 + 7 = 14. The final answer is 14.,14,arithmetic,easy,smoke,train,math,smoke
row_008,You are a careful math tutor.,What is 8 + 8?,8 + 8 = 16. The final answer is 16.,16,arithmetic,easy,smoke,validation,math,smoke
row_009,You are a careful math tutor.,What is 9 + 9?,9 + 9 = 18. The final answer is 18.,18,arithmetic,easy,smoke,holdout,math,smoke
row_010,You are a careful math tutor.,What is 10 + 10?,10 + 10 = 20. The final answer is 20.,20,arithmetic,easy,smoke,holdout,math,smoke
CSV

UPLOAD_RESPONSE="$(curl -fsS -b "$COOKIE_JAR" -F "file=@$CSV_FILE" -F "dataset_type=math_sft" -F "title=Smoke Math" -F "slug=smoke-math" "$BASE_URL/api/datasets/upload")"
DATASET_ID="$(python3 - <<PY
import json
print(json.loads("""$UPLOAD_RESPONSE""")["dataset_id"])
PY
)"
curl -fsS -b "$COOKIE_JAR" -X POST "$BASE_URL/api/datasets/$DATASET_ID/approve" >/dev/null
curl -fsS -b "$COOKIE_JAR" -H "Content-Type: application/json" \
  -d "{\"model_slug\":\"lfm25-12b-base\",\"benchmarks\":[\"gsm8k\"],\"limit\":10,\"dry_run\":true}" \
  "$BASE_URL/api/jobs/benchmark" >/dev/null

TRAIN_RESPONSE="$(curl -fsS -b "$COOKIE_JAR" -H "Content-Type: application/json" \
  -d "{\"model_slug\":\"lfm25-12b-base\",\"dataset_id\":\"$DATASET_ID\",\"mode\":\"lora\",\"preset\":\"smoke\",\"output_name\":\"morrigan-smoke-lora\",\"max_steps\":1,\"dry_run\":true}" \
  "$BASE_URL/api/jobs/fine-tune")"
TRAIN_JOB_ID="$(python3 - <<PY
import json
print(json.loads("""$TRAIN_RESPONSE""")["job_id"])
PY
)"
for _ in $(seq 1 50); do
  STATUS="$(curl -fsS -b "$COOKIE_JAR" "$BASE_URL/api/jobs" | python3 -c 'import json, sys; job_id = sys.argv[1]; jobs = json.load(sys.stdin); job = next((item for item in jobs if item["job_id"] == job_id), {}); print(job.get("status", "missing"))' "$TRAIN_JOB_ID")"
  if [[ "$STATUS" == "succeeded" ]]; then
    break
  fi
  if [[ "$STATUS" == "failed" || "$STATUS" == "cancelled" ]]; then
    echo "Training smoke ended with status $STATUS" >&2
    exit 1
  fi
  sleep 0.2
done

echo "Acceptance smoke requests submitted."
