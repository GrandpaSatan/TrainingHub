from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Any

from traininghub.core.config import Settings
from traininghub.core.database import connect, row_to_dict, rows_to_dicts
from traininghub.core.id_utils import make_job_id, slugify
from traininghub.core.security import utc_now
from traininghub.services.datasets import get_approved_version
from traininghub.services.gpu import choose_gpu_ids
from traininghub.services.inference_run import shutdown_inference_for_training
from traininghub.services.training import validate_training_payload


JOB_PREFIXES = {
    "benchmark": "bm",
    "benchmark_mmlu": "bm",
    "benchmark_hellaswag": "bm",
    "benchmark_arc": "bm",
    "benchmark_ifeval": "bm",
    "benchmark_code": "bm",
    "quantize": "qt",
    "dataset_import": "di",
    "model_download": "md",
    "model_upload": "mu",
    "generate": "gen",
    "cleanup": "cl",
    "convert_gguf": "cv",
    "train_lora": "tr",
    "train_qlora": "tr",
    "train_full": "tr",
    "extract_capability": "ec",
    "align_capability": "ac",
}

WORKER_MODULES = {
    "benchmark": "traininghub.workers.run_benchmark_math",
    "benchmark_mmlu": "traininghub.workers.run_benchmark_mmlu",
    "benchmark_hellaswag": "traininghub.workers.run_benchmark_hellaswag",
    "benchmark_arc": "traininghub.workers.run_benchmark_arc",
    "benchmark_ifeval": "traininghub.workers.run_benchmark_ifeval",
    "benchmark_code": "traininghub.workers.run_benchmark_code",
    "quantize": "traininghub.workers.run_quantize_gguf",
    "dataset_import": "traininghub.workers.run_import_dataset",
    "model_download": "traininghub.workers.run_download_model",
    "model_upload": "traininghub.workers.run_upload_model_hf",
    "generate": "traininghub.workers.run_generate_examples",
    "cleanup": "traininghub.workers.run_cleanup",
    "convert_gguf": "traininghub.workers.run_convert_gguf",
    "train_lora": "traininghub.workers.run_train_lora",
    "train_qlora": "traininghub.workers.run_train_lora",
    "train_full": "traininghub.workers.run_train_full",
    "extract_capability": "traininghub.workers.run_extract_capability",
    "align_capability": "traininghub.workers.run_align_capability",
}

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
TRAINING_JOB_TYPES = {"train_lora", "train_qlora", "train_full"}
DEFAULT_PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"


class JobValidationError(ValueError):
    pass


def list_jobs(database_path: Path) -> list[dict[str, Any]]:
    with connect(database_path) as conn:
        rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 100").fetchall()
    jobs = rows_to_dicts(rows)
    for job in jobs:
        job["payload"] = json.loads(job.pop("payload_json") or "{}")
        job["gpu_ids"] = json.loads(job["gpu_ids"] or "[]")
    return jobs


def get_job(database_path: Path, job_id: str) -> dict[str, Any] | None:
    with connect(database_path) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    job = row_to_dict(row)
    if job:
        job["payload"] = json.loads(job.pop("payload_json") or "{}")
        job["gpu_ids"] = json.loads(job["gpu_ids"] or "[]")
    return job


def create_and_start_job(settings: Settings, job_type: str, slug: str, payload: dict[str, Any]) -> dict[str, Any]:
    if job_type not in WORKER_MODULES:
        raise JobValidationError(f"Unsupported job type: {job_type}")
    _validate_job_request(settings, job_type, payload)
    prefix = JOB_PREFIXES[job_type]
    job_id = make_job_id(prefix, slug)
    work_dir = settings.data_root / "jobs" / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    payload_path = work_dir / "payload.json"
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    gpu_ids = choose_gpu_ids(settings.database_path, payload.get("gpu_ids"))
    now = utc_now()
    with connect(settings.database_path) as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, job_type, status, slug, payload_json, work_dir, worker_module,
                worker_pid, gpu_ids, created_at, started_at, finished_at, terminal_message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                job_type,
                "queued",
                slugify(slug, "job"),
                json.dumps(payload, sort_keys=True),
                str(work_dir),
                WORKER_MODULES[job_type],
                None,
                json.dumps(gpu_ids),
                now,
                None,
                None,
                "",
            ),
        )
        conn.execute(
            """
            INSERT INTO job_events (job_id, created_at, event_type, level, message, data_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (job_id, now, "queued", "info", "Job queued.", json.dumps({"job_type": job_type})),
        )
    if job_type in TRAINING_JOB_TYPES:
        shutdown_result = shutdown_inference_for_training()
        with connect(settings.database_path) as conn:
            conn.execute(
                """
                INSERT INTO job_events (job_id, created_at, event_type, level, message, data_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    utc_now(),
                    "inference_shutdown",
                    "info",
                    "Inference stopped for training.",
                    json.dumps(shutdown_result, sort_keys=True),
                ),
            )
    _spawn_worker(settings, job_id, WORKER_MODULES[job_type], payload_path, work_dir, gpu_ids)
    return get_job(settings.database_path, job_id) or {"job_id": job_id}


def _validate_job_request(settings: Settings, job_type: str, payload: dict[str, Any]) -> None:
    if job_type in TRAINING_JOB_TYPES:
        try:
            validate_training_payload(settings.database_path, job_type, payload)
        except ValueError as exc:
            raise JobValidationError(str(exc)) from exc
    if job_type == "benchmark" and payload.get("dataset_id"):
        approved_version = get_approved_version(settings.database_path, payload["dataset_id"])
        if not approved_version:
            raise JobValidationError("Dataset must be approved before use as a benchmark holdout.")
        payload["dataset_version_id"] = approved_version["version_id"]
        payload["dataset_jsonl_path"] = approved_version["jsonl_path"]
    if job_type in {"extract_capability", "align_capability"}:
        try:
            from traininghub.services.capability_transfers import validate_capability_job_payload

            validate_capability_job_payload(settings, job_type, payload)
        except ValueError as exc:
            raise JobValidationError(str(exc)) from exc


def _spawn_worker(
    settings: Settings,
    job_id: str,
    worker_module: str,
    payload_path: Path,
    work_dir: Path,
    gpu_ids: list[int],
) -> None:
    env = _worker_env(settings, job_id, work_dir, gpu_ids)
    stdout_path = work_dir / "stdout.log"
    stderr_path = work_dir / "stderr.log"
    stdout_handle = stdout_path.open("ab")
    stderr_handle = stderr_path.open("ab")
    command = [
        settings.worker_python,
        "-m",
        worker_module,
        "--job-id",
        job_id,
        "--job-dir",
        str(work_dir),
        "--payload",
        str(payload_path),
    ]
    process = subprocess.Popen(
        command,
        stdout=stdout_handle,
        stderr=stderr_handle,
        env=env,
        cwd=str(settings.app_root),
        start_new_session=True,
    )
    with connect(settings.database_path) as conn:
        conn.execute(
            "UPDATE jobs SET status = 'running', started_at = ?, worker_pid = ? WHERE job_id = ?",
            (utc_now(), process.pid, job_id),
        )
        conn.execute(
            """
            INSERT INTO job_events (job_id, created_at, event_type, level, message, data_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                utc_now(),
                "started",
                "info",
                "Worker process started.",
                json.dumps({"pid": process.pid, "command": command, "gpu_ids": gpu_ids}),
            ),
        )


def _worker_env(settings: Settings, job_id: str, work_dir: Path, gpu_ids: list[int]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(settings.app_root / "backend") + os.pathsep + env.get("PYTHONPATH", ""),
            "TRAININGHUB_DATABASE_PATH": str(settings.database_path),
            "TRAININGHUB_DATA_ROOT": str(settings.data_root),
            "TRAININGHUB_JOB_ID": job_id,
            "TRAININGHUB_JOB_DIR": str(work_dir),
            "TRAININGHUB_GPU_IDS": ",".join(str(gpu_id) for gpu_id in gpu_ids),
            "CUDA_VISIBLE_DEVICES": ",".join(str(gpu_id) for gpu_id in gpu_ids),
            "TRAININGHUB_ENABLE_REAL_WORKERS": "1" if settings.real_workers_enabled else "0",
            "PYTORCH_CUDA_ALLOC_CONF": env.get("PYTORCH_CUDA_ALLOC_CONF", DEFAULT_PYTORCH_CUDA_ALLOC_CONF),
        }
    )
    return env


def cancel_job(settings: Settings, job_id: str) -> dict[str, Any] | None:
    job = get_job(settings.database_path, job_id)
    if not job:
        return None
    if job["status"] in TERMINAL_STATUSES:
        return job
    work_dir = Path(job["work_dir"])
    (work_dir / "cancel.requested").write_text("cancelled by user\n", encoding="utf-8")
    pid = job.get("worker_pid")
    if pid:
        try:
            os.killpg(int(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    now = utc_now()
    with connect(settings.database_path) as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = 'cancelled', finished_at = ?, terminal_message = ?
            WHERE job_id = ?
            """,
            (now, "Cancelled by user.", job_id),
        )
        conn.execute(
            """
            INSERT INTO job_events (job_id, created_at, event_type, level, message, data_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (job_id, now, "cancelled", "warning", "Cancellation requested.", "{}"),
        )
    return get_job(settings.database_path, job_id)


async def stream_job_events(database_path: Path, job_id: str):
    last_event_id = 0
    while True:
        with connect(database_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM job_events
                WHERE job_id = ? AND id > ?
                ORDER BY id ASC
                """,
                (job_id, last_event_id),
            ).fetchall()
            status_row = conn.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        for row in rows:
            last_event_id = row["id"]
            event = row_to_dict(row) or {}
            event["data"] = json.loads(event.pop("data_json") or "{}")
            yield f"event: {event['event_type']}\ndata: {json.dumps(event, sort_keys=True)}\n\n"
        if status_row is None or (status_row["status"] in TERMINAL_STATUSES and not rows):
            yield "event: close\ndata: {}\n\n"
            return
        await asyncio.sleep(1)
