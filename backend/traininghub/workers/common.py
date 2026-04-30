from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from traininghub import __version__
from traininghub.core.database import connect
from traininghub.core.security import utc_now


class WorkerCancelled(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--payload", required=True)
    return parser.parse_args()


def load_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class WorkerContext:
    def __init__(self, job_id: str, job_dir: Path, database_path: Path) -> None:
        self.job_id = job_id
        self.job_dir = job_dir
        self.database_path = database_path
        self.events_path = job_dir / "events.jsonl"
        self.metrics_path = job_dir / "metrics.jsonl"
        self.cancel_path = job_dir / "cancel.requested"
        self._cancelled = False
        self._completion_message: str | None = None
        self._completion_data: dict[str, Any] = {}

    def install_signal_handlers(self) -> None:
        def _handle_signal(signum: int, _frame: object) -> None:
            self._cancelled = True
            self.cancel_path.write_text(f"signal {signum}\n", encoding="utf-8")
            raise WorkerCancelled(f"Received signal {signum}.")

        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

    def check_cancelled(self) -> None:
        if self._cancelled or self.cancel_path.exists():
            raise WorkerCancelled("Cancellation requested.")

    def event(self, event_type: str, message: str, level: str = "info", data: dict[str, Any] | None = None) -> None:
        payload = {
            "job_id": self.job_id,
            "created_at": utc_now(),
            "event_type": event_type,
            "level": level,
            "message": message,
            "data": data or {},
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        with connect(self.database_path) as conn:
            conn.execute(
                """
                INSERT INTO job_events (job_id, created_at, event_type, level, message, data_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    self.job_id,
                    payload["created_at"],
                    event_type,
                    level,
                    message,
                    json.dumps(payload["data"], sort_keys=True),
                ),
            )

    def metric(self, values: dict[str, Any]) -> None:
        payload = {"created_at": utc_now(), **values}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self.event("metric", "Metric update.", data=values)

    def finish(self, status: str, message: str, data: dict[str, Any] | None = None) -> None:
        with connect(self.database_path) as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, finished_at = ?, terminal_message = ?
                WHERE job_id = ?
                """,
                (status, utc_now(), message, self.job_id),
            )
        self.event(status, message, "error" if status == "failed" else "info", data or {})

    def set_completion_summary(self, message: str, data: dict[str, Any]) -> None:
        self._completion_message = message
        self._completion_data = data

    def completion_message(self) -> str:
        return self._completion_message or "Job completed successfully."

    def completion_data(self) -> dict[str, Any]:
        return self._completion_data

    def write_metadata(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.job_dir / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def run_command(self, command: list[str], env: dict[str, str] | None = None) -> None:
        self.event("command", "Starting command.", data={"command": command})
        with (self.job_dir / "command_stdout.log").open("ab") as stdout_handle:
            with (self.job_dir / "command_stderr.log").open("ab") as stderr_handle:
                process = subprocess.Popen(
                    command,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    env=env or os.environ.copy(),
                    cwd=str(self.job_dir),
                    start_new_session=True,
                )
                while process.poll() is None:
                    self.check_cancelled()
                    time.sleep(1)
                if process.returncode != 0:
                    raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(command)}")
        self.event("command_complete", "Command completed.", data={"command": command})

    def register_artifact(
        self,
        path: Path,
        artifact_type: str,
        display_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        checksum = sha256_file(path) if path.is_file() else sha256_text(str(path))
        artifact_id = f"{self.job_id}_{artifact_type}_{sha256_text(str(path))[:12]}"
        size_bytes = path.stat().st_size if path.is_file() else directory_size(path)
        with connect(self.database_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO artifacts (
                    artifact_id, job_id, artifact_type, display_name, path,
                    size_bytes, checksum_sha256, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    self.job_id,
                    artifact_type,
                    display_name,
                    str(path),
                    size_bytes,
                    checksum,
                    json.dumps(metadata or {}, sort_keys=True),
                    utc_now(),
                ),
            )
        artifact = {
            "artifact_id": artifact_id,
            "artifact_type": artifact_type,
            "display_name": display_name,
            "path": str(path),
            "size_bytes": size_bytes,
            "checksum_sha256": checksum,
        }
        self.event("artifact", "Artifact registered.", data=artifact)
        return artifact


def run_worker(main: Callable[[WorkerContext, dict[str, Any]], None]) -> int:
    args = parse_args()
    database_path = Path(os.environ["TRAININGHUB_DATABASE_PATH"])
    context = WorkerContext(args.job_id, Path(args.job_dir), database_path)
    context.install_signal_handlers()
    payload = load_payload(args.payload)
    try:
        context.event("worker_start", "Worker started.", data=runtime_snapshot(payload))
        main(context, payload)
        context.finish("succeeded", context.completion_message(), context.completion_data())
        return 0
    except WorkerCancelled as exc:
        context.finish("cancelled", str(exc))
        return 130
    except Exception as exc:
        context.event("worker_error", str(exc), "error", {"exception_type": type(exc).__name__})
        context.finish("failed", str(exc))
        return 1


def runtime_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "app_version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "command": sys.argv,
        "env": {
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", ""),
            "PYTORCH_CUDA_ALLOC_CONF": os.getenv("PYTORCH_CUDA_ALLOC_CONF", ""),
            "TRAININGHUB_ENABLE_REAL_WORKERS": os.getenv("TRAININGHUB_ENABLE_REAL_WORKERS", "0"),
        },
        "payload_keys": sorted(payload.keys()),
        "hardware": hardware_snapshot(),
    }


def hardware_snapshot() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {"gpus": []}
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4:
            gpus.append({"index": parts[0], "name": parts[1], "memory_total_mb": parts[2], "driver_version": parts[3]})
    return {"gpus": gpus}


def real_workers_enabled() -> bool:
    return os.getenv("TRAININGHUB_ENABLE_REAL_WORKERS", os.getenv("TRAININGHUB_ENABLE_REAL_TRAINING", "0")) == "1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def directory_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
