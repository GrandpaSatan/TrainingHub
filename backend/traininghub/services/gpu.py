from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from traininghub.core.database import connect


def query_gpus() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    gpus: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        index, name, memory_total, memory_used, utilization, temperature = parts
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mb": int(memory_total),
                "memory_used_mb": int(memory_used),
                "utilization_gpu_percent": int(utilization),
                "temperature_c": int(temperature),
            }
        )
    return gpus


def disk_usage(path: Path) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}


def running_gpu_ids(database_path: Path) -> set[int]:
    with connect(database_path) as conn:
        rows = conn.execute("SELECT gpu_ids FROM jobs WHERE status = 'running'").fetchall()
    gpu_ids: set[int] = set()
    for row in rows:
        try:
            values = json.loads(row["gpu_ids"] or "[]")
        except json.JSONDecodeError:
            values = []
        gpu_ids.update(int(value) for value in values)
    return gpu_ids


def choose_gpu_ids(database_path: Path, requested_gpu_ids: list[int] | None = None) -> list[int]:
    if requested_gpu_ids:
        return requested_gpu_ids
    gpus = query_gpus()
    leased = running_gpu_ids(database_path)
    available = [gpu["index"] for gpu in gpus if gpu["index"] not in leased]
    if available:
        return [available[0]]
    if gpus:
        return [gpus[0]["index"]]
    return []

