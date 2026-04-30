from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from traininghub.core.database import connect


class GpuAllocationError(RuntimeError):
    pass


@dataclass(frozen=True)
class GpuAllocation:
    gpu_ids: list[int]
    requested_gpu_count: int
    available_gpu_ids: list[int]
    leased_gpu_ids: list[int]
    strategy: str
    allow_overlap: bool

    def as_event_data(self) -> dict[str, Any]:
        return {
            "gpu_ids": self.gpu_ids,
            "requested_gpu_count": self.requested_gpu_count,
            "available_gpu_ids": self.available_gpu_ids,
            "leased_gpu_ids": self.leased_gpu_ids,
            "strategy": self.strategy,
            "allow_overlap": self.allow_overlap,
        }


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
                "memory_free_mb": int(memory_total) - int(memory_used),
                "utilization_gpu_percent": int(utilization),
                "temperature_c": int(temperature),
            }
        )
    return gpus


def disk_usage(path: Path) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}


def running_gpu_ids(database_path: Path) -> set[int]:
    return set(active_gpu_leases(database_path).keys())


def active_gpu_leases(database_path: Path) -> dict[int, str]:
    with connect(database_path) as conn:
        rows = conn.execute("SELECT job_id, worker_pid, gpu_ids FROM jobs WHERE status = 'running'").fetchall()
    leases: dict[int, str] = {}
    for row in rows:
        pid = row["worker_pid"]
        if pid and not _pid_is_running(int(pid)):
            continue
        try:
            values = json.loads(row["gpu_ids"] or "[]")
        except json.JSONDecodeError:
            values = []
        for value in values:
            leases[int(value)] = row["job_id"]
    return leases


def choose_gpu_ids(database_path: Path, requested_gpu_ids: list[int] | None = None) -> list[int]:
    return choose_gpu_allocation(database_path, requested_gpu_ids=requested_gpu_ids).gpu_ids


def choose_gpu_allocation(
    database_path: Path,
    requested_gpu_ids: list[int] | None = None,
    requested_gpu_count: int = 1,
    allow_overlap: bool = False,
    require_gpu: bool = False,
    strategy: str = "single",
) -> GpuAllocation:
    requested_gpu_count = max(0, requested_gpu_count)
    gpus = query_gpus()
    gpu_by_id = {int(gpu["index"]): gpu for gpu in gpus}
    leased_gpu_ids = sorted(running_gpu_ids(database_path))
    available = [gpu for gpu in gpus if int(gpu["index"]) not in leased_gpu_ids]
    available_gpu_ids = [int(gpu["index"]) for gpu in available]

    if requested_gpu_ids:
        gpu_ids = [int(gpu_id) for gpu_id in requested_gpu_ids]
        unknown = [gpu_id for gpu_id in gpu_ids if gpus and gpu_id not in gpu_by_id]
        if unknown:
            raise GpuAllocationError(f"Unknown GPU id(s): {', '.join(str(gpu_id) for gpu_id in unknown)}.")
        overlapping = sorted(set(gpu_ids).intersection(leased_gpu_ids))
        if overlapping and not allow_overlap:
            raise GpuAllocationError(
                f"GPU id(s) {', '.join(str(gpu_id) for gpu_id in overlapping)} are already assigned to running jobs."
            )
        return GpuAllocation(
            gpu_ids=gpu_ids,
            requested_gpu_count=len(gpu_ids),
            available_gpu_ids=available_gpu_ids,
            leased_gpu_ids=leased_gpu_ids,
            strategy=strategy,
            allow_overlap=allow_overlap,
        )

    if requested_gpu_count == 0:
        return GpuAllocation([], 0, available_gpu_ids, leased_gpu_ids, strategy, allow_overlap)
    if not gpus:
        if require_gpu:
            raise GpuAllocationError("No NVIDIA GPUs were detected for this job.")
        return GpuAllocation([], requested_gpu_count, [], leased_gpu_ids, strategy, allow_overlap)

    candidates = gpus if allow_overlap else available
    if len(candidates) < requested_gpu_count:
        raise GpuAllocationError(
            f"{requested_gpu_count} idle GPU(s) required for this job, but only {len(candidates)} are available. "
            "Wait for running GPU jobs to finish or request a smaller allocation."
        )

    selected = sorted(candidates, key=lambda gpu: (int(gpu["memory_free_mb"]), -int(gpu["index"])), reverse=True)[
        :requested_gpu_count
    ]
    gpu_ids = sorted(int(gpu["index"]) for gpu in selected)
    return GpuAllocation(
        gpu_ids=gpu_ids,
        requested_gpu_count=requested_gpu_count,
        available_gpu_ids=available_gpu_ids,
        leased_gpu_ids=leased_gpu_ids,
        strategy=strategy,
        allow_overlap=allow_overlap,
    )


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
