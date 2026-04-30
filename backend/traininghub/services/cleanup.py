from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Any

from traininghub.core.database import connect, row_to_dict, rows_to_dicts
from traininghub.core.id_utils import make_job_id
from traininghub.core.security import utc_now


IMMEDIATE_MODEL_PATH = Path("/home/jhernandez/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-MXFP4_MOE.gguf")
SCAN_ROOTS = [
    Path("/home/jhernandez/fine-tuning"),
    Path("/home/jhernandez/models"),
    Path("/home/jhernandez/llama.cpp/models"),
]


def scan_cleanup(database_path: Path, data_root: Path, include_immediate: bool = True) -> dict[str, Any]:
    manifest_id = make_job_id("cl", "cleanup-scan")
    quarantine_dir = data_root / "cleanup" / manifest_id / "quarantine"
    items: list[dict[str, Any]] = []
    if include_immediate:
        process = process_on_port(8080)
        if process:
            items.append(
                {
                    "path": "",
                    "action": "stop_process",
                    "approved": False,
                    "reason": "Current llama-server process listening on port 8080.",
                    "process": process,
                }
            )
        if IMMEDIATE_MODEL_PATH.exists():
            items.append(
                {
                    "path": str(IMMEDIATE_MODEL_PATH),
                    "action": "quarantine",
                    "approved": False,
                    "reason": "Previously served Qwen3.6 GGUF approved for removal or quarantine.",
                    "size_bytes": IMMEDIATE_MODEL_PATH.stat().st_size,
                }
            )
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.iterdir():
            if path == IMMEDIATE_MODEL_PATH.parent:
                continue
            if _is_candidate_artifact(path):
                items.append(
                    {
                        "path": str(path),
                        "action": "quarantine",
                        "approved": False,
                        "reason": "Unregistered training or model artifact candidate.",
                        "size_bytes": _path_size(path),
                    }
                )
    manifest = {
        "manifest_id": manifest_id,
        "status": "draft",
        "created_at": utc_now(),
        "quarantine_dir": str(quarantine_dir),
        "items": items,
        "policy": "Only approved manifest items may be moved or deleted.",
    }
    with connect(database_path) as conn:
        conn.execute(
            """
            INSERT INTO cleanup_manifests (manifest_id, status, manifest_json, created_at, applied_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (manifest_id, "draft", json.dumps(manifest, sort_keys=True), manifest["created_at"], None),
        )
    return manifest


def list_manifests(database_path: Path) -> list[dict[str, Any]]:
    with connect(database_path) as conn:
        rows = conn.execute("SELECT * FROM cleanup_manifests ORDER BY created_at DESC").fetchall()
    manifests = rows_to_dicts(rows)
    for manifest in manifests:
        manifest["manifest"] = json.loads(manifest.pop("manifest_json") or "{}")
    return manifests


def get_manifest(database_path: Path, manifest_id: str) -> dict[str, Any] | None:
    with connect(database_path) as conn:
        row = conn.execute("SELECT * FROM cleanup_manifests WHERE manifest_id = ?", (manifest_id,)).fetchone()
    manifest = row_to_dict(row)
    if manifest:
        manifest["manifest"] = json.loads(manifest.pop("manifest_json") or "{}")
    return manifest


def approve_manifest_items(database_path: Path, manifest_id: str, approved_paths: list[str]) -> dict[str, Any] | None:
    record = get_manifest(database_path, manifest_id)
    if not record:
        return None
    manifest = record["manifest"]
    approved_set = set(approved_paths)
    for item in manifest.get("items", []):
        if item.get("path") in approved_set:
            item["approved"] = True
    with connect(database_path) as conn:
        conn.execute(
            "UPDATE cleanup_manifests SET manifest_json = ?, status = ? WHERE manifest_id = ?",
            (json.dumps(manifest, sort_keys=True), "approved", manifest_id),
        )
    return manifest


def process_on_port(port: int) -> dict[str, Any] | None:
    commands = [
        ["bash", "-lc", f"ss -ltnp 'sport = :{port}' || true"],
        ["bash", "-lc", f"lsof -nP -iTCP:{port} -sTCP:LISTEN || true"],
    ]
    for command in commands:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        output = result.stdout.strip()
        if output and "LISTEN" in output:
            return {"port": port, "command": command, "output": output}
    return None


def stop_process_on_port(port: int) -> dict[str, Any]:
    pids = _pids_on_port(port)
    stopped = []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            stopped.append(pid)
        except ProcessLookupError:
            pass
    return {"port": port, "stopped_pids": stopped}


def _pids_on_port(port: int) -> list[int]:
    try:
        result = subprocess.run(["bash", "-lc", f"lsof -tiTCP:{port} -sTCP:LISTEN"], capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    return [int(line) for line in result.stdout.splitlines() if line.strip().isdigit()]


def _is_candidate_artifact(path: Path) -> bool:
    name = path.name.lower()
    return path.is_dir() and any(marker in name for marker in ["checkpoint", "adapter", "qwen", "gemma", "lfm", "failed"])


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def apply_immediate_cleanup(data_root: Path) -> dict[str, Any]:
    manifest = {
        "manifest_id": make_job_id("cl", "immediate-llama-server"),
        "created_at": utc_now(),
        "quarantine_dir": str(data_root / "cleanup" / "immediate-llama-server" / "quarantine"),
        "items": [],
    }
    process = process_on_port(8080)
    if process:
        manifest["items"].append({"path": "", "action": "stop_process", "approved": True, "process": process})
        stop_process_on_port(8080)
    if IMMEDIATE_MODEL_PATH.exists():
        quarantine_dir = Path(manifest["quarantine_dir"])
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        destination = quarantine_dir / IMMEDIATE_MODEL_PATH.name
        shutil.move(str(IMMEDIATE_MODEL_PATH), str(destination))
        manifest["items"].append(
            {
                "path": str(IMMEDIATE_MODEL_PATH),
                "action": "quarantine",
                "approved": True,
                "destination": str(destination),
            }
        )
        parent = IMMEDIATE_MODEL_PATH.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
            manifest["items"].append({"path": str(parent), "action": "delete_empty_parent", "approved": True})
    return manifest

