#!/usr/bin/env bash
set -euo pipefail

APPLY="0"
if [[ "${1:-}" == "--apply" ]]; then
  APPLY="1"
fi

python3 - "$APPLY" <<'PY'
from __future__ import annotations

import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path


apply_changes = sys.argv[1] == "1"
home = Path("/home/jhernandez")
app_root = Path(os.getenv("TRAININGHUB_APP_ROOT", str(home / "traininghub"))).resolve()
data_root = Path(os.getenv("TRAININGHUB_DATA_ROOT", str(home / "traininghub-data"))).resolve()
db_path = Path(os.getenv("TRAININGHUB_DATABASE_PATH", str(data_root / "traininghub.sqlite3"))).resolve()
manifest_id = time.strftime("cl_%Y%m%d_%H%M%S_traininghub-only-cleanup", time.gmtime())
manifest_dir = data_root / "cleanup" / manifest_id
manifest_dir.mkdir(parents=True, exist_ok=True)
manifest_path = manifest_dir / "manifest.json"


def run(command: list[str]) -> dict[str, str | int]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except Exception as exc:
        return {"returncode": 1, "stdout": "", "stderr": str(exc)}


def disk_snapshot() -> dict[str, int]:
    usage = shutil.disk_usage("/")
    return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}


def path_size(path: Path) -> int:
    if not path.exists() and not path.is_symlink():
        return 0
    if path.is_file() or path.is_symlink():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file() or child.is_symlink():
                total += child.stat().st_size
        except OSError:
            pass
    return total


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False
    except FileNotFoundError:
        return str(path).startswith(str(root))


def protected_traininghub_paths() -> list[Path]:
    protected = [app_root, data_root, db_path]
    if not db_path.exists():
        return protected
    try:
        conn = sqlite3.connect(db_path)
        for table, column in [
            ("artifacts", "path"),
            ("dataset_versions", "raw_csv_path"),
            ("dataset_versions", "jsonl_path"),
        ]:
            try:
                rows = conn.execute(f"SELECT {column} FROM {table}").fetchall()
            except sqlite3.Error:
                continue
            for (value,) in rows:
                if value:
                    protected.append(Path(value).expanduser())
        conn.close()
    except sqlite3.Error:
        pass
    return protected


protected_paths = protected_traininghub_paths()


def deletion_blocked(path: Path) -> str:
    resolved = path.resolve() if path.exists() else path
    if resolved == app_root or resolved == data_root or resolved == home:
        return "protected root"
    if is_relative_to(resolved, app_root):
        return "inside TrainingHub app root"
    if is_relative_to(resolved, data_root) and "quarantine" not in resolved.parts:
        return "inside TrainingHub data root"
    for protected in protected_paths:
        if protected and is_relative_to(protected, resolved):
            return f"contains TrainingHub referenced path: {protected}"
    return ""


def collect_processes() -> list[dict[str, str | int]]:
    result = run(["ps", "-u", "jhernandez", "-o", "pid=,ppid=,etimes=,stat=,comm=,args="])
    rows = []
    for line in result["stdout"].splitlines():
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        pid, ppid, etimes, stat, comm, args = parts
        rows.append({"pid": int(pid), "ppid": int(ppid), "etimes": int(etimes), "stat": stat, "comm": comm, "args": args})
    return rows


def process_is_keep(proc: dict[str, str | int], current_tree: set[int]) -> bool:
    args = str(proc["args"])
    comm = str(proc["comm"])
    pid = int(proc["pid"])
    if pid in current_tree:
        return True
    keep_tokens = [
        "traininghub.main:app",
        "traininghub/.venv",
        "systemd --user",
        "(sd-pam)",
        "sshd-session",
        "ps -u jhernandez",
        "morrigan_cleanup_traininghub_only",
    ]
    if any(token in args for token in keep_tokens):
        return True
    if comm in {"sshd-session", "systemd", "(sd-pam)"}:
        return True
    return False


def process_is_stop_candidate(proc: dict[str, str | int]) -> bool:
    args = str(proc["args"]).lower()
    comm = str(proc["comm"]).lower()
    if "traininghub" in args:
        return False
    patterns = [
        "llama-server",
        "fine-tuning/scripts/",
        "fine-tuning/.venv/bin/python",
        "bench_math.py",
        "torchrun",
        "accelerate",
        "lm_eval",
        "jupyter",
        "notebook",
        "streamlit",
        "gradio",
        "vite --host",
        "pnpm dev",
        "npm run dev",
    ]
    if any(pattern in args for pattern in patterns):
        return True
    if comm in {"python", "python3"} and any(token in args for token in ["train", "benchmark", "server.py", "app.py"]):
        return True
    return False


def current_process_tree(processes: list[dict[str, str | int]]) -> set[int]:
    tree = {os.getpid(), os.getppid()}
    changed = True
    while changed:
        changed = False
        for proc in processes:
            pid = int(proc["pid"])
            ppid = int(proc["ppid"])
            if pid in tree or ppid not in tree:
                continue
            tree.add(pid)
            changed = True
    return tree


def stop_processes(processes: list[dict[str, str | int]]) -> list[dict[str, object]]:
    current_tree = current_process_tree(processes)
    actions = []
    candidates = [proc for proc in processes if not process_is_keep(proc, current_tree) and process_is_stop_candidate(proc)]
    for proc in candidates:
        pid = int(proc["pid"])
        action = {"action": "stop_process", "pid": pid, "args": proc["args"], "applied": False}
        if apply_changes:
            try:
                os.kill(pid, signal.SIGTERM)
                action["applied"] = True
                action["signal"] = "SIGTERM"
            except ProcessLookupError:
                action["applied"] = True
                action["signal"] = "already_exited"
            except PermissionError as exc:
                action["error"] = str(exc)
        actions.append(action)
    if apply_changes and candidates:
        time.sleep(3)
        remaining = {int(proc["pid"]) for proc in collect_processes()}
        for action in actions:
            pid = int(action["pid"])
            if pid not in remaining:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                action["signal"] = "SIGKILL"
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                action["error"] = str(exc)
    return actions


def planned_delete_paths() -> list[Path]:
    paths = [
        home / "fine-tuning",
        home / ".cache" / "huggingface",
        home / ".cache" / "pip",
        home / ".cache" / "evalplus",
        home / ".cache" / "torch_extensions",
        home / ".cache" / "torch",
        home / ".cache" / "matplotlib",
        home / ".cache" / "gdown",
        home / ".cache" / "puccinialin",
        home / ".nv" / "ComputeCache",
        home / ".triton",
        home / ".rustup",
        home / ".cargo",
        home / ".pyenv",
        home / ".local",
        home / "unsloth_compiled_cache",
        home / "benchmarks",
        home / "lfm2.5",
        home / "models",
        home / "prismml-llama.cpp",
        home / "coding-bench",
        home / "venv",
        app_root / ".pytest_cache",
        app_root / "frontend" / "node_modules",
    ]
    for quarantine_dir in (data_root / "cleanup").glob("cl_*/quarantine"):
        paths.append(quarantine_dir)
    return paths


def delete_paths(paths: list[Path]) -> list[dict[str, object]]:
    actions = []
    for path in paths:
        if not path.exists() and not path.is_symlink():
            continue
        reason = deletion_blocked(path)
        action = {
            "action": "delete_path",
            "path": str(path),
            "size_bytes": path_size(path),
            "applied": False,
            "blocked_reason": reason,
        }
        if not reason and apply_changes:
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                action["applied"] = True
            except Exception as exc:
                action["error"] = str(exc)
        actions.append(action)
    return actions


before_processes = collect_processes()
manifest: dict[str, object] = {
    "manifest_id": manifest_id,
    "apply": apply_changes,
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "policy": "Keep TrainingHub, SSH/session/system processes, and TrainingHub-referenced DB paths. Stop unrelated user workloads. Delete safe caches and obsolete non-TrainingHub trees.",
    "app_root": str(app_root),
    "data_root": str(data_root),
    "database_path": str(db_path),
    "disk_before": disk_snapshot(),
    "listeners_before": run(["bash", "-lc", "ss -ltnp || true"]),
    "processes_before": before_processes,
    "protected_paths": [str(path) for path in protected_paths],
}

manifest["process_actions"] = stop_processes(before_processes)
manifest["delete_actions"] = delete_paths(planned_delete_paths())
manifest["disk_after"] = disk_snapshot()
manifest["listeners_after"] = run(["bash", "-lc", "ss -ltnp || true"])
manifest["processes_after"] = collect_processes()
before_free = int(manifest["disk_before"]["free_bytes"])  # type: ignore[index]
after_free = int(manifest["disk_after"]["free_bytes"])  # type: ignore[index]
manifest["freed_bytes"] = max(0, after_free - before_free)

manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({
    "manifest_path": str(manifest_path),
    "apply": apply_changes,
    "freed_bytes": manifest["freed_bytes"],
    "process_actions": len(manifest["process_actions"]),  # type: ignore[arg-type]
    "delete_actions": len(manifest["delete_actions"]),  # type: ignore[arg-type]
}, indent=2, sort_keys=True))
PY
