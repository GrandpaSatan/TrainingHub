from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

from traininghub.workers.common import WorkerContext, run_worker


def main(context: WorkerContext, payload: dict[str, Any]) -> None:
    manifest_path = Path(payload["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    applied: list[dict[str, Any]] = []
    for item in manifest.get("items", []):
        context.check_cancelled()
        action = item.get("action")
        path = Path(item["path"])
        if not item.get("approved", False):
            continue
        if action == "quarantine":
            quarantine_dir = Path(manifest["quarantine_dir"])
            quarantine_dir.mkdir(parents=True, exist_ok=True)
            destination = quarantine_dir / path.name
            if path.exists():
                shutil.move(str(path), str(destination))
            applied.append({"path": str(path), "action": action, "destination": str(destination)})
        elif action == "delete":
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            applied.append({"path": str(path), "action": action})
        elif action == "stop_process":
            context.event("cleanup", "Process stop actions are handled by deployment scripts.", data=item)
        else:
            context.event("cleanup_skip", "Unknown cleanup action skipped.", "warning", item)
    report_path = context.job_dir / "cleanup_report.json"
    report_path.write_text(json.dumps({"applied": applied}, indent=2, sort_keys=True), encoding="utf-8")
    context.register_artifact(report_path, "cleanup_report", "Cleanup execution report", {"manifest_path": str(manifest_path)})


if __name__ == "__main__":
    sys.exit(run_worker(main))

