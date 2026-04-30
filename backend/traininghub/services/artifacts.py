from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from traininghub.core.database import connect, row_to_dict, rows_to_dicts


def list_artifacts(database_path: Path) -> list[dict[str, Any]]:
    with connect(database_path) as conn:
        rows = conn.execute("SELECT * FROM artifacts ORDER BY created_at DESC").fetchall()
    artifacts = rows_to_dicts(rows)
    for artifact in artifacts:
        artifact["metadata"] = json.loads(artifact.pop("metadata_json") or "{}")
    return artifacts


def get_artifact(database_path: Path, artifact_id: str) -> dict[str, Any] | None:
    with connect(database_path) as conn:
        row = conn.execute("SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)).fetchone()
    artifact = row_to_dict(row)
    if artifact:
        artifact["metadata"] = json.loads(artifact.pop("metadata_json") or "{}")
    return artifact

