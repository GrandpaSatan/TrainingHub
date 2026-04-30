from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from traininghub.api.dependencies import current_user, settings_dependency
from traininghub.core.config import Settings
from traininghub.services.artifacts import get_artifact, list_artifacts


router = APIRouter(prefix="/api/artifacts", tags=["artifacts"])


@router.get("")
def artifacts(
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> list[dict[str, Any]]:
    return list_artifacts(settings.database_path)


@router.get("/{artifact_id}/download")
def download_artifact(
    artifact_id: str,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> FileResponse:
    artifact = get_artifact(settings.database_path, artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found.")
    path = Path(artifact["path"])
    if not path.is_file():
        raise HTTPException(status_code=400, detail="Artifact is a directory or missing.")
    return FileResponse(path, filename=path.name)

