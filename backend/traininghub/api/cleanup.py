from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from traininghub.api.dependencies import current_user, settings_dependency
from traininghub.core.config import Settings
from traininghub.services.cleanup import approve_manifest_items, get_manifest, list_manifests, scan_cleanup
from traininghub.services.jobs import create_and_start_job


router = APIRouter(prefix="/api/cleanup", tags=["cleanup"])


class ScanRequest(BaseModel):
    include_immediate: bool = True


class ApplyRequest(BaseModel):
    manifest_id: str
    approved_paths: list[str]
    dry_run: bool = False


@router.get("/manifests")
def manifests(
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> list[dict[str, Any]]:
    return list_manifests(settings.database_path)


@router.post("/scan")
def scan(
    payload: ScanRequest,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, Any]:
    return scan_cleanup(settings.database_path, settings.data_root, payload.include_immediate)


@router.post("/apply")
def apply(
    payload: ApplyRequest,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, Any]:
    manifest = approve_manifest_items(settings.database_path, payload.manifest_id, payload.approved_paths)
    if not manifest:
        raise HTTPException(status_code=404, detail="Manifest not found.")
    manifest_path = settings.data_root / "cleanup" / payload.manifest_id / "manifest.approved.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return create_and_start_job(
        settings,
        "cleanup",
        "cleanup-apply",
        {"manifest_id": payload.manifest_id, "manifest_path": str(manifest_path), "dry_run": payload.dry_run},
    )

