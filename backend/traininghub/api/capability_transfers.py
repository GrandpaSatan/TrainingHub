from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from traininghub.api.dependencies import current_user, settings_dependency
from traininghub.core.config import Settings
from traininghub.services.capability_transfers import (
    CapabilityTransferError,
    activate_transfer,
    create_transfer,
    deactivate_transfer,
    delete_transfer,
    get_transfer,
    list_transfers,
    queue_alignment,
)


router = APIRouter(prefix="/api/capability-transfers", tags=["capability-transfers"])


class CapabilityTransferCreateRequest(BaseModel):
    display_name: str = ""
    source_model_slug: str = ""
    source_runtime: str = "transformers"
    source_artifact_id: str = ""
    target_model_slug: str = ""
    target_runtime: str = "transformers"
    target_artifact_id: str = ""
    calibration_dataset_id: str
    layer_targets: str | list[int] = "all"
    contrast_mode: str = "prompt_pair"
    rank: int = Field(default=16, ge=1, le=256)
    dry_run: bool = False


class CapabilityTransferAlignRequest(BaseModel):
    rank: int = Field(default=16, ge=1, le=256)
    layer_pairs: list[list[int]] = Field(default_factory=list)


class CapabilityTransferActivateRequest(BaseModel):
    alpha: float = Field(default=1.0, ge=0.0, le=4.0)
    layer_targets: str | list[int] = "all"


@router.get("")
def transfers(
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> list[dict[str, Any]]:
    return list_transfers(settings.database_path)


@router.get("/{transfer_id}")
def transfer_detail(
    transfer_id: str,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, Any]:
    transfer = get_transfer(settings.database_path, transfer_id)
    if not transfer:
        raise HTTPException(status_code=404, detail="Capability transfer not found.")
    return transfer


@router.post("")
def create(
    payload: CapabilityTransferCreateRequest,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, Any]:
    try:
        return create_transfer(settings, payload.model_dump())
    except CapabilityTransferError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{transfer_id}/align")
def align(
    transfer_id: str,
    payload: CapabilityTransferAlignRequest,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, Any]:
    try:
        return queue_alignment(settings, transfer_id, payload.model_dump())
    except CapabilityTransferError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{transfer_id}/activate")
def activate(
    transfer_id: str,
    payload: CapabilityTransferActivateRequest,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, Any]:
    try:
        return activate_transfer(settings, transfer_id, payload.model_dump())
    except CapabilityTransferError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{transfer_id}/deactivate")
def deactivate(
    transfer_id: str,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, Any]:
    return deactivate_transfer(settings, transfer_id)


@router.delete("/{transfer_id}")
def delete(
    transfer_id: str,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, Any]:
    try:
        return delete_transfer(settings, transfer_id)
    except CapabilityTransferError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
