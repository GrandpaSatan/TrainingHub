from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from traininghub.api.dependencies import current_user, settings_dependency
from traininghub.core.config import Settings
from traininghub.services.gpu import disk_usage, query_gpus


router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/gpus")
def gpus(_user: Annotated[dict[str, Any], Depends(current_user)]) -> list[dict[str, Any]]:
    return query_gpus()


@router.get("/disk")
def disk(
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, int]:
    return disk_usage(settings.data_root)

