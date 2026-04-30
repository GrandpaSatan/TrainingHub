from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from traininghub.api.dependencies import current_user
from traininghub.services.hub import HubResolveError, resolve_hub_resource


router = APIRouter(prefix="/api/hub", tags=["hub"])


class HubResolveRequest(BaseModel):
    input: str
    resource_type: Literal["auto", "model", "dataset"] = "auto"
    revision: str | None = None


@router.post("/resolve")
def resolve_hub(
    payload: HubResolveRequest,
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, Any]:
    try:
        return resolve_hub_resource(payload.input, payload.resource_type, payload.revision)
    except HubResolveError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
