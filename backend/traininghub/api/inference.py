from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Event
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from traininghub.api.dependencies import current_user, settings_dependency
from traininghub.core.config import Settings
from traininghub.services.inference import (
    InferenceTargetError,
    get_active_inference_target,
    list_inference_options,
    set_active_inference_target,
)
from traininghub.services.inference_run import InferenceRunError, SamplingConfig, run_prompt


router = APIRouter(prefix="/api/inference", tags=["inference"])


class InferenceTargetRequest(BaseModel):
    target_type: str
    model_slug: str = ""
    artifact_id: str = ""


class InferenceRunRequest(BaseModel):
    prompt: str = Field(min_length=1)
    system: str = ""
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    stop: list[str] = Field(default_factory=list, max_length=8)
    repetition_penalty: float = Field(default=1.08, ge=1.0, le=2.0)
    no_repeat_ngram_size: int = Field(default=3, ge=0, le=12)
    do_sample: bool = True
    dry_run: bool = False


@router.get("/target")
def get_target(
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, Any] | None:
    return get_active_inference_target(settings.database_path)


@router.post("/target")
def set_target(
    payload: InferenceTargetRequest,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, Any]:
    try:
        return set_active_inference_target(settings.database_path, payload.model_dump())
    except InferenceTargetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/options")
def options(
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> list[dict[str, Any]]:
    return list_inference_options(settings.database_path)


@router.post("/run")
def run(
    payload: InferenceRunRequest,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> StreamingResponse:
    target = get_active_inference_target(settings.database_path)
    if target is None:
        raise HTTPException(status_code=400, detail="No active inference target is configured. Select a model on the Models page.")
    sampling = SamplingConfig(
        system=payload.system,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
        stop=tuple(payload.stop),
        repetition_penalty=payload.repetition_penalty,
        no_repeat_ngram_size=payload.no_repeat_ngram_size,
        do_sample=payload.do_sample,
        dry_run=payload.dry_run,
    )
    cancel_event = Event()

    return StreamingResponse(
        _stream_inference_events(target, payload.prompt, sampling, cancel_event, settings.database_path),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _stream_inference_events(
    target: dict[str, Any],
    prompt: str,
    sampling: SamplingConfig,
    cancel_event: Event,
    database_path: Path,
):
    started_at = time.monotonic()
    token_count = 0
    try:
        async for token in run_prompt(target, prompt, sampling, cancel_event, database_path):
            token_count += 1
            yield _sse("token", {"token": token})
        yield _sse(
            "done",
            {
                "target": {
                    "target_type": target.get("target_type", ""),
                    "model_slug": target.get("model_slug", ""),
                    "artifact_id": target.get("artifact_id", ""),
                    "display_name": target.get("display_name", ""),
                },
                "elapsed_ms": round((time.monotonic() - started_at) * 1000),
                "tokens": token_count,
            },
        )
    except InferenceRunError as exc:
        yield _sse("error", {"message": str(exc)})
    except GeneratorExit:
        cancel_event.set()
        raise
    except Exception as exc:  # pragma: no cover - defensive SSE boundary.
        yield _sse("error", {"message": f"Inference failed: {exc}"})
    finally:
        cancel_event.set()


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, sort_keys=True)}\n\n"
