from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from traininghub.api.dependencies import current_user, settings_dependency
from traininghub.core.config import Settings
from traininghub.core.database import connect, row_to_dict
from traininghub.services.benchmark_catalog import benchmark_job_type_for
from traininghub.services.jobs import JobValidationError, cancel_job, create_and_start_job, get_job, list_jobs, stream_job_events


router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class GenerateJobRequest(BaseModel):
    teacher_model: str = "local"
    seed_prompt: str
    target_count: int = Field(default=100, ge=1, le=5000)
    category_mix: dict[str, float] = Field(default_factory=dict)
    difficulty_mix: dict[str, float] = Field(default_factory=dict)
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256
    output_schema: str = "chat_sft"
    validation_strictness: str = "normal"
    use_teacher_model: bool = False
    dry_run: bool = False


class BenchmarkJobRequest(BaseModel):
    model_slug: str
    checkpoint_path: str | None = None
    benchmarks: list[str] = Field(default_factory=lambda: ["gsm8k", "math-500"])
    limit: int = Field(default=10, ge=1, le=10000)
    maj_k: int = Field(default=1, ge=1, le=64)
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 512
    prompt_template: str = "default_cot"
    seed: int = 7
    batch_size: int = 1
    gpu_ids: list[int] | None = None
    dry_run: bool = False


class QuantizeJobRequest(BaseModel):
    source_gguf: str
    quant_type: str = "Q4_K_M"
    llama_cpp_root: str | None = None
    quantize_binary: str | None = None
    dry_run: bool = False


class ConvertGgufJobRequest(BaseModel):
    source_checkpoint: str
    output_name: str = "model"
    outtype: str = "f16"
    llama_cpp_root: str | None = None
    dry_run: bool = False


class TrainingJobRequest(BaseModel):
    model_slug: str
    dataset_id: str
    mode: Literal["lora", "qlora", "full"] = "lora"
    preset: Literal["smoke", "standard", "custom"] = "smoke"
    output_name: str | None = None
    epochs: float | None = None
    max_steps: int | None = None
    learning_rate: float | None = None
    per_device_train_batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    max_sequence_length: int | None = None
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None
    target_modules: list[str] = Field(default_factory=list)
    merge_adapter: bool | None = None
    gpu_ids: list[int] | None = None
    dry_run: bool = False


@router.get("")
def jobs(
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> list[dict[str, Any]]:
    return list_jobs(settings.database_path)


@router.post("/generate")
def generate_job(
    payload: GenerateJobRequest,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, Any]:
    job_payload = payload.model_dump()
    if job_payload["teacher_model"] == "local" and settings.default_teacher_model != "local":
        job_payload["teacher_model"] = settings.default_teacher_model
        job_payload["use_teacher_model"] = True
    if job_payload["teacher_model"] != "local":
        job_payload["use_teacher_model"] = True
    return _create(settings, "generate", "generate-examples", job_payload)


@router.post("/fine-tune")
def fine_tune_job(
    payload: TrainingJobRequest,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, Any]:
    job_type = f"train_{payload.mode}"
    return _create(settings, job_type, f"train-{payload.model_slug}-{payload.preset}", payload.model_dump(exclude_none=True))


@router.post("/benchmark")
def benchmark_job(
    payload: BenchmarkJobRequest,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, Any]:
    benchmark_ids = payload.benchmarks or ["gsm8k", "math-500"]
    try:
        job_type = benchmark_job_type_for(benchmark_ids)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    model = _model(settings, payload.model_slug)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")
    if not bool(model["supports_benchmark"]):
        raise HTTPException(status_code=400, detail="Model does not support benchmarks.")
    job_payload = payload.model_dump()
    job_payload["benchmarks"] = benchmark_ids
    job_payload["model_id"] = model["provider_id"]
    return _create(settings, job_type, f"benchmark-{payload.model_slug}", job_payload)


@router.post("/quantize")
def quantize_job(
    payload: QuantizeJobRequest,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, Any]:
    return _create(settings, "quantize", f"quantize-{payload.quant_type}", payload.model_dump(exclude_none=True))


@router.post("/convert-gguf")
def convert_gguf_job(
    payload: ConvertGgufJobRequest,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, Any]:
    return _create(settings, "convert_gguf", f"convert-{payload.output_name}", payload.model_dump(exclude_none=True))


@router.post("/{job_id}/cancel")
def cancel(
    job_id: str,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, Any]:
    job = cancel_job(settings, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@router.get("/{job_id}/events")
def events(
    job_id: str,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> StreamingResponse:
    if not get_job(settings.database_path, job_id):
        raise HTTPException(status_code=404, detail="Job not found.")
    return StreamingResponse(stream_job_events(settings.database_path, job_id), media_type="text/event-stream")


def _create(settings: Settings, job_type: str, slug: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return create_and_start_job(settings, job_type, slug, payload)
    except JobValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _model(settings: Settings, model_slug: str) -> dict[str, Any] | None:
    with connect(settings.database_path) as conn:
        row = conn.execute("SELECT * FROM model_registry WHERE slug = ?", (model_slug,)).fetchone()
    model = row_to_dict(row)
    if model:
        model["metadata"] = json.loads(model.pop("metadata_json") or "{}")
    return model
