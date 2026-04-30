from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from traininghub.api.dependencies import current_user, settings_dependency
from traininghub.core.config import Settings
from traininghub.core.database import connect, rows_to_dicts
from traininghub.services.gpu import disk_usage
from traininghub.services.hub import HubResolveError, estimate_model_download
from traininghub.services.jobs import JobValidationError, create_and_start_job
from traininghub.services.model_files import local_provider_path, local_provider_is_runnable
from traininghub.services.model_registry import DEFAULT_MODEL_SLUGS, delete_model_record, model_is_deletable


router = APIRouter(prefix="/api/models", tags=["models"])
MIN_MODEL_DOWNLOAD_MARGIN_BYTES = 5 * 1024**3
MODEL_DOWNLOAD_MARGIN_FRACTION = 0.10


class ImportModelRequest(BaseModel):
    slug: str
    provider_id: str
    display_name: str
    family: str = "custom"
    parameter_count: str = "unknown"
    supports_lora: bool = True
    supports_qlora: bool = True
    supports_full_finetune: bool = False
    supports_bf16_inference: bool = True
    supports_benchmark: bool = True
    supports_quantization: bool = True
    supports_gguf_path: bool = True
    hardware_note: str = "Custom model. Verify memory before BF16 inference or route to GGUF."
    default_dtype: str = "bf16"
    max_sequence_length: int = 2048
    metadata: dict[str, Any] = Field(default_factory=dict)


class DownloadHfModelRequest(BaseModel):
    repo_id: str
    revision: str | None = None
    confirmed_sha: str | None = None
    slug: str | None = None
    display_name: str | None = None
    family: str | None = None
    parameter_count: str = "unknown"
    default_dtype: str = "auto"
    max_sequence_length: int = 2048
    include_patterns: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=list)
    dry_run: bool = False


class DownloadUrlModelRequest(BaseModel):
    url: str
    filename: str | None = None
    slug: str | None = None
    display_name: str | None = None
    family: str | None = None
    parameter_count: str = "unknown"
    default_dtype: str = "auto"
    max_sequence_length: int = 2048
    dry_run: bool = False


class UploadHfModelRequest(BaseModel):
    repo_id: str
    model_slug: str | None = None
    artifact_id: str | None = None
    source_path: str | None = None
    private: bool = True
    large_folder: bool = False
    path_in_repo: str | None = None
    include_patterns: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=list)
    commit_message: str = "Upload TrainingHub model artifact"
    dry_run: bool = False


@router.get("")
def list_models(
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> list[dict[str, Any]]:
    with connect(settings.database_path) as conn:
        rows = conn.execute("SELECT * FROM model_registry ORDER BY family, display_name").fetchall()
        artifact_rows = conn.execute("SELECT * FROM artifacts ORDER BY created_at DESC").fetchall()
    models = rows_to_dicts(rows)
    artifacts = rows_to_dicts(artifact_rows)
    for artifact in artifacts:
        artifact["metadata"] = json.loads(artifact.pop("metadata_json") or "{}")
    visible_models: list[dict[str, Any]] = []
    for model in models:
        model["metadata"] = json.loads(model.pop("metadata_json") or "{}")
        if _is_unrunnable_trained_checkpoint(model):
            continue
        model["supports_bf16_inference"] = bool(model["supports_bf16_inference"])
        model["supports_benchmark"] = bool(model["supports_benchmark"])
        model["supports_quantization"] = bool(model["supports_quantization"])
        model["supports_gguf_path"] = bool(model["supports_gguf_path"])
        model["supports_lora"] = bool(model["supports_lora"])
        model["supports_qlora"] = bool(model["supports_qlora"])
        model["supports_full_finetune"] = bool(model["supports_full_finetune"])
        model.pop("is_saga", None)
        model["deletable"] = model_is_deletable(model["slug"])
        model["seeded"] = model["slug"] in DEFAULT_MODEL_SLUGS
        model["local_path"] = str(model["metadata"].get("local_path") or "")
        model["local_size_bytes"] = _model_disk_usage(settings.data_root, model, artifacts)
        visible_models.append(model)
    return visible_models


def _is_unrunnable_trained_checkpoint(model: dict[str, Any]) -> bool:
    metadata = model["metadata"]
    provider_id = str(model.get("provider_id") or "")
    return (
        metadata.get("route") == "trained_model"
        and local_provider_path(provider_id) is not None
        and not local_provider_is_runnable(provider_id)
    )


@router.post("/import")
def import_model(
    payload: ImportModelRequest,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, Any]:
    with connect(settings.database_path) as conn:
        conn.execute("DELETE FROM model_delete_tombstones WHERE slug = ?", (payload.slug,))
        conn.execute(
            """
            INSERT INTO model_registry (
                slug, provider_id, display_name, family, parameter_count,
                supports_lora, supports_qlora, supports_full_finetune,
                supports_bf16_inference, supports_benchmark, supports_quantization,
                supports_gguf_path, is_saga,
                hardware_note, default_dtype, max_sequence_length, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(slug) DO UPDATE SET
                provider_id = excluded.provider_id,
                display_name = excluded.display_name,
                family = excluded.family,
                parameter_count = excluded.parameter_count,
                supports_lora = excluded.supports_lora,
                supports_qlora = excluded.supports_qlora,
                supports_full_finetune = excluded.supports_full_finetune,
                supports_bf16_inference = excluded.supports_bf16_inference,
                supports_benchmark = excluded.supports_benchmark,
                supports_quantization = excluded.supports_quantization,
                supports_gguf_path = excluded.supports_gguf_path,
                is_saga = excluded.is_saga,
                hardware_note = excluded.hardware_note,
                default_dtype = excluded.default_dtype,
                max_sequence_length = excluded.max_sequence_length,
                metadata_json = excluded.metadata_json
            """,
            (
                payload.slug,
                payload.provider_id,
                payload.display_name,
                payload.family,
                payload.parameter_count,
                int(payload.supports_lora),
                int(payload.supports_qlora),
                int(payload.supports_full_finetune),
                int(payload.supports_bf16_inference),
                int(payload.supports_benchmark),
                int(payload.supports_quantization),
                int(payload.supports_gguf_path),
                0,
                payload.hardware_note,
                payload.default_dtype,
                payload.max_sequence_length,
                json.dumps(payload.metadata, sort_keys=True),
            ),
        )
    return {"slug": payload.slug, "status": "imported"}


@router.post("/download-hf")
def download_hf_model(
    payload: DownloadHfModelRequest,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, Any]:
    job_payload = payload.model_dump(exclude_none=True)
    _apply_hf_download_preflight(settings, job_payload)
    job_payload["source_type"] = "hf"
    return _create_job(settings, "model_download", f"download-{payload.slug or payload.repo_id}", job_payload)


@router.post("/download-url")
def download_url_model(
    payload: DownloadUrlModelRequest,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, Any]:
    job_payload = payload.model_dump(exclude_none=True)
    job_payload["source_type"] = "url"
    return _create_job(settings, "model_download", f"download-{payload.slug or payload.url}", job_payload)


@router.post("/upload-hf")
def upload_hf_model(
    payload: UploadHfModelRequest,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, Any]:
    if not payload.model_slug and not payload.artifact_id and not payload.source_path:
        raise HTTPException(status_code=400, detail="model_slug, artifact_id, or source_path is required.")
    return _create_job(settings, "model_upload", f"upload-{payload.repo_id}", payload.model_dump(exclude_none=True))


@router.post("/{model_slug}/upload-hf")
def upload_model_slug_hf(
    model_slug: str,
    payload: UploadHfModelRequest,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, Any]:
    job_payload = payload.model_dump(exclude_none=True)
    job_payload["model_slug"] = model_slug
    return _create_job(settings, "model_upload", f"upload-{payload.repo_id}", job_payload)


@router.delete("/{model_slug}")
def delete_model(
    model_slug: str,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, Any]:
    result = delete_model_record(settings.database_path, settings.data_root, model_slug)
    if result is None:
        raise HTTPException(status_code=404, detail="Model not found.")
    if result.get("blocked"):
        raise HTTPException(status_code=400, detail=result["detail"])
    return result


def _create_job(settings: Settings, job_type: str, slug: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return create_and_start_job(settings, job_type, slug, payload)
    except JobValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _apply_hf_download_preflight(settings: Settings, job_payload: dict[str, Any]) -> None:
    try:
        estimate = estimate_model_download(
            str(job_payload["repo_id"]),
            job_payload.get("revision"),
            _patterns(job_payload.get("include_patterns")),
            _patterns(job_payload.get("exclude_patterns")),
        )
        if job_payload.get("confirmed_sha") and estimate["sha"] != str(job_payload["confirmed_sha"]):
            raise HubResolveError(
                "The Hugging Face repository changed after confirmation. Run Find again before starting the download.",
                status_code=409,
            )
    except HubResolveError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    if estimate["matched_file_count"] <= 0:
        raise HTTPException(status_code=400, detail="No Hugging Face model files matched the selected include/exclude patterns.")

    job_payload["repo_id"] = estimate["repo_id"]
    job_payload["include_patterns"] = estimate["include_patterns"]
    job_payload["exclude_patterns"] = estimate["exclude_patterns"]
    job_payload["download_estimate"] = estimate
    if not job_payload.get("dry_run", False):
        _ensure_download_fits(settings, estimate)


def _ensure_download_fits(settings: Settings, estimate: dict[str, Any]) -> None:
    estimated_size = int(estimate.get("estimated_size_bytes") or 0)
    if estimated_size <= 0:
        return
    margin = max(MIN_MODEL_DOWNLOAD_MARGIN_BYTES, int(estimated_size * MODEL_DOWNLOAD_MARGIN_FRACTION))
    margin = int(os.getenv("TRAININGHUB_MODEL_DOWNLOAD_MARGIN_BYTES", str(margin)))
    free_bytes = int(disk_usage(settings.data_root)["free_bytes"])
    required = estimated_size + margin
    estimate["safety_margin_bytes"] = margin
    estimate["free_bytes"] = free_bytes
    if required > free_bytes:
        raise HTTPException(
            status_code=400,
            detail=(
                "Not enough disk space for this model download. "
                f"Estimated download is {_format_bytes(estimated_size)}, safety margin is {_format_bytes(margin)}, "
                f"and {_format_bytes(free_bytes)} is free under TRAININGHUB_DATA_ROOT."
            ),
        )


def _patterns(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _format_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    amount = float(value)
    index = 0
    while amount >= 1024 and index < len(units) - 1:
        amount /= 1024
        index += 1
    return f"{amount:.1f} {units[index]}" if index else f"{int(amount)} {units[index]}"


def _model_disk_usage(data_root: Path, model: dict[str, Any], artifacts: list[dict[str, Any]]) -> int:
    paths: list[Path] = []
    local_path = str(model.get("metadata", {}).get("local_path") or "")
    if local_path:
        paths.append(Path(local_path))
    artifact_ids = {str(artifact_id) for artifact_id in model.get("metadata", {}).get("artifact_ids", []) if str(artifact_id)}
    for artifact in artifacts:
        metadata = artifact.get("metadata", {})
        if artifact["artifact_id"] in artifact_ids or metadata.get("model_slug") == model["slug"]:
            paths.append(Path(str(artifact["path"])))
    return _deduped_path_size(paths, data_root)


def _deduped_path_size(paths: list[Path], data_root: Path) -> int:
    root = data_root.resolve()
    selected: list[Path] = []
    for path in paths:
        resolved = path.resolve(strict=False)
        if not _is_relative_to(resolved, root) or not resolved.exists():
            continue
        if any(_is_relative_to(resolved, parent) for parent in selected):
            continue
        selected = [parent for parent in selected if not _is_relative_to(parent, resolved)]
        selected.append(resolved)
    return sum(_path_size(path) for path in selected)


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
