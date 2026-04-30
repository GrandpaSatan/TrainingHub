from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from traininghub.api.dependencies import current_user, settings_dependency
from traininghub.core.config import Settings
from traininghub.services.datasets import (
    approve_dataset_version,
    build_template,
    create_dataset_from_canonical_jsonl,
    create_dataset_version,
    delete_dataset_version,
    purge_rejected_datasets,
    read_dataset_records,
    read_review_sample,
    list_datasets,
)
from traininghub.services.artifacts import get_artifact
from traininghub.services.hub import HubResolveError, ensure_confirmed_hub_sha
from traininghub.services.inference import get_active_inference_target
from traininghub.services.jobs import JobValidationError, create_and_start_job


router = APIRouter(prefix="/api/datasets", tags=["datasets"])


class TemplateRequest(BaseModel):
    dataset_type: str = "math_sft"


class ImportGeneratedRequest(BaseModel):
    artifact_id: str
    title: str = "Generated review dataset"
    slug: str = "generated-review"
    dataset_type: str = "math_sft"


class ImportHfDatasetRequest(BaseModel):
    repo_id: str
    config_name: str | None = None
    revision: str | None = None
    confirmed_sha: str | None = None
    split: str = "train"
    title: str = "Hugging Face dataset"
    slug: str = "hf-dataset"
    dataset_type: str = "math_sft"
    max_rows: int | None = Field(default=None, ge=1)
    clean_with_inference: bool = False
    delete_raw_after_clean: bool = True
    cleaning_model: str | None = None
    prompt_field: str | None = None
    response_field: str | None = None
    system_field: str | None = None
    final_answer_field: str | None = None
    split_field: str | None = None
    default_split: str = "holdout"


class ImportUrlDatasetRequest(BaseModel):
    url: str
    title: str = "URL dataset"
    slug: str = "url-dataset"
    dataset_type: str = "math_sft"
    max_rows: int | None = Field(default=None, ge=1)
    clean_with_inference: bool = False
    delete_raw_after_clean: bool = True
    cleaning_model: str | None = None
    prompt_field: str | None = None
    response_field: str | None = None
    system_field: str | None = None
    final_answer_field: str | None = None
    split_field: str | None = None
    default_split: str = "holdout"


@router.get("")
def datasets(
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> list[dict[str, Any]]:
    purge_rejected_datasets(settings.database_path, settings.data_root)
    return list_datasets(settings.database_path)


@router.get("/{dataset_id}/records")
def dataset_records(
    dataset_id: str,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
    offset: int = 0,
    limit: int = 50,
    split: str | None = None,
    query: str | None = None,
) -> dict[str, Any]:
    result = read_dataset_records(settings.database_path, dataset_id, max(offset, 0), min(max(limit, 1), 500), split, query)
    if result is None:
        return {"records": [], "detail": "Dataset not found."}
    return result


@router.get("/{dataset_id}/review-sample")
def dataset_review_sample(
    dataset_id: str,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
    sample_size: int = 100,
) -> dict[str, Any]:
    result = read_review_sample(settings.database_path, dataset_id, min(max(sample_size, 1), 500))
    if result is None:
        return {"records": [], "detail": "Dataset not found."}
    return result


@router.post("/template")
def template(
    payload: TemplateRequest,
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> Response:
    content = build_template(payload.dataset_type)
    return Response(
        content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{payload.dataset_type}_template.csv"'},
    )


@router.post("/upload")
async def upload_dataset(
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
    file: Annotated[UploadFile, File()],
    dataset_type: Annotated[str, Form()] = "math_sft",
    title: Annotated[str, Form()] = "Uploaded dataset",
    slug: Annotated[str, Form()] = "uploaded-dataset",
    max_sequence_length: Annotated[int, Form()] = 2048,
) -> dict[str, Any]:
    csv_bytes = await file.read()
    return create_dataset_version(
        settings.database_path,
        settings.data_root,
        csv_bytes,
        dataset_type,
        title,
        slug,
        max_sequence_length,
        source=file.filename or "upload.csv",
    )


@router.post("/import-generated")
def import_generated_dataset(
    payload: ImportGeneratedRequest,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, Any]:
    artifact = get_artifact(settings.database_path, payload.artifact_id)
    if not artifact or artifact["artifact_type"] != "generated_dataset":
        return {"created": False, "validation": {"valid": False, "errors": [{"message": "Generated dataset artifact not found."}]}}
    return create_dataset_from_canonical_jsonl(
        settings.database_path,
        settings.data_root,
        Path(artifact["path"]),
        payload.dataset_type,
        payload.title,
        payload.slug,
        payload.artifact_id,
    )


@router.post("/import-hf")
def import_hf_dataset(
    payload: ImportHfDatasetRequest,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, Any]:
    _ensure_confirmed_sha(payload.repo_id, "dataset", payload.confirmed_sha, payload.revision)
    job_payload = {"source_type": "hf", **payload.model_dump(exclude_none=True)}
    _attach_cleaning_model(settings, job_payload)
    return _create_job(settings, "dataset_import", f"hf-{payload.slug}", job_payload)


@router.post("/import-url")
def import_url_dataset(
    payload: ImportUrlDatasetRequest,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, Any]:
    job_payload = {"source_type": "url", **payload.model_dump(exclude_none=True)}
    _attach_cleaning_model(settings, job_payload)
    return _create_job(settings, "dataset_import", f"url-{payload.slug}", job_payload)


@router.post("/{dataset_id}/approve")
def approve_dataset(
    dataset_id: str,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, Any]:
    version = approve_dataset_version(settings.database_path, dataset_id)
    if not version:
        return {"approved": False, "detail": "Dataset not found."}
    return {"approved": True, "version": version}


@router.post("/{dataset_id}/reject")
def reject_dataset(
    dataset_id: str,
    settings: Annotated[Settings, Depends(settings_dependency)],
    _user: Annotated[dict[str, Any], Depends(current_user)],
) -> dict[str, Any]:
    result = delete_dataset_version(settings.database_path, settings.data_root, dataset_id)
    if not result:
        return {"rejected": False, "detail": "Dataset not found."}
    return {"rejected": True, "deleted": True, **result}


def _create_job(settings: Settings, job_type: str, slug: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return create_and_start_job(settings, job_type, slug, payload)
    except JobValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _attach_cleaning_model(settings: Settings, payload: dict[str, Any]) -> None:
    if not payload.get("clean_with_inference") or payload.get("cleaning_model"):
        return
    target = get_active_inference_target(settings.database_path)
    payload["cleaning_model"] = target.get("display_name") or target.get("provider_id") or target.get("path") or "local-inference"


def _ensure_confirmed_sha(repo_id: str, resource_type: str, confirmed_sha: str | None, revision: str | None) -> None:
    try:
        ensure_confirmed_hub_sha(repo_id, resource_type, confirmed_sha, revision)
    except HubResolveError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
