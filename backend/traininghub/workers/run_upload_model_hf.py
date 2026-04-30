from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from traininghub.core.database import connect, row_to_dict
from traininghub.workers.common import WorkerContext, run_worker


def main(context: WorkerContext, payload: dict[str, Any]) -> None:
    source_path = _resolve_source_path(context, payload)
    repo_id = str(payload["repo_id"])
    report = {
        "repo_id": repo_id,
        "source_path": str(source_path),
        "private": bool(payload.get("private", True)),
        "dry_run": bool(payload.get("dry_run", False)),
        "large_folder": bool(payload.get("large_folder", False)),
    }
    if payload.get("dry_run", False):
        report_path = context.write_metadata("hf_upload_report.json", report)
        context.register_artifact(report_path, "hf_upload_report", "Hugging Face upload dry-run report", report)
        return

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required to upload models to Hugging Face.")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for Hugging Face uploads.") from exc

    api = HfApi(token=token)
    context.event("hf_repo", "Ensuring Hugging Face model repo exists.", data={"repo_id": repo_id})
    api.create_repo(repo_id=repo_id, repo_type="model", private=bool(payload.get("private", True)), exist_ok=True)

    commit_message = str(payload.get("commit_message") or "Upload TrainingHub model artifact")
    if source_path.is_file():
        path_in_repo = str(payload.get("path_in_repo") or source_path.name)
        context.event("hf_upload", "Uploading model file to Hugging Face.", data={"path_in_repo": path_in_repo})
        result = api.upload_file(
            path_or_fileobj=str(source_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message=commit_message,
        )
    elif payload.get("large_folder", False):
        context.event("hf_upload", "Uploading large model folder to Hugging Face.", data={"folder_path": str(source_path)})
        api.upload_large_folder(
            repo_id=repo_id,
            folder_path=str(source_path),
            repo_type="model",
            private=bool(payload.get("private", True)),
            allow_patterns=_patterns(payload.get("include_patterns")) or None,
            ignore_patterns=_patterns(payload.get("exclude_patterns")) or None,
        )
        result = "upload_large_folder completed"
    else:
        context.event("hf_upload", "Uploading model folder to Hugging Face.", data={"folder_path": str(source_path)})
        result = api.upload_folder(
            folder_path=str(source_path),
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message=commit_message,
            allow_patterns=_patterns(payload.get("include_patterns")) or None,
            ignore_patterns=_patterns(payload.get("exclude_patterns")) or None,
        )

    report["result"] = str(result)
    report_path = context.write_metadata("hf_upload_report.json", report)
    context.register_artifact(report_path, "hf_upload_report", "Hugging Face upload report", report)
    context.event("model_uploaded", "Model uploaded to Hugging Face.", data={"repo_id": repo_id, "source_path": str(source_path)})


def _resolve_source_path(context: WorkerContext, payload: dict[str, Any]) -> Path:
    if payload.get("source_path"):
        return _existing_path(payload["source_path"])
    if payload.get("artifact_id"):
        with connect(context.database_path) as conn:
            row = conn.execute("SELECT * FROM artifacts WHERE artifact_id = ?", (payload["artifact_id"],)).fetchone()
        artifact = row_to_dict(row)
        if not artifact:
            raise RuntimeError("Artifact not found.")
        return _existing_path(artifact["path"])
    if payload.get("model_slug"):
        model_slug = str(payload["model_slug"])
        with connect(context.database_path) as conn:
            row = conn.execute("SELECT * FROM model_registry WHERE slug = ?", (model_slug,)).fetchone()
        model = row_to_dict(row)
        if not model:
            raise RuntimeError("Model not found.")
        metadata = json.loads(model.get("metadata_json") or "{}")
        local_path = metadata.get("local_path")
        if local_path:
            return _existing_path(local_path)
        candidate = Path(os.environ["TRAININGHUB_DATA_ROOT"]) / "models" / model_slug
        return _existing_path(candidate)
    raise RuntimeError("model_slug, artifact_id, or source_path is required for upload.")


def _existing_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        raise RuntimeError(f"Upload source path does not exist: {path}")
    return path


def _patterns(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


if __name__ == "__main__":
    sys.exit(run_worker(main))
