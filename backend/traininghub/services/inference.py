from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from traininghub.core.database import connect, row_to_dict, rows_to_dicts
from traininghub.core.security import utc_now
from traininghub.services.model_files import local_provider_path, local_provider_is_runnable
from traininghub.services.model_registry import fallback_inference_target


SETTING_KEY = "active_inference_target"
RUNNABLE_GGUF_TYPES = {"gguf_fp16", "gguf_quantized"}


class InferenceTargetError(ValueError):
    pass


def default_inference_target(database_path: Path) -> dict[str, Any] | None:
    return fallback_inference_target(database_path)


def get_active_inference_target(database_path: Path) -> dict[str, Any] | None:
    with connect(database_path) as conn:
        row = conn.execute("SELECT value_json FROM settings WHERE key = ?", (SETTING_KEY,)).fetchone()
    if row is None:
        return _replace_with_fallback_target(database_path)
    try:
        target = json.loads(row["value_json"])
    except json.JSONDecodeError:
        return _replace_with_fallback_target(database_path)
    try:
        validate_inference_target(database_path, target)
    except InferenceTargetError:
        return _replace_with_fallback_target(database_path)
    return target


def set_active_inference_target(database_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    target = validate_inference_target(database_path, payload)
    target["updated_at"] = utc_now()
    with connect(database_path) as conn:
        conn.execute(
            """
            INSERT INTO settings (key, value_json)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
            """,
            (SETTING_KEY, json.dumps(target, sort_keys=True)),
        )
    return target


def _replace_with_fallback_target(database_path: Path) -> dict[str, Any] | None:
    target = default_inference_target(database_path)
    with connect(database_path) as conn:
        if target:
            target["updated_at"] = utc_now()
            conn.execute(
                """
                INSERT INTO settings (key, value_json)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
                """,
                (SETTING_KEY, json.dumps(target, sort_keys=True)),
            )
        else:
            conn.execute("DELETE FROM settings WHERE key = ?", (SETTING_KEY,))
    return target


def validate_inference_target(database_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    target_type = payload.get("target_type")
    capability_transfer_id = str(payload.get("capability_transfer_id") or "")
    if target_type == "base_model":
        model_slug = payload.get("model_slug")
        if not model_slug:
            raise InferenceTargetError("model_slug is required for base_model targets.")
        model = _get_model(database_path, model_slug)
        if not model:
            raise InferenceTargetError("Model not found.")
        if not model["supports_bf16_inference"]:
            raise InferenceTargetError("This model is not valid for BF16/FP16 local inference on Morrigan. Select a GGUF artifact instead.")
        if local_provider_path(model["provider_id"]) and not local_provider_is_runnable(model["provider_id"]):
            raise InferenceTargetError("This local checkpoint is missing model weights or tokenizer files. Run real training before selecting it for inference.")
        transfer_id = _validate_transfer_for_target(database_path, capability_transfer_id, target_type, model_slug, "")
        return _base_model_target(model, updated_at=payload.get("updated_at", 0), capability_transfer_id=transfer_id)
    if target_type == "gguf_artifact":
        artifact_id = payload.get("artifact_id")
        if not artifact_id:
            raise InferenceTargetError("artifact_id is required for gguf_artifact targets.")
        artifact = _get_artifact(database_path, artifact_id)
        if not artifact:
            raise InferenceTargetError("Artifact not found.")
        if artifact["artifact_type"] not in RUNNABLE_GGUF_TYPES:
            raise InferenceTargetError("Artifact is not a runnable GGUF inference target.")
        model_slug = str(payload.get("model_slug") or artifact["metadata"].get("model_slug", ""))
        transfer_id = _validate_transfer_for_target(database_path, capability_transfer_id, target_type, model_slug, artifact["artifact_id"])
        target = {
            "target_type": "gguf_artifact",
            "model_slug": model_slug,
            "artifact_id": artifact["artifact_id"],
            "display_name": artifact["display_name"],
            "provider_id": "",
            "path": artifact["path"],
            "updated_at": int(payload.get("updated_at", 0)),
        }
        if transfer_id:
            target["capability_transfer_id"] = transfer_id
        return target
    raise InferenceTargetError("target_type must be base_model or gguf_artifact.")


def list_inference_options(database_path: Path) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    with connect(database_path) as conn:
        models = rows_to_dicts(conn.execute("SELECT * FROM model_registry ORDER BY family, display_name").fetchall())
        artifacts = rows_to_dicts(conn.execute("SELECT * FROM artifacts ORDER BY created_at DESC").fetchall())
    for model in models:
        model["metadata"] = json.loads(model.pop("metadata_json") or "{}")
        model["supports_bf16_inference"] = bool(model["supports_bf16_inference"])
        local_checkpoint_missing = bool(local_provider_path(model["provider_id"])) and not local_provider_is_runnable(model["provider_id"])
        enabled = bool(model["supports_bf16_inference"]) and not local_checkpoint_missing
        options.append(
            {
                "target_type": "base_model",
                "model_slug": model["slug"],
                "artifact_id": "",
                "display_name": model["display_name"],
                "description": model["hardware_note"],
                "enabled": enabled,
                "disabled_reason": "" if enabled else _disabled_model_reason(model, local_checkpoint_missing),
                "provider_id": model["provider_id"],
                "path": "",
            }
        )
    for artifact in artifacts:
        artifact["metadata"] = json.loads(artifact.pop("metadata_json") or "{}")
        enabled = artifact["artifact_type"] in RUNNABLE_GGUF_TYPES
        options.append(
            {
                "target_type": "gguf_artifact",
                "model_slug": str(artifact["metadata"].get("model_slug", "")),
                "artifact_id": artifact["artifact_id"],
                "display_name": artifact["display_name"],
                "description": f"{artifact['artifact_type']} at {artifact['path']}",
                "enabled": enabled,
                "disabled_reason": "" if enabled else "Only GGUF artifacts are valid local inference targets.",
                "provider_id": "",
                "path": artifact["path"],
            }
        )
    return options


def _disabled_model_reason(model: dict[str, Any], local_checkpoint_missing: bool = False) -> str:
    if local_checkpoint_missing:
        return "Local checkpoint is missing model weights or tokenizer files. Run real training before selecting it for inference."
    return model["hardware_note"]


def _base_model_target(model: dict[str, Any], updated_at: int, capability_transfer_id: str = "") -> dict[str, Any]:
    target = {
        "target_type": "base_model",
        "model_slug": model["slug"],
        "artifact_id": "",
        "display_name": model["display_name"],
        "provider_id": model["provider_id"],
        "path": "",
        "updated_at": int(updated_at),
    }
    if capability_transfer_id:
        target["capability_transfer_id"] = capability_transfer_id
    return target


def _validate_transfer_for_target(
    database_path: Path,
    transfer_id: str,
    target_type: str,
    model_slug: str,
    artifact_id: str,
) -> str:
    if not transfer_id:
        return ""
    with connect(database_path) as conn:
        row = conn.execute("SELECT * FROM capability_transfers WHERE transfer_id = ?", (transfer_id,)).fetchone()
    transfer = row_to_dict(row)
    if not transfer or transfer["status"] != "ready":
        raise InferenceTargetError("Capability transfer is not ready.")
    config = json.loads(transfer.pop("config_json") or "{}")
    if target_type == "base_model":
        if transfer["target_runtime"] != "transformers" or transfer["target_model_slug"] != model_slug:
            raise InferenceTargetError("Capability transfer target does not match this base model.")
        return transfer_id
    target_artifact_id = str(config.get("target_artifact_id") or "")
    if transfer["target_runtime"] != "llama_cpp" or (target_artifact_id and target_artifact_id != artifact_id):
        raise InferenceTargetError("Capability transfer target does not match this GGUF artifact.")
    return transfer_id


def _get_model(database_path: Path, model_slug: str) -> dict[str, Any] | None:
    with connect(database_path) as conn:
        row = conn.execute("SELECT * FROM model_registry WHERE slug = ?", (model_slug,)).fetchone()
    model = row_to_dict(row)
    if not model:
        return None
    model["supports_bf16_inference"] = bool(model["supports_bf16_inference"])
    model["metadata"] = json.loads(model.pop("metadata_json") or "{}")
    return model


def _get_artifact(database_path: Path, artifact_id: str) -> dict[str, Any] | None:
    with connect(database_path) as conn:
        row = conn.execute("SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)).fetchone()
    artifact = row_to_dict(row)
    if not artifact:
        return None
    artifact["metadata"] = json.loads(artifact.pop("metadata_json") or "{}")
    return artifact
