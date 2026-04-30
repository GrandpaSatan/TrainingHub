from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from traininghub.core.config import Settings
from traininghub.core.database import connect, row_to_dict, rows_to_dicts
from traininghub.core.id_utils import make_job_id, slugify
from traininghub.core.security import utc_now
from traininghub.services.calibration_pairs import CalibrationDatasetError, validate_calibration_dataset
from traininghub.services.datasets import get_approved_version
from traininghub.services.deletion import safe_remove_traininghub_path
from traininghub.services.inference import RUNNABLE_GGUF_TYPES, get_active_inference_target, set_active_inference_target


TRANSFER_STATUSES = {"extracting", "extracted", "aligning", "ready", "failed", "deleted"}
TRANSFER_RUNTIMES = {"transformers", "llama_cpp"}
ACTIVE_INFERENCE_SETTING_KEY = "active_inference_target"


class CapabilityTransferError(ValueError):
    pass


def list_transfers(database_path: Path, include_deleted: bool = False) -> list[dict[str, Any]]:
    where = "" if include_deleted else "WHERE status != 'deleted'"
    with connect(database_path) as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM capability_transfers
            {where}
            ORDER BY created_at DESC
            """
        ).fetchall()
    return [_hydrate_transfer(row) for row in rows_to_dicts(rows)]


def get_transfer(database_path: Path, transfer_id: str, include_deleted: bool = False) -> dict[str, Any] | None:
    with connect(database_path) as conn:
        row = conn.execute("SELECT * FROM capability_transfers WHERE transfer_id = ?", (transfer_id,)).fetchone()
    transfer = _hydrate_transfer(row_to_dict(row))
    if not transfer or (transfer["status"] == "deleted" and not include_deleted):
        return None
    return transfer


def create_transfer(settings: Settings, payload: dict[str, Any]) -> dict[str, Any]:
    source = _validate_endpoint_ref(settings.database_path, "source", payload)
    target = _validate_endpoint_ref(settings.database_path, "target", payload)
    rank = _int_range(payload.get("rank", 16), 1, 256, "rank")
    layer_targets = _normalize_layer_targets(payload.get("layer_targets", "all"))
    contrast_mode = str(payload.get("contrast_mode") or "prompt_pair")
    if contrast_mode not in {"prompt_pair", "system_pair"}:
        raise CapabilityTransferError("contrast_mode must be prompt_pair or system_pair.")
    calibration = _approved_calibration(settings.database_path, str(payload.get("calibration_dataset_id") or ""), contrast_mode)
    display_name = str(payload.get("display_name") or "").strip() or f"{source['display_name']} to {target['display_name']}"
    transfer_id = make_job_id("ct", display_name)
    now = utc_now()
    config = {
        "calibration_dataset_id": calibration["dataset_id"],
        "calibration_dataset_version_id": calibration["version_id"],
        "calibration_dataset_jsonl_path": calibration["jsonl_path"],
        "contrast_mode": contrast_mode,
        "rank": rank,
        "source_artifact_id": source.get("artifact_id", ""),
        "target_artifact_id": target.get("artifact_id", ""),
        "source_display_name": source["display_name"],
        "target_display_name": target["display_name"],
        "source_model_id": source.get("model_id", source["model_slug"]),
        "target_model_id": target.get("model_id", target["model_slug"]),
        "source_artifact_path": source.get("path", ""),
        "target_artifact_path": target.get("path", ""),
        "dry_run": bool(payload.get("dry_run", False)),
        "degraded_mode": source["runtime"] == "llama_cpp" or target["runtime"] == "llama_cpp",
        "extract_job_id": "",
        "align_job_id": "",
        "layer_pairs": [],
    }
    with connect(settings.database_path) as conn:
        conn.execute(
            """
            INSERT INTO capability_transfers (
                transfer_id, display_name, source_model_slug, source_runtime,
                target_model_slug, target_runtime, vector_artifact_id,
                alignment_artifact_id, alpha, layer_targets_json, status,
                config_json, created_at, updated_at, deleted_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transfer_id,
                display_name,
                source["model_slug"],
                source["runtime"],
                target["model_slug"],
                target["runtime"],
                None,
                None,
                1.0,
                json.dumps(layer_targets, sort_keys=True),
                "extracting",
                json.dumps(config, sort_keys=True),
                now,
                now,
                None,
            ),
        )

    try:
        from traininghub.services.jobs import create_and_start_job

        job = create_and_start_job(
            settings,
            "extract_capability",
            f"extract-{slugify(display_name, 'capability')}",
            {
                "transfer_id": transfer_id,
                "source_model_slug": source["model_slug"],
                "source_runtime": source["runtime"],
                "source_artifact_id": source.get("artifact_id", ""),
                "source_model_id": source.get("model_id", source["model_slug"]),
                "source_artifact_path": source.get("path", ""),
                "calibration_dataset_id": calibration["dataset_id"],
                "dataset_version_id": calibration["version_id"],
                "dataset_jsonl_path": calibration["jsonl_path"],
                "contrast_mode": contrast_mode,
                "layer_targets": layer_targets,
                "dry_run": bool(payload.get("dry_run", False)),
            },
        )
        _patch_config(settings.database_path, transfer_id, {"extract_job_id": job.get("job_id", "")})
    except Exception:
        update_transfer_status(settings.database_path, transfer_id, "failed")
        raise
    transfer = get_transfer(settings.database_path, transfer_id)
    if not transfer:
        raise CapabilityTransferError("Capability transfer was not created.")
    return transfer


def queue_alignment(settings: Settings, transfer_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    transfer = _require_transfer(settings.database_path, transfer_id)
    if transfer["status"] != "extracted":
        raise CapabilityTransferError("Capability extraction must complete before alignment.")
    if not transfer.get("vector_artifact_id"):
        raise CapabilityTransferError("Capability vector artifact is missing.")
    rank = _int_range(payload.get("rank", transfer["config"].get("rank", 16)), 1, 256, "rank")
    layer_pairs = _normalize_layer_pairs(payload.get("layer_pairs") or transfer["config"].get("layer_pairs") or [])
    config_patch = {"rank": rank, "layer_pairs": layer_pairs}
    _patch_config(settings.database_path, transfer_id, config_patch)
    update_transfer_status(settings.database_path, transfer_id, "aligning")

    try:
        from traininghub.services.jobs import create_and_start_job

        refreshed = _require_transfer(settings.database_path, transfer_id)
        job = create_and_start_job(
            settings,
            "align_capability",
            f"align-{slugify(refreshed['display_name'], 'capability')}",
            {
                "transfer_id": transfer_id,
                "source_model_slug": refreshed["source_model_slug"],
                "target_model_slug": refreshed["target_model_slug"],
                "source_runtime": refreshed["source_runtime"],
                "target_runtime": refreshed["target_runtime"],
                "source_artifact_id": refreshed["config"].get("source_artifact_id", ""),
                "target_artifact_id": refreshed["config"].get("target_artifact_id", ""),
                "source_model_id": refreshed["config"].get("source_model_id", refreshed["source_model_slug"]),
                "target_model_id": refreshed["config"].get("target_model_id", refreshed["target_model_slug"]),
                "source_artifact_path": refreshed["config"].get("source_artifact_path", ""),
                "target_artifact_path": refreshed["config"].get("target_artifact_path", ""),
                "vector_artifact_id": refreshed["vector_artifact_id"],
                "calibration_dataset_id": refreshed["config"].get("calibration_dataset_id", ""),
                "dataset_version_id": refreshed["config"].get("calibration_dataset_version_id", ""),
                "dataset_jsonl_path": refreshed["config"].get("calibration_dataset_jsonl_path", ""),
                "rank": rank,
                "layer_pairs": layer_pairs,
                "dry_run": bool(refreshed["config"].get("dry_run", False)),
            },
        )
        _patch_config(settings.database_path, transfer_id, {"align_job_id": job.get("job_id", "")})
    except Exception:
        update_transfer_status(settings.database_path, transfer_id, "failed")
        raise
    return _require_transfer(settings.database_path, transfer_id)


def activate_transfer(settings: Settings, transfer_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    transfer = _require_transfer(settings.database_path, transfer_id)
    if transfer["status"] != "ready":
        raise CapabilityTransferError("Only ready capability transfers can be activated.")
    alpha = _float_range(payload.get("alpha", transfer["alpha"]), 0.0, 4.0, "alpha")
    layer_targets = _normalize_layer_targets(payload.get("layer_targets", transfer["layer_targets"]))
    _update_alpha_and_layers(settings.database_path, transfer_id, alpha, layer_targets)
    refreshed = _require_transfer(settings.database_path, transfer_id)
    target_payload = _target_payload_for_transfer(refreshed)
    target_payload["capability_transfer_id"] = transfer_id
    active_target = set_active_inference_target(settings.database_path, target_payload)
    return {
        "transfer": refreshed,
        "active_target": active_target,
        "warning": _degraded_warning(refreshed),
    }


def deactivate_transfer(settings: Settings, transfer_id: str) -> dict[str, Any]:
    current = get_active_inference_target(settings.database_path)
    if current.get("capability_transfer_id") == transfer_id:
        current = {key: value for key, value in current.items() if key != "capability_transfer_id"}
        set_active_inference_target(settings.database_path, current)
    transfer = get_transfer(settings.database_path, transfer_id, include_deleted=True)
    return {"transfer": transfer, "active_target": get_active_inference_target(settings.database_path)}


def delete_transfer(settings: Settings, transfer_id: str) -> dict[str, Any]:
    transfer = _require_transfer(settings.database_path, transfer_id, include_deleted=True)
    removed_paths: list[str] = []
    artifact_ids = [value for value in [transfer.get("vector_artifact_id"), transfer.get("alignment_artifact_id")] if value]
    with connect(settings.database_path) as conn:
        artifacts = rows_to_dicts(
            conn.execute(
                f"SELECT * FROM artifacts WHERE artifact_id IN ({','.join('?' for _ in artifact_ids)})",
                artifact_ids,
            ).fetchall()
            if artifact_ids
            else []
        )
    for artifact in artifacts:
        removed_paths.extend(safe_remove_traininghub_path(artifact["path"], settings.data_root))
    now = utc_now()
    with connect(settings.database_path) as conn:
        if artifact_ids:
            conn.executemany("DELETE FROM artifacts WHERE artifact_id = ?", [(artifact_id,) for artifact_id in artifact_ids])
        conn.execute(
            """
            UPDATE capability_transfers
            SET status = 'deleted',
                vector_artifact_id = NULL,
                alignment_artifact_id = NULL,
                updated_at = ?,
                deleted_at = ?
            WHERE transfer_id = ?
            """,
            (now, now, transfer_id),
        )
    deactivate_transfer(settings, transfer_id)
    return {"deleted": True, "transfer_id": transfer_id, "removed_paths": removed_paths, "removed_artifacts": artifact_ids}


def update_transfer_status(database_path: Path, transfer_id: str, status: str) -> None:
    if status not in TRANSFER_STATUSES:
        raise CapabilityTransferError(f"Unsupported transfer status: {status}")
    with connect(database_path) as conn:
        conn.execute(
            "UPDATE capability_transfers SET status = ?, updated_at = ? WHERE transfer_id = ?",
            (status, utc_now(), transfer_id),
        )


def mark_extracted(database_path: Path, transfer_id: str, vector_artifact_id: str, metadata: dict[str, Any] | None = None) -> None:
    if metadata:
        _patch_config(database_path, transfer_id, metadata)
    with connect(database_path) as conn:
        conn.execute(
            """
            UPDATE capability_transfers
            SET vector_artifact_id = ?, status = 'extracted', updated_at = ?
            WHERE transfer_id = ? AND status != 'deleted'
            """,
            (vector_artifact_id, utc_now(), transfer_id),
        )


def mark_aligned(database_path: Path, transfer_id: str, alignment_artifact_id: str, metadata: dict[str, Any] | None = None) -> None:
    if metadata:
        _patch_config(database_path, transfer_id, metadata)
    with connect(database_path) as conn:
        conn.execute(
            """
            UPDATE capability_transfers
            SET alignment_artifact_id = ?, status = 'ready', updated_at = ?
            WHERE transfer_id = ? AND status != 'deleted'
            """,
            (alignment_artifact_id, utc_now(), transfer_id),
        )


def validate_capability_job_payload(settings: Settings, job_type: str, payload: dict[str, Any]) -> None:
    transfer_id = str(payload.get("transfer_id") or "")
    transfer = _require_transfer(settings.database_path, transfer_id)
    if job_type == "extract_capability" and transfer["status"] not in {"extracting", "failed"}:
        raise CapabilityTransferError("Capability extraction job is not valid for the current transfer status.")
    if job_type == "align_capability" and transfer["status"] not in {"aligning", "failed"}:
        raise CapabilityTransferError("Capability alignment job is not valid for the current transfer status.")
    calibration_dataset_id = str(payload.get("calibration_dataset_id") or transfer["config"].get("calibration_dataset_id") or "")
    contrast_mode = str(payload.get("contrast_mode") or transfer["config"].get("contrast_mode") or "prompt_pair")
    _approved_calibration(settings.database_path, calibration_dataset_id, contrast_mode)
    if job_type == "align_capability":
        vector_id = str(payload.get("vector_artifact_id") or "")
        if not vector_id or not _get_artifact(settings.database_path, vector_id):
            raise CapabilityTransferError("Capability vector artifact not found.")


def transfer_for_inference(database_path: Path, transfer_id: str) -> dict[str, Any] | None:
    transfer = get_transfer(database_path, transfer_id)
    if not transfer or transfer["status"] != "ready":
        return None
    vector = _get_artifact(database_path, str(transfer.get("vector_artifact_id") or ""))
    alignment = _get_artifact(database_path, str(transfer.get("alignment_artifact_id") or ""))
    transfer["vector_artifact"] = vector
    transfer["alignment_artifact"] = alignment
    return transfer if vector and alignment else None


def _hydrate_transfer(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    config = _json(row.pop("config_json", "{}"), {})
    layer_targets = _json(row.pop("layer_targets_json", '"all"'), "all")
    row["config"] = config
    row["layer_targets"] = layer_targets
    row["alpha"] = float(row.get("alpha") or 0.0)
    row["extract_job_id"] = config.get("extract_job_id", "")
    row["align_job_id"] = config.get("align_job_id", "")
    row["source_artifact_id"] = config.get("source_artifact_id", "")
    row["target_artifact_id"] = config.get("target_artifact_id", "")
    row["source_display_name"] = config.get("source_display_name") or row.get("source_model_slug", "")
    row["target_display_name"] = config.get("target_display_name") or row.get("target_model_slug", "")
    row["degraded_mode"] = bool(config.get("degraded_mode") or row.get("source_runtime") == "llama_cpp" or row.get("target_runtime") == "llama_cpp")
    return row


def _require_transfer(database_path: Path, transfer_id: str, include_deleted: bool = False) -> dict[str, Any]:
    if not transfer_id:
        raise CapabilityTransferError("transfer_id is required.")
    transfer = get_transfer(database_path, transfer_id, include_deleted=include_deleted)
    if not transfer:
        raise CapabilityTransferError("Capability transfer not found.")
    return transfer


def _validate_endpoint_ref(database_path: Path, role: str, payload: dict[str, Any]) -> dict[str, str]:
    runtime = str(payload.get(f"{role}_runtime") or "transformers")
    if runtime not in TRANSFER_RUNTIMES:
        raise CapabilityTransferError(f"{role}_runtime must be transformers or llama_cpp.")
    model_slug = str(payload.get(f"{role}_model_slug") or "")
    artifact_id = str(payload.get(f"{role}_artifact_id") or "")
    if runtime == "transformers":
        model = _get_model(database_path, model_slug)
        if not model:
            raise CapabilityTransferError(f"{role} model not found.")
        if not bool(model.get("supports_bf16_inference")):
            raise CapabilityTransferError(f"{role} model is not valid for local Transformers inference.")
        return {
            "runtime": runtime,
            "model_slug": model["slug"],
            "model_id": model["provider_id"],
            "artifact_id": "",
            "path": "",
            "display_name": model["display_name"],
        }
    artifact = _get_artifact(database_path, artifact_id or model_slug)
    if not artifact:
        raise CapabilityTransferError(f"{role} GGUF artifact not found.")
    if artifact["artifact_type"] not in RUNNABLE_GGUF_TYPES:
        raise CapabilityTransferError(f"{role} artifact is not a runnable GGUF artifact.")
    artifact_model_slug = str(artifact["metadata"].get("model_slug") or model_slug or artifact["artifact_id"])
    return {
        "runtime": runtime,
        "model_slug": artifact_model_slug,
        "model_id": artifact_model_slug,
        "artifact_id": artifact["artifact_id"],
        "path": artifact["path"],
        "display_name": artifact["display_name"],
    }


def _target_payload_for_transfer(transfer: dict[str, Any]) -> dict[str, Any]:
    if transfer["target_runtime"] == "transformers":
        return {"target_type": "base_model", "model_slug": transfer["target_model_slug"], "artifact_id": ""}
    artifact_id = str(transfer["config"].get("target_artifact_id") or transfer.get("target_artifact_id") or "")
    if not artifact_id:
        raise CapabilityTransferError("Target GGUF artifact is missing.")
    return {"target_type": "gguf_artifact", "model_slug": transfer["target_model_slug"], "artifact_id": artifact_id}


def _approved_calibration(database_path: Path, dataset_id: str, contrast_mode: str) -> dict[str, Any]:
    if not dataset_id:
        raise CapabilityTransferError("calibration_dataset_id is required.")
    approved = get_approved_version(database_path, dataset_id)
    if not approved:
        raise CapabilityTransferError("Calibration dataset must be approved before use.")
    with connect(database_path) as conn:
        dataset = conn.execute("SELECT dataset_type FROM datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
    if not dataset or dataset["dataset_type"] != "capability_calibration":
        raise CapabilityTransferError("Calibration dataset must use dataset_type capability_calibration.")
    try:
        summary = validate_calibration_dataset(Path(str(approved["jsonl_path"])), contrast_mode)
    except CalibrationDatasetError as exc:
        raise CapabilityTransferError(str(exc)) from exc
    approved["calibration_pair_count"] = summary["pair_count"]
    approved["dataset_id"] = dataset_id
    return approved


def _get_model(database_path: Path, model_slug: str) -> dict[str, Any] | None:
    with connect(database_path) as conn:
        row = conn.execute("SELECT * FROM model_registry WHERE slug = ?", (model_slug,)).fetchone()
    model = row_to_dict(row)
    if not model:
        return None
    model["metadata"] = _json(model.pop("metadata_json") or "{}", {})
    model["supports_bf16_inference"] = bool(model.get("supports_bf16_inference"))
    return model


def _get_artifact(database_path: Path, artifact_id: str) -> dict[str, Any] | None:
    if not artifact_id:
        return None
    with connect(database_path) as conn:
        row = conn.execute("SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)).fetchone()
    artifact = row_to_dict(row)
    if not artifact:
        return None
    artifact["metadata"] = _json(artifact.pop("metadata_json") or "{}", {})
    return artifact


def _patch_config(database_path: Path, transfer_id: str, patch: dict[str, Any]) -> None:
    transfer = _require_transfer(database_path, transfer_id, include_deleted=True)
    config = {**transfer["config"], **patch}
    with connect(database_path) as conn:
        conn.execute(
            "UPDATE capability_transfers SET config_json = ?, updated_at = ? WHERE transfer_id = ?",
            (json.dumps(config, sort_keys=True), utc_now(), transfer_id),
        )


def _update_alpha_and_layers(database_path: Path, transfer_id: str, alpha: float, layer_targets: str | list[int]) -> None:
    with connect(database_path) as conn:
        conn.execute(
            """
            UPDATE capability_transfers
            SET alpha = ?, layer_targets_json = ?, updated_at = ?
            WHERE transfer_id = ? AND status != 'deleted'
            """,
            (alpha, json.dumps(layer_targets, sort_keys=True), utc_now(), transfer_id),
        )


def _normalize_layer_targets(raw: Any) -> str | list[int]:
    if raw is None or raw == "":
        return "all"
    if isinstance(raw, str):
        raw = raw.strip()
        if raw in {"all", "last", "every-4"}:
            return raw
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = [part.strip() for part in raw.split(",") if part.strip()]
        return _normalize_layer_targets(parsed)
    if isinstance(raw, list):
        values = sorted({int(item) for item in raw})
        if any(value < 0 for value in values):
            raise CapabilityTransferError("Layer targets must be non-negative.")
        return values
    raise CapabilityTransferError("layer_targets must be all, last, every-4, or a list of layer indexes.")


def _normalize_layer_pairs(raw: Any) -> list[list[int]]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CapabilityTransferError("layer_pairs must be JSON pairs.") from exc
    if not isinstance(raw, list):
        raise CapabilityTransferError("layer_pairs must be a list.")
    pairs: list[list[int]] = []
    for item in raw:
        if not isinstance(item, list | tuple) or len(item) != 2:
            raise CapabilityTransferError("Each layer pair must be [source_layer, target_layer].")
        source_layer = int(item[0])
        target_layer = int(item[1])
        if source_layer < 0 or target_layer < 0:
            raise CapabilityTransferError("Layer pairs must be non-negative.")
        pairs.append([source_layer, target_layer])
    return pairs


def _int_range(raw: Any, low: int, high: int, name: str) -> int:
    value = int(raw)
    if value < low or value > high:
        raise CapabilityTransferError(f"{name} must be between {low} and {high}.")
    return value


def _float_range(raw: Any, low: float, high: float, name: str) -> float:
    value = float(raw)
    if value < low or value > high:
        raise CapabilityTransferError(f"{name} must be between {low} and {high}.")
    return value


def _degraded_warning(transfer: dict[str, Any]) -> str:
    if transfer["target_runtime"] == "llama_cpp" or transfer["source_runtime"] == "llama_cpp":
        return "GGUF transfers run in last-layer degraded mode; use HF Transformers targets for per-layer steering."
    return ""


def _json(value: str, fallback: Any) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback
