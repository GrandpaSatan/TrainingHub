from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from traininghub.core.security import utc_now
from traininghub.services.deletion import safe_remove_traininghub_path
from traininghub.services.model_files import local_provider_is_runnable, local_provider_path


DEFAULT_MODELS: list[dict[str, Any]] = [
    {
        "slug": "lfm25-12b-base",
        "provider_id": "LiquidAI/LFM2.5-1.2B-Base",
        "display_name": "LFM2.5 1.2B Base",
        "family": "lfm25",
        "parameter_count": "1.2B",
        "supports_bf16_inference": True,
        "supports_benchmark": True,
        "supports_quantization": True,
        "supports_gguf_path": True,
        "supports_full_finetune": True,
        "hardware_note": "Small base model. BF16 diagnostics, benchmark, conversion, and quantized GGUF paths are viable.",
        "default_dtype": "bf16",
        "max_sequence_length": 3072,
        "metadata": {"route": "base_model"},
    },
    {
        "slug": "lfm25-12b-instruct",
        "provider_id": "LiquidAI/LFM2.5-1.2B-Instruct",
        "display_name": "LFM2.5 1.2B Instruct",
        "family": "lfm25",
        "parameter_count": "1.2B",
        "supports_bf16_inference": True,
        "supports_benchmark": True,
        "supports_quantization": True,
        "supports_gguf_path": True,
        "supports_full_finetune": True,
        "hardware_note": "Small base model. BF16 diagnostics, benchmark, conversion, and quantized GGUF paths are viable.",
        "default_dtype": "bf16",
        "max_sequence_length": 3072,
        "metadata": {"route": "base_model"},
    },
    {
        "slug": "gemma4-e2b-it",
        "provider_id": "google/gemma-4-E2B-it",
        "display_name": "Gemma 4 E2B IT",
        "family": "gemma4",
        "parameter_count": "E2B",
        "supports_bf16_inference": True,
        "supports_benchmark": True,
        "supports_quantization": True,
        "supports_gguf_path": True,
        "hardware_note": "Small base model. BF16 diagnostics and quantized GGUF paths are viable on Morrigan.",
        "default_dtype": "bf16",
        "max_sequence_length": 3072,
        "metadata": {"route": "base_model"},
    },
    {
        "slug": "gemma4-e4b-it",
        "provider_id": "google/gemma-4-E4B-it",
        "display_name": "Gemma 4 E4B IT",
        "family": "gemma4",
        "parameter_count": "E4B",
        "supports_bf16_inference": True,
        "supports_benchmark": True,
        "supports_quantization": True,
        "supports_gguf_path": True,
        "hardware_note": "Small base model. Use conservative BF16 diagnostics or route to GGUF for serving.",
        "default_dtype": "bf16",
        "max_sequence_length": 2048,
        "metadata": {"route": "base_model"},
    },
    {
        "slug": "qwen3-4b",
        "provider_id": "Qwen/Qwen3-4B",
        "display_name": "Qwen3 4B",
        "family": "qwen3",
        "parameter_count": "4B",
        "supports_bf16_inference": True,
        "supports_benchmark": True,
        "supports_quantization": True,
        "supports_gguf_path": True,
        "hardware_note": "Small base model. BF16 diagnostics, benchmark, and quantized GGUF paths are supported.",
        "default_dtype": "bf16",
        "max_sequence_length": 2048,
        "metadata": {"route": "base_model"},
    },
    {
        "slug": "qwen35-35b-a3b",
        "provider_id": "Qwen/Qwen3.5-35B-A3B",
        "display_name": "Qwen3.5 35B A3B",
        "family": "qwen35",
        "parameter_count": "35B-A3B",
        "supports_bf16_inference": False,
        "supports_benchmark": True,
        "supports_quantization": True,
        "supports_gguf_path": True,
        "hardware_note": "Quantized GGUF only on Morrigan's two 12 GB GPUs. BF16/FP16 inference is disabled.",
        "default_dtype": "bf16",
        "max_sequence_length": 2048,
        "metadata": {"large_model": True, "route": "quantized_gguf"},
    },
    {
        "slug": "qwen36-35b-a3b",
        "provider_id": "Qwen/Qwen3.6-35B-A3B",
        "display_name": "Qwen3.6 35B A3B",
        "family": "qwen36",
        "parameter_count": "35B-A3B",
        "supports_bf16_inference": False,
        "supports_benchmark": True,
        "supports_quantization": True,
        "supports_gguf_path": True,
        "hardware_note": "Quantized GGUF only on Morrigan's two 12 GB GPUs. BF16/FP16 inference is disabled.",
        "default_dtype": "bf16",
        "max_sequence_length": 2048,
        "metadata": {"large_model": True, "route": "quantized_gguf"},
    },
    {
        "slug": "gemma4-26b-a4b-it",
        "provider_id": "google/gemma-4-26B-A4B-it",
        "display_name": "Gemma 4 26B A4B IT",
        "family": "gemma4",
        "parameter_count": "26B-A4B",
        "supports_bf16_inference": False,
        "supports_benchmark": True,
        "supports_quantization": True,
        "supports_gguf_path": True,
        "hardware_note": "Quantized GGUF only on Morrigan's two 12 GB GPUs. BF16/FP16 inference is disabled.",
        "default_dtype": "bf16",
        "max_sequence_length": 2048,
        "metadata": {"large_model": True, "route": "quantized_gguf"},
    },
    {
        "slug": "gemma4-31b-it",
        "provider_id": "google/gemma-4-31B-it",
        "display_name": "Gemma 4 31B IT",
        "family": "gemma4",
        "parameter_count": "31B",
        "supports_bf16_inference": False,
        "supports_benchmark": True,
        "supports_quantization": True,
        "supports_gguf_path": True,
        "hardware_note": "Quantized GGUF only on Morrigan's two 12 GB GPUs. BF16/FP16 inference is disabled.",
        "default_dtype": "bf16",
        "max_sequence_length": 2048,
        "metadata": {"large_model": True, "route": "quantized_gguf"},
    },
]


DEFAULT_MODEL_SLUGS = {model["slug"] for model in DEFAULT_MODELS}
ACTIVE_INFERENCE_SETTING_KEY = "active_inference_target"
RUNNABLE_GGUF_TYPES = {"gguf_fp16", "gguf_quantized"}


def delete_model_record(database_path: Path, data_root: Path, model_slug: str) -> dict[str, Any] | None:
    from traininghub.core.database import connect, row_to_dict

    with connect(database_path) as conn:
        row = conn.execute("SELECT * FROM model_registry WHERE slug = ?", (model_slug,)).fetchone()
        model = row_to_dict(row)
        if model is None:
            return None
        metadata = json.loads(model.pop("metadata_json") or "{}")
        artifacts = _model_artifacts(conn, model_slug, metadata)
        artifact_ids = [artifact["artifact_id"] for artifact in artifacts]
        candidate_paths = [artifact["path"] for artifact in artifacts]
        if metadata.get("local_path"):
            candidate_paths.insert(0, str(metadata["local_path"]))

        deleted_paths: list[str] = []
        for candidate in candidate_paths:
            deleted_paths.extend(path for path in safe_remove_traininghub_path(candidate, data_root) if path not in deleted_paths)

        if artifact_ids:
            conn.executemany("DELETE FROM artifacts WHERE artifact_id = ?", [(artifact_id,) for artifact_id in artifact_ids])
        if model_slug in DEFAULT_MODEL_SLUGS:
            conn.execute(
                """
                INSERT INTO model_delete_tombstones (slug, provider_id, display_name, deleted_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(slug) DO UPDATE SET
                    provider_id = excluded.provider_id,
                    display_name = excluded.display_name,
                    deleted_at = excluded.deleted_at
                """,
                (model_slug, model["provider_id"], model["display_name"], utc_now()),
            )
        conn.execute("DELETE FROM model_registry WHERE slug = ?", (model_slug,))
        _clear_active_target_if_needed(conn, model_slug, set(artifact_ids))
        return {
            "deleted": True,
            "blocked": False,
            "detail": "Model deleted.",
            "deleted_paths": deleted_paths,
            "removed_records": {"models": 1, "artifacts": len(artifact_ids)},
        }


def model_is_deletable(model_slug: str) -> bool:
    return True


def fallback_inference_target(database_path: Path) -> dict[str, Any] | None:
    from traininghub.core.database import connect

    with connect(database_path) as conn:
        return fallback_inference_target_from_connection(conn)


def fallback_inference_target_from_connection(conn: Any) -> dict[str, Any] | None:
    preferred = _base_model_target_for_slug(conn, "lfm25-12b-instruct")
    if preferred:
        return preferred
    rows = conn.execute("SELECT * FROM model_registry ORDER BY family, display_name").fetchall()
    for row in rows:
        model = dict(row)
        if _model_row_is_runnable(model):
            return _base_model_target(model)
    artifact_rows = conn.execute("SELECT * FROM artifacts ORDER BY created_at DESC").fetchall()
    for row in artifact_rows:
        artifact = dict(row)
        if artifact["artifact_type"] not in RUNNABLE_GGUF_TYPES:
            continue
        metadata = _json_dict(artifact.pop("metadata_json", "{}"))
        if artifact.get("path") and not Path(str(artifact["path"])).exists():
            continue
        return {
            "target_type": "gguf_artifact",
            "model_slug": str(metadata.get("model_slug", "")),
            "artifact_id": artifact["artifact_id"],
            "display_name": artifact["display_name"],
            "provider_id": "",
            "path": artifact["path"],
            "updated_at": 0,
        }
    return None


def _model_artifacts(conn: Any, model_slug: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    from traininghub.core.database import rows_to_dicts

    artifact_ids = {str(artifact_id) for artifact_id in metadata.get("artifact_ids", []) if str(artifact_id)}
    artifacts = rows_to_dicts(conn.execute("SELECT * FROM artifacts ORDER BY created_at DESC").fetchall())
    selected = []
    for artifact in artifacts:
        artifact_metadata = json.loads(artifact.pop("metadata_json") or "{}")
        if artifact["artifact_id"] in artifact_ids or artifact_metadata.get("model_slug") == model_slug:
            artifact["metadata"] = artifact_metadata
            selected.append(artifact)
    return selected


def _clear_active_target_if_needed(conn: Any, model_slug: str, artifact_ids: set[str]) -> None:
    row = conn.execute("SELECT value_json FROM settings WHERE key = ?", (ACTIVE_INFERENCE_SETTING_KEY,)).fetchone()
    if row is None:
        return
    try:
        target = json.loads(row["value_json"])
    except json.JSONDecodeError:
        conn.execute("DELETE FROM settings WHERE key = ?", (ACTIVE_INFERENCE_SETTING_KEY,))
        return
    if target.get("model_slug") == model_slug or target.get("artifact_id") in artifact_ids:
        fallback = fallback_inference_target_from_connection(conn)
        if fallback:
            conn.execute(
                """
                INSERT INTO settings (key, value_json)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
                """,
                (ACTIVE_INFERENCE_SETTING_KEY, json.dumps(fallback, sort_keys=True)),
            )
        else:
            conn.execute("DELETE FROM settings WHERE key = ?", (ACTIVE_INFERENCE_SETTING_KEY,))


def _base_model_target_for_slug(conn: Any, model_slug: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM model_registry WHERE slug = ?", (model_slug,)).fetchone()
    if row is None:
        return None
    model = dict(row)
    if not _model_row_is_runnable(model):
        return None
    return _base_model_target(model)


def _base_model_target(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_type": "base_model",
        "model_slug": model["slug"],
        "artifact_id": "",
        "display_name": model["display_name"],
        "provider_id": model["provider_id"],
        "path": "",
        "updated_at": 0,
    }


def _model_row_is_runnable(model: dict[str, Any]) -> bool:
    if not bool(model.get("supports_bf16_inference")):
        return False
    provider_id = str(model.get("provider_id") or "")
    return not (local_provider_path(provider_id) and not local_provider_is_runnable(provider_id))


def _json_dict(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
