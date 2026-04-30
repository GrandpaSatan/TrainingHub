from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from traininghub.core.database import connect, row_to_dict
from traininghub.core.id_utils import slugify
from traininghub.services.datasets import get_approved_version
from traininghub.services.model_files import checkpoint_has_inference_files


TRAINING_JOB_TYPES = {
    "train_lora": ("lora", "supports_lora"),
    "train_qlora": ("qlora", "supports_qlora"),
    "train_full": ("full", "supports_full_finetune"),
}

TRAINING_PRESETS = {"smoke", "standard", "custom"}
MORRIGAN_SAFE_LORA_PARAM_BILLIONS = 4.0
MORRIGAN_SAFE_LORA_MAX_SEQUENCE_LENGTH = 1024
_BILLION_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*B", re.IGNORECASE)


def validate_training_payload(database_path: Path, job_type: str, payload: dict[str, Any]) -> None:
    if job_type not in TRAINING_JOB_TYPES:
        raise ValueError(f"Unsupported training job type: {job_type}")
    mode, capability = TRAINING_JOB_TYPES[job_type]
    model_slug = str(payload.get("model_slug") or "").strip()
    dataset_id = str(payload.get("dataset_id") or "").strip()
    if not model_slug:
        raise ValueError("model_slug is required for training jobs.")
    if not dataset_id:
        raise ValueError("dataset_id is required for training jobs.")

    model = get_training_model(database_path, model_slug)
    if not model:
        raise ValueError("Model not found.")
    if not bool(model[capability]):
        raise ValueError(f"{model['display_name']} does not support {mode.upper()} training.")

    approved_version = get_approved_version(database_path, dataset_id)
    if not approved_version:
        raise ValueError("Dataset must be approved before training.")

    preset = str(payload.get("preset") or "smoke")
    if preset not in TRAINING_PRESETS:
        raise ValueError("preset must be smoke, standard, or custom.")

    payload["mode"] = mode
    payload["model_provider_id"] = model["provider_id"]
    payload["model_display_name"] = model["display_name"]
    payload["model_family"] = model["family"]
    payload["model_default_dtype"] = model["default_dtype"]
    payload["dataset_version_id"] = approved_version["version_id"]
    payload["dataset_jsonl_path"] = approved_version["jsonl_path"]
    payload["output_name"] = slugify(str(payload.get("output_name") or f"{model_slug}-{mode}-{preset}"), f"{model_slug}-{mode}")
    payload["max_sequence_length"] = _training_sequence_length(model, mode, payload.get("max_sequence_length"))
    _apply_training_defaults(payload, mode, preset)
    _validate_memory_safe_training(payload, model, mode)


def get_training_model(database_path: Path, model_slug: str) -> dict[str, Any] | None:
    with connect(database_path) as conn:
        row = conn.execute("SELECT * FROM model_registry WHERE slug = ?", (model_slug,)).fetchone()
    model = row_to_dict(row)
    if not model:
        return None
    model["metadata"] = json.loads(model.pop("metadata_json") or "{}")
    for key in ["supports_lora", "supports_qlora", "supports_full_finetune"]:
        model[key] = bool(model[key])
    return model


def register_trained_model(
    database_path: Path,
    model_slug: str,
    provider_id: str,
    display_name: str,
    family: str,
    parameter_count: str,
    local_path: Path,
    artifact_ids: list[str],
    source_job_id: str,
    max_sequence_length: int,
    default_dtype: str,
) -> None:
    if not checkpoint_has_inference_files(local_path):
        return
    metadata = {
        "route": "trained_model",
        "local_path": str(local_path),
        "artifact_ids": artifact_ids,
        "training_job_id": source_job_id,
    }
    with connect(database_path) as conn:
        conn.execute(
            """
            INSERT INTO model_registry (
                slug, provider_id, display_name, family, parameter_count,
                supports_lora, supports_qlora, supports_full_finetune,
                supports_bf16_inference, supports_benchmark, supports_quantization,
                supports_gguf_path, is_saga, hardware_note, default_dtype,
                max_sequence_length, metadata_json
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
                model_slug,
                provider_id,
                display_name,
                family,
                parameter_count,
                1,
                1,
                0,
                1,
                1,
                1,
                1,
                0,
                "TrainingHub-trained checkpoint. Benchmark, convert, quantize, or serve with local inference after validating memory use.",
                default_dtype,
                max_sequence_length,
                json.dumps(metadata, sort_keys=True),
            ),
        )


def _apply_training_defaults(payload: dict[str, Any], mode: str, preset: str) -> None:
    if preset == "smoke":
        defaults = {
            "epochs": 1.0,
            "max_steps": 1,
            "learning_rate": 2e-4 if mode != "full" else 5e-5,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "merge_adapter": mode == "lora",
        }
    else:
        defaults = {
            "epochs": 1.0,
            "max_steps": 0,
            "learning_rate": 2e-4 if mode != "full" else 2e-5,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "merge_adapter": mode == "lora",
        }
    for key, value in defaults.items():
        if payload.get(key) in [None, ""]:
            payload[key] = value
    if not payload.get("target_modules") and mode != "full":
        payload["target_modules"] = ["q_proj", "v_proj"]


def _training_sequence_length(model: dict[str, Any], mode: str, requested: Any) -> int:
    if requested not in [None, ""]:
        value = int(requested)
    else:
        value = int(model["max_sequence_length"])
        if mode == "lora" and _model_parameter_billions(model) >= MORRIGAN_SAFE_LORA_PARAM_BILLIONS:
            value = min(value, MORRIGAN_SAFE_LORA_MAX_SEQUENCE_LENGTH)
    if value <= 0:
        raise ValueError("max_sequence_length must be positive.")
    return value


def _validate_memory_safe_training(payload: dict[str, Any], model: dict[str, Any], mode: str) -> None:
    if payload.get("dry_run", False) or mode != "lora":
        return
    parameter_billions = _model_parameter_billions(model)
    if (
        parameter_billions >= MORRIGAN_SAFE_LORA_PARAM_BILLIONS
        and int(payload["max_sequence_length"]) > MORRIGAN_SAFE_LORA_MAX_SEQUENCE_LENGTH
    ):
        raise ValueError(
            f"{model['display_name']} LoRA at sequence length {payload['max_sequence_length']} exceeds Morrigan's 12 GB GPU budget. "
            f"Use QLoRA or set max_sequence_length <= {MORRIGAN_SAFE_LORA_MAX_SEQUENCE_LENGTH}."
        )


def _model_parameter_billions(model: dict[str, Any]) -> float:
    sources = [
        model.get("parameter_count"),
        model.get("provider_id"),
        model.get("display_name"),
        model.get("slug"),
    ]
    for source in sources:
        value = _parse_parameter_billions(source)
        if value > 0:
            return value
    return 0.0


def _parse_parameter_billions(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        number = float(text)
    except ValueError:
        number = 0.0
    if number > 0:
        return number / 1e9 if number >= 1e6 else number
    match = _BILLION_PATTERN.search(text)
    if not match:
        return 0.0
    return float(match.group(1))
