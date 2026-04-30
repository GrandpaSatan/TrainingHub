from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from traininghub.services.artifacts import get_artifact
from traininghub.services.capability_transfers import mark_aligned, update_transfer_status
from traininghub.services.model_introspection import normalize_layer_targets, proportional_layer_pairs, transformer_layers
from traininghub.services.inference_run import _local_model_path_from_provider_id
from traininghub.workers.common import WorkerContext, real_workers_enabled, run_worker


def main(context: WorkerContext, payload: dict[str, Any]) -> None:
    transfer_id = str(payload["transfer_id"])
    try:
        if payload.get("dry_run") or not real_workers_enabled():
            artifact = _dry_run_align(context, payload)
        elif payload.get("target_runtime") == "llama_cpp":
            artifact = _align_llama_cpp_degraded(context, payload)
        else:
            artifact = _align_transformers(context, payload)
        mark_aligned(context.database_path, transfer_id, artifact["artifact_id"], artifact.get("metadata", {}))
        context.set_completion_summary(
            "Capability alignment completed.",
            {"transfer_id": transfer_id, "alignment_artifact_id": artifact["artifact_id"]},
        )
    except Exception:
        update_transfer_status(context.database_path, transfer_id, "failed")
        raise


def _dry_run_align(context: WorkerContext, payload: dict[str, Any]) -> dict[str, Any]:
    vector_path = _artifact_path(context, str(payload["vector_artifact_id"]))
    vectors = np.load(vector_path)
    layer_keys = sorted(key for key in vectors.files if key.startswith("layer_") and key != "layer_last")
    if not layer_keys:
        layer_keys = ["layer_last"]
    pairs = payload.get("layer_pairs") or [[_layer_number(key, index), _layer_number(key, index)] for index, key in enumerate(layer_keys)]
    arrays: dict[str, np.ndarray] = {}
    for index, pair in enumerate(pairs):
        source_key = f"layer_{pair[0]}"
        vector = vectors[source_key] if source_key in vectors else vectors[layer_keys[min(index, len(layer_keys) - 1)]]
        hidden = int(vector.shape[0])
        arrays[f"pair_{index}"] = np.eye(hidden, dtype=np.float32)
    if payload.get("target_runtime") == "llama_cpp":
        arrays["logit_bias"] = np.zeros_like(vectors[layer_keys[0]], dtype=np.float32)
    path = context.job_dir / "alignment_map.npz"
    np.savez_compressed(path, **arrays)
    metadata = {
        "source_model_slug": payload.get("source_model_slug", ""),
        "target_model_slug": payload.get("target_model_slug", ""),
        "rank": int(payload.get("rank") or 16),
        "layer_pairs": pairs,
        "calibration_size": 4,
        "dry_run": True,
        "degraded_mode": payload.get("target_runtime") == "llama_cpp",
    }
    artifact = context.register_artifact(path, "alignment_map", "Capability alignment map", metadata)
    artifact["metadata"] = metadata
    context.metric({"layer_pairs": len(pairs), "rank": metadata["rank"], "dry_run": True})
    return artifact


def _align_transformers(context: WorkerContext, payload: dict[str, Any]) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    vector_artifact = get_artifact(context.database_path, str(payload["vector_artifact_id"]))
    if not vector_artifact:
        raise RuntimeError("Capability vector artifact not found.")
    vectors = np.load(vector_artifact["path"])
    prompts = _load_alignment_prompts(Path(str(payload["dataset_jsonl_path"])))
    if not prompts:
        raise RuntimeError("Calibration dataset has no usable alignment prompts.")
    source_id = _resolve_model_source(str(payload.get("source_model_id") or payload["source_model_slug"]))
    target_id = _resolve_model_source(str(payload.get("target_model_id") or payload["target_model_slug"]))
    source_tokenizer = AutoTokenizer.from_pretrained(source_id, trust_remote_code=True)
    target_tokenizer = AutoTokenizer.from_pretrained(target_id, trust_remote_code=True)
    source_model = AutoModelForCausalLM.from_pretrained(
        source_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    target_model = AutoModelForCausalLM.from_pretrained(
        target_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    source_model.eval()
    target_model.eval()
    source_layers = transformer_layers(source_model)
    target_layers = transformer_layers(target_model)
    layer_pairs = payload.get("layer_pairs") or _default_pairs(vectors, len(source_layers), len(target_layers))
    rank = int(payload.get("rank") or 16)
    arrays: dict[str, np.ndarray] = {}
    errors: list[float] = []
    for pair_index, pair in enumerate(layer_pairs):
        context.check_cancelled()
        source_layer, target_layer = int(pair[0]), int(pair[1])
        source_matrix = []
        target_matrix = []
        for prompt_index, prompt in enumerate(prompts, start=1):
            source_matrix.append(_capture_layer(source_model, source_tokenizer, prompt, source_layer))
            target_matrix.append(_capture_layer(target_model, target_tokenizer, prompt, target_layer))
            if prompt_index == 1 or prompt_index == len(prompts) or prompt_index % 8 == 0:
                context.event(
                    "progress",
                    "Alignment calibration prompt processed.",
                    data={"step": prompt_index, "total": len(prompts), "pair_index": pair_index},
                )
        x_source = np.vstack(source_matrix).astype(np.float32)
        x_target = np.vstack(target_matrix).astype(np.float32)
        full_map, *_ = np.linalg.lstsq(x_source, x_target, rcond=None)
        u, singular_values, vt = np.linalg.svd(full_map, full_matrices=False)
        keep = min(rank, singular_values.shape[0])
        low_rank = (u[:, :keep] * singular_values[:keep]) @ vt[:keep, :]
        arrays[f"pair_{pair_index}"] = low_rank.astype(np.float32)
        reconstructed = x_source @ low_rank
        error = float(np.linalg.norm(x_target - reconstructed) / max(np.linalg.norm(x_target), 1e-9))
        errors.append(error)
        context.metric({"pair_index": pair_index, "source_layer": source_layer, "target_layer": target_layer, "reconstruction_error": error})
    path = context.job_dir / "alignment_map.npz"
    np.savez_compressed(path, **arrays)
    metadata = {
        "source_model_slug": payload.get("source_model_slug", ""),
        "target_model_slug": payload.get("target_model_slug", ""),
        "rank": rank,
        "layer_pairs": layer_pairs,
        "calibration_size": len(prompts),
        "mean_reconstruction_error": float(np.mean(errors)) if errors else 0.0,
    }
    artifact = context.register_artifact(path, "alignment_map", "Capability alignment map", metadata)
    artifact["metadata"] = metadata
    return artifact


def _align_llama_cpp_degraded(context: WorkerContext, payload: dict[str, Any]) -> dict[str, Any]:
    context.event(
        "worker_warning",
        "GGUF target alignment is degraded to a last-layer logits bias map.",
        "warning",
        {"target_runtime": "llama_cpp"},
    )
    vector_path = _artifact_path(context, str(payload["vector_artifact_id"]))
    vectors = np.load(vector_path)
    vector = vectors["layer_last"] if "layer_last" in vectors else vectors[vectors.files[0]]
    arrays = {"logit_bias": np.zeros_like(vector, dtype=np.float32)}
    path = context.job_dir / "alignment_map.npz"
    np.savez_compressed(path, **arrays)
    metadata = {
        "source_model_slug": payload.get("source_model_slug", ""),
        "target_model_slug": payload.get("target_model_slug", ""),
        "rank": int(payload.get("rank") or 16),
        "layer_pairs": [["last", "logits"]],
        "calibration_size": 0,
        "degraded_mode": True,
    }
    artifact = context.register_artifact(path, "alignment_map", "Capability alignment map", metadata)
    artifact["metadata"] = metadata
    return artifact


def _artifact_path(context: WorkerContext, artifact_id: str) -> Path:
    artifact = get_artifact(context.database_path, artifact_id)
    if not artifact:
        raise RuntimeError("Capability vector artifact not found.")
    return Path(str(artifact["path"]))


def _default_pairs(vectors: Any, source_layer_count: int, target_layer_count: int) -> list[list[int]]:
    source_layers = [_layer_number(key, index) for index, key in enumerate(vectors.files) if key.startswith("layer_") and key != "layer_last"]
    if not source_layers:
        source_layers = normalize_layer_targets("last", source_layer_count)
    return proportional_layer_pairs(source_layer_count, target_layer_count, source_layers)


def _capture_layer(model: Any, tokenizer: Any, text: str, layer_index: int) -> np.ndarray:
    import torch

    captures: dict[str, Any] = {}
    layers = transformer_layers(model)

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        captures["value"] = hidden.detach()[:, -1, :].float().cpu()

    handle = layers[layer_index].register_forward_hook(hook)
    try:
        inputs = tokenizer(text, return_tensors="pt")
        device = _model_input_device(model)
        if device is not None:
            inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()
    return captures["value"].numpy()[0]


def _load_alignment_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt") or row.get("prompt_present") or row.get("continuation_prefix") or "").strip()
            if prompt:
                prompts.append(prompt)
    return prompts


def _layer_number(key: str, fallback: int) -> int:
    try:
        return int(key.removeprefix("layer_"))
    except ValueError:
        return fallback


def _resolve_model_source(model_slug: str) -> str:
    local = _local_model_path_from_provider_id(model_slug)
    if local is not None:
        return str(local)
    return model_slug


def _model_input_device(model: Any) -> Any | None:
    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        return device
    try:
        for parameter in model.parameters():
            if str(parameter.device) != "meta":
                return parameter.device
    except Exception:
        return None
    return None


if __name__ == "__main__":
    raise SystemExit(run_worker(main))
