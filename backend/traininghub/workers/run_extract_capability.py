from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from traininghub.services.capability_transfers import mark_extracted, update_transfer_status
from traininghub.services.model_introspection import normalize_layer_targets, transformer_layers
from traininghub.services.inference_run import _local_model_path_from_provider_id
from traininghub.workers.common import WorkerContext, real_workers_enabled, run_worker


def main(context: WorkerContext, payload: dict[str, Any]) -> None:
    transfer_id = str(payload["transfer_id"])
    try:
        if payload.get("dry_run") or not real_workers_enabled():
            artifact = _dry_run_extract(context, payload)
        elif payload.get("source_runtime") == "llama_cpp":
            artifact = _extract_llama_cpp(context, payload)
        else:
            artifact = _extract_transformers(context, payload)
        mark_extracted(context.database_path, transfer_id, artifact["artifact_id"], artifact.get("metadata", {}))
        context.set_completion_summary(
            "Capability extraction completed.",
            {"transfer_id": transfer_id, "vector_artifact_id": artifact["artifact_id"]},
        )
    except Exception:
        update_transfer_status(context.database_path, transfer_id, "failed")
        raise


def _dry_run_extract(context: WorkerContext, payload: dict[str, Any]) -> dict[str, Any]:
    context.event("worker_warning", "Dry-run capability extraction is producing deterministic placeholder vectors.", data={"dry_run": True})
    rng = np.random.default_rng(260406377)
    selected_layers = [0, 1, 2, 3] if payload.get("layer_targets") != "last" else [3]
    arrays = {f"layer_{layer}": _unit(rng.normal(size=(16,)).astype(np.float32)) for layer in selected_layers}
    arrays["layer_last"] = arrays[f"layer_{selected_layers[-1]}"]
    path = context.job_dir / "capability_vector.npz"
    np.savez_compressed(path, **arrays)
    metadata = {
        "source_model_slug": payload.get("source_model_slug", ""),
        "source_runtime": payload.get("source_runtime", "transformers"),
        "layer_count": len(selected_layers),
        "hidden_size": 16,
        "contrast_mode": payload.get("contrast_mode", "prompt_pair"),
        "calibration_size": 4,
        "dry_run": True,
    }
    artifact = context.register_artifact(path, "capability_vector", "Capability vector", metadata)
    artifact["metadata"] = metadata
    context.metric({"calibration_pairs": 4, "selected_layers": len(selected_layers)})
    return artifact


def _extract_transformers(context: WorkerContext, payload: dict[str, Any]) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pairs = _load_calibration_pairs(Path(str(payload["dataset_jsonl_path"])), str(payload.get("contrast_mode") or "prompt_pair"))
    if not pairs:
        raise RuntimeError("Calibration dataset has no usable capability contrast pairs.")
    model_id = _resolve_model_source(str(payload.get("source_model_id") or payload["source_model_slug"]))
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    layers = transformer_layers(model)
    selected_layers = normalize_layer_targets(payload.get("layer_targets", "all"), len(layers))
    if not selected_layers:
        raise RuntimeError("No valid source layers selected.")
    sums: dict[int, np.ndarray] = {}
    count = 0
    for index, pair in enumerate(pairs, start=1):
        context.check_cancelled()
        present = _capture_layers(model, tokenizer, pair["present"], selected_layers)
        absent = _capture_layers(model, tokenizer, pair["absent"], selected_layers)
        for layer in selected_layers:
            delta = present[layer] - absent[layer]
            sums[layer] = delta if layer not in sums else sums[layer] + delta
        count += 1
        if index == 1 or index == len(pairs) or index % 8 == 0:
            context.event("progress", "Capability contrast pair processed.", data={"step": index, "total": len(pairs)})
    arrays = {f"layer_{layer}": _unit(sums[layer] / max(count, 1)).astype(np.float32) for layer in selected_layers}
    arrays["layer_last"] = arrays[f"layer_{selected_layers[-1]}"]
    path = context.job_dir / "capability_vector.npz"
    np.savez_compressed(path, **arrays)
    hidden_size = int(next(iter(arrays.values())).shape[0])
    metadata = {
        "source_model_slug": payload.get("source_model_slug", ""),
        "source_runtime": "transformers",
        "layer_count": len(layers),
        "selected_layers": selected_layers,
        "hidden_size": hidden_size,
        "contrast_mode": payload.get("contrast_mode", "prompt_pair"),
        "calibration_size": count,
    }
    artifact = context.register_artifact(path, "capability_vector", "Capability vector", metadata)
    artifact["metadata"] = metadata
    context.metric({"calibration_pairs": count, "selected_layers": len(selected_layers), "hidden_size": hidden_size})
    return artifact


def _extract_llama_cpp(context: WorkerContext, payload: dict[str, Any]) -> dict[str, Any]:
    context.event(
        "worker_warning",
        "GGUF extraction is using last-layer embedding mode; fidelity is degraded versus per-layer Transformers extraction.",
        "warning",
    )
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError("llama-cpp-python is required for GGUF capability extraction.") from exc
    pairs = _load_calibration_pairs(Path(str(payload["dataset_jsonl_path"])), str(payload.get("contrast_mode") or "prompt_pair"))
    model_path = Path(str(payload.get("source_artifact_path") or "")).expanduser()
    if not model_path.is_file():
        raise RuntimeError(f"GGUF source artifact not found: {model_path}")
    llm = Llama(model_path=str(model_path), embedding=True)
    total: np.ndarray | None = None
    for index, pair in enumerate(pairs, start=1):
        present = _llama_embedding(llm, pair["present"])
        absent = _llama_embedding(llm, pair["absent"])
        delta = present - absent
        total = delta if total is None else total + delta
        if index == 1 or index == len(pairs) or index % 8 == 0:
            context.event("progress", "GGUF contrast pair processed.", data={"step": index, "total": len(pairs)})
    if total is None:
        raise RuntimeError("Calibration dataset has no usable capability contrast pairs.")
    vector = _unit(total / len(pairs)).astype(np.float32)
    path = context.job_dir / "capability_vector.npz"
    np.savez_compressed(path, layer_last=vector)
    metadata = {
        "source_model_slug": payload.get("source_model_slug", ""),
        "source_runtime": "llama_cpp",
        "layer_count": 1,
        "selected_layers": ["last"],
        "hidden_size": int(vector.shape[0]),
        "contrast_mode": payload.get("contrast_mode", "prompt_pair"),
        "calibration_size": len(pairs),
        "degraded_mode": True,
    }
    artifact = context.register_artifact(path, "capability_vector", "Capability vector", metadata)
    artifact["metadata"] = metadata
    return artifact


def _capture_layers(model: Any, tokenizer: Any, text: str, selected_layers: list[int]) -> dict[int, np.ndarray]:
    import torch

    captures: dict[int, Any] = {}
    handles = []
    layers = transformer_layers(model)
    for layer_index in selected_layers:
        def hook(_module: Any, _inputs: Any, output: Any, layer_index: int = layer_index) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captures[layer_index] = hidden.detach()[:, -1, :].float().cpu()

        handles.append(layers[layer_index].register_forward_hook(hook))
    try:
        inputs = tokenizer(text, return_tensors="pt")
        device = _model_input_device(model)
        if device is not None:
            inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
    return {layer: captures[layer].numpy()[0] for layer in selected_layers}


def _load_calibration_pairs(path: Path, contrast_mode: str) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            prefix = str(row.get("continuation_prefix") or row.get("prefix") or "")
            if contrast_mode == "system_pair":
                present = str(row.get("system_present") or "").strip()
                absent = str(row.get("system_absent") or "").strip()
                prompt = str(row.get("prompt") or prefix).strip()
                if present and absent and prompt:
                    pairs.append({"present": f"{present}\n\n{prompt}", "absent": f"{absent}\n\n{prompt}"})
                    continue
            present = str(row.get("prompt_present") or "").strip()
            absent = str(row.get("prompt_absent") or "").strip()
            if present and absent:
                pairs.append({"present": present + prefix, "absent": absent + prefix})
                continue
            raise RuntimeError(f"Calibration row {line_number} is missing prompt_present/prompt_absent fields.")
    return pairs


def _resolve_model_source(model_slug: str) -> str:
    local = _local_model_path_from_provider_id(model_slug)
    if local is not None:
        return str(local)
    return model_slug


def _llama_embedding(llm: Any, text: str) -> np.ndarray:
    result = llm.create_embedding(text)
    data = result.get("data") or []
    if not data:
        raise RuntimeError("llama-cpp-python returned no embedding.")
    embedding = data[0].get("embedding")
    if not embedding:
        raise RuntimeError("llama-cpp-python returned an empty embedding.")
    return np.asarray(embedding, dtype=np.float32)


def _unit(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value)
    return value if norm == 0 else value / norm


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
