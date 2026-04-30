from __future__ import annotations

import asyncio
import gc
import json
import os
import re
import time
from asyncio.subprocess import PIPE
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, AsyncIterator, Iterator

import numpy as np

from traininghub.services.capability_transfers import transfer_for_inference
from traininghub.services.model_introspection import transformer_layers


class InferenceRunError(RuntimeError):
    pass


@dataclass(frozen=True)
class SamplingConfig:
    system: str = ""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256
    stop: tuple[str, ...] = ()
    repetition_penalty: float = 1.08
    no_repeat_ngram_size: int = 3
    do_sample: bool = True
    dry_run: bool = False


@dataclass
class _CachedTransformersModel:
    tokenizer: Any
    model: Any
    expires_at: float


@dataclass
class _CachedLlamaModel:
    model: Any
    expires_at: float


@dataclass
class _CachedTransferDirections:
    directions: dict[int, np.ndarray]
    logit_bias: np.ndarray | None
    expires_at: float


_TRANSFORMERS_CACHE: dict[str, _CachedTransformersModel] = {}
_LLAMA_CACHE: dict[str, _CachedLlamaModel] = {}
_TRANSFER_CACHE: dict[str, _CachedTransferDirections] = {}
_ACTIVE_INFERENCE_CANCEL_EVENTS: set[Event] = set()
_ACTIVE_INFERENCE_LOCK = Lock()
_DEFAULT_CACHE_TTL_SECONDS = 300
_LOCAL_PROVIDER_PREFIX = "local:"


def cancel_active_inference_runs() -> int:
    with _ACTIVE_INFERENCE_LOCK:
        cancel_events = tuple(_ACTIVE_INFERENCE_CANCEL_EVENTS)
    for cancel_event in cancel_events:
        cancel_event.set()
    return len(cancel_events)


def clear_transformers_cache() -> int:
    cache_entries = len(_TRANSFORMERS_CACHE) + len(_LLAMA_CACHE) + len(_TRANSFER_CACHE)
    _TRANSFORMERS_CACHE.clear()
    _LLAMA_CACHE.clear()
    _TRANSFER_CACHE.clear()
    gc.collect()
    _release_cuda_memory()
    return cache_entries


def shutdown_inference_for_training() -> dict[str, Any]:
    active_runs_cancelled = cancel_active_inference_runs()
    deadline = time.monotonic() + _shutdown_grace_seconds()
    while _active_inference_run_count() and time.monotonic() < deadline:
        time.sleep(0.05)
    return {
        "active_runs_cancelled": active_runs_cancelled,
        "active_runs_remaining": _active_inference_run_count(),
        "cache_entries_cleared": clear_transformers_cache(),
    }


async def run_prompt(
    target: dict[str, Any],
    prompt: str,
    sampling: SamplingConfig,
    cancel_event: Event | None = None,
    database_path: Path | None = None,
) -> AsyncIterator[str]:
    if not prompt.strip():
        raise InferenceRunError("prompt is required.")

    stop_event = cancel_event or Event()
    _register_active_inference(stop_event)
    try:
        if sampling.dry_run:
            stream = _run_dry_prompt(target, prompt, stop_event)
        else:
            target_type = str(target.get("target_type", ""))
            if target_type == "base_model":
                stream = _run_transformers_prompt(target, prompt, sampling, stop_event, database_path)
            elif target_type == "gguf_artifact":
                stream = _run_llama_cpp_prompt(target, prompt, sampling, stop_event, database_path)
            else:
                raise InferenceRunError("active inference target must be a base_model or gguf_artifact.")

        async for token in _apply_stop_sequences(stream, sampling.stop, stop_event):
            yield token
    finally:
        stop_event.set()
        _unregister_active_inference(stop_event)


async def _run_dry_prompt(target: dict[str, Any], prompt: str, cancel_event: Event) -> AsyncIterator[str]:
    display_name = str(target.get("display_name") or target.get("model_slug") or "active target")
    text = (
        f"Dry-run response from {display_name}. "
        f"I received your prompt: {prompt.strip()[:180]}. "
        "Streaming is wired correctly; enable a local inference runtime to generate real model output."
    )
    for token in re.findall(r"\S+\s*|\n", text):
        if cancel_event.is_set():
            return
        await asyncio.sleep(0.003)
        yield token


async def _run_transformers_prompt(
    target: dict[str, Any],
    prompt: str,
    sampling: SamplingConfig,
    cancel_event: Event,
    database_path: Path | None,
) -> AsyncIterator[str]:
    runtime = os.getenv("TRAININGHUB_INFERENCE_RUNTIME", "").strip().casefold()
    if runtime != "transformers":
        raise InferenceRunError("Set TRAININGHUB_INFERENCE_RUNTIME=transformers to run base-model chat.")

    try:
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
    except ImportError as exc:
        raise InferenceRunError("Transformers is required for base-model chat.") from exc

    tokenizer, model = _load_transformers_target(target)
    prompt_text = _format_transformers_prompt(tokenizer, sampling.system, prompt)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_device = _model_input_device(model)
    if input_device is not None:
        inputs = {key: value.to(input_device) for key, value in inputs.items()}

    class _CancelCriteria(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            return cancel_event.is_set()

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs: dict[str, Any] = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": sampling.max_tokens,
        "do_sample": sampling.do_sample and sampling.temperature > 0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": sampling.repetition_penalty,
        "no_repeat_ngram_size": sampling.no_repeat_ngram_size,
        "stopping_criteria": StoppingCriteriaList([_CancelCriteria()]),
    }
    if generation_kwargs["do_sample"]:
        generation_kwargs["temperature"] = sampling.temperature
        generation_kwargs["top_p"] = sampling.top_p

    errors: list[BaseException] = []
    hook_handles = _attach_steering_hooks(model, target, database_path)

    def generate() -> None:
        try:
            model.generate(**generation_kwargs)
        except BaseException as exc:  # pragma: no cover - exercised only with a real runtime.
            errors.append(exc)
            streamer.on_finalized_text("", stream_end=True)

    try:
        thread = Thread(target=generate, daemon=True)
        thread.start()
        iterator = iter(streamer)
        sentinel = object()

        while True:
            item = await asyncio.to_thread(_next_stream_item, iterator, sentinel)
            if item is sentinel:
                break
            if cancel_event.is_set():
                break
            if item:
                yield str(item)

        cancel_event.set()
        thread.join(timeout=0)
        if errors:
            raise InferenceRunError(str(errors[0])) from errors[0]
    finally:
        for handle in hook_handles:
            handle.remove()


async def _run_llama_cpp_prompt(
    target: dict[str, Any],
    prompt: str,
    sampling: SamplingConfig,
    cancel_event: Event,
    database_path: Path | None,
) -> AsyncIterator[str]:
    if os.getenv("TRAININGHUB_LLAMA_RUNTIME", "").strip().casefold() == "cli":
        async for token in _run_llama_cli_prompt(target, prompt, sampling, cancel_event):
            yield token
        return
    try:
        llm = _load_llama_target(target)
    except ImportError:
        async for token in _run_llama_cli_prompt(target, prompt, sampling, cancel_event):
            yield token
        return

    completion_kwargs: dict[str, Any] = {
        "prompt": _format_prompt(sampling.system, prompt),
        "max_tokens": sampling.max_tokens,
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "repeat_penalty": sampling.repetition_penalty,
        "stream": True,
    }
    logits_processor = _build_llama_logits_processor(target, database_path)
    if logits_processor is not None:
        completion_kwargs["logits_processor"] = logits_processor
    iterator = None
    try:
        try:
            iterator = iter(llm.create_completion(**completion_kwargs))
        except TypeError:
            completion_kwargs.pop("logits_processor", None)
            iterator = iter(llm.create_completion(**completion_kwargs))
        sentinel = object()
        while True:
            if cancel_event.is_set():
                break
            item = await asyncio.to_thread(_next_stream_item, iterator, sentinel)
            if item is sentinel:
                break
            token = _llama_stream_text(item)
            if token:
                yield token
    finally:
        cancel_event.set()


async def _run_llama_cli_prompt(
    target: dict[str, Any],
    prompt: str,
    sampling: SamplingConfig,
    cancel_event: Event,
) -> AsyncIterator[str]:
    model_path = Path(str(target.get("path") or "")).expanduser()
    if not model_path.exists():
        raise InferenceRunError(f"GGUF artifact not found: {model_path}")
    llama_cli = _llama_cli_path()
    if not llama_cli.exists():
        raise InferenceRunError(f"llama.cpp CLI not found: {llama_cli}")

    command = [
        str(llama_cli),
        "-m",
        str(model_path),
        "-p",
        _format_prompt(sampling.system, prompt),
        "-n",
        str(sampling.max_tokens),
        "--temp",
        str(sampling.temperature),
        "--top-p",
        str(sampling.top_p),
        "--repeat-penalty",
        str(sampling.repetition_penalty),
        "--no-display-prompt",
    ]
    gpu_layers = os.getenv("LLAMA_CPP_N_GPU_LAYERS", "999")
    if gpu_layers:
        command.extend(["-ngl", gpu_layers])

    process = await asyncio.create_subprocess_exec(*command, stdout=PIPE, stderr=PIPE)
    assert process.stdout is not None
    assert process.stderr is not None
    stderr_task = asyncio.create_task(process.stderr.read())
    try:
        while True:
            if cancel_event.is_set():
                break
            chunk = await process.stdout.read(1)
            if not chunk:
                break
            yield chunk.decode("utf-8", errors="ignore")
        return_code = await process.wait()
        stderr = (await stderr_task).decode("utf-8", errors="replace").strip()
        if return_code != 0:
            raise InferenceRunError(stderr or f"llama-cli exited with status {return_code}.")
    finally:
        cancel_event.set()
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except asyncio.TimeoutError:
                process.kill()
        if not stderr_task.done():
            stderr_task.cancel()


def _load_transformers_target(target: dict[str, Any]) -> tuple[Any, Any]:
    cache_key = _target_cache_key(target)
    now = time.monotonic()
    cached = _TRANSFORMERS_CACHE.get(cache_key)
    if cached and cached.expires_at > now:
        cached.expires_at = now + _cache_ttl_seconds()
        return cached.tokenizer, cached.model

    _prune_transformers_cache(now)
    model_id = _resolve_transformers_model_source(target)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise InferenceRunError("Transformers and torch are required for base-model chat.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    _TRANSFORMERS_CACHE[cache_key] = _CachedTransformersModel(tokenizer, model, now + _cache_ttl_seconds())
    return tokenizer, model


def _load_llama_target(target: dict[str, Any]) -> Any:
    model_path = Path(str(target.get("path") or "")).expanduser()
    if not model_path.exists():
        raise InferenceRunError(f"GGUF artifact not found: {model_path}")
    try:
        from llama_cpp import Llama
    except ImportError:
        raise
    cache_key = _target_cache_key(target)
    now = time.monotonic()
    cached = _LLAMA_CACHE.get(cache_key)
    if cached and cached.expires_at > now:
        cached.expires_at = now + _cache_ttl_seconds()
        return cached.model
    _prune_llama_cache(now)
    gpu_layers = _int_env("LLAMA_CPP_N_GPU_LAYERS", 999)
    llm = Llama(model_path=str(model_path), n_gpu_layers=gpu_layers)
    _LLAMA_CACHE[cache_key] = _CachedLlamaModel(llm, now + _cache_ttl_seconds())
    return llm


def _resolve_transformers_model_source(target: dict[str, Any]) -> str:
    source = str(target.get("provider_id") or target.get("path") or target.get("model_slug") or "").strip()
    if not source:
        raise InferenceRunError("Active base-model target has no provider_id.")

    local_path = _local_model_path_from_provider_id(source)
    if local_path is None:
        return source
    if not local_path.exists():
        raise InferenceRunError(f"Local checkpoint not found: {local_path}")
    return str(local_path)


def _local_model_path_from_provider_id(provider_id: str) -> Path | None:
    if not provider_id.startswith(_LOCAL_PROVIDER_PREFIX):
        return None
    path_value = provider_id.removeprefix(_LOCAL_PROVIDER_PREFIX).strip()
    if not path_value:
        raise InferenceRunError("Local checkpoint provider_id is missing a path.")
    return Path(path_value).expanduser()


def _prune_transformers_cache(now: float) -> None:
    expired_keys = [key for key, cached in _TRANSFORMERS_CACHE.items() if cached.expires_at <= now]
    for key in expired_keys:
        _TRANSFORMERS_CACHE.pop(key, None)


def _prune_llama_cache(now: float) -> None:
    expired_keys = [key for key, cached in _LLAMA_CACHE.items() if cached.expires_at <= now]
    for key in expired_keys:
        _LLAMA_CACHE.pop(key, None)


def _attach_steering_hooks(model: Any, target: dict[str, Any], database_path: Path | None = None) -> list[Any]:
    transfer = _resolve_active_transfer(target, database_path)
    if not transfer or transfer["target_runtime"] != "transformers" or float(transfer.get("alpha") or 0.0) == 0.0:
        return []
    directions = _load_transfer_directions(transfer).directions
    if not directions:
        return []
    layers = transformer_layers(model)
    handles = []
    alpha = float(transfer["alpha"])
    for target_layer, direction in directions.items():
        if target_layer < 0 or target_layer >= len(layers):
            continue

        def hook(_module: Any, _inputs: Any, output: Any, direction: np.ndarray = direction, alpha: float = alpha) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            try:
                import torch
            except ImportError:
                return output
            delta = torch.as_tensor(direction, device=hidden.device, dtype=hidden.dtype)
            if delta.shape[-1] != hidden.shape[-1]:
                return output
            steered = hidden.clone()
            steered[:, -1:, :] = steered[:, -1:, :] + (alpha * delta).view(1, 1, -1)
            if isinstance(output, tuple):
                return (steered, *output[1:])
            return steered

        handles.append(layers[target_layer].register_forward_hook(hook))
    return handles


def _resolve_active_transfer(target: dict[str, Any], database_path: Path | None = None) -> dict[str, Any] | None:
    transfer_id = str(target.get("capability_transfer_id") or "")
    if not transfer_id:
        return None
    resolved_database_path = database_path or _database_path()
    if resolved_database_path is None:
        raise InferenceRunError("Active capability transfer requires TRAININGHUB_DATABASE_PATH.")
    transfer = transfer_for_inference(resolved_database_path, transfer_id)
    if not transfer:
        raise InferenceRunError("Active capability transfer is missing, deleted, or not ready.")
    if transfer["target_runtime"] == "transformers" and target.get("model_slug") != transfer["target_model_slug"]:
        raise InferenceRunError("Active capability transfer does not match the selected base model.")
    if transfer["target_runtime"] == "llama_cpp":
        target_artifact_id = str(transfer["config"].get("target_artifact_id") or "")
        if target_artifact_id and target.get("artifact_id") != target_artifact_id:
            raise InferenceRunError("Active capability transfer does not match the selected GGUF artifact.")
    return transfer


def _load_transfer_directions(transfer: dict[str, Any]) -> _CachedTransferDirections:
    cache_key = (
        f"{transfer['transfer_id']}:{transfer.get('vector_artifact_id')}:{transfer.get('alignment_artifact_id')}:"
        f"{transfer.get('alpha')}:{json.dumps(transfer.get('layer_targets'), sort_keys=True)}:{transfer.get('updated_at')}"
    )
    now = time.monotonic()
    cached = _TRANSFER_CACHE.get(cache_key)
    if cached and cached.expires_at > now:
        cached.expires_at = now + _cache_ttl_seconds()
        return cached
    vector_artifact = transfer.get("vector_artifact") or {}
    alignment_artifact = transfer.get("alignment_artifact") or {}
    vectors = np.load(vector_artifact["path"])
    alignment = np.load(alignment_artifact["path"])
    layer_pairs = transfer.get("config", {}).get("layer_pairs") or []
    layer_targets = transfer.get("layer_targets", "all")
    directions: dict[int, np.ndarray] = {}
    for index, pair in enumerate(layer_pairs):
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        try:
            source_layer, target_layer = int(pair[0]), int(pair[1])
        except (TypeError, ValueError):
            continue
        if not _target_layer_enabled(target_layer, layer_targets):
            continue
        vector_key = f"layer_{source_layer}"
        map_key = f"pair_{index}"
        if vector_key not in vectors or map_key not in alignment:
            continue
        source_direction = vectors[vector_key]
        projection = alignment[map_key]
        if source_direction.shape[-1] == projection.shape[0]:
            directions[target_layer] = (source_direction @ projection).astype(np.float32)
    logit_bias = alignment["logit_bias"].astype(np.float32) if "logit_bias" in alignment else None
    cached = _CachedTransferDirections(directions, logit_bias, now + _cache_ttl_seconds())
    _TRANSFER_CACHE[cache_key] = cached
    return cached


def _target_layer_enabled(target_layer: int, layer_targets: Any) -> bool:
    if layer_targets in {"all", "every-4"}:
        return True
    if layer_targets == "last":
        return True
    if isinstance(layer_targets, list):
        return target_layer in {int(item) for item in layer_targets}
    return True


def _build_llama_logits_processor(target: dict[str, Any], database_path: Path | None = None) -> Any | None:
    transfer = _resolve_active_transfer(target, database_path)
    if not transfer or transfer["target_runtime"] != "llama_cpp" or float(transfer.get("alpha") or 0.0) == 0.0:
        return None
    cached = _load_transfer_directions(transfer)
    if cached.logit_bias is None:
        return None
    alpha = float(transfer["alpha"])
    bias = cached.logit_bias

    def processor(_input_ids: Any, scores: Any) -> Any:
        if getattr(scores, "shape", [0])[-1] != bias.shape[-1]:
            return scores
        return scores + (alpha * bias)

    return processor


def _llama_stream_text(item: Any) -> str:
    if not isinstance(item, dict):
        return str(item)
    choices = item.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if isinstance(choice, dict):
        return str(choice.get("text") or choice.get("delta", {}).get("content") or "")
    return ""


def _register_active_inference(cancel_event: Event) -> None:
    with _ACTIVE_INFERENCE_LOCK:
        _ACTIVE_INFERENCE_CANCEL_EVENTS.add(cancel_event)


def _unregister_active_inference(cancel_event: Event) -> None:
    with _ACTIVE_INFERENCE_LOCK:
        _ACTIVE_INFERENCE_CANCEL_EVENTS.discard(cancel_event)


def _active_inference_run_count() -> int:
    with _ACTIVE_INFERENCE_LOCK:
        return len(_ACTIVE_INFERENCE_CANCEL_EVENTS)


def _release_cuda_memory() -> None:
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except RuntimeError:
        return


def _shutdown_grace_seconds() -> float:
    value = os.getenv("TRAININGHUB_INFERENCE_SHUTDOWN_GRACE_SECONDS", "2")
    try:
        return max(float(value), 0.0)
    except ValueError:
        return 2.0


def _cache_ttl_seconds() -> int:
    return int(os.getenv("TRAININGHUB_INFERENCE_CACHE_TTL_SECONDS", str(_DEFAULT_CACHE_TTL_SECONDS)))


def _database_path() -> Path | None:
    value = os.getenv("TRAININGHUB_DATABASE_PATH", "").strip()
    return Path(value) if value else None


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _target_cache_key(target: dict[str, Any]) -> str:
    target_type = str(target.get("target_type") or "")
    if target_type == "gguf_artifact":
        return f"gguf:{target.get('artifact_id')}:{target.get('path')}"
    return f"base:{target.get('model_slug')}:{target.get('provider_id')}"


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


def _next_stream_item(iterator: Iterator[str], sentinel: object) -> str | object:
    try:
        return next(iterator)
    except StopIteration:
        return sentinel


async def _apply_stop_sequences(tokens: AsyncIterator[str], stop: tuple[str, ...], cancel_event: Event) -> AsyncIterator[str]:
    stop_sequences = tuple(sequence for sequence in stop if sequence)
    if not stop_sequences:
        async for token in tokens:
            yield token
        return

    keep = max(len(sequence) for sequence in stop_sequences) - 1
    buffer = ""
    async for token in tokens:
        buffer += token
        stop_index = _first_stop_index(buffer, stop_sequences)
        if stop_index is not None:
            cancel_event.set()
            if stop_index > 0:
                yield buffer[:stop_index]
            return
        if keep > 0 and len(buffer) > keep:
            flush_size = len(buffer) - keep
            yield buffer[:flush_size]
            buffer = buffer[flush_size:]
        elif keep == 0 and buffer:
            yield buffer
            buffer = ""
    if buffer:
        yield buffer


def _first_stop_index(text: str, stop: tuple[str, ...]) -> int | None:
    indexes = [text.find(sequence) for sequence in stop if sequence in text]
    if not indexes:
        return None
    return min(indexes)


def _format_prompt(system: str, prompt: str) -> str:
    system = system.strip()
    prompt = prompt.strip()
    if system:
        return f"System:\n{system}\n\nUser:\n{prompt}\n\nAssistant:\n"
    return f"User:\n{prompt}\n\nAssistant:\n"


def _format_transformers_prompt(tokenizer: Any, system: str, prompt: str) -> str:
    messages = []
    system = system.strip()
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt.strip()})
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            rendered = apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if isinstance(rendered, str) and rendered.strip():
                return rendered
        except Exception:
            pass
    return _format_prompt(system, prompt)


def _llama_cli_path() -> Path:
    if os.getenv("LLAMA_CPP_CLI"):
        return Path(os.environ["LLAMA_CPP_CLI"]).expanduser()
    root = Path(os.getenv("LLAMA_CPP_ROOT", "/home/jhernandez/llama.cpp")).expanduser()
    for candidate in [root / "build" / "bin" / "llama-cli", root / "llama-cli"]:
        if candidate.exists():
            return candidate
    return root / "build" / "bin" / "llama-cli"
