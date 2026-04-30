from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from traininghub.core.database import connect
from traininghub.core.id_utils import slugify
from traininghub.services.hub import TRANSFORMERS_MODEL_INCLUDE_PATTERNS, ensure_confirmed_hub_sha
from traininghub.workers.common import WorkerContext, run_worker


def main(context: WorkerContext, payload: dict[str, Any]) -> None:
    source_type = str(payload.get("source_type", "")).strip()
    if source_type == "hf" and not payload.get("dry_run", False):
        ensure_confirmed_hub_sha(
            str(payload["repo_id"]),
            "model",
            payload.get("confirmed_sha"),
            payload.get("revision"),
        )
    model_slug = _model_slug(payload)
    model_dir = Path(os.environ["TRAININGHUB_DATA_ROOT"]) / "models" / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)

    if payload.get("dry_run", False):
        report = _report(payload, model_slug, model_dir, [], [])
        report_path = context.write_metadata("model_download_report.json", report)
        context.register_artifact(report_path, "model_download_report", "Model download dry-run report", report)
        return

    if source_type == "hf":
        if payload.get("download_estimate"):
            context.event("download_preflight", "Hugging Face download preflight accepted.", data=payload["download_estimate"])
        source_path = _download_hf_model(context, payload, model_dir)
    elif source_type == "url":
        source_path = _download_url_model(context, payload, model_dir)
    else:
        raise RuntimeError("source_type must be hf or url.")

    gguf_files = sorted(path for path in model_dir.rglob("*.gguf") if path.is_file())
    artifacts = []
    if gguf_files:
        for gguf_path in gguf_files:
            artifacts.append(
                context.register_artifact(
                    gguf_path,
                    "gguf_quantized",
                    f"{_display_name(payload, model_slug)} GGUF",
                    {"model_slug": model_slug, "source_type": source_type, "source_path": str(source_path)},
                )
            )
    else:
        artifacts.append(
            context.register_artifact(
                model_dir,
                "downloaded_model",
                _display_name(payload, model_slug),
                {"model_slug": model_slug, "source_type": source_type, "source_path": str(source_path)},
            )
        )

    _upsert_model_registry(context, payload, model_slug, model_dir, artifacts)
    report = _report(payload, model_slug, model_dir, gguf_files, artifacts)
    report_path = context.write_metadata("model_download_report.json", report)
    context.register_artifact(report_path, "model_download_report", "Model download report", report)
    context.event(
        "model_downloaded",
        "Model downloaded and registered.",
        data={"model_slug": model_slug, "artifact_count": len(artifacts), "local_path": str(model_dir)},
    )


def _download_hf_model(context: WorkerContext, payload: dict[str, Any], model_dir: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for Hugging Face model downloads.") from exc

    include_patterns = _patterns(payload.get("include_patterns")) or list(TRANSFORMERS_MODEL_INCLUDE_PATTERNS)
    exclude_patterns = _patterns(payload.get("exclude_patterns"))
    context.event(
        "download",
        "Downloading model from Hugging Face.",
        data={"repo_id": payload["repo_id"], "revision": payload.get("revision"), "local_dir": str(model_dir)},
    )
    downloaded_path = snapshot_download(
        repo_id=payload["repo_id"],
        repo_type="model",
        revision=payload.get("revision") or None,
        allow_patterns=include_patterns or None,
        ignore_patterns=exclude_patterns or None,
        local_dir=str(model_dir),
        token=os.getenv("HF_TOKEN") or None,
    )
    return Path(downloaded_path)


def _download_url_model(context: WorkerContext, payload: dict[str, Any], model_dir: Path) -> Path:
    url = str(payload["url"])
    filename = str(payload.get("filename") or _url_filename(url) or "model.bin")
    target_path = model_dir / filename
    context.event("download", "Downloading model from URL.", data={"url": url, "target_path": str(target_path)})
    bytes_written = 0
    with urllib.request.urlopen(url, timeout=60) as response:
        with target_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                context.check_cancelled()
                handle.write(chunk)
                bytes_written += len(chunk)
                if bytes_written % (128 * 1024 * 1024) < 1024 * 1024:
                    context.metric({"bytes_downloaded": bytes_written})
    context.metric({"bytes_downloaded": bytes_written})
    return target_path


def _upsert_model_registry(
    context: WorkerContext,
    payload: dict[str, Any],
    model_slug: str,
    model_dir: Path,
    artifacts: list[dict[str, Any]],
) -> None:
    source_type = str(payload.get("source_type"))
    repo_id = str(payload.get("repo_id") or "")
    url = str(payload.get("url") or "")
    has_gguf = any(artifact["artifact_type"] == "gguf_quantized" for artifact in artifacts)
    estimated_size = int((payload.get("download_estimate") or {}).get("estimated_size_bytes") or 0)
    large_transformers = bool(not has_gguf and estimated_size >= _large_transformers_threshold_bytes())
    metadata = {
        "route": "downloaded_gguf" if has_gguf else "downloaded_model",
        "local_path": str(model_dir),
        "download_source": source_type,
        "repo_id": repo_id,
        "url": url,
        "artifact_ids": [artifact["artifact_id"] for artifact in artifacts],
        "download_job_id": context.job_id,
        "download_estimate": payload.get("download_estimate") or {},
        "large_model": large_transformers,
    }
    with connect(context.database_path) as conn:
        conn.execute("DELETE FROM model_delete_tombstones WHERE slug = ?", (model_slug,))
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
                repo_id or f"local:{url}",
                _display_name(payload, model_slug),
                str(payload.get("family") or model_slug.split("-")[0] or "downloaded"),
                str(payload.get("parameter_count") or "unknown"),
                0 if has_gguf else 1,
                0 if has_gguf else 1,
                0,
                0 if has_gguf or large_transformers else 1,
                1,
                0 if has_gguf else 1,
                1,
                0,
                _hardware_note(has_gguf, large_transformers),
                str(payload.get("default_dtype") or "auto"),
                int(payload.get("max_sequence_length") or 2048),
                json.dumps(metadata, sort_keys=True),
            ),
        )


def _report(
    payload: dict[str, Any],
    model_slug: str,
    model_dir: Path,
    gguf_files: list[Path],
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "source_type": payload.get("source_type"),
        "repo_id": payload.get("repo_id"),
        "url": payload.get("url"),
        "model_slug": model_slug,
        "display_name": _display_name(payload, model_slug),
        "local_path": str(model_dir),
        "gguf_files": [str(path) for path in gguf_files],
        "artifacts": artifacts,
        "dry_run": bool(payload.get("dry_run", False)),
        "download_estimate": payload.get("download_estimate") or {},
    }


def _model_slug(payload: dict[str, Any]) -> str:
    value = payload.get("slug") or payload.get("repo_id") or _url_filename(str(payload.get("url") or "")) or "downloaded-model"
    return slugify(str(value).split("/")[-1], "downloaded-model")


def _display_name(payload: dict[str, Any], model_slug: str) -> str:
    return str(payload.get("display_name") or payload.get("repo_id") or model_slug)


def _patterns(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _url_filename(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = Path(urllib.parse.unquote(parsed.path)).name
    return name


def _large_transformers_threshold_bytes() -> int:
    return int(os.getenv("TRAININGHUB_LARGE_TRANSFORMERS_BYTES", str(24 * 1024**3)))


def _hardware_note(has_gguf: bool, large_transformers: bool) -> str:
    if has_gguf:
        return "Downloaded GGUF model. Select the GGUF artifact directly for Morrigan inference."
    if large_transformers:
        return "Downloaded large Transformers model. BF16/FP16 inference is disabled on Morrigan; convert or quantize before serving."
    return "Downloaded model. GGUF artifacts can be selected directly for Morrigan inference."


if __name__ == "__main__":
    sys.exit(run_worker(main))
