from __future__ import annotations

import csv
import io
import json
import os
import shutil
import sys
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from traininghub.services.hub import ensure_confirmed_hub_sha
from traininghub.services.datasets import create_dataset_from_canonical_records
from traininghub.workers.common import WorkerContext, run_worker


PROMPT_FIELDS = ["prompt", "question", "problem", "input", "instruction", "query"]
RESPONSE_FIELDS = ["response", "answer", "solution", "output", "completion", "target"]
SYSTEM_FIELDS = ["system", "system_prompt"]
FINAL_ANSWER_FIELDS = ["final_answer", "answer", "target", "label"]
ALLOWED_SPLITS = {"train", "validation", "holdout"}


def main(context: WorkerContext, payload: dict[str, Any]) -> None:
    source_type = str(payload.get("source_type", "")).strip()
    raw_paths: list[Path] = []
    if source_type == "hf":
        ensure_confirmed_hub_sha(
            str(payload["repo_id"]),
            "dataset",
            payload.get("confirmed_sha"),
            payload.get("revision"),
        )
        source_rows = _load_hf_dataset(context, payload)
        source_label = payload["repo_id"]
        raw_paths.append(_raw_cache_dir(context))
    elif source_type == "url":
        source_rows = _load_url_dataset(context, payload)
        source_label = payload["url"]
        raw_paths.append(_raw_cache_dir(context))
    else:
        raise RuntimeError("source_type must be hf or url.")

    if payload.get("clean_with_inference"):
        context.event(
            "cleaning",
            "Formatting imported rows with the selected local inference target.",
            data={"cleaning_model": payload.get("cleaning_model") or "local-inference"},
        )
    max_rows = payload.get("max_rows")
    max_rows_int = int(max_rows) if max_rows else None
    records: list[dict[str, Any]] = []
    skipped = 0
    for index, row in enumerate(source_rows):
        context.check_cancelled()
        if max_rows_int is not None and len(records) >= max_rows_int:
            break
        record = _canonicalize_row(row, index, source_label, payload)
        if record is None:
            skipped += 1
            continue
        if payload.get("clean_with_inference"):
            record = _clean_record(record, payload)
        records.append(record)
        if len(records) % 25 == 0:
            context.metric({"accepted_records": len(records), "skipped_records": skipped})

    if not records:
        raise RuntimeError("Dataset import produced no valid prompt/response records.")

    result = create_dataset_from_canonical_records(
        context.database_path,
        Path(os.environ["TRAININGHUB_DATA_ROOT"]),
        records,
        str(payload.get("dataset_type") or "math_sft"),
        str(payload.get("title") or "Imported dataset"),
        str(payload.get("slug") or "imported-dataset"),
        source_label,
        keep_source_copy=False,
    )
    cleanup_report = _delete_raw_paths(context, raw_paths) if payload.get("delete_raw_after_clean", True) else {"deleted": [], "retained": [str(path) for path in raw_paths]}
    report = {
        "source_type": source_type,
        "source": source_label,
        "created": bool(result.get("created")),
        "dataset_id": result.get("dataset_id"),
        "version_id": result.get("version_id"),
        "accepted_records": len(records),
        "skipped_records": skipped,
        "clean_with_inference": bool(payload.get("clean_with_inference")),
        "cleaning_model": payload.get("cleaning_model") or "",
        "approval_required": True,
        "review_sample_size": min(100, len(records)),
        "raw_cleanup": cleanup_report,
    }
    report_path = context.write_metadata("dataset_import_report.json", report)
    context.register_artifact(report_path, "dataset_import_report", "Dataset import report", report)
    cleanup_path = context.write_metadata("dataset_raw_cleanup_report.json", cleanup_report)
    context.register_artifact(cleanup_path, "dataset_raw_cleanup_report", "Dataset raw cleanup report", cleanup_report)
    if not result.get("created"):
        errors = result.get("validation", {}).get("errors", [])
        raise RuntimeError(f"Dataset import failed validation: {errors[:3]}")
    context.event(
        "dataset_imported",
        "Dataset imported and queued for human approval.",
        data={"dataset_id": result["dataset_id"], "version_id": result["version_id"], "row_count": len(records)},
    )


def _load_hf_dataset(context: WorkerContext, payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The datasets package is required for Hugging Face dataset imports.") from exc

    cache_dir = _raw_cache_dir(context) / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    context.event(
        "download",
        "Loading Hugging Face dataset.",
        data={"repo_id": payload["repo_id"], "split": payload.get("split"), "raw_cache": str(cache_dir)},
    )
    dataset = load_dataset(
        path=payload["repo_id"],
        name=payload.get("config_name") or None,
        split=payload.get("split") or "train",
        revision=payload.get("revision") or None,
        cache_dir=str(cache_dir),
        token=os.getenv("HF_TOKEN") or None,
    )
    return dataset


def _load_url_dataset(context: WorkerContext, payload: dict[str, Any]) -> list[dict[str, Any]]:
    url = str(payload["url"])
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read()
    raw_path = _raw_cache_dir(context) / f"source{_url_suffix(url) or '.data'}"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(content)
    suffix = _url_suffix(url)
    if suffix == ".csv":
        text = content.decode("utf-8-sig")
        return [dict(row) for row in csv.DictReader(io.StringIO(text))]
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        for line in content.decode("utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        payload_json = json.loads(content.decode("utf-8"))
        if isinstance(payload_json, list):
            return [dict(row) for row in payload_json if isinstance(row, dict)]
        if isinstance(payload_json, dict):
            for key in ["data", "rows", "examples", "train"]:
                value = payload_json.get(key)
                if isinstance(value, list):
                    return [dict(row) for row in value if isinstance(row, dict)]
        raise RuntimeError("JSON dataset must be a list of records or contain a data/rows/examples/train list.")
    raise RuntimeError("URL imports support CSV, JSONL, NDJSON, and JSON files.")


def _url_suffix(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.casefold()
    for suffix in [".jsonl", ".ndjson", ".json", ".csv"]:
        if path.endswith(suffix):
            return suffix
    return ""


def _canonicalize_row(row: Any, index: int, source_label: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    messages = row.get("messages")
    prompt_from_messages, response_from_messages, system_from_messages = _messages_to_fields(messages)
    prompt = _field(row, payload.get("prompt_field"), PROMPT_FIELDS) or prompt_from_messages
    response = _field(row, payload.get("response_field"), RESPONSE_FIELDS) or response_from_messages
    if not prompt or not response:
        return None
    system = _field(row, payload.get("system_field"), SYSTEM_FIELDS) or system_from_messages
    final_answer = _field(row, payload.get("final_answer_field"), FINAL_ANSWER_FIELDS)
    split_value = _field(row, payload.get("split_field"), ["split"]) or payload.get("default_split") or "holdout"
    split = _normalize_split(split_value)
    row_id = _field(row, "id", ["id", "uuid", "example_id"]) or f"imported_{index + 1:06d}"
    category = _field(row, "category", ["category", "subject", "topic"]) or "imported"
    difficulty = _field(row, "difficulty", ["difficulty", "level"]) or "unknown"
    notes = _field(row, "notes", ["notes", "rationale"]) or ""
    return {
        "messages": [
            *([{"role": "system", "content": system}] if system else []),
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "metadata": {
            "id": row_id,
            "final_answer": final_answer,
            "category": category,
            "difficulty": difficulty,
            "source": source_label,
            "split": split,
            "tags": _tags(row.get("tags")),
            "notes": notes,
        },
    }


def _messages_to_fields(messages: Any) -> tuple[str, str, str]:
    if not isinstance(messages, list):
        return "", "", ""
    prompt = ""
    response = ""
    system = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").casefold()
        content = _stringify(message.get("content"))
        if role == "system" and not system:
            system = content
        elif role == "user" and not prompt:
            prompt = content
        elif role == "assistant" and not response:
            response = content
    return prompt, response, system


def _field(row: dict[str, Any], explicit: Any, candidates: list[str]) -> str:
    field_names = [str(explicit)] if explicit else []
    field_names.extend(candidates)
    for field_name in field_names:
        if not field_name or field_name not in row:
            continue
        value = _stringify(row.get(field_name)).strip()
        if value:
            return value
    return ""


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def _normalize_split(value: str) -> str:
    normalized = str(value).strip().casefold()
    if normalized in {"valid", "dev", "eval"}:
        normalized = "validation"
    if normalized in {"test", "benchmark"}:
        normalized = "holdout"
    return normalized if normalized in ALLOWED_SPLITS else "holdout"


def _tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return ["imported"]


def _raw_cache_dir(context: WorkerContext) -> Path:
    return context.job_dir / "raw"


def _clean_record(record: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    cleaned = json.loads(json.dumps(record))
    messages = cleaned.get("messages", [])
    for message in messages:
        if isinstance(message, dict):
            message["content"] = " ".join(str(message.get("content") or "").split())
    metadata = cleaned.setdefault("metadata", {})
    metadata["source"] = " ".join(str(metadata.get("source") or "").split())
    metadata["notes"] = " ".join(str(metadata.get("notes") or "Cleaned and formatted by TrainingHub acquisition.").split())
    metadata["cleaning_model"] = str(payload.get("cleaning_model") or "local-inference")
    metadata["cleaning_status"] = "formatted"
    return cleaned


def _delete_raw_paths(context: WorkerContext, paths: list[Path]) -> dict[str, Any]:
    deleted: list[str] = []
    retained: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            deleted.append(str(path))
        except OSError:
            retained.append(str(path))
    context.event("raw_cleanup", "Raw acquisition files deleted.", data={"deleted": deleted, "retained": retained})
    return {"deleted": deleted, "retained": retained}


if __name__ == "__main__":
    sys.exit(run_worker(main))
