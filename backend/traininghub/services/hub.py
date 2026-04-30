from __future__ import annotations

import os
import urllib.parse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from fnmatch import fnmatch
from pathlib import PurePosixPath
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError


class HubResolveError(ValueError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class ParsedHubInput:
    repo_id: str
    hinted_type: str | None


TRANSFORMERS_MODEL_INCLUDE_PATTERNS = [
    "*.safetensors",
    "*.safetensors.index.json",
    "*.json",
    "tokenizer*",
    "*.model",
    "*.txt",
    "*.jinja",
]
GGUF_MODEL_INCLUDE_PATTERNS = ["*.gguf", "*.json", "tokenizer*"]


def resolve_hub_resource(input_value: str, resource_type: str = "auto", revision: str | None = None) -> dict[str, Any]:
    parsed = parse_hub_input(input_value)
    target_type = _target_type(resource_type, parsed.hinted_type)
    if target_type == "auto":
        return _resolve_auto(parsed.repo_id, revision)
    if target_type not in {"model", "dataset"}:
        raise HubResolveError("resource_type must be auto, model, or dataset.")
    return _resolve_typed(parsed.repo_id, target_type, revision)


def ensure_confirmed_hub_sha(
    repo_id: str,
    resource_type: str,
    confirmed_sha: str | None,
    revision: str | None = None,
) -> None:
    if not confirmed_sha:
        return
    resolved = _resolve_typed(repo_id, resource_type, revision)
    current_sha = str(resolved.get("sha") or "")
    if current_sha != confirmed_sha:
        raise HubResolveError(
            "The Hugging Face repository changed after confirmation. Run Find again before starting the download.",
            status_code=409,
        )


def default_model_include_patterns(file_names: list[str]) -> list[str]:
    if any(name.casefold().endswith(".gguf") for name in file_names):
        return list(GGUF_MODEL_INCLUDE_PATTERNS)
    return list(TRANSFORMERS_MODEL_INCLUDE_PATTERNS)


def estimate_model_download(
    repo_id: str,
    revision: str | None = None,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> dict[str, Any]:
    api = HfApi(token=_hub_token())
    try:
        info = api.model_info(repo_id=repo_id, revision=revision, files_metadata=True, token=_hub_token())
    except RepositoryNotFoundError as exc:
        raise HubResolveError(
            "Hugging Face model repository was not found or is private without an available token.",
            status_code=404,
        ) from exc
    except RevisionNotFoundError as exc:
        raise HubResolveError("The requested Hugging Face revision was not found.", status_code=404) from exc
    except HubResolveError:
        raise
    except Exception as exc:
        raise HubResolveError(f"Unable to contact Hugging Face: {exc}", status_code=502) from exc

    sibling_records = _sibling_records(info)
    file_names = [record["name"] for record in sibling_records]
    allow_patterns = include_patterns or default_model_include_patterns(file_names)
    ignore_patterns = exclude_patterns or []
    match = _estimate_file_match(sibling_records, allow_patterns, ignore_patterns)
    return {
        "repo_id": str(getattr(info, "id", repo_id)),
        "sha": str(getattr(info, "sha", "")),
        "revision": revision or "",
        "include_patterns": allow_patterns,
        "exclude_patterns": ignore_patterns,
        **match,
    }


def parse_hub_input(input_value: str) -> ParsedHubInput:
    value = input_value.strip()
    if not value:
        raise HubResolveError("Enter a Hugging Face URL or repository name.")
    parsed = urllib.parse.urlparse(value)
    if parsed.netloc in {"huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co"}:
        return _parse_hub_path(parsed.path)
    if value.startswith("datasets/"):
        repo_id = value.removeprefix("datasets/").strip("/")
        return ParsedHubInput(_clean_repo_id(repo_id), "dataset")
    if value.startswith("models/"):
        repo_id = value.removeprefix("models/").strip("/")
        return ParsedHubInput(_clean_repo_id(repo_id), "model")
    return ParsedHubInput(_clean_repo_id(value), None)


def _parse_hub_path(path: str) -> ParsedHubInput:
    parts = [part for part in PurePosixPath(urllib.parse.unquote(path)).parts if part not in {"/", ""}]
    if not parts:
        raise HubResolveError("The Hugging Face URL does not include a repository name.")
    if parts[0] == "datasets":
        return ParsedHubInput(_clean_repo_id("/".join(parts[1:3])), "dataset")
    if parts[0] == "models":
        return ParsedHubInput(_clean_repo_id("/".join(parts[1:3])), "model")
    if parts[0] == "spaces":
        raise HubResolveError("Spaces are not supported by TrainingHub acquisition.", status_code=400)
    return ParsedHubInput(_clean_repo_id("/".join(parts[:2])), "model")


def _clean_repo_id(repo_id: str) -> str:
    clean = repo_id.strip().strip("/")
    clean = clean.split("?")[0].split("#")[0]
    stop_words = {"tree", "blob", "resolve"}
    parts = [part for part in clean.split("/") if part]
    for index, part in enumerate(parts):
        if part in stop_words:
            parts = parts[:index]
            break
    if len(parts) > 2:
        parts = parts[:2]
    clean = "/".join(parts)
    if not clean:
        raise HubResolveError("The Hugging Face input does not include a repository name.")
    return clean


def _target_type(resource_type: str, hinted_type: str | None) -> str:
    if resource_type != "auto":
        return resource_type
    return hinted_type or "auto"


def _resolve_auto(repo_id: str, revision: str | None) -> dict[str, Any]:
    try:
        return _resolve_typed(repo_id, "model", revision)
    except HubResolveError as model_error:
        try:
            return _resolve_typed(repo_id, "dataset", revision)
        except HubResolveError:
            raise model_error


def _resolve_typed(repo_id: str, resource_type: str, revision: str | None) -> dict[str, Any]:
    api = HfApi(token=_hub_token())
    try:
        if resource_type == "model":
            info = api.model_info(repo_id=repo_id, revision=revision, files_metadata=True, token=_hub_token())
        elif resource_type == "dataset":
            info = api.dataset_info(repo_id=repo_id, revision=revision, files_metadata=True, token=_hub_token())
        else:
            raise HubResolveError("resource_type must be model or dataset.")
    except RepositoryNotFoundError as exc:
        raise HubResolveError(
            f"Hugging Face {resource_type} repository was not found or is private without an available token.",
            status_code=404,
        ) from exc
    except RevisionNotFoundError as exc:
        raise HubResolveError("The requested Hugging Face revision was not found.", status_code=404) from exc
    except HubResolveError:
        raise
    except Exception as exc:
        raise HubResolveError(f"Unable to contact Hugging Face: {exc}", status_code=502) from exc
    return _info_response(info, resource_type)


def _info_response(info: Any, resource_type: str) -> dict[str, Any]:
    sibling_records = _sibling_records(info)
    siblings = [record["name"] for record in sibling_records]
    summary = _model_summary(info, sibling_records) if resource_type == "model" else _dataset_summary(info, siblings)
    return {
        "found": True,
        "resource_type": resource_type,
        "repo_id": str(getattr(info, "id", "")),
        "sha": str(getattr(info, "sha", "")),
        "last_modified": _iso(getattr(info, "last_modified", None)),
        "private": bool(getattr(info, "private", False)),
        "gated": getattr(info, "gated", False) or False,
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
        "tags": [str(tag) for tag in (getattr(info, "tags", None) or [])],
        "siblings": siblings[:80],
        "summary": summary,
    }


def _model_summary(info: Any, sibling_records: list[dict[str, Any]]) -> dict[str, Any]:
    config = getattr(info, "config", None) or {}
    safetensors = getattr(info, "safetensors", None)
    siblings = [record["name"] for record in sibling_records]
    include_patterns = default_model_include_patterns(siblings)
    default_estimate = _estimate_file_match(sibling_records, include_patterns, [])
    return {
        "library": getattr(info, "library_name", None),
        "pipeline": getattr(info, "pipeline_tag", None),
        "architectures": config.get("architectures") if isinstance(config, dict) else None,
        "safetensors_parameters": getattr(safetensors, "total", None),
        "has_gguf": bool(getattr(info, "gguf", None)) or any(name.casefold().endswith(".gguf") for name in siblings),
        "file_count": len(siblings),
        "file_types": _file_types(siblings),
        "default_include_patterns": include_patterns,
        "estimated_download_size_bytes": default_estimate["estimated_size_bytes"],
        "matched_file_count": default_estimate["matched_file_count"],
        "total_size_bytes": default_estimate["total_size_bytes"],
    }


def _dataset_summary(info: Any, siblings: list[str]) -> dict[str, Any]:
    card_data = getattr(info, "card_data", None)
    configs = _card_value(card_data, "configs")
    return {
        "configs": configs if isinstance(configs, list) else [],
        "splits": _dataset_splits_from_card(card_data),
        "sample_fields": _dataset_field_hints(siblings),
        "file_count": len(siblings),
        "file_types": _file_types(siblings),
        "has_viewer_files": any(name.casefold().endswith((".parquet", ".jsonl", ".json", ".csv", ".arrow")) for name in siblings),
    }


def _dataset_splits_from_card(card_data: Any) -> list[str]:
    dataset_info = _card_value(card_data, "dataset_info")
    if not isinstance(dataset_info, dict):
        return []
    splits = dataset_info.get("splits")
    if isinstance(splits, list):
        return [str(item.get("name") or item.get("split") or item) for item in splits]
    if isinstance(splits, dict):
        return [str(key) for key in splits.keys()]
    return []


def _dataset_field_hints(siblings: list[str]) -> list[str]:
    names = " ".join(siblings).casefold()
    hints = []
    for field_name in ["messages", "prompt", "question", "problem", "instruction", "response", "answer", "solution", "final_answer", "split"]:
        if field_name in names:
            hints.append(field_name)
    return hints[:10]


def _file_types(siblings: list[str]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for name in siblings:
        suffix = PurePosixPath(name).suffix.casefold() or "[none]"
        counter[suffix] += 1
    return dict(counter.most_common(8))


def _sibling_records(info: Any) -> list[dict[str, Any]]:
    records = []
    for sibling in getattr(info, "siblings", []) or []:
        name = str(getattr(sibling, "rfilename", ""))
        if not name:
            continue
        size = getattr(sibling, "size", None)
        records.append({"name": name, "size": int(size) if isinstance(size, int) else 0})
    return records


def _estimate_file_match(
    sibling_records: list[dict[str, Any]],
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> dict[str, Any]:
    matched = []
    for record in sibling_records:
        name = str(record["name"])
        included = not include_patterns or any(fnmatch(name, pattern) for pattern in include_patterns)
        excluded = any(fnmatch(name, pattern) for pattern in exclude_patterns)
        if included and not excluded:
            matched.append(record)
    estimated_size = sum(int(record.get("size") or 0) for record in matched)
    total_size = sum(int(record.get("size") or 0) for record in sibling_records)
    return {
        "estimated_size_bytes": estimated_size,
        "total_size_bytes": total_size,
        "matched_file_count": len(matched),
        "matched_files": [str(record["name"]) for record in matched[:80]],
    }


def _card_value(card_data: Any, key: str) -> Any:
    if card_data is None:
        return None
    if isinstance(card_data, dict):
        return card_data.get(key)
    return getattr(card_data, key, None)


def _iso(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value) if value else None


def _hub_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or None
