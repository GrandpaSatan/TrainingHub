from __future__ import annotations

import json
from pathlib import Path


LOCAL_PROVIDER_PREFIX = "local:"
TOKENIZER_FILENAMES = {"tokenizer.json", "tokenizer.model", "vocab.json"}
WEIGHT_PATTERNS = ("*.safetensors", "pytorch_model*.bin")


def local_provider_path(provider_id: str) -> Path | None:
    if not provider_id.startswith(LOCAL_PROVIDER_PREFIX):
        return None
    path_value = provider_id.removeprefix(LOCAL_PROVIDER_PREFIX).strip()
    if not path_value:
        return None
    return Path(path_value).expanduser()


def checkpoint_has_inference_files(path: Path) -> bool:
    if not path.exists() or not path.is_dir() or _is_smoke_checkpoint(path):
        return False
    return _has_weight_file(path) and _has_tokenizer_file(path)


def local_provider_is_runnable(provider_id: str) -> bool:
    path = local_provider_path(provider_id)
    return path is not None and checkpoint_has_inference_files(path)


def _is_smoke_checkpoint(path: Path) -> bool:
    config_path = path / "config.json"
    if not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(config.get("traininghub_smoke"))


def _has_weight_file(path: Path) -> bool:
    return any(next(path.glob(pattern), None) is not None for pattern in WEIGHT_PATTERNS)


def _has_tokenizer_file(path: Path) -> bool:
    return any((path / filename).exists() for filename in TOKENIZER_FILENAMES)
