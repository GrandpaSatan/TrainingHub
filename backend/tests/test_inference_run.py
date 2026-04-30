from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from traininghub.services import inference_run
from traininghub.services.inference_run import (
    InferenceRunError,
    SamplingConfig,
    _CachedTransformersModel,
    _TRANSFORMERS_CACHE,
    _resolve_transformers_model_source,
    cancel_active_inference_runs,
    clear_transformers_cache,
    run_prompt,
    shutdown_inference_for_training,
)


def test_transformers_model_source_resolves_local_provider_id(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()

    source = _resolve_transformers_model_source({"provider_id": f"local:{checkpoint_dir}", "model_slug": "trained-model"})

    assert source == str(checkpoint_dir)


def test_transformers_model_source_preserves_hub_repo_id() -> None:
    source = _resolve_transformers_model_source({"provider_id": "LiquidAI/LFM2.5-1.2B-Instruct"})

    assert source == "LiquidAI/LFM2.5-1.2B-Instruct"


def test_transformers_model_source_rejects_missing_local_checkpoint(tmp_path: Path) -> None:
    missing_checkpoint = tmp_path / "missing-checkpoint"

    with pytest.raises(InferenceRunError, match="Local checkpoint not found"):
        _resolve_transformers_model_source({"provider_id": f"local:{missing_checkpoint}"})


def test_active_transfer_uses_explicit_database_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    database_path = tmp_path / "traininghub.sqlite3"
    seen: list[tuple[Path, str]] = []

    def fake_transfer_for_inference(path: Path, transfer_id: str) -> dict[str, object]:
        seen.append((path, transfer_id))
        return {
            "transfer_id": transfer_id,
            "target_runtime": "transformers",
            "target_model_slug": "target-model",
            "config": {},
        }

    monkeypatch.delenv("TRAININGHUB_DATABASE_PATH", raising=False)
    monkeypatch.setattr(inference_run, "transfer_for_inference", fake_transfer_for_inference)

    transfer = inference_run._resolve_active_transfer(
        {"capability_transfer_id": "ct_example", "model_slug": "target-model"},
        database_path,
    )

    assert transfer and transfer["transfer_id"] == "ct_example"
    assert seen == [(database_path, "ct_example")]


def test_clear_transformers_cache_releases_cached_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    released = []
    monkeypatch.setattr(inference_run, "_release_cuda_memory", lambda: released.append(True))
    _TRANSFORMERS_CACHE.clear()
    _TRANSFORMERS_CACHE["base:model"] = _CachedTransformersModel(object(), object(), 999.0)

    assert clear_transformers_cache() == 1

    assert _TRANSFORMERS_CACHE == {}
    assert released == [True]


def test_shutdown_inference_for_training_cancels_active_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRAININGHUB_INFERENCE_SHUTDOWN_GRACE_SECONDS", "0")

    async def scenario() -> None:
        stream = run_prompt(
            {"target_type": "base_model", "display_name": "Dry run"},
            "hello",
            SamplingConfig(dry_run=True),
        )
        first = await anext(stream)
        assert first

        result = shutdown_inference_for_training()

        assert result["active_runs_cancelled"] == 1
        assert result["cache_entries_cleared"] == 0
        assert [token async for token in stream] == []
        assert cancel_active_inference_runs() == 0

    asyncio.run(scenario())
