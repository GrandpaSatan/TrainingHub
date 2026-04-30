from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import upload_and_approve_dataset, wait_for_job
from traininghub.core.config import get_settings
from traininghub.core.database import connect
from traininghub.core.security import utc_now
from traininghub.services import gpu as gpu_service
from traininghub.services.jobs import _worker_env


def _gpu(index: int, free_mb: int) -> dict[str, int | str]:
    return {
        "index": index,
        "name": "RTX 3060",
        "memory_total_mb": 12288,
        "memory_used_mb": 12288 - free_mb,
        "memory_free_mb": free_mb,
        "utilization_gpu_percent": 0,
        "temperature_c": 40,
    }


def _insert_running_gpu_job(database_path: Path, job_id: str, gpu_ids: list[int]) -> None:
    now = utc_now()
    with connect(database_path) as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, job_type, status, slug, payload_json, work_dir, worker_module,
                worker_pid, gpu_ids, created_at, started_at, finished_at, terminal_message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                "generate",
                "running",
                job_id,
                "{}",
                "/tmp/job",
                "traininghub.workers.run_generate_examples",
                os.getpid(),
                f"[{','.join(str(gpu_id) for gpu_id in gpu_ids)}]",
                now,
                now,
                None,
                "",
            ),
        )


def test_lora_training_smoke_registers_artifacts(authed: TestClient) -> None:
    dataset_id = upload_and_approve_dataset(authed)
    response = authed.post(
        "/api/jobs/fine-tune",
        json={
            "mode": "lora",
            "model_slug": "lfm25-12b-base",
            "dataset_id": dataset_id,
            "preset": "smoke",
            "output_name": "rag-lora-smoke",
            "max_steps": 1,
            "dry_run": True,
        },
    )
    assert response.status_code == 200, response.text
    job = wait_for_job(authed, response.json()["job_id"])
    assert job["status"] == "succeeded"
    artifacts = authed.get("/api/artifacts").json()
    artifact_types = {artifact["artifact_type"] for artifact in artifacts}
    assert {"training_adapter", "training_merged_checkpoint", "training_report"}.issubset(artifact_types)
    models = authed.get("/api/models").json()
    assert "rag-lora-smoke" not in {item["slug"] for item in models}
    options = authed.get("/api/inference/options").json()
    assert all(item["model_slug"] != "rag-lora-smoke" for item in options)


def test_training_job_stops_inference_before_worker_start(authed: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    shutdown_calls = []

    def shutdown() -> dict[str, int]:
        shutdown_calls.append("shutdown")
        return {"active_runs_cancelled": 0, "active_runs_remaining": 0, "cache_entries_cleared": 0}

    monkeypatch.setattr("traininghub.services.jobs.shutdown_inference_for_training", shutdown)
    dataset_id = upload_and_approve_dataset(authed)

    response = authed.post(
        "/api/jobs/fine-tune",
        json={
            "mode": "lora",
            "model_slug": "lfm25-12b-base",
            "dataset_id": dataset_id,
            "preset": "smoke",
            "output_name": "shutdown-lora-smoke",
            "max_steps": 1,
            "dry_run": True,
        },
    )

    assert response.status_code == 200, response.text
    job = wait_for_job(authed, response.json()["job_id"])
    assert job["status"] == "succeeded"
    assert shutdown_calls == ["shutdown"]


def test_lora_training_rejects_4b_2048_context(authed: TestClient) -> None:
    dataset_id = upload_and_approve_dataset(authed)

    response = authed.post(
        "/api/jobs/fine-tune",
        json={
            "mode": "lora",
            "model_slug": "qwen3-4b",
            "dataset_id": dataset_id,
            "preset": "custom",
            "output_name": "qwen-4b-unsafe-lora",
            "max_steps": 1,
            "max_sequence_length": 2048,
        },
    )

    assert response.status_code == 400
    assert "Use QLoRA" in response.json()["detail"]


def test_lora_training_defaults_4b_context_to_1024(authed: TestClient) -> None:
    dataset_id = upload_and_approve_dataset(authed)

    response = authed.post(
        "/api/jobs/fine-tune",
        json={
            "mode": "lora",
            "model_slug": "qwen3-4b",
            "dataset_id": dataset_id,
            "preset": "smoke",
            "output_name": "qwen-4b-safe-lora-smoke",
            "dry_run": True,
        },
    )

    assert response.status_code == 200, response.text
    job = wait_for_job(authed, response.json()["job_id"])
    assert job["status"] == "succeeded"
    assert job["payload"]["max_sequence_length"] == 1024


def test_worker_env_sets_expandable_cuda_allocator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    settings = get_settings()

    env = _worker_env(settings, "job_123", tmp_path / "job_123", [0])

    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_worker_env_records_gpu_strategy(tmp_path: Path) -> None:
    settings = get_settings()

    env = _worker_env(settings, "job_123", tmp_path / "job_123", [0, 1], "balanced_model_parallel", "balanced")

    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert env["TRAININGHUB_GPU_IDS"] == "0,1"
    assert env["TRAININGHUB_GPU_STRATEGY"] == "balanced_model_parallel"
    assert env["TRAININGHUB_TRAINING_DEVICE_MAP"] == "balanced"


def test_gpu_allocator_prefers_idle_gpu_with_most_free_vram(authed: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(gpu_service, "query_gpus", lambda: [_gpu(0, 2048), _gpu(1, 8192)])

    allocation = gpu_service.choose_gpu_allocation(settings.database_path)

    assert allocation.gpu_ids == [1]
    assert allocation.strategy == "single"


def test_gpu_allocator_ignores_leased_gpu(authed: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(gpu_service, "query_gpus", lambda: [_gpu(0, 8192), _gpu(1, 2048)])
    _insert_running_gpu_job(settings.database_path, "running-gpu-0", [0])

    allocation = gpu_service.choose_gpu_allocation(settings.database_path)

    assert allocation.gpu_ids == [1]
    assert allocation.leased_gpu_ids == [0]


def test_full_training_real_workers_requests_balanced_two_gpu_plan(
    authed: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAININGHUB_ENABLE_REAL_WORKERS", "1")
    monkeypatch.setattr(gpu_service, "query_gpus", lambda: [_gpu(0, 8192), _gpu(1, 8192)])
    monkeypatch.setattr(
        "traininghub.services.jobs.shutdown_inference_for_training",
        lambda: {"active_runs_cancelled": 0, "active_runs_remaining": 0, "cache_entries_cleared": 0},
    )
    monkeypatch.setattr("traininghub.services.jobs._spawn_worker", lambda *args, **kwargs: None)
    dataset_id = upload_and_approve_dataset(authed)

    response = authed.post(
        "/api/jobs/fine-tune",
        json={
            "mode": "full",
            "model_slug": "lfm25-12b-base",
            "dataset_id": dataset_id,
            "preset": "standard",
            "output_name": "full-balanced",
            "max_steps": 1,
        },
    )

    assert response.status_code == 200, response.text
    job = response.json()
    assert job["gpu_ids"] == [0, 1]
    assert job["payload"]["training_launch_mode"] == "single_process_model_parallel"
    assert job["payload"]["training_device_map"] == "balanced"
    assert job["payload"]["gpu_allocation"]["strategy"] == "balanced_model_parallel"


def test_full_training_rejects_when_two_idle_gpus_are_not_available(
    authed: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAININGHUB_ENABLE_REAL_WORKERS", "1")
    monkeypatch.setattr(gpu_service, "query_gpus", lambda: [_gpu(0, 8192), _gpu(1, 8192)])
    monkeypatch.setattr(
        "traininghub.services.jobs.shutdown_inference_for_training",
        lambda: {"active_runs_cancelled": 0, "active_runs_remaining": 0, "cache_entries_cleared": 0},
    )
    settings = get_settings()
    _insert_running_gpu_job(settings.database_path, "running-gpu-0", [0])
    dataset_id = upload_and_approve_dataset(authed)

    response = authed.post(
        "/api/jobs/fine-tune",
        json={
            "mode": "full",
            "model_slug": "lfm25-12b-base",
            "dataset_id": dataset_id,
            "preset": "standard",
            "output_name": "full-blocked",
            "max_steps": 1,
        },
    )

    assert response.status_code == 400
    assert "2 idle GPU(s) required" in response.json()["detail"]


def test_non_training_job_does_not_stop_inference(authed: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    shutdown_calls = []
    monkeypatch.setattr(
        "traininghub.services.jobs.shutdown_inference_for_training",
        lambda: shutdown_calls.append("shutdown") or {"active_runs_cancelled": 0},
    )

    response = authed.post(
        "/api/jobs/benchmark",
        json={"model_slug": "lfm25-12b-base", "benchmarks": ["gsm8k"], "limit": 1, "dry_run": True},
    )

    assert response.status_code == 200, response.text
    job = wait_for_job(authed, response.json()["job_id"])
    assert job["status"] == "succeeded"
    assert shutdown_calls == []


def test_full_training_rejects_unsupported_large_model(authed: TestClient) -> None:
    dataset_id = upload_and_approve_dataset(authed)
    response = authed.post(
        "/api/jobs/fine-tune",
        json={"mode": "full", "model_slug": "qwen36-35b-a3b", "dataset_id": dataset_id, "preset": "smoke"},
    )
    assert response.status_code == 400
    assert "does not support FULL training" in response.json()["detail"]


def test_benchmark_job_smoke_stores_result(authed: TestClient) -> None:
    response = authed.post(
        "/api/jobs/benchmark",
        json={"model_slug": "lfm25-12b-base", "benchmarks": ["gsm8k", "math-500"], "limit": 3, "dry_run": True},
    )
    assert response.status_code == 200, response.text
    job = wait_for_job(authed, response.json()["job_id"])
    assert job["status"] == "succeeded"
    artifacts = authed.get("/api/artifacts").json()
    assert any(artifact["artifact_type"] == "benchmark_results" for artifact in artifacts)


def test_benchmark_catalog_and_results_endpoint(authed: TestClient) -> None:
    catalog = authed.get("/api/benchmarks/catalog")
    assert catalog.status_code == 200, catalog.text
    benchmark_ids = {item["id"] for item in catalog.json()}
    assert len(benchmark_ids) == 11
    assert {"mmlu", "hellaswag", "arc", "ifeval", "humaneval"}.issubset(benchmark_ids)

    response = authed.post(
        "/api/jobs/benchmark",
        json={"model_slug": "lfm25-12b-base", "benchmarks": ["mmlu", "humaneval"], "limit": 2, "dry_run": True},
    )
    assert response.status_code == 200, response.text
    job = wait_for_job(authed, response.json()["job_id"])
    assert job["status"] == "succeeded"

    filtered = authed.get("/api/benchmarks/results", params={"model_slug": "lfm25-12b-base", "benchmark": "mmlu"})
    assert filtered.status_code == 200, filtered.text
    rows = filtered.json()
    assert len(rows) == 1
    assert rows[0]["benchmark_name"] == "mmlu"
    assert rows[0]["metrics"]["pass_at_1"] is not None
    assert rows[0]["artifact_id"]


def test_benchmark_job_rejects_unknown_catalog_id(authed: TestClient) -> None:
    response = authed.post(
        "/api/jobs/benchmark",
        json={"model_slug": "lfm25-12b-base", "benchmarks": ["missing-bench"], "limit": 2, "dry_run": True},
    )
    assert response.status_code == 400
    assert "Unsupported benchmark id" in response.json()["detail"]


def test_generate_examples_imports_as_review_dataset(authed: TestClient) -> None:
    response = authed.post(
        "/api/jobs/generate",
        json={"teacher_model": "local", "seed_prompt": "Create addition examples.", "target_count": 3, "dry_run": True},
    )
    assert response.status_code == 200, response.text
    job = wait_for_job(authed, response.json()["job_id"])
    assert job["status"] == "succeeded"
    artifact = next(item for item in authed.get("/api/artifacts").json() if item["artifact_type"] == "generated_dataset")
    imported = authed.post(
        "/api/datasets/import-generated",
        json={"artifact_id": artifact["artifact_id"], "title": "Generated Review", "slug": "generated-review"},
    )
    assert imported.status_code == 200, imported.text
    body = imported.json()
    assert body["created"] is True
    approve = authed.post(f"/api/datasets/{body['dataset_id']}/approve")
    assert approve.status_code == 200


def test_model_url_download_registers_gguf_for_inference(authed: TestClient, tmp_path: Path) -> None:
    source = tmp_path / "tiny.Q4_K_M.gguf"
    source.write_bytes(b"GGUF test placeholder")
    response = authed.post(
        "/api/models/download-url",
        json={
            "url": source.as_uri(),
            "slug": "tiny-gguf",
            "display_name": "Tiny GGUF",
        },
    )
    assert response.status_code == 200, response.text
    job = wait_for_job(authed, response.json()["job_id"])
    assert job["status"] == "succeeded"
    models = authed.get("/api/models").json()
    model = next(item for item in models if item["slug"] == "tiny-gguf")
    assert model["supports_bf16_inference"] is False
    artifacts = authed.get("/api/artifacts").json()
    artifact = next(item for item in artifacts if item["artifact_type"] == "gguf_quantized")
    assert artifact["metadata"]["model_slug"] == "tiny-gguf"
    options = authed.get("/api/inference/options").json()
    assert any(item["artifact_id"] == artifact["artifact_id"] and item["enabled"] for item in options)


def test_model_upload_hf_dry_run_uses_local_source_without_token(authed: TestClient, tmp_path: Path) -> None:
    source = tmp_path / "model.gguf"
    source.write_bytes(b"GGUF test placeholder")
    response = authed.post(
        "/api/models/upload-hf",
        json={
            "repo_id": "example/traininghub-test",
            "source_path": str(source),
            "private": True,
            "dry_run": True,
        },
    )
    assert response.status_code == 200, response.text
    job = wait_for_job(authed, response.json()["job_id"])
    assert job["status"] == "succeeded"
    artifacts = authed.get("/api/artifacts").json()
    assert any(artifact["artifact_type"] == "hf_upload_report" for artifact in artifacts)


def test_cleanup_scan_creates_manifest(authed: TestClient) -> None:
    response = authed.post("/api/cleanup/scan", json={"include_immediate": True})
    assert response.status_code == 200
    body = response.json()
    assert body["manifest_id"].startswith("cl_")
    assert body["policy"].startswith("Only approved")
