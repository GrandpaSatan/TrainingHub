from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError

from conftest import wait_for_job


class FakeSibling:
    def __init__(self, name: str, size: int | None = None) -> None:
        self.rfilename = name
        self.size = size


class FakeHfApi:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def model_info(self, repo_id: str, **kwargs: object) -> SimpleNamespace:
        revision = kwargs.get("revision")
        if repo_id == "missing/model":
            raise RepositoryNotFoundError("missing", response=_response(404))
        if revision == "bad-revision":
            raise RevisionNotFoundError("bad revision", response=_response(404))
        if repo_id == "Qwen/Qwen3.5-27B":
            return SimpleNamespace(
                id=repo_id,
                sha="qwen-sha",
                last_modified=datetime(2026, 4, 24, tzinfo=timezone.utc),
                private=False,
                gated=False,
                downloads=6200000,
                likes=963,
                tags=["transformers", "qwen3_5", "image-text-to-text"],
                siblings=[
                    FakeSibling("config.json", 4000),
                    FakeSibling("chat_template.jinja", 7000),
                    FakeSibling("merges.txt", 3_300_000),
                    FakeSibling("model.safetensors-00001-of-00011.safetensors", 5_200_000_000),
                    FakeSibling("model.safetensors.index.json", 126_000),
                    FakeSibling("tokenizer.json", 12_800_000),
                    FakeSibling("vocab.json", 6_700_000),
                ],
                library_name="transformers",
                pipeline_tag="image-text-to-text",
                safetensors=SimpleNamespace(total=27_781_400_000),
                gguf=None,
                config={"architectures": ["Qwen3_5ForConditionalGeneration"]},
            )
        return SimpleNamespace(
            id=repo_id,
            sha="model-sha",
            last_modified=datetime(2026, 1, 2, tzinfo=timezone.utc),
            private=False,
            gated=False,
            downloads=12,
            likes=3,
            tags=["transformers", "text-generation"],
            siblings=[FakeSibling("config.json"), FakeSibling("model.safetensors")],
            library_name="transformers",
            pipeline_tag="text-generation",
            safetensors=SimpleNamespace(total=1000),
            gguf=None,
            config={"architectures": ["TinyModel"]},
        )

    def dataset_info(self, repo_id: str, **kwargs: object) -> SimpleNamespace:
        revision = kwargs.get("revision")
        if repo_id == "missing/dataset":
            raise RepositoryNotFoundError("missing", response=_response(404))
        if revision == "bad-revision":
            raise RevisionNotFoundError("bad revision", response=_response(404))
        return SimpleNamespace(
            id=repo_id,
            sha="dataset-sha",
            last_modified=datetime(2026, 1, 3, tzinfo=timezone.utc),
            private=False,
            gated=False,
            downloads=21,
            likes=5,
            tags=["math", "jsonl"],
            siblings=[FakeSibling("data/train.jsonl"), FakeSibling("README.md")],
            card_data={"dataset_info": {"splits": [{"name": "train"}]}, "configs": [{"config_name": "default"}]},
        )


def test_resolve_public_model_and_dataset_url(authed: TestClient, monkeypatch) -> None:
    import traininghub.services.hub as hub_service

    monkeypatch.setattr(hub_service, "HfApi", FakeHfApi)

    model = authed.post("/api/hub/resolve", json={"input": "acme/tiny-model", "resource_type": "model"})
    assert model.status_code == 200, model.text
    model_body = model.json()
    assert model_body["resource_type"] == "model"
    assert model_body["repo_id"] == "acme/tiny-model"
    assert model_body["sha"] == "model-sha"
    assert model_body["summary"]["pipeline"] == "text-generation"

    dataset = authed.post("/api/hub/resolve", json={"input": "https://huggingface.co/datasets/acme/math/tree/main", "resource_type": "auto"})
    assert dataset.status_code == 200, dataset.text
    dataset_body = dataset.json()
    assert dataset_body["resource_type"] == "dataset"
    assert dataset_body["repo_id"] == "acme/math"
    assert dataset_body["sha"] == "dataset-sha"
    assert dataset_body["summary"]["splits"] == ["train"]


def test_resolve_clear_errors_and_revision_mismatch(authed: TestClient, monkeypatch) -> None:
    import traininghub.services.hub as hub_service

    monkeypatch.setattr(hub_service, "HfApi", FakeHfApi)

    missing = authed.post("/api/hub/resolve", json={"input": "missing/model", "resource_type": "model"})
    assert missing.status_code == 404
    assert "not found" in missing.json()["detail"]

    bad_revision = authed.post("/api/hub/resolve", json={"input": "acme/tiny-model", "resource_type": "model", "revision": "bad-revision"})
    assert bad_revision.status_code == 404
    assert "revision" in bad_revision.json()["detail"]

    stale = authed.post(
        "/api/models/download-hf",
        json={"repo_id": "acme/tiny-model", "confirmed_sha": "stale-sha", "dry_run": True},
    )
    assert stale.status_code == 409
    assert "changed after confirmation" in stale.json()["detail"]


def test_model_confirmed_sha_can_queue_dry_run_download(authed: TestClient, monkeypatch) -> None:
    import traininghub.services.hub as hub_service

    monkeypatch.setattr(hub_service, "HfApi", FakeHfApi)

    response = authed.post(
        "/api/models/download-hf",
        json={
            "repo_id": "acme/tiny-model",
            "confirmed_sha": "model-sha",
            "slug": "tiny-model",
            "display_name": "Tiny Model",
            "dry_run": True,
        },
    )
    assert response.status_code == 200, response.text
    job = wait_for_job(authed, response.json()["job_id"])
    assert job["status"] == "succeeded"


def test_qwen_model_download_uses_auxiliary_tokenizer_patterns(authed: TestClient, monkeypatch) -> None:
    import traininghub.services.hub as hub_service

    monkeypatch.setattr(hub_service, "HfApi", FakeHfApi)

    resolved = authed.post("/api/hub/resolve", json={"input": "Qwen/Qwen3.5-27B", "resource_type": "model"})
    assert resolved.status_code == 200, resolved.text
    summary = resolved.json()["summary"]
    assert "*.txt" in summary["default_include_patterns"]
    assert "*.jinja" in summary["default_include_patterns"]
    assert summary["estimated_download_size_bytes"] > 5_000_000_000

    response = authed.post(
        "/api/models/download-hf",
        json={
            "repo_id": "Qwen/Qwen3.5-27B",
            "confirmed_sha": "qwen-sha",
            "slug": "qwen35-27b",
            "display_name": "Qwen3.5 27B",
            "dry_run": True,
        },
    )
    assert response.status_code == 200, response.text
    job = wait_for_job(authed, response.json()["job_id"])
    assert job["status"] == "succeeded"
    payload = job["payload"]
    assert "*.txt" in payload["include_patterns"]
    assert "*.jinja" in payload["include_patterns"]
    assert "merges.txt" in payload["download_estimate"]["matched_files"]
    assert "chat_template.jinja" in payload["download_estimate"]["matched_files"]


def test_model_download_preflight_blocks_when_disk_is_too_small(authed: TestClient, monkeypatch) -> None:
    import traininghub.api.models as models_api
    import traininghub.services.hub as hub_service

    monkeypatch.setattr(hub_service, "HfApi", FakeHfApi)
    monkeypatch.setattr(models_api, "disk_usage", lambda _path: {"total_bytes": 10_000, "used_bytes": 1_000, "free_bytes": 10_000})

    response = authed.post(
        "/api/models/download-hf",
        json={"repo_id": "Qwen/Qwen3.5-27B", "confirmed_sha": "qwen-sha"},
    )
    assert response.status_code == 400
    assert "Not enough disk space" in response.json()["detail"]


def test_dataset_clean_import_deletes_raw_cache_and_keeps_canonical_data(authed: TestClient, tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text(
        "\n".join(
            json.dumps({"question": f" What is {index} + 1? ", "answer": f" The final answer is {index + 1}. ", "split": "train"})
            for index in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    response = authed.post(
        "/api/datasets/import-url",
        json={
            "url": source.as_uri(),
            "title": "Clean URL Review",
            "slug": "clean-url-review",
            "dataset_type": "math_sft",
            "max_rows": 3,
            "clean_with_inference": True,
            "delete_raw_after_clean": True,
            "cleaning_model": "Morrigan local",
        },
    )
    assert response.status_code == 200, response.text
    job = wait_for_job(authed, response.json()["job_id"])
    assert job["status"] == "succeeded"
    assert not (Path(job["work_dir"]) / "raw").exists()

    dataset = next(item for item in authed.get("/api/datasets").json() if item["title"] == "Clean URL Review")
    assert dataset["approved"] is False
    sample = authed.get(f"/api/datasets/{dataset['dataset_id']}/review-sample?sample_size=1").json()
    metadata = sample["records"][0]["metadata"]
    assert metadata["cleaning_model"] == "Morrigan local"
    assert metadata["cleaning_status"] == "formatted"
    assert Path(dataset["jsonl_path"]).name == "canonical.jsonl"

    artifacts = authed.get("/api/artifacts").json()
    assert any(artifact["artifact_type"] == "dataset_raw_cleanup_report" for artifact in artifacts)


def _response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=httpx.Request("GET", "https://huggingface.co/test"))
