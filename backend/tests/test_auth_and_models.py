from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient


def test_login_logout_and_me(client: TestClient) -> None:
    assert client.get("/api/me").status_code == 401
    login = client.post("/api/auth/login", json={"username": "admin", "password": "traininghub"})
    assert login.status_code == 200
    assert client.get("/api/me").json()["username"] == "admin"
    logout = client.post("/api/auth/logout")
    assert logout.status_code == 200
    assert client.get("/api/me").status_code == 401


def test_model_registry_blocks_large_full_finetune(authed: TestClient) -> None:
    models = authed.get("/api/models").json()
    qwen = next(model for model in models if model["slug"] == "qwen36-35b-a3b")
    assert qwen["supports_bf16_inference"] is False
    assert qwen["supports_gguf_path"] is True
    assert qwen["supports_lora"] is True
    assert qwen["supports_qlora"] is True
    assert qwen["supports_full_finetune"] is False
    assert "12 GB" in qwen["hardware_note"]


def test_active_inference_target_defaults_and_updates_base_model(authed: TestClient) -> None:
    current = authed.get("/api/inference/target")
    assert current.status_code == 200
    assert current.json()["target_type"] == "base_model"
    assert current.json()["model_slug"] == "lfm25-12b-instruct"

    updated = authed.post("/api/inference/target", json={"target_type": "base_model", "model_slug": "qwen3-4b"})
    assert updated.status_code == 200, updated.text
    assert updated.json()["model_slug"] == "qwen3-4b"

    persisted = authed.get("/api/inference/target")
    assert persisted.json()["model_slug"] == "qwen3-4b"


def test_active_inference_rejects_large_bf16_model(authed: TestClient) -> None:
    response = authed.post("/api/inference/target", json={"target_type": "base_model", "model_slug": "qwen36-35b-a3b"})
    assert response.status_code == 400
    assert "GGUF" in response.json()["detail"]


def test_active_inference_rejects_smoke_checkpoint(authed: TestClient, tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "smoke-checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text(json.dumps({"traininghub_smoke": True}), encoding="utf-8")
    (checkpoint_dir / "README.md").write_text("TrainingHub smoke checkpoint placeholder.\n", encoding="utf-8")
    database_path = Path(os.environ["TRAININGHUB_DATABASE_PATH"])
    with sqlite3.connect(database_path) as conn:
        conn.execute(
            """
            INSERT INTO model_registry (
                slug, provider_id, display_name, family, parameter_count,
                supports_bf16_inference, supports_benchmark, supports_quantization,
                supports_gguf_path, is_saga,
                hardware_note, default_dtype, max_sequence_length, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "smoke-checkpoint",
                f"local:{checkpoint_dir}",
                "Smoke Checkpoint",
                "test",
                "trained",
                1,
                1,
                1,
                1,
                0,
                "Smoke placeholder.",
                "bf16",
                2048,
                json.dumps({"route": "trained_model", "local_path": str(checkpoint_dir)}),
            ),
        )

    response = authed.post("/api/inference/target", json={"target_type": "base_model", "model_slug": "smoke-checkpoint"})
    assert response.status_code == 400
    assert "missing model weights or tokenizer files" in response.json()["detail"]

    options = authed.get("/api/inference/options").json()
    option = next(item for item in options if item["model_slug"] == "smoke-checkpoint")
    assert option["enabled"] is False
    models = authed.get("/api/models").json()
    assert "smoke-checkpoint" not in {item["slug"] for item in models}


def test_models_list_includes_local_checkpoint_with_weights_and_tokenizer(authed: TestClient, tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "real-checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").write_bytes(b"weights")
    (checkpoint_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    database_path = Path(os.environ["TRAININGHUB_DATABASE_PATH"])
    with sqlite3.connect(database_path) as conn:
        conn.execute(
            """
            INSERT INTO model_registry (
                slug, provider_id, display_name, family, parameter_count,
                supports_bf16_inference, supports_benchmark, supports_quantization,
                supports_gguf_path, is_saga,
                hardware_note, default_dtype, max_sequence_length, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "real-checkpoint",
                f"local:{checkpoint_dir}",
                "Real Checkpoint",
                "test",
                "trained",
                1,
                1,
                1,
                1,
                0,
                "Runnable checkpoint.",
                "bf16",
                2048,
                json.dumps({"route": "trained_model", "local_path": str(checkpoint_dir)}),
            ),
        )

    models = authed.get("/api/models").json()
    model = next(item for item in models if item["slug"] == "real-checkpoint")
    assert model["supports_bf16_inference"] is True


def test_active_inference_allows_gguf_artifact(authed: TestClient, tmp_path: Path) -> None:
    artifact_path = tmp_path / "model.Q4_K_M.gguf"
    artifact_path.write_text("gguf", encoding="utf-8")
    database_path = Path(os.environ["TRAININGHUB_DATABASE_PATH"])
    with sqlite3.connect(database_path) as conn:
        conn.execute(
            """
            INSERT INTO artifacts (
                artifact_id, job_id, artifact_type, display_name, path,
                size_bytes, checksum_sha256, metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "artifact_gguf",
                None,
                "gguf_quantized",
                "Smoke GGUF",
                str(artifact_path),
                artifact_path.stat().st_size,
                "sha256",
                json.dumps({"model_slug": "lfm25-12b-base"}),
                1,
            ),
        )
    response = authed.post("/api/inference/target", json={"target_type": "gguf_artifact", "artifact_id": "artifact_gguf"})
    assert response.status_code == 200, response.text
    assert response.json()["target_type"] == "gguf_artifact"
    assert response.json()["path"] == str(artifact_path)


def test_inference_run_requires_auth(client: TestClient) -> None:
    response = client.post("/api/inference/run", json={"prompt": "Say hello.", "dry_run": True})
    assert response.status_code == 401


def test_inference_run_dry_run_streams_sse(authed: TestClient) -> None:
    response = authed.post(
        "/api/inference/run",
        json={
            "prompt": "Say hello.",
            "system": "You are concise.",
            "max_tokens": 12,
            "dry_run": True,
        },
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: token" in response.text
    assert "event: done" in response.text
    assert "Dry-run " in response.text
    assert "response " in response.text


def test_default_model_delete_tombstones_seed_and_survives_reseed(authed: TestClient) -> None:
    response = authed.delete("/api/models/lfm25-12b-instruct")
    assert response.status_code == 200
    body = response.json()
    assert body["deleted"] is True
    assert body["removed_records"] == {"models": 1, "artifacts": 0}
    assert "lfm25-12b-instruct" not in {model["slug"] for model in authed.get("/api/models").json()}
    assert authed.get("/api/inference/target").json()["model_slug"] != "lfm25-12b-instruct"

    from traininghub.core.config import get_settings
    from traininghub.core.database import init_db

    init_db(get_settings())
    assert "lfm25-12b-instruct" not in {model["slug"] for model in authed.get("/api/models").json()}


def test_downloaded_model_delete_removes_files_artifacts_and_active_target(authed: TestClient) -> None:
    database_path = Path(os.environ["TRAININGHUB_DATABASE_PATH"])
    data_root = database_path.parent
    model_dir = data_root / "models" / "custom-delete"
    model_dir.mkdir(parents=True)
    artifact_path = model_dir / "custom.Q4_K_M.gguf"
    artifact_path.write_text("gguf", encoding="utf-8")

    with sqlite3.connect(database_path) as conn:
        conn.execute(
            """
            INSERT INTO model_registry (
                slug, provider_id, display_name, family, parameter_count,
                supports_bf16_inference, supports_benchmark, supports_quantization,
                supports_gguf_path, is_saga,
                hardware_note, default_dtype, max_sequence_length, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "custom-delete",
                "local:custom-delete",
                "Custom Delete",
                "custom",
                "unknown",
                0,
                1,
                1,
                1,
                0,
                "Downloaded custom model.",
                "auto",
                2048,
                json.dumps({"local_path": str(model_dir), "artifact_ids": ["artifact_delete"]}),
            ),
        )
        conn.execute(
            """
            INSERT INTO artifacts (
                artifact_id, job_id, artifact_type, display_name, path,
                size_bytes, checksum_sha256, metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "artifact_delete",
                None,
                "gguf_quantized",
                "Custom Delete GGUF",
                str(artifact_path),
                artifact_path.stat().st_size,
                "sha256",
                json.dumps({"model_slug": "custom-delete"}),
                1,
            ),
        )

    target = authed.post("/api/inference/target", json={"target_type": "gguf_artifact", "artifact_id": "artifact_delete"})
    assert target.status_code == 200, target.text

    response = authed.delete("/api/models/custom-delete")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["deleted"] is True
    assert body["removed_records"] == {"models": 1, "artifacts": 1}
    assert not model_dir.exists()
    assert "custom-delete" not in {model["slug"] for model in authed.get("/api/models").json()}
    assert authed.get("/api/inference/target").json()["model_slug"] == "lfm25-12b-instruct"
