from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("TRAININGHUB_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("TRAININGHUB_APP_ROOT", str(root))
    monkeypatch.setenv("TRAININGHUB_DATABASE_PATH", str(tmp_path / "data" / "traininghub.sqlite3"))
    monkeypatch.setenv("TRAININGHUB_WORKER_PYTHON", os.environ.get("PYTHON", "python3"))
    monkeypatch.setenv("TRAININGHUB_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("TRAININGHUB_ADMIN_PASSWORD", "traininghub")
    from traininghub.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def authed(client: TestClient) -> TestClient:
    response = client.post("/api/auth/login", json={"username": "admin", "password": "traininghub"})
    assert response.status_code == 200, response.text
    return client


def valid_math_csv() -> bytes:
    return (
        "id,system,prompt,response,final_answer,category,difficulty,source,split,tags,notes\n"
        "row_001,You are careful,What is 1 + 1?,The final answer is 2.,2,arithmetic,easy,manual,train,math,\n"
        "row_002,You are careful,What is 2 + 2?,The final answer is 4.,4,arithmetic,easy,manual,validation,math,\n"
    ).encode("utf-8")


def valid_calibration_csv() -> bytes:
    return (
        "id,prompt_present,prompt_absent,continuation_prefix,system_present,system_absent,prompt,source,split,tags,notes\n"
        "cal_001,Think step by step: What is 1 + 1?,Answer directly: What is 1 + 1?,,,,,manual,holdout,calibration,\n"
        "cal_002,Think step by step: What is 2 + 2?,Answer directly: What is 2 + 2?,,,,,manual,holdout,calibration,\n"
    ).encode("utf-8")


def valid_system_calibration_csv() -> bytes:
    return (
        "id,prompt_present,prompt_absent,continuation_prefix,system_present,system_absent,prompt,source,split,tags,notes\n"
        "cal_001,,,,You are a careful tutor.,You answer briefly.,What is 1 + 1?,manual,holdout,calibration,system pair\n"
        "cal_002,,,,You reason aloud.,You give terse answers.,What is 2 + 2?,manual,holdout,calibration,system pair\n"
    ).encode("utf-8")


def upload_and_approve_dataset(client: TestClient) -> str:
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("math.csv", valid_math_csv(), "text/csv")},
        data={"dataset_type": "math_sft", "title": "Math", "slug": "math", "max_sequence_length": "2048"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["created"] is True
    dataset_id = body["dataset_id"]
    approve = client.post(f"/api/datasets/{dataset_id}/approve")
    assert approve.status_code == 200, approve.text
    assert approve.json()["approved"] is True
    return dataset_id


def upload_and_approve_calibration_dataset(client: TestClient) -> str:
    return upload_and_approve_calibration_csv(client, valid_calibration_csv(), "capability-calibration")


def upload_and_approve_system_calibration_dataset(client: TestClient) -> str:
    return upload_and_approve_calibration_csv(client, valid_system_calibration_csv(), "system-capability-calibration")


def upload_and_approve_calibration_csv(client: TestClient, csv_bytes: bytes, slug: str) -> str:
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("calibration.csv", csv_bytes, "text/csv")},
        data={
            "dataset_type": "capability_calibration",
            "title": "Capability Calibration",
            "slug": slug,
            "max_sequence_length": "2048",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["created"] is True
    dataset_id = body["dataset_id"]
    approve = client.post(f"/api/datasets/{dataset_id}/approve")
    assert approve.status_code == 200, approve.text
    assert approve.json()["approved"] is True
    return dataset_id


def wait_for_job(client: TestClient, job_id: str, timeout: float = 10.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get("/api/jobs")
        assert response.status_code == 200, response.text
        jobs = response.json()
        job = next(item for item in jobs if item["job_id"] == job_id)
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(0.2)
    raise AssertionError(f"Job {job_id} did not finish in time.")
