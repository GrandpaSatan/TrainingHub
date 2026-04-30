from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from conftest import valid_math_csv
from conftest import wait_for_job


def test_template_download(authed: TestClient) -> None:
    response = authed.post("/api/datasets/template", json={"dataset_type": "math_sft"})
    assert response.status_code == 200
    assert "prompt,response,final_answer" in response.text


def test_upload_validation_errors(authed: TestClient) -> None:
    csv_bytes = (
        "id,system,prompt,response,final_answer,category,difficulty,source,split,tags,notes\n"
        "row_001,,What is 1 + 1?,Answer,,arithmetic,easy,manual,invalid,math,\n"
        "row_001,,What is 1 + 1?,Answer,,arithmetic,easy,manual,train,math,\n"
    ).encode("utf-8")
    response = authed.post(
        "/api/datasets/upload",
        files={"file": ("bad.csv", csv_bytes, "text/csv")},
        data={"dataset_type": "math_sft", "title": "Bad", "slug": "bad", "max_sequence_length": "2048"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["created"] is False
    codes = {error["code"] for error in body["validation"]["errors"]}
    assert {"required", "invalid_split", "duplicate_id"}.issubset(codes)


def test_upload_and_approve_dataset(authed: TestClient) -> None:
    response = authed.post(
        "/api/datasets/upload",
        files={"file": ("math.csv", valid_math_csv(), "text/csv")},
        data={"dataset_type": "math_sft", "title": "Math", "slug": "math", "max_sequence_length": "2048"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["created"] is True
    assert body["validation"]["accepted_count"] == 2
    dataset_id = body["dataset_id"]
    approve = authed.post(f"/api/datasets/{dataset_id}/approve")
    assert approve.status_code == 200
    datasets = authed.get("/api/datasets").json()
    assert datasets[0]["approved"] is True
    sample = authed.get(f"/api/datasets/{dataset_id}/review-sample?sample_size=100")
    assert sample.status_code == 200
    assert sample.json()["required_review_sample_size"] == 2


def test_import_url_dataset_requires_review_before_use(authed: TestClient, tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    with source.open("w", encoding="utf-8") as handle:
        for index in range(120):
            handle.write(
                json.dumps(
                    {
                        "question": f"What is {index} + 1?",
                        "answer": f"The final answer is {index + 1}.",
                        "split": "train",
                    }
                )
                + "\n"
            )
    response = authed.post(
        "/api/datasets/import-url",
        json={
            "url": source.as_uri(),
            "title": "URL Review",
            "slug": "url-review",
            "dataset_type": "math_sft",
            "max_rows": 120,
        },
    )
    assert response.status_code == 200, response.text
    job = wait_for_job(authed, response.json()["job_id"])
    assert job["status"] == "succeeded"
    dataset = next(item for item in authed.get("/api/datasets").json() if item["title"] == "URL Review")
    assert dataset["approved"] is False
    jsonl_path = Path(dataset["jsonl_path"])
    assert jsonl_path.exists()
    sample = authed.get(f"/api/datasets/{dataset['dataset_id']}/review-sample?sample_size=100").json()
    assert sample["required_review_sample_size"] == 100
    assert len(sample["records"]) == 100
    reject = authed.post(f"/api/datasets/{dataset['dataset_id']}/reject")
    assert reject.status_code == 200
    assert reject.json()["deleted"] is True
    assert reject.json()["version"]["review_sample_size"] == 100
    assert dataset["dataset_id"] not in {item["dataset_id"] for item in authed.get("/api/datasets").json()}
    assert not jsonl_path.exists()


def test_existing_rejected_dataset_is_purged_on_list(authed: TestClient) -> None:
    response = authed.post(
        "/api/datasets/upload",
        files={"file": ("math.csv", valid_math_csv(), "text/csv")},
        data={"dataset_type": "math_sft", "title": "Rejected", "slug": "rejected", "max_sequence_length": "2048"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    dataset_id = body["dataset_id"]
    jsonl_path = Path(body["jsonl_path"])
    assert jsonl_path.exists()

    from os import environ

    with sqlite3.connect(environ["TRAININGHUB_DATABASE_PATH"]) as conn:
        conn.execute(
            "UPDATE dataset_versions SET reviewed_at = ?, review_sample_size = ? WHERE dataset_id = ?",
            (1, 1, dataset_id),
        )

    datasets = authed.get("/api/datasets")
    assert datasets.status_code == 200
    assert dataset_id not in {item["dataset_id"] for item in datasets.json()}
    assert not jsonl_path.exists()
