from __future__ import annotations

from fastapi.testclient import TestClient

from conftest import upload_and_approve_dataset, wait_for_job


def test_capability_transfer_dry_run_lifecycle(authed: TestClient) -> None:
    dataset_id = upload_and_approve_dataset(authed)

    create = authed.post(
        "/api/capability-transfers",
        json={
            "display_name": "Dry CoT Transfer",
            "source_model_slug": "lfm25-12b-instruct",
            "source_runtime": "transformers",
            "target_model_slug": "lfm25-12b-instruct",
            "target_runtime": "transformers",
            "calibration_dataset_id": dataset_id,
            "layer_targets": "all",
            "contrast_mode": "prompt_pair",
            "rank": 8,
            "dry_run": True,
        },
    )
    assert create.status_code == 200, create.text
    transfer = create.json()
    assert transfer["status"] == "extracting"
    assert transfer["extract_job_id"]

    extract_job = wait_for_job(authed, transfer["extract_job_id"])
    assert extract_job["status"] == "succeeded"

    detail = authed.get(f"/api/capability-transfers/{transfer['transfer_id']}")
    assert detail.status_code == 200, detail.text
    transfer = detail.json()
    assert transfer["status"] == "extracted"
    assert transfer["vector_artifact_id"]

    align = authed.post(f"/api/capability-transfers/{transfer['transfer_id']}/align", json={"rank": 8, "layer_pairs": []})
    assert align.status_code == 200, align.text
    transfer = align.json()
    assert transfer["status"] == "aligning"
    assert transfer["align_job_id"]

    align_job = wait_for_job(authed, transfer["align_job_id"])
    assert align_job["status"] == "succeeded"

    ready = authed.get(f"/api/capability-transfers/{transfer['transfer_id']}").json()
    assert ready["status"] == "ready"
    assert ready["alignment_artifact_id"]

    activate = authed.post(f"/api/capability-transfers/{transfer['transfer_id']}/activate", json={"alpha": 1.2, "layer_targets": "last"})
    assert activate.status_code == 200, activate.text
    active_target = activate.json()["active_target"]
    assert active_target["capability_transfer_id"] == transfer["transfer_id"]

    target = authed.get("/api/inference/target")
    assert target.status_code == 200, target.text
    assert target.json()["capability_transfer_id"] == transfer["transfer_id"]

    delete = authed.delete(f"/api/capability-transfers/{transfer['transfer_id']}")
    assert delete.status_code == 200, delete.text
    assert delete.json()["deleted"] is True

    target_after_delete = authed.get("/api/inference/target")
    assert "capability_transfer_id" not in target_after_delete.json()


def test_capability_transfer_requires_auth(client: TestClient) -> None:
    response = client.get("/api/capability-transfers")

    assert response.status_code == 401


def test_capability_transfer_rejects_unapproved_dataset(authed: TestClient) -> None:
    response = authed.post(
        "/api/capability-transfers",
        json={
            "display_name": "No Dataset",
            "source_model_slug": "lfm25-12b-instruct",
            "source_runtime": "transformers",
            "target_model_slug": "lfm25-12b-instruct",
            "target_runtime": "transformers",
            "calibration_dataset_id": "missing",
            "layer_targets": "all",
            "contrast_mode": "prompt_pair",
            "rank": 8,
            "dry_run": True,
        },
    )

    assert response.status_code == 400
    assert "Calibration dataset" in response.json()["detail"]
