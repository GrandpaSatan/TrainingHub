from __future__ import annotations

import csv
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

from traininghub.core.database import connect, row_to_dict, rows_to_dicts
from traininghub.core.id_utils import make_dataset_id, slugify
from traininghub.core.security import utc_now
from traininghub.services.deletion import safe_remove_traininghub_path


CSV_COLUMNS = [
    "id",
    "system",
    "prompt",
    "response",
    "final_answer",
    "category",
    "difficulty",
    "source",
    "split",
    "tags",
    "notes",
]

CALIBRATION_CSV_COLUMNS = [
    "id",
    "prompt_present",
    "prompt_absent",
    "continuation_prefix",
    "system_present",
    "system_absent",
    "prompt",
    "source",
    "split",
    "tags",
    "notes",
]

ALLOWED_SPLITS = {"train", "validation", "holdout"}
KNOWN_BENCHMARK_PROMPT_FRAGMENTS = [
    "janet's ducks lay 16 eggs per day",
    "a robe takes 2 bolts of blue fiber",
    "every day ryan spends",
    "find the domain of the expression",
]


def build_template(dataset_type: str = "math_sft") -> str:
    output = io.StringIO()
    if dataset_type == "capability_calibration":
        writer = csv.DictWriter(output, fieldnames=CALIBRATION_CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "id": "calibration_001",
                "prompt_present": "Solve this step by step before giving the answer: What is 17 + 25?",
                "prompt_absent": "Give only the final answer: What is 17 + 25?",
                "continuation_prefix": "",
                "system_present": "",
                "system_absent": "",
                "prompt": "",
                "source": "manual",
                "split": "holdout",
                "tags": "calibration,reasoning",
                "notes": "Prompt-pair contrast example.",
            }
        )
        return output.getvalue()

    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    writer.writerow(
        {
            "id": "example_001",
            "system": "You are a careful math tutor.",
            "prompt": "What is 2 + 2?",
            "response": "2 + 2 = 4. The final answer is 4.",
            "final_answer": "4" if dataset_type in {"math_sft", "holdout"} else "",
            "category": "arithmetic",
            "difficulty": "easy",
            "source": "manual",
            "split": "train",
            "tags": "math,addition",
            "notes": "",
        }
    )
    return output.getvalue()


def validate_csv_bytes(csv_bytes: bytes, dataset_type: str, max_sequence_length: int) -> dict[str, Any]:
    try:
        text = csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        return {
            "valid": False,
            "errors": [{"row_number": 0, "field": "file", "code": "malformed_csv", "message": str(exc)}],
            "warnings": [],
            "rows": [],
            "accepted_count": 0,
        }
    return validate_csv_text(text, dataset_type, max_sequence_length)


def validate_csv_text(csv_text: str, dataset_type: str, max_sequence_length: int) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()

    try:
        reader = csv.DictReader(io.StringIO(csv_text))
        if reader.fieldnames is None:
            raise csv.Error("missing header row")
        expected_columns = _csv_columns(dataset_type)
        missing_columns = [column for column in expected_columns if column not in reader.fieldnames]
        if missing_columns:
            errors.append(
                {
                    "row_number": 0,
                    "field": "header",
                    "code": "missing_columns",
                    "message": f"Missing columns: {', '.join(missing_columns)}",
                }
            )
            return {"valid": False, "errors": errors, "warnings": warnings, "rows": [], "accepted_count": 0}

        for row_number, row in enumerate(reader, start=2):
            cleaned = {column: (row.get(column) or "").strip() for column in expected_columns}
            if dataset_type == "capability_calibration":
                if not cleaned["split"]:
                    cleaned["split"] = "holdout"
                row_errors = _validate_calibration_row(cleaned, row_number, max_sequence_length, seen_ids, seen_pairs)
            else:
                row_errors = _validate_row(cleaned, row_number, dataset_type, max_sequence_length, seen_ids, seen_pairs)
            errors.extend(row_errors)
            if not row_errors:
                _append_leakage_warnings(cleaned, row_number, warnings)
            rows.append(cleaned)
    except csv.Error as exc:
        errors.append({"row_number": 0, "field": "file", "code": "malformed_csv", "message": str(exc)})

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "rows": rows,
        "accepted_count": 0 if errors else len(rows),
    }


def _csv_columns(dataset_type: str) -> list[str]:
    if dataset_type == "capability_calibration":
        return CALIBRATION_CSV_COLUMNS
    return CSV_COLUMNS


def _validate_row(
    row: dict[str, str],
    row_number: int,
    dataset_type: str,
    max_sequence_length: int,
    seen_ids: set[str],
    seen_pairs: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for field in ["prompt", "response", "split"]:
        if not row[field]:
            errors.append(_error(row_number, field, "required", "Field is required."))
    if dataset_type in {"math_sft", "holdout"} and not row["final_answer"]:
        errors.append(_error(row_number, "final_answer", "required", "Math records require a final answer."))
    if row["split"] and row["split"] not in ALLOWED_SPLITS:
        errors.append(_error(row_number, "split", "invalid_split", "Use train, validation, or holdout."))
    if row["id"]:
        if row["id"] in seen_ids:
            errors.append(_error(row_number, "id", "duplicate_id", "Duplicate id in this upload."))
        seen_ids.add(row["id"])
    pair = (row["prompt"].casefold(), row["response"].casefold())
    if pair != ("", ""):
        if pair in seen_pairs:
            errors.append(
                _error(row_number, "prompt", "duplicate_prompt_response", "Duplicate prompt/response pair in this upload.")
            )
        seen_pairs.add(pair)
    approximate_tokens = max(1, len((row["system"] + row["prompt"] + row["response"]).split()))
    if approximate_tokens > max_sequence_length:
        errors.append(
            _error(
                row_number,
                "prompt",
                "too_long",
                f"Approximate token count {approximate_tokens} exceeds max sequence length {max_sequence_length}.",
            )
        )
    return errors


def _validate_calibration_row(
    row: dict[str, str],
    row_number: int,
    max_sequence_length: int,
    seen_ids: set[str],
    seen_pairs: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    prompt_present = row["prompt_present"]
    prompt_absent = row["prompt_absent"]
    system_present = row["system_present"]
    system_absent = row["system_absent"]
    prompt = row["prompt"]
    has_prompt_pair = bool(prompt_present and prompt_absent)
    has_system_pair = bool(system_present and system_absent and prompt)
    if not has_prompt_pair and not has_system_pair:
        errors.append(
            _error(
                row_number,
                "prompt_present",
                "required_contrast_pair",
                "Provide prompt_present and prompt_absent, or system_present, system_absent, and prompt.",
            )
        )
    if row["split"] and row["split"] not in ALLOWED_SPLITS:
        errors.append(_error(row_number, "split", "invalid_split", "Use train, validation, or holdout."))
    if row["id"]:
        if row["id"] in seen_ids:
            errors.append(_error(row_number, "id", "duplicate_id", "Duplicate id in this upload."))
        seen_ids.add(row["id"])
    pair = (
        (prompt_present or f"{system_present}\n{prompt}").casefold(),
        (prompt_absent or f"{system_absent}\n{prompt}").casefold(),
    )
    if pair != ("", ""):
        if pair in seen_pairs:
            errors.append(_error(row_number, "prompt_present", "duplicate_contrast_pair", "Duplicate contrast pair in this upload."))
        seen_pairs.add(pair)
    approximate_tokens = max(
        1,
        len((prompt_present + prompt_absent + row["continuation_prefix"] + system_present + system_absent + prompt).split()),
    )
    if approximate_tokens > max_sequence_length:
        errors.append(
            _error(
                row_number,
                "prompt_present",
                "too_long",
                f"Approximate token count {approximate_tokens} exceeds max sequence length {max_sequence_length}.",
            )
        )
    return errors


def _append_leakage_warnings(row: dict[str, str], row_number: int, warnings: list[dict[str, Any]]) -> None:
    prompt = row["prompt"].casefold()
    source = row["source"].casefold()
    split = row["split"].casefold()
    tags = row["tags"].casefold()
    if split == "train" and any(name in source or name in tags for name in ["gsm8k", "math-500", "math500"]):
        warnings.append(
            _warning(
                row_number,
                "source",
                "potential_benchmark_leakage",
                "Training row references a benchmark source.",
            )
        )
    for fragment in KNOWN_BENCHMARK_PROMPT_FRAGMENTS:
        if fragment in prompt:
            warnings.append(
                _warning(row_number, "prompt", "potential_benchmark_leakage", "Prompt resembles a known benchmark item.")
            )


def _error(row_number: int, field: str, code: str, message: str) -> dict[str, Any]:
    return {"row_number": row_number, "field": field, "code": code, "message": message}


def _warning(row_number: int, field: str, code: str, message: str) -> dict[str, Any]:
    return {"row_number": row_number, "field": field, "code": code, "message": message}


def canonical_record(row: dict[str, str], dataset_type: str = "math_sft") -> dict[str, Any]:
    if dataset_type == "capability_calibration":
        record = {
            key: row[key]
            for key in [
                "prompt_present",
                "prompt_absent",
                "continuation_prefix",
                "system_present",
                "system_absent",
                "prompt",
            ]
            if row.get(key)
        }
        tags = [tag.strip() for tag in row["tags"].split(",") if tag.strip()]
        record["metadata"] = {
            "id": row["id"],
            "source": row["source"],
            "split": row["split"] or "holdout",
            "tags": tags,
            "notes": row["notes"],
            "dataset_type": dataset_type,
        }
        return record

    messages = []
    if row["system"]:
        messages.append({"role": "system", "content": row["system"]})
    messages.extend(
        [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ]
    )
    tags = [tag.strip() for tag in row["tags"].split(",") if tag.strip()]
    return {
        "messages": messages,
        "metadata": {
            "id": row["id"],
            "final_answer": row["final_answer"],
            "category": row["category"],
            "difficulty": row["difficulty"],
            "source": row["source"],
            "split": row["split"],
            "tags": tags,
            "notes": row["notes"],
        },
    }


def create_dataset_version(
    database_path: Path,
    data_root: Path,
    csv_bytes: bytes,
    dataset_type: str,
    title: str,
    slug: str,
    max_sequence_length: int,
    source: str,
) -> dict[str, Any]:
    validation = validate_csv_bytes(csv_bytes, dataset_type, max_sequence_length)
    if not validation["valid"]:
        return {"created": False, "validation": validation}

    dataset_slug = slugify(slug or title, "dataset")
    dataset_id = make_dataset_id(dataset_slug)
    version_number = 1
    version_id = f"{dataset_id}_v{version_number:03d}"
    version_dir = data_root / "datasets" / dataset_id / "versions" / version_id
    version_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_path = version_dir / "source.csv"
    jsonl_path = version_dir / "canonical.jsonl"
    raw_csv_path.write_bytes(csv_bytes)
    rows = validation["rows"]
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(canonical_record(row, dataset_type), sort_keys=True) + "\n")

    split_counts = Counter(row["split"] for row in rows)
    now = utc_now()
    with connect(database_path) as conn:
        conn.execute(
            """
            INSERT INTO datasets (dataset_id, slug, dataset_type, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (dataset_id, dataset_slug, dataset_type, title, now, now),
        )
        conn.execute(
            """
            INSERT INTO dataset_versions (
                version_id, dataset_id, version, approved, created_at, approved_at,
                raw_csv_path, jsonl_path, validation_json, row_count, split_counts_json, source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version_id,
                dataset_id,
                version_number,
                0,
                now,
                None,
                str(raw_csv_path),
                str(jsonl_path),
                json.dumps(validation, sort_keys=True),
                len(rows),
                json.dumps(dict(split_counts), sort_keys=True),
                source,
            ),
        )
    return {
        "created": True,
        "dataset_id": dataset_id,
        "version_id": version_id,
        "validation": validation,
        "jsonl_path": str(jsonl_path),
    }


def create_dataset_from_canonical_jsonl(
    database_path: Path,
    data_root: Path,
    source_jsonl_path: Path,
    dataset_type: str,
    title: str,
    slug: str,
    source: str,
    keep_source_copy: bool = True,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with source_jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(_error(line_number, "file", "malformed_jsonl", str(exc)))
                continue
            if "messages" not in record or "metadata" not in record:
                errors.append(_error(line_number, "record", "invalid_canonical_record", "Record requires messages and metadata."))
                continue
            rows.append(record)
    validation = {
        "valid": not errors,
        "errors": errors,
        "warnings": [],
        "rows": [],
        "accepted_count": 0 if errors else len(rows),
    }
    if errors:
        return {"created": False, "validation": validation}

    dataset_slug = slugify(slug or title, "generated-dataset")
    dataset_id = make_dataset_id(dataset_slug)
    version_number = 1
    version_id = f"{dataset_id}_v{version_number:03d}"
    version_dir = data_root / "datasets" / dataset_id / "versions" / version_id
    version_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = version_dir / "canonical.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    raw_csv_path = version_dir / "source.generated.jsonl"
    if keep_source_copy:
        raw_csv_path.write_text(source_jsonl_path.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        raw_csv_path = jsonl_path
    split_counts = Counter(row.get("metadata", {}).get("split", "holdout") for row in rows)
    now = utc_now()
    with connect(database_path) as conn:
        conn.execute(
            """
            INSERT INTO datasets (dataset_id, slug, dataset_type, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (dataset_id, dataset_slug, dataset_type, title, now, now),
        )
        conn.execute(
            """
            INSERT INTO dataset_versions (
                version_id, dataset_id, version, approved, created_at, approved_at,
                raw_csv_path, jsonl_path, validation_json, row_count, split_counts_json, source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version_id,
                dataset_id,
                version_number,
                0,
                now,
                None,
                str(raw_csv_path),
                str(jsonl_path),
                json.dumps(validation, sort_keys=True),
                len(rows),
                json.dumps(dict(split_counts), sort_keys=True),
                source,
            ),
        )
    return {
        "created": True,
        "dataset_id": dataset_id,
        "version_id": version_id,
        "validation": validation,
        "jsonl_path": str(jsonl_path),
    }


def create_dataset_from_canonical_records(
    database_path: Path,
    data_root: Path,
    records: list[dict[str, Any]],
    dataset_type: str,
    title: str,
    slug: str,
    source: str,
    keep_source_copy: bool = False,
) -> dict[str, Any]:
    staging_dir = data_root / "uploads" / "canonical-imports"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staging_path = staging_dir / f"{slugify(slug or title, 'dataset')}-{utc_now()}.jsonl"
    with staging_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    try:
        return create_dataset_from_canonical_jsonl(
            database_path,
            data_root,
            staging_path,
            dataset_type,
            title,
            slug,
            source,
            keep_source_copy=keep_source_copy,
        )
    finally:
        staging_path.unlink(missing_ok=True)


def approve_dataset_version(database_path: Path, dataset_id: str) -> dict[str, Any] | None:
    now = utc_now()
    with connect(database_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM dataset_versions
            WHERE dataset_id = ?
            ORDER BY version DESC
            LIMIT 1
            """,
            (dataset_id,),
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE dataset_versions SET approved = 1, approved_at = ?, reviewed_at = ?, review_sample_size = ? WHERE version_id = ?",
            (now, now, min(100, int(row["row_count"])), row["version_id"]),
        )
        updated = conn.execute("SELECT * FROM dataset_versions WHERE version_id = ?", (row["version_id"],)).fetchone()
        return row_to_dict(updated)


def delete_dataset_version(database_path: Path, data_root: Path, dataset_id: str) -> dict[str, Any] | None:
    now = utc_now()
    with connect(database_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM dataset_versions
            WHERE dataset_id = ?
            ORDER BY version DESC
            LIMIT 1
            """,
            (dataset_id,),
        ).fetchone()
        if row is None:
            return None
        version = row_to_dict(row) or {}
        version["approved"] = False
        version["approved_at"] = None
        version["reviewed_at"] = now
        version["review_sample_size"] = min(100, int(row["row_count"]))
        dataset_root = data_root / "datasets" / dataset_id
        candidate_paths = [dataset_root, Path(row["raw_csv_path"]), Path(row["jsonl_path"])]
        deleted_paths: list[str] = []
        for candidate in candidate_paths:
            deleted_paths.extend(path for path in safe_remove_traininghub_path(candidate, data_root) if path not in deleted_paths)
        conn.execute("DELETE FROM datasets WHERE dataset_id = ?", (dataset_id,))
        return {
            "version": version,
            "deleted_paths": deleted_paths,
            "removed_records": {"datasets": 1, "dataset_versions": 1},
        }


def reject_dataset_version(database_path: Path, dataset_id: str) -> dict[str, Any] | None:
    now = utc_now()
    with connect(database_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM dataset_versions
            WHERE dataset_id = ?
            ORDER BY version DESC
            LIMIT 1
            """,
            (dataset_id,),
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE dataset_versions SET approved = 0, approved_at = NULL, reviewed_at = ?, review_sample_size = ? WHERE version_id = ?",
            (now, min(100, int(row["row_count"])), row["version_id"]),
        )
        updated = conn.execute("SELECT * FROM dataset_versions WHERE version_id = ?", (row["version_id"],)).fetchone()
        return row_to_dict(updated)


def purge_rejected_datasets(database_path: Path, data_root: Path) -> dict[str, Any]:
    with connect(database_path) as conn:
        rows = conn.execute(
            """
            SELECT v.dataset_id
            FROM dataset_versions v
            WHERE v.approved = 0 AND v.reviewed_at IS NOT NULL
            """
        ).fetchall()
    deleted = []
    deleted_paths: list[str] = []
    for row in rows:
        result = delete_dataset_version(database_path, data_root, row["dataset_id"])
        if result:
            deleted.append(row["dataset_id"])
            deleted_paths.extend(path for path in result["deleted_paths"] if path not in deleted_paths)
    return {"deleted": deleted, "deleted_paths": deleted_paths}


def list_datasets(database_path: Path) -> list[dict[str, Any]]:
    with connect(database_path) as conn:
        rows = conn.execute(
            """
            SELECT
                d.dataset_id,
                d.slug,
                d.dataset_type,
                d.title,
                d.created_at,
                d.updated_at,
                v.version_id,
                v.version,
                v.approved,
                v.reviewed_at,
                v.review_sample_size,
                v.row_count,
                v.split_counts_json,
                v.validation_json,
                v.jsonl_path
            FROM datasets d
            LEFT JOIN dataset_versions v ON v.dataset_id = d.dataset_id
            WHERE v.version = (
                SELECT MAX(version) FROM dataset_versions latest WHERE latest.dataset_id = d.dataset_id
            )
            ORDER BY d.created_at DESC
            """
        ).fetchall()
    datasets = rows_to_dicts(rows)
    for dataset in datasets:
        dataset["approved"] = bool(dataset["approved"])
        dataset["split_counts"] = json.loads(dataset.pop("split_counts_json") or "{}")
        dataset["validation"] = json.loads(dataset.pop("validation_json") or "{}")
    return datasets


def get_approved_version(database_path: Path, dataset_id: str) -> dict[str, Any] | None:
    with connect(database_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM dataset_versions
            WHERE dataset_id = ? AND approved = 1
            ORDER BY version DESC
            LIMIT 1
            """,
            (dataset_id,),
        ).fetchone()
        return row_to_dict(row)


def get_latest_version(database_path: Path, dataset_id: str) -> dict[str, Any] | None:
    with connect(database_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM dataset_versions
            WHERE dataset_id = ?
            ORDER BY version DESC
            LIMIT 1
            """,
            (dataset_id,),
        ).fetchone()
        return row_to_dict(row)


def read_dataset_records(
    database_path: Path,
    dataset_id: str,
    offset: int = 0,
    limit: int = 50,
    split: str | None = None,
    query: str | None = None,
) -> dict[str, Any] | None:
    version = get_latest_version(database_path, dataset_id)
    if not version:
        return None
    records: list[dict[str, Any]] = []
    total_matching = 0
    query_normalized = (query or "").casefold().strip()
    with Path(version["jsonl_path"]).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            record = json.loads(line)
            metadata = record.get("metadata", {})
            if split and metadata.get("split") != split:
                continue
            if query_normalized and query_normalized not in json.dumps(record, sort_keys=True).casefold():
                continue
            if total_matching >= offset and len(records) < limit:
                records.append(_dataset_record_view(index, record))
            total_matching += 1
    return {
        "dataset_id": dataset_id,
        "version_id": version["version_id"],
        "approved": bool(version["approved"]),
        "row_count": int(version["row_count"]),
        "total_matching": total_matching,
        "offset": offset,
        "limit": limit,
        "records": records,
    }


def read_review_sample(database_path: Path, dataset_id: str, sample_size: int = 100) -> dict[str, Any] | None:
    result = read_dataset_records(database_path, dataset_id, offset=0, limit=sample_size)
    if result is None:
        return None
    result["sample_size"] = len(result["records"])
    result["required_review_sample_size"] = min(100, int(result["row_count"]))
    return result


def _dataset_record_view(index: int, record: dict[str, Any]) -> dict[str, Any]:
    if "messages" not in record:
        metadata = record.get("metadata", {})
        return {
            "index": index,
            "system": str(record.get("system_present") or ""),
            "prompt": str(record.get("prompt_present") or record.get("prompt") or ""),
            "response": str(record.get("prompt_absent") or record.get("system_absent") or ""),
            "metadata": metadata,
        }
    messages = record.get("messages", [])
    metadata = record.get("metadata", {})
    role_content = {message.get("role", ""): message.get("content", "") for message in messages if isinstance(message, dict)}
    return {
        "index": index,
        "system": role_content.get("system", ""),
        "prompt": role_content.get("user", ""),
        "response": role_content.get("assistant", ""),
        "metadata": metadata,
    }
