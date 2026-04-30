from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from traininghub.core.config import Settings
from traininghub.core.security import password_hash, utc_now
from traininghub.services.model_registry import DEFAULT_MODELS


def connect(database_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(database_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(settings: Settings) -> None:
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(settings.database_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS model_registry (
                slug TEXT PRIMARY KEY,
                provider_id TEXT NOT NULL,
                display_name TEXT NOT NULL,
                family TEXT NOT NULL,
                parameter_count TEXT NOT NULL,
                supports_lora INTEGER NOT NULL DEFAULT 0,
                supports_qlora INTEGER NOT NULL DEFAULT 0,
                supports_full_finetune INTEGER NOT NULL DEFAULT 0,
                supports_bf16_inference INTEGER NOT NULL,
                supports_benchmark INTEGER NOT NULL,
                supports_quantization INTEGER NOT NULL,
                supports_gguf_path INTEGER NOT NULL,
                is_saga INTEGER NOT NULL DEFAULT 0,
                hardware_note TEXT NOT NULL,
                default_dtype TEXT NOT NULL,
                max_sequence_length INTEGER NOT NULL,
                metadata_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS model_delete_tombstones (
                slug TEXT PRIMARY KEY,
                provider_id TEXT NOT NULL,
                display_name TEXT NOT NULL,
                deleted_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT PRIMARY KEY,
                slug TEXT NOT NULL,
                dataset_type TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dataset_versions (
                version_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
                version INTEGER NOT NULL,
                approved INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                approved_at INTEGER,
                raw_csv_path TEXT NOT NULL,
                jsonl_path TEXT NOT NULL,
                validation_json TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                split_counts_json TEXT NOT NULL,
                source TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                slug TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                work_dir TEXT NOT NULL,
                worker_module TEXT NOT NULL,
                worker_pid INTEGER,
                gpu_ids TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                started_at INTEGER,
                finished_at INTEGER,
                terminal_message TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS job_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                created_at INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                data_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                job_id TEXT REFERENCES jobs(job_id) ON DELETE SET NULL,
                artifact_type TEXT NOT NULL,
                display_name TEXT NOT NULL,
                path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                checksum_sha256 TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS benchmark_results (
                result_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                model_slug TEXT NOT NULL,
                benchmark_name TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                result_path TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS capability_transfers (
                transfer_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                source_model_slug TEXT NOT NULL,
                source_runtime TEXT NOT NULL,
                target_model_slug TEXT NOT NULL,
                target_runtime TEXT NOT NULL,
                vector_artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE SET NULL,
                alignment_artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE SET NULL,
                alpha REAL NOT NULL DEFAULT 1.0,
                layer_targets_json TEXT NOT NULL,
                status TEXT NOT NULL,
                config_json TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                deleted_at INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_capability_transfers_status
            ON capability_transfers(status);

            CREATE TABLE IF NOT EXISTS cleanup_manifests (
                manifest_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                manifest_json TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                applied_at INTEGER
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            );

            """
        )
        _ensure_model_registry_columns(conn)
        _ensure_dataset_version_columns(conn)
        _ensure_capability_transfer_columns(conn)
        _seed_user(conn, settings)
        _seed_models(conn)
        _remove_retired_seed_models(conn)


def _seed_user(conn: sqlite3.Connection, settings: Settings) -> None:
    count = conn.execute("SELECT COUNT(*) FROM users WHERE username = ?", (settings.admin_username,)).fetchone()[0]
    if count and os.getenv("TRAININGHUB_ADMIN_PASSWORD"):
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?",
            (password_hash(settings.admin_password), settings.admin_username),
        )
        return
    if count:
        return
    conn.execute(
        "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
        (settings.admin_username, password_hash(settings.admin_password), utc_now()),
    )


def _seed_models(conn: sqlite3.Connection) -> None:
    tombstones = {
        row["slug"]
        for row in conn.execute("SELECT slug FROM model_delete_tombstones").fetchall()
    }
    for model in DEFAULT_MODELS:
        if model["slug"] in tombstones:
            continue
        conn.execute(
            """
            INSERT INTO model_registry (
                slug, provider_id, display_name, family, parameter_count,
                supports_lora, supports_qlora, supports_full_finetune,
                supports_bf16_inference, supports_benchmark, supports_quantization,
                supports_gguf_path, is_saga,
                hardware_note, default_dtype, max_sequence_length, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(slug) DO UPDATE SET
                provider_id = excluded.provider_id,
                display_name = excluded.display_name,
                family = excluded.family,
                parameter_count = excluded.parameter_count,
                supports_lora = excluded.supports_lora,
                supports_qlora = excluded.supports_qlora,
                supports_full_finetune = excluded.supports_full_finetune,
                supports_bf16_inference = excluded.supports_bf16_inference,
                supports_benchmark = excluded.supports_benchmark,
                supports_quantization = excluded.supports_quantization,
                supports_gguf_path = excluded.supports_gguf_path,
                is_saga = excluded.is_saga,
                hardware_note = excluded.hardware_note,
                default_dtype = excluded.default_dtype,
                max_sequence_length = excluded.max_sequence_length,
                metadata_json = excluded.metadata_json
            """,
            (
                model["slug"],
                model["provider_id"],
                model["display_name"],
                model["family"],
                model["parameter_count"],
                int(model.get("supports_lora", True)),
                int(model.get("supports_qlora", True)),
                int(model.get("supports_full_finetune", False)),
                int(model["supports_bf16_inference"]),
                int(model["supports_benchmark"]),
                int(model["supports_quantization"]),
                int(model["supports_gguf_path"]),
                0,
                model["hardware_note"],
                model["default_dtype"],
                int(model["max_sequence_length"]),
                json.dumps(model["metadata"], sort_keys=True),
            ),
        )


def _ensure_model_registry_columns(conn: sqlite3.Connection) -> None:
    rows = conn.execute("PRAGMA table_info(model_registry)").fetchall()
    existing = {row["name"] for row in rows}
    additions = {
        "supports_lora": "INTEGER NOT NULL DEFAULT 0",
        "supports_qlora": "INTEGER NOT NULL DEFAULT 0",
        "supports_full_finetune": "INTEGER NOT NULL DEFAULT 0",
        "supports_bf16_inference": "INTEGER NOT NULL DEFAULT 0",
        "supports_gguf_path": "INTEGER NOT NULL DEFAULT 1",
        "is_saga": "INTEGER NOT NULL DEFAULT 0",
    }
    for column, definition in additions.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE model_registry ADD COLUMN {column} {definition}")


def _remove_retired_seed_models(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT slug, metadata_json FROM model_registry WHERE slug = ?", ("lfm-saga-v4",)).fetchall()
    for row in rows:
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except json.JSONDecodeError:
            metadata = {}
        if metadata.get("route") == "saga_import":
            conn.execute("DELETE FROM model_registry WHERE slug = ?", (row["slug"],))


def _ensure_dataset_version_columns(conn: sqlite3.Connection) -> None:
    rows = conn.execute("PRAGMA table_info(dataset_versions)").fetchall()
    existing = {row["name"] for row in rows}
    additions = {
        "reviewed_at": "INTEGER",
        "review_sample_size": "INTEGER NOT NULL DEFAULT 0",
    }
    for column, definition in additions.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE dataset_versions ADD COLUMN {column} {definition}")


def _ensure_capability_transfer_columns(conn: sqlite3.Connection) -> None:
    rows = conn.execute("PRAGMA table_info(capability_transfers)").fetchall()
    if not rows:
        return
    existing = {row["name"] for row in rows}
    additions = {
        "deleted_at": "INTEGER",
    }
    for column, definition in additions.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE capability_transfers ADD COLUMN {column} {definition}")


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [row_to_dict(row) or {} for row in rows]
