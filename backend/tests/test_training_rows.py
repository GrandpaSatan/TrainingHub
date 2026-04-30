from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from traininghub.workers import run_train_full, run_train_lora


class FakeTokenizer:
    def __call__(self, text: str, truncation: bool, max_length: int) -> dict[str, Any]:
        assert truncation is True
        input_ids = list(range(min(len(text.split()), max_length)))
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}


def test_full_training_rows_leave_labels_to_collator(tmp_path: Path) -> None:
    path = _write_rows(tmp_path)

    rows = run_train_full._read_training_rows(path, FakeTokenizer(), 8)

    assert rows
    assert all("labels" not in row for row in rows)
    assert all(len(row["input_ids"]) <= 8 for row in rows)


def test_lora_training_rows_leave_labels_to_collator(tmp_path: Path) -> None:
    path = _write_rows(tmp_path)

    rows = run_train_lora._read_training_rows(path, FakeTokenizer(), 8)

    assert rows
    assert all("labels" not in row for row in rows)
    assert all(len(row["input_ids"]) <= 8 for row in rows)


def _write_rows(tmp_path: Path) -> Path:
    path = tmp_path / "canonical.jsonl"
    records = [
        {"messages": [{"role": "user", "content": "short"}, {"role": "assistant", "content": "answer"}]},
        {
            "messages": [
                {"role": "user", "content": "this is a longer prompt with several words"},
                {"role": "assistant", "content": "this answer is also longer"},
            ]
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return path
