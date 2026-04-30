from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CalibrationDatasetError(ValueError):
    pass


def load_calibration_pairs(path: Path, contrast_mode: str) -> list[dict[str, str]]:
    if contrast_mode not in {"prompt_pair", "system_pair"}:
        raise CalibrationDatasetError("contrast_mode must be prompt_pair or system_pair.")
    pairs: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CalibrationDatasetError(f"Calibration row {line_number} is malformed JSON: {exc.msg}.") from exc
            pairs.append(_calibration_pair(row, contrast_mode, line_number))
    if not pairs:
        raise CalibrationDatasetError("Calibration dataset has no usable capability contrast pairs.")
    return pairs


def validate_calibration_dataset(path: Path, contrast_mode: str) -> dict[str, Any]:
    pairs = load_calibration_pairs(path, contrast_mode)
    return {"valid": True, "pair_count": len(pairs), "contrast_mode": contrast_mode}


def _calibration_pair(row: dict[str, Any], contrast_mode: str, line_number: int) -> dict[str, str]:
    prefix = str(row.get("continuation_prefix") or row.get("prefix") or "")
    if contrast_mode == "system_pair":
        present = str(row.get("system_present") or "").strip()
        absent = str(row.get("system_absent") or "").strip()
        prompt = str(row.get("prompt") or prefix).strip()
        if present and absent and prompt:
            return {"present": f"{present}\n\n{prompt}", "absent": f"{absent}\n\n{prompt}"}
        raise CalibrationDatasetError(
            f"Calibration row {line_number} requires system_present, system_absent, and prompt for system_pair mode."
        )

    present = str(row.get("prompt_present") or "").strip()
    absent = str(row.get("prompt_absent") or "").strip()
    if present and absent:
        return {"present": present + prefix, "absent": absent + prefix}
    raise CalibrationDatasetError(
        f"Calibration row {line_number} requires prompt_present and prompt_absent for prompt_pair mode."
    )
