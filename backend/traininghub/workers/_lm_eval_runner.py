from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from traininghub.core.database import connect
from traininghub.core.security import utc_now
from traininghub.services.benchmark_catalog import BenchmarkDefinition, require_benchmark_definitions
from traininghub.workers.common import WorkerContext, real_workers_enabled


PRIMARY_METRIC_NAMES = (
    "exact_match,flexible-extract",
    "exact_match",
    "acc,none",
    "acc_norm,none",
    "acc",
    "pass_at_1",
)


def run_benchmark_worker(context: WorkerContext, payload: dict[str, Any], artifact_label: str) -> None:
    benchmark_ids = payload.get("benchmarks") or ["gsm8k", "math-500"]
    definitions = require_benchmark_definitions(benchmark_ids)
    limit = int(payload.get("limit", 10))
    maj_k = int(payload.get("maj_k", 1))
    results_path = context.job_dir / "benchmark_results.json"
    metadata = _metadata(payload, definitions, limit, maj_k)
    context.write_metadata("benchmark_command.json", metadata)

    if real_workers_enabled() and not payload.get("dry_run", False):
        results = _run_lm_eval(context, payload, definitions, limit)
    else:
        results = _run_smoke_benchmark(context, definitions, limit, maj_k)

    results_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    context.register_artifact(results_path, "benchmark_results", artifact_label, metadata)
    _store_results(context, payload, results_path, results)


def _metadata(
    payload: dict[str, Any],
    definitions: list[BenchmarkDefinition],
    limit: int,
    maj_k: int,
) -> dict[str, Any]:
    return {
        "model_slug": payload["model_slug"],
        "model_id": payload.get("model_id") or payload.get("checkpoint_path"),
        "benchmarks": [definition.id for definition in definitions],
        "lm_eval_tasks": {definition.id: list(definition.lm_eval_tasks) for definition in definitions},
        "limit": limit,
        "maj_k": maj_k,
        "prompt_template": payload.get("prompt_template", "default_cot"),
        "generation": {
            "temperature": payload.get("temperature", 0.0),
            "top_p": payload.get("top_p", 1.0),
            "max_new_tokens": payload.get("max_new_tokens", 512),
            "seed": payload.get("seed", 7),
        },
    }


def _run_smoke_benchmark(
    context: WorkerContext,
    definitions: list[BenchmarkDefinition],
    limit: int,
    maj_k: int,
) -> dict[str, Any]:
    started = time.time()
    benchmark_results = {}
    for definition in definitions:
        context.check_cancelled()
        failures = []
        correct = 0
        for index in range(limit):
            passed = (index + len(definition.id)) % 5 != 0
            correct += int(passed)
            if not passed and len(failures) < 5:
                failures.append(
                    {
                        "sample_id": f"{definition.id}_{index}",
                        "prompt": f"Smoke {definition.label} sample {index}",
                        "expected": "accepted",
                        "prediction": "rejected",
                    }
                )
            context.metric(
                {
                    "benchmark": definition.id,
                    "completed": index + 1,
                    "limit": limit,
                    "score": round(correct / max(index + 1, 1), 4),
                }
            )
        pass_at_1 = correct / max(limit, 1)
        benchmark_results[definition.id] = {
            "pass_at_1": pass_at_1,
            "accuracy": pass_at_1,
            "maj_at_k": min(1.0, pass_at_1 + (0.03 if maj_k > 1 else 0.0)),
            "runtime_seconds": round(time.time() - started, 3),
            "tokens_per_second": None,
            "peak_gpu_memory_mb": None,
            "failure_examples": failures,
        }
    return {"benchmarks": benchmark_results, "summary": _summary(benchmark_results), "mode": "smoke"}


def _run_lm_eval(
    context: WorkerContext,
    payload: dict[str, Any],
    definitions: list[BenchmarkDefinition],
    limit: int,
) -> dict[str, Any]:
    tasks = [task for definition in definitions for task in definition.lm_eval_tasks]
    output_path = context.job_dir / "lm_eval_output.json"
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={payload.get('checkpoint_path') or payload['model_id']},trust_remote_code=True",
        "--tasks",
        ",".join(tasks),
        "--batch_size",
        str(payload.get("batch_size", 1)),
        "--limit",
        str(limit),
        "--output_path",
        str(output_path),
        "--log_samples",
    ]
    context.run_command(command)
    raw = json.loads(_read_lm_eval_output(output_path).read_text(encoding="utf-8"))
    raw_results = raw.get("results", {})
    results = {}
    for definition in definitions:
        task_metrics = {task: raw_results.get(task, {}) for task in definition.lm_eval_tasks}
        task_scores = [_first_metric(metrics, PRIMARY_METRIC_NAMES) for metrics in task_metrics.values()]
        scored = [score for score in task_scores if score is not None]
        pass_at_1 = statistics.mean(scored) if scored else None
        results[definition.id] = {
            "pass_at_1": pass_at_1,
            "accuracy": pass_at_1,
            "maj_at_k": None,
            "runtime_seconds": raw.get("runtime_seconds"),
            "tokens_per_second": None,
            "peak_gpu_memory_mb": None,
            "failure_examples": [],
            "raw_task_metrics": task_metrics,
        }
    return {"benchmarks": results, "summary": _summary(results), "mode": "lm_eval", "raw_path": str(output_path)}


def _read_lm_eval_output(output_path: Path) -> Path:
    if output_path.is_file():
        return output_path
    if output_path.is_dir():
        candidates = sorted(output_path.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]
    raise RuntimeError(f"LM Evaluation Harness did not write JSON output at {output_path}.")


def _first_metric(metrics: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = metrics.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and any(key == name or key.startswith(f"{name},") for name in names):
            return float(value)
    return None


def _summary(results: dict[str, Any]) -> dict[str, Any]:
    values = [metrics["pass_at_1"] for metrics in results.values() if metrics.get("pass_at_1") is not None]
    return {"mean_pass_at_1": statistics.mean(values) if values else None}


def _store_results(context: WorkerContext, payload: dict[str, Any], result_path: Path, results: dict[str, Any]) -> None:
    with connect(context.database_path) as conn:
        for benchmark_name, metrics in results.get("benchmarks", {}).items():
            result_id = f"{context.job_id}_{benchmark_name}"
            conn.execute(
                """
                INSERT OR REPLACE INTO benchmark_results (
                    result_id, job_id, model_slug, benchmark_name, metrics_json, result_path, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result_id,
                    context.job_id,
                    payload["model_slug"],
                    benchmark_name,
                    json.dumps(metrics, sort_keys=True),
                    str(result_path),
                    utc_now(),
                ),
            )
