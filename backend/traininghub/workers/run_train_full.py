from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from traininghub.core.id_utils import slugify
from traininghub.services.training import register_trained_model
from traininghub.workers.common import WorkerContext, real_workers_enabled, run_worker
from traininghub.workers.training_telemetry import (
    TrainingTelemetryCallback,
    normalize_training_summary,
    synthetic_training_summary,
    training_summary_message,
)


def main(context: WorkerContext, payload: dict[str, Any]) -> None:
    if payload["mode"] != "full":
        raise RuntimeError("run_train_full only supports full fine-tuning jobs.")
    output_name = slugify(str(payload["output_name"]), "trained-checkpoint")
    checkpoint_dir = context.job_dir / output_name / "checkpoint"
    context.write_metadata("training_request.json", payload)

    if real_workers_enabled() and not payload.get("dry_run", False):
        training_summary = _train_real(context, payload, checkpoint_dir)
    else:
        _write_smoke_checkpoint(payload, checkpoint_dir)
        training_summary = synthetic_training_summary(payload)
    completion_message = training_summary_message(training_summary)
    context.set_completion_summary(completion_message, training_summary)

    checkpoint_artifact = context.register_artifact(
        checkpoint_dir,
        "training_checkpoint",
        f"{payload['output_name']} checkpoint",
        _artifact_metadata(payload),
    )
    register_trained_model(
        context.database_path,
        output_name,
        f"local:{checkpoint_dir}",
        f"{payload['output_name']} checkpoint",
        str(payload["model_family"]),
        "trained",
        checkpoint_dir,
        [checkpoint_artifact["artifact_id"]],
        context.job_id,
        int(payload["max_sequence_length"]),
        str(payload["model_default_dtype"]),
    )
    report = {
        "job_id": context.job_id,
        "mode": payload["mode"],
        "preset": payload["preset"],
        "model_slug": payload["model_slug"],
        "model_provider_id": payload["model_provider_id"],
        "dataset_id": payload["dataset_id"],
        "dataset_version_id": payload["dataset_version_id"],
        "checkpoint_path": str(checkpoint_dir),
        "dry_run": bool(payload.get("dry_run", False)),
        "training_summary": training_summary,
    }
    report_path = context.write_metadata("training_report.json", report)
    report_artifact = context.register_artifact(report_path, "training_report", f"{payload['output_name']} training report", report)
    context.event(
        "training_complete",
        completion_message,
        data={"artifact_ids": [checkpoint_artifact["artifact_id"], report_artifact["artifact_id"]], **training_summary},
    )


def _train_real(context: WorkerContext, payload: dict[str, Any], checkpoint_dir: Path) -> dict[str, Any]:
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError(f"Full fine-tuning requires the optional TrainingHub ML stack: {exc}") from exc

    model_id = str(payload["model_provider_id"])
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device_map = _training_device_map(payload, torch)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        trust_remote_code=True,
    )
    _enable_memory_saving_training(model)
    rows = _read_training_rows(Path(payload["dataset_jsonl_path"]), tokenizer, int(payload["max_sequence_length"]))
    training_args = TrainingArguments(
        output_dir=str(context.job_dir / "trainer"),
        num_train_epochs=float(payload["epochs"]),
        max_steps=int(payload["max_steps"]) if int(payload["max_steps"]) > 0 else -1,
        per_device_train_batch_size=int(payload["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(payload["gradient_accumulation_steps"]),
        learning_rate=float(payload["learning_rate"]),
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    telemetry = TrainingTelemetryCallback(context)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(rows),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[telemetry],
    )
    context.event(
        "training_start",
        "Starting full fine-tuning.",
        data={
            "row_count": len(rows),
            "gpu_ids": payload.get("resolved_gpu_ids", []),
            "launch_mode": payload.get("training_launch_mode", "single_process"),
            "training_device_map": device_map,
        },
    )
    started_at = telemetry.started_at
    train_output = trainer.train()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(checkpoint_dir))
    context.check_cancelled()
    return normalize_training_summary(train_output, telemetry, started_at)


def _training_device_map(payload: dict[str, Any], torch: Any) -> str | None:
    if not torch.cuda.is_available():
        return None
    device_map = str(payload.get("training_device_map") or "").strip()
    return device_map or "auto"


def _enable_memory_saving_training(model: Any) -> None:
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = False
    if not hasattr(model, "gradient_checkpointing_enable"):
        return
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()


def _read_training_rows(path: Path, tokenizer: Any, max_sequence_length: int) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            text = _record_text(record, tokenizer)
            tokenized = tokenizer(text, truncation=True, max_length=max_sequence_length)
            rows.append(dict(tokenized))
    if not rows:
        raise RuntimeError("Approved dataset contains no training rows.")
    return rows


def _record_text(record: dict[str, Any], tokenizer: Any) -> str:
    messages = record.get("messages") or []
    if messages and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            pass
    return "\n".join(f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages)


def _write_smoke_checkpoint(payload: dict[str, Any], checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "config.json").write_text(
        json.dumps({"base_model_name_or_path": payload["model_provider_id"], "traininghub_smoke": True}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (checkpoint_dir / "README.md").write_text("TrainingHub smoke full checkpoint placeholder.\n", encoding="utf-8")


def _artifact_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": "full",
        "model_slug": payload["output_name"],
        "base_model_slug": payload["model_slug"],
        "base_model_id": payload["model_provider_id"],
        "dataset_id": payload["dataset_id"],
        "dataset_version_id": payload["dataset_version_id"],
        "output_name": payload["output_name"],
    }


if __name__ == "__main__":
    sys.exit(run_worker(main))
