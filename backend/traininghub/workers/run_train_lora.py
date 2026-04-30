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
    mode = str(payload["mode"])
    if mode not in {"lora", "qlora"}:
        raise RuntimeError("run_train_lora only supports lora and qlora modes.")
    output_name = slugify(str(payload["output_name"]), "trained-adapter")
    output_dir = context.job_dir / output_name
    adapter_dir = output_dir / "adapter"
    merged_dir = output_dir / "merged_checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)
    context.write_metadata("training_request.json", payload)

    if real_workers_enabled() and not payload.get("dry_run", False):
        training_summary = _train_real(context, payload, adapter_dir, merged_dir)
    else:
        _write_smoke_adapter(payload, adapter_dir)
        if payload.get("merge_adapter", False):
            _write_smoke_merged_checkpoint(payload, merged_dir)
        training_summary = synthetic_training_summary(payload)
    completion_message = training_summary_message(training_summary)
    context.set_completion_summary(completion_message, training_summary)

    artifact_ids = []
    adapter_artifact = context.register_artifact(adapter_dir, "training_adapter", f"{payload['output_name']} adapter", _artifact_metadata(payload, mode))
    artifact_ids.append(adapter_artifact["artifact_id"])
    if merged_dir.exists():
        merged_artifact = context.register_artifact(
            merged_dir,
            "training_merged_checkpoint",
            f"{payload['output_name']} merged checkpoint",
            _artifact_metadata(payload, mode),
        )
        artifact_ids.append(merged_artifact["artifact_id"])
        _register_model(context, payload, merged_dir, merged_artifact["artifact_id"])

    report = _training_report(context, payload, adapter_dir, merged_dir if merged_dir.exists() else None)
    report["training_summary"] = training_summary
    report_path = context.write_metadata("training_report.json", report)
    report_artifact = context.register_artifact(report_path, "training_report", f"{payload['output_name']} training report", report)
    artifact_ids.append(report_artifact["artifact_id"])
    context.event("training_complete", completion_message, data={"artifact_ids": artifact_ids, **training_summary})


def _train_real(context: WorkerContext, payload: dict[str, Any], adapter_dir: Path, merged_dir: Path) -> dict[str, Any]:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError(f"Training requires the optional TrainingHub ML stack: {exc}") from exc

    model_id = str(payload["model_provider_id"])
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if payload["mode"] == "qlora":
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("QLoRA training requires BitsAndBytesConfig from transformers.") from exc
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    if payload["mode"] == "qlora":
        model = prepare_model_for_kbit_training(model)
    _enable_memory_saving_training(model)

    lora_config = LoraConfig(
        r=int(payload["lora_rank"]),
        lora_alpha=int(payload["lora_alpha"]),
        lora_dropout=float(payload["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(payload.get("target_modules") or ["q_proj", "v_proj"]),
    )
    model = get_peft_model(model, lora_config)

    rows = _read_training_rows(Path(payload["dataset_jsonl_path"]), tokenizer, int(payload["max_sequence_length"]))
    dataset = Dataset.from_list(rows)
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
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[telemetry],
    )
    context.event("training_start", "Starting adapter training.", data={"row_count": len(rows), "mode": payload["mode"]})
    started_at = telemetry.started_at
    train_output = trainer.train()
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(adapter_dir))
    context.check_cancelled()

    if payload.get("merge_adapter", False):
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged = model.merge_and_unload()
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
    return normalize_training_summary(train_output, telemetry, started_at)


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
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _write_smoke_adapter(payload: dict[str, Any], adapter_dir: Path) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": payload["model_provider_id"],
                "peft_type": payload["mode"].upper(),
                "r": payload["lora_rank"],
                "lora_alpha": payload["lora_alpha"],
                "target_modules": payload.get("target_modules", []),
                "traininghub_smoke": True,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (adapter_dir / "README.md").write_text("TrainingHub smoke adapter placeholder.\n", encoding="utf-8")


def _write_smoke_merged_checkpoint(payload: dict[str, Any], merged_dir: Path) -> None:
    merged_dir.mkdir(parents=True, exist_ok=True)
    (merged_dir / "config.json").write_text(
        json.dumps({"base_model_name_or_path": payload["model_provider_id"], "traininghub_smoke": True}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (merged_dir / "README.md").write_text("TrainingHub smoke merged checkpoint placeholder.\n", encoding="utf-8")


def _artifact_metadata(payload: dict[str, Any], mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "model_slug": payload["model_slug"],
        "base_model_id": payload["model_provider_id"],
        "dataset_id": payload["dataset_id"],
        "dataset_version_id": payload["dataset_version_id"],
        "output_name": payload["output_name"],
    }


def _training_report(context: WorkerContext, payload: dict[str, Any], adapter_dir: Path, merged_dir: Path | None) -> dict[str, Any]:
    return {
        "job_id": context.job_id,
        "mode": payload["mode"],
        "preset": payload["preset"],
        "model_slug": payload["model_slug"],
        "model_provider_id": payload["model_provider_id"],
        "dataset_id": payload["dataset_id"],
        "dataset_version_id": payload["dataset_version_id"],
        "adapter_path": str(adapter_dir),
        "merged_checkpoint_path": str(merged_dir) if merged_dir else "",
        "dry_run": bool(payload.get("dry_run", False)),
    }


def _register_model(context: WorkerContext, payload: dict[str, Any], merged_dir: Path, artifact_id: str) -> None:
    model_slug = slugify(str(payload["output_name"]), "trained-model")
    register_trained_model(
        context.database_path,
        model_slug,
        f"local:{merged_dir}",
        f"{payload['output_name']} merged",
        str(payload["model_family"]),
        "trained",
        merged_dir,
        [artifact_id],
        context.job_id,
        int(payload["max_sequence_length"]),
        str(payload["model_default_dtype"]),
    )


if __name__ == "__main__":
    sys.exit(run_worker(main))
