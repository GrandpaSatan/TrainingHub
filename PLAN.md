# TrainingHub Multi-Family Training, Hugging Face Downloads, and Native Dataset Import Plan

## Summary

TrainingHub should own the end-to-end local workflow for acquiring models, acquiring datasets, validating training data, and queueing LoRA, QLoRA, or full fine-tuning jobs across multiple model families.

This replaces the old “training is unsupported” and external-import handoff behavior with a real TrainingHub-owned queue while preserving the existing dashboard, inference selector, benchmarks, quantization, cleanup, and job infrastructure.

## Key Changes

- Add a first-class **Training** page with model selection, dataset selection, training mode tabs for `LoRA`, `QLoRA`, and `Full`, preset selection, advanced settings, queue controls, and live job status.
- Re-enable `/api/jobs/fine-tune` as a real queueing endpoint instead of returning HTTP 410.
- Support multiple model families through registry capabilities:
  - `supports_lora`
  - `supports_qlora`
  - `supports_full_finetune`
  - training notes / safe defaults
- Add Hugging Face acquisition flows:
  - model download job using `huggingface_hub.snapshot_download`
  - dataset download job using `datasets.load_dataset`
  - local registry entries for downloaded model and dataset sources
- Add native dataset upload/import:
  - keep CSV support
  - add JSONL, JSON, Parquet, Arrow, and TXT source uploads
  - provide mapping/preview/validation before approval
  - convert approved sources into TrainingHub’s canonical JSONL training format
- Add real workers:
  - model download worker
  - dataset download worker
  - dataset import/validation worker
  - LoRA/QLoRA training worker
  - full fine-tuning worker
- Use the existing job/event/log/cancel infrastructure for all new long-running work.

## Implementation Details

- Backend model registry:
  - expose training capabilities in model API responses
  - seed conservative defaults per family
  - allow downloaded Hugging Face models to be registered with editable capabilities
  - disable full fine-tuning by default except for explicitly supported small/base models

- Backend APIs:
  - `POST /api/jobs/model-download`
  - `GET /api/model-downloads`
  - `POST /api/jobs/dataset-download`
  - `GET /api/dataset-sources`
  - `POST /api/datasets/upload-native`
  - `POST /api/jobs/fine-tune`
  - optional helper endpoint for training presets if not embedded in model metadata

- Training request shape should include:
  - model registry slug or downloaded model id
  - approved dataset id/version
  - mode: `lora`, `qlora`, or `full`
  - preset: `smoke`, `standard`, or `custom`
  - optional advanced settings such as epochs, max steps, learning rate, batch size, gradient accumulation, LoRA rank, target modules, output name, and GPU selection

- Dataset validation:
  - require an approved canonical dataset before queueing training
  - accepted conversational shape: `messages`
  - accepted supervised shape: prompt/instruction plus response/output
  - Parquet/Arrow imports should use `datasets`/`pyarrow`, not ad hoc parsing
  - TXT uploads are allowed as sources but must be mapped into valid prompt/response rows before training

- Worker environment:
  - keep the API/backend environment lightweight
  - provision a separate TrainingHub ML worker venv for heavy dependencies
  - include `torch`, `transformers`, `datasets`, `peft`, `trl`, `accelerate`, `bitsandbytes`, `huggingface_hub`, `pyarrow`, and `safetensors`
  - configure workers with `TRAININGHUB_WORKER_PYTHON`
  - require `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` for private Hugging Face repos

- Frontend:
  - add `Training` to primary navigation
  - extend `Models` with Hugging Face model download/import UI
  - extend `Datasets` with Hugging Face dataset download and native upload/mapping UI
  - surface unsupported modes clearly by disabling them with model-specific notes
  - show queued/running/completed training jobs on Dashboard and Training pages
  - keep the UI dark-themed consistently

## Acceptance Criteria

- User can download a Hugging Face model from the UI and see it registered locally.
- User can download a Hugging Face dataset from the UI and convert/approve it for training.
- User can upload JSONL, JSON, Parquet, Arrow, TXT, or CSV dataset sources.
- User can map native dataset fields, preview rows, validate the dataset, and approve it.
- User can queue LoRA or QLoRA training for supported model families.
- User can queue full fine-tuning only for models explicitly marked safe/supported.
- Unsupported model/mode combinations are blocked in both UI and backend.
- Training jobs appear in the same job/status/log system as benchmarks and quantization.
- Canceling a training/download/import job terminates the worker process cleanly.
- Existing benchmark, quantization, inference-target, and cleanup flows continue to work.

## Test Plan

- Backend tests:
  - training endpoint accepts valid supported model + approved dataset
  - training endpoint rejects unsupported mode/model combinations
  - training endpoint rejects unapproved datasets
  - model download job validates Hugging Face repo/revision input
  - dataset download job records dataset source metadata
  - native JSONL/JSON/Parquet/TXT imports produce validation reports
  - cancellation updates job state correctly

- Frontend tests/build:
  - Training page renders model, dataset, mode, preset, and queue controls
  - unsupported training modes are disabled
  - Hugging Face model download form submits valid jobs
  - Hugging Face dataset download form submits valid jobs
  - native upload/mapping flow renders and validates
  - `pnpm --dir frontend build` succeeds

- Smoke verification:
  - upload a tiny JSONL dataset
  - approve it
  - queue a LoRA smoke run with `max_steps=1`
  - confirm job logs, final status, and output artifact registration

## Assumptions

- TrainingHub owns local training orchestration end to end.
- Existing “training unsupported” code should be removed, not hidden behind labels.
- V1 training support should prioritize PEFT LoRA/QLoRA and conservative full fine-tuning.
- Hugging Face credentials come from environment variables, not from storing tokens in the UI.
- Downloaded models and datasets should be versioned locally instead of overwriting existing directories by default.
