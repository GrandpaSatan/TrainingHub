# TrainingHub Implementation Plan

## Scope

TrainingHub owns the full local workflow for model acquisition, dataset acquisition, dataset validation, training, benchmarking, conversion, quantization, local inference selection, upload, and cleanup on Morrigan.

The application no longer treats any model family as a special external import path. Models trained for RAG purposes are regular TrainingHub outputs: train them from approved datasets, register the resulting artifacts, then route them through benchmark, GGUF conversion, quantization, or inference.

## Runtime Shape

- `backend`: FastAPI app, SQLite metadata, authentication, job queue, model registry, dataset registry, artifact registry, cleanup manifests, and system telemetry.
- `frontend`: React/Vite UI for dashboard, datasets, generation review, benchmarks, training, quantization, model registry, cleanup, and knowledge-base pages.
- `worker`: Python subprocess jobs for dataset import, example generation, benchmarking, model transfer, LoRA/QLoRA training, full fine-tuning, GGUF conversion, quantization, upload, and cleanup.
- `data root`: `/home/jhernandez/traininghub-data` on Morrigan unless overridden with `TRAININGHUB_DATA_ROOT`.
- `app root`: `/home/jhernandez/traininghub` on Morrigan unless overridden with `TRAININGHUB_APP_ROOT`.

## Data Flow

1. User downloads/registers a base model or selects an existing registry entry.
2. User uploads/imports a dataset source.
3. TrainingHub normalizes the source into canonical JSONL.
4. User reviews and approves the dataset version.
5. User queues a training job from the Training page.
6. Scheduler validates model capability plus dataset approval, creates the job, assigns GPUs, and launches a worker subprocess.
7. Worker writes append-only events, metrics, metadata, and artifacts.
8. Finished training artifacts are registered as adapters, merged checkpoints, full checkpoints, and reports.
9. User can benchmark, convert to GGUF, quantize, select for local inference where applicable, upload to Hugging Face, or clean up artifacts.

## Naming

### Job IDs

- Benchmarks: `bm_YYYYMMDD_HHMMSS_slug`
- Dataset imports: `di_YYYYMMDD_HHMMSS_slug`
- Model downloads: `md_YYYYMMDD_HHMMSS_slug`
- Model uploads: `mu_YYYYMMDD_HHMMSS_slug`
- Example generation: `gen_YYYYMMDD_HHMMSS_slug`
- Training: `tr_YYYYMMDD_HHMMSS_slug`
- Conversion: `cv_YYYYMMDD_HHMMSS_slug`
- Quantization: `qt_YYYYMMDD_HHMMSS_slug`
- Cleanup: `cl_YYYYMMDD_HHMMSS_slug`

### Training Job Types

- `train_lora`
- `train_qlora`
- `train_full`

### Training Artifact Types

- `training_adapter`
- `training_checkpoint`
- `training_merged_checkpoint`
- `training_report`

## Model Registry

Each model registry row records:

- `slug`
- `provider_id`
- `display_name`
- `family`
- `parameter_count`
- `supports_lora`
- `supports_qlora`
- `supports_full_finetune`
- `supports_bf16_inference`
- `supports_benchmark`
- `supports_quantization`
- `supports_gguf_path`
- `hardware_note`
- `default_dtype`
- `max_sequence_length`
- `metadata_json`

Unsupported operations must be blocked in both UI and backend. Full fine-tuning is disabled by default except for explicitly marked small or safe models.

## Dataset Registry

Datasets are versioned. A dataset version must be approved before it can be used for training or benchmark holdouts.

Canonical records are JSONL objects with chat-style `messages` and `metadata`. Training workers consume the approved canonical JSONL path attached by backend validation.

## Training Page

The Training page must provide:

- Base model selection.
- Approved dataset selection.
- Mode selection: LoRA, QLoRA, or Full.
- Preset selection: smoke, standard, or custom.
- Core settings: max steps, learning rate, batch size, gradient accumulation, LoRA rank, and adapter merge.
- Queue action.
- Recent training jobs.
- Registered training artifacts.

The UI must disable unsupported model/mode combinations with a clear reason.

## Backend APIs

Required API surfaces:

- `POST /api/jobs/fine-tune`
- `POST /api/jobs/benchmark`
- `POST /api/jobs/convert-gguf`
- `POST /api/jobs/quantize`
- `POST /api/models/download-hf`
- `POST /api/models/download-url`
- `POST /api/models/upload-hf`
- `POST /api/datasets/upload`
- `POST /api/datasets/import-hf`
- `POST /api/datasets/import-url`
- `POST /api/datasets/{dataset_id}/approve`
- `POST /api/cleanup/scan`
- `POST /api/cleanup/apply`

Training requests include:

- `model_slug`
- `dataset_id`
- `mode`
- `preset`
- optional `output_name`
- optional optimizer/training parameters
- optional `gpu_ids`
- optional `dry_run`

Backend validation must attach:

- `model_provider_id`
- `model_display_name`
- `model_family`
- `model_default_dtype`
- `dataset_version_id`
- `dataset_jsonl_path`
- resolved training defaults

## Workers

Workers must:

- Write structured job events.
- Capture stdout/stderr.
- Support cancellation.
- Register artifacts through the shared artifact table.
- Avoid deleting or overwriting unrelated paths.

Training workers:

- `run_train_lora`: handles LoRA and QLoRA adapter training.
- `run_train_full`: handles explicitly supported full fine-tuning.

When real workers are disabled, training workers produce deterministic smoke artifacts so the queue, logs, registry, and UI can be tested without loading large ML dependencies.

When real workers are enabled, training workers use the optional ML environment configured by `TRAININGHUB_WORKER_PYTHON` and require the `ml` dependency set: `torch`, `transformers`, `datasets`, `peft`, `trl`, `accelerate`, `bitsandbytes`, and `safetensors`.

## Acceptance Criteria

- User can download or register a model and see training capabilities in the registry.
- User can upload/import, review, and approve a dataset.
- User can queue a LoRA or QLoRA training job for supported models.
- User can queue full fine-tuning only for models explicitly marked supported.
- Training jobs appear in the shared job/log/event system.
- Completed training jobs register adapter/checkpoint/report artifacts.
- Merged or full checkpoints can be routed to GGUF conversion.
- Registered trained checkpoints can be benchmarked and selected for local inference where memory allows.
- Unsupported model/mode combinations are rejected in the UI and backend.
- Existing benchmark, quantization, inference, model upload/download, dataset, and cleanup flows continue to work.

## Verification

Run:

```bash
python3 -m pytest backend/tests -q
pnpm --dir frontend build
```

Optional Morrigan smoke:

```bash
TRAININGHUB_ADMIN_PASSWORD='change-me' scripts/morrigan_acceptance_smoke.sh
```
