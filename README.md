# TrainingHub

Standalone FastAPI and React web platform for Morrigan training workflows.

TrainingHub owns the local model lifecycle end to end: acquire models and datasets, approve canonical training data, queue LoRA/QLoRA/full fine-tuning jobs, then benchmark, convert, quantize, upload, or clean up the resulting artifacts.

## Local Run

```bash
scripts/run_dev.sh
```

Backend API runs on `http://127.0.0.1:7860`; Vite runs on `http://127.0.0.1:5173`.

Default local login:

- Username: `admin`
- Password: `traininghub`

Set `TRAININGHUB_ADMIN_PASSWORD` before first startup for a different password.

## Production Build

```bash
scripts/build_frontend.sh
TRAININGHUB_APP_ROOT="$PWD" TRAININGHUB_DATA_ROOT="$PWD/.traininghub-data" TRAININGHUB_INFERENCE_RUNTIME=transformers scripts/run_backend.sh
```

The built React app is served by FastAPI when `frontend/dist` exists.
Set `TRAININGHUB_INFERENCE_RUNTIME=transformers` when you want Chat to run base-model inference.

Capability Transfer adds an experimental UNLOCK-style inference-time steering workflow under the Transfer page. It extracts capability vectors, aligns them into a target model, and activates the ready transfer through the active inference target; it does not fine-tune or rewrite model weights. HF Transformers targets use per-layer hooks. GGUF targets use `llama-cpp-python` when available and fall back to `llama-cli`; set `TRAININGHUB_INSTALL_LLAMA_CPP=1` during Morrigan deploy to install the optional binding, and set `TRAININGHUB_LLAMA_RUNTIME=cli` to force the old CLI path. For CUDA llama-cpp-python builds, export the appropriate `CMAKE_ARGS` before installation, for example `CMAKE_ARGS="-DGGML_CUDA=on"`.

## Morrigan Deploy

```bash
TRAININGHUB_ADMIN_PASSWORD='change-me' scripts/deploy_morrigan.sh
```

To also perform the approved immediate `llama-server` cleanup on Morrigan:

```bash
TRAININGHUB_ADMIN_PASSWORD='change-me' scripts/deploy_morrigan.sh --cleanup-llama
```

The cleanup script records manifests under `/home/jhernandez/traininghub-data/cleanup`.

## Verification

```bash
python3 -m pytest backend/tests -q
pnpm --dir frontend build
```

Optional Morrigan smoke:

```bash
TRAININGHUB_ADMIN_PASSWORD='change-me' scripts/morrigan_acceptance_smoke.sh
```
