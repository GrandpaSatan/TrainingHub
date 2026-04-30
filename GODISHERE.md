# UNLOCK Capability Transfer — TrainingHub Feature Plan

## Context

You want to make a smaller model behave like a larger one without retraining or model-file shrinking. The arxiv paper at 2604.06377 — *The Master Key Hypothesis: Unlocking Cross-Model Capability Transfer via Linear Subspace Alignment* — proposes the **UNLOCK** framework, which is **inference-time activation steering**, not compression. UNLOCK has three offline+inference stages:

1. **Capability Extraction** — run paired prompts (capability-present vs capability-absent) through a *source* (larger) model, contrast residual-stream activations layer-by-layer, save a per-layer **capability direction vector**.
2. **Linear Alignment** — run a small calibration set through both the *source* and the *target* (smaller) model, capture paired activations, fit a low-rank linear map per layer (closed form via SVD / least-squares — **no gradient training**) that projects source-space directions into target-space.
3. **Inference Injection** — at generation time on the target model, register forward hooks that add `alpha * aligned_direction[layer]` into the residual stream at chosen layers.

Net effect (confirmed with you in plan-mode questions): a small target model gains the source's capability (e.g. CoT reasoning) without changing its parameter count or file size. This **fits TrainingHub's scope** — no SGD, no fine-tuning. Project memory says training lives in NeuralNest `nn-train`; this feature does not train, so it stays here.

You picked **HF transformers + llama-cpp-python** for runtime support. Honest constraint up front: per-layer steering needs Python-level forward hooks. HF gives that natively. llama.cpp doesn't expose per-layer tensor callbacks from Python — the realistic GGUF path is **last-layer-only steering** via `llama_cpp.Llama(logits_processor=...)`. The plan is explicit about this.

---

## Scope summary

| Capability | HF transformers (`base_model`) | GGUF (`gguf_artifact`, via `llama-cpp-python`) |
|---|---|---|
| Per-layer activation extraction | Yes — forward hooks on each transformer block | **No** — last-hidden-state only (`embedding=True`) |
| Per-layer aligned injection | Yes — pre-forward hook adds aligned direction to residual | **No** — final-logit `logits_processor` only |
| Streaming chat with steering on | Yes — `TextIteratorStreamer` + hooks | Yes — token callback + logits processor |

GGUF targets are supported but flagged in the UI as **"Last-layer mode — degraded fidelity vs paper."** Users with the linked HF source model already on disk get a one-click "Use HF source for full per-layer steering" toggle.

---

## Backend design

### Database (`backend/traininghub/core/database.py`)

Add two tables (idempotent migration in the `init_database()` path):

```sql`
CREATE TABLE IF NOT EXISTS capability_transfers (
  transfer_id           TEXT PRIMARY KEY,
  display_name          TEXT NOT NULL,
  source_model_slug     TEXT NOT NULL,      -- model_registry.slug OR artifact_id of GGUF source
  source_runtime        TEXT NOT NULL,      -- 'transformers' | 'llama_cpp'
  target_model_slug     TEXT NOT NULL,
  target_runtime        TEXT NOT NULL,
  vector_artifact_id    TEXT,               -- artifacts.artifact_id of capability_vector
  alignment_artifact_id TEXT,               -- artifacts.artifact_id of alignment_map
  alpha                 REAL NOT NULL DEFAULT 1.0,
  layer_targets_json    TEXT NOT NULL,      -- e.g. [12, 16, 20] or "all" or "last"
  status                TEXT NOT NULL,      -- 'extracting' | 'aligning' | 'ready' | 'failed'
  config_json           TEXT NOT NULL,      -- {rank, calibration_dataset_id, contrast_mode, ...}
  created_at            TEXT NOT NULL,
  updated_at            TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_capability_transfers_status ON capability_transfers(status);
```

No separate `capability_vectors` table — per-layer tensors live inside the `capability_vector` artifact `.npz` (`{layer_0: ndarray, layer_1: ndarray, ...}`); ditto for `alignment_map`. The artifacts table is the source of truth.

### New artifact types (registered via `WorkerContext.register_artifact`)

- `capability_vector` — `.npz` file. Metadata: `{source_model_slug, layer_count, hidden_size, contrast_mode, calibration_size}`.
- `alignment_map` — `.npz` file. Metadata: `{source_model_slug, target_model_slug, rank, layer_pairs_json, calibration_size}`.

Add both strings to the artifact-type enum used by `services/artifacts.py` and the cleanup matcher.

### New job types & worker modules (`backend/traininghub/services/jobs.py`)

Extend `JOB_PREFIXES` and `WORKER_MODULES`:

```python
JOB_PREFIXES["extract_capability"] = "ec"
JOB_PREFIXES["align_capability"]   = "ac"

WORKER_MODULES["extract_capability"] = "traininghub.workers.run_extract_capability"
WORKER_MODULES["align_capability"]   = "traininghub.workers.run_align_capability"
```

Validation in `_validate_job_request`: confirm referenced models / GGUF artifacts exist, and that `calibration_dataset_id` resolves to an approved dataset (reuse `services.datasets.get_approved_version`, same as benchmark validation at `services/jobs.py:129-134`).

### New worker: `workers/run_extract_capability.py`

Follows the `WorkerContext` pattern from `workers/common.py:171-188`. Inputs (from `payload.json`):

```json
{
  "transfer_id": "...",
  "source_model_slug": "...",
  "source_runtime": "transformers" | "llama_cpp",
  "calibration_dataset_id": "...",
  "contrast_mode": "prompt_pair" | "system_pair",
  "layer_targets": "all" | "last" | [12,16,20]
}
```

Algorithm (HF path):
1. Load source model with the same incantation as `services/inference_run.py:198-204` (`AutoModelForCausalLM`, `bfloat16`, `device_map="auto"`).
2. Read calibration dataset (JSONL) — each row has `prompt_present`, `prompt_absent` (or `system_present`/`system_absent` for `system_pair` mode) plus a shared continuation prefix.
3. Register pre-forward hooks on each `model.model.layers[i]` (architecture-aware lookup helper in a new `services/model_introspection.py` — covers Llama, Qwen, Mistral, LFM2 families).
4. For each pair: forward both prompts, capture residual at chosen layers at the last prompt token. Accumulate `direction_i = mean(present_i - absent_i)` across the calibration set.
5. L2-normalize per layer, save as `.npz` artifact, register as `capability_vector`, update `capability_transfers.vector_artifact_id` and emit `progress` events every N pairs.

GGUF path (degraded): use `llama_cpp.Llama(model_path, embedding=True)` to get final hidden state per prompt; only `layer_targets="last"` is valid. Surface a `worker_warning` event.

`context.event()` writes both `progress` (with `{step, total}`) and `metric` events so the dashboard's existing live-metric pane (`App.tsx:932`) tails it.

### New worker: `workers/run_align_capability.py`

Inputs:

```json
{
  "transfer_id": "...",
  "source_model_slug": "...",
  "target_model_slug": "...",
  "source_runtime": "...",
  "target_runtime": "...",
  "vector_artifact_id": "...",
  "calibration_dataset_id": "...",
  "rank": 16,
  "layer_pairs": [[12,8], [16,12], [20,16]]   // source_layer -> target_layer
}
```

Algorithm:
1. Load source + target. If both are HF, use forward hooks on both. If target is GGUF, refuse with a `worker_error` (alignment per-layer needs hooks on the target).
2. For each calibration row, run the *same* prompt through both models, capture residuals at the configured layer pairs at last-token position. Build paired matrices `X_src` (N × d_src) and `X_tgt` (N × d_tgt) per pair.
3. Fit low-rank linear map per pair: `W ≈ argmin ||X_tgt - X_src · W||` with rank-r constraint via truncated SVD on `X_src^T X_tgt`. Closed form, ~seconds. **No SGD.** Use `numpy.linalg.svd`.
4. Save `{pair_0: W_0, pair_1: W_1, ...}` as `.npz`, register as `alignment_map`, update `capability_transfers.alignment_artifact_id`, set status `ready`.

If source is HF and target is HF and architectures match (same hidden size, same layer count) the rank-r alignment is approximately identity for matched layers — the worker still runs it and reports the residual reconstruction error as a metric.

### Inference path (`backend/traininghub/services/inference_run.py`)

Two surgical changes:

**1. Replace `llama-cli` subprocess with `llama-cpp-python`.** New helper `_run_llama_cpp_prompt(target, prompt, sampling, hook_config)` that:
- Maintains a `_LLAMA_CACHE` (parallel to `_TRANSFORMERS_CACHE` at line 35) of `Llama(model_path, n_gpu_layers=int(os.getenv("LLAMA_CPP_N_GPU_LAYERS","999")), embedding=False)` instances keyed by `artifact_id`.
- Streams via the `Llama.create_completion(stream=True)` token generator.
- When `hook_config` is present and `target_runtime == "llama_cpp"`: pass a `logits_processor` that adds `alpha * projected_direction` to the final logits (last-layer-only mode).
- Keep `_run_llama_cli_prompt` as a fallback if the import fails, gated by env var `TRAININGHUB_LLAMA_RUNTIME=cli` for users who want the old path.

**2. Add steering hooks to `_run_transformers_prompt` (lines 70-122).** Resolve the active transfer (if any) before `model.generate`:

```python
transfer = _resolve_active_transfer(target)  # reads active_inference_target.capability_transfer_id
if transfer:
    hook_handles = _attach_steering_hooks(model, transfer)  # pre-forward hooks per chosen target layer
try:
    model.generate(**generation_kwargs)
finally:
    for h in hook_handles: h.remove()
```

`_attach_steering_hooks` lazy-loads the alignment `.npz`, projects each per-layer source direction into the target hidden space (`v_target = v_source @ W`), and the hook adds `alpha * v_target` to the residual stream at every forward pass. Direction tensors are cached in `_TRANSFER_CACHE` keyed by `transfer_id`.

### Active inference target schema (`backend/traininghub/services/inference.py`)

Extend the `active_inference_target` value to optionally include `capability_transfer_id`. `validate_inference_target` (line 63) accepts it, `_base_model_target` and the GGUF branch carry it through. Frontend reads/writes it via the existing `/api/inference/target` endpoint.

### New API module: `backend/traininghub/api/capability_transfers.py`

Mounted on `/api/capability-transfers` from `main.py`:

- `GET /` — list transfers (joins to artifacts for vector + alignment paths, returns status).
- `POST /` — create a transfer record (status `extracting`), kick off `extract_capability` job. Body: `{display_name, source_*, target_*, calibration_dataset_id, layer_targets, contrast_mode, rank}`.
- `POST /{id}/align` — kick off `align_capability` job (only valid when extraction is complete).
- `POST /{id}/activate` — sets `active_inference_target.capability_transfer_id = id`. Validates target compatibility (HF: full; GGUF: last-layer only — emit a warning in the response).
- `POST /{id}/deactivate` — clears `capability_transfer_id`.
- `DELETE /{id}` — soft-delete, removes vector + alignment artifacts via existing `services/deletion.py`.
- `GET /{id}` — single transfer detail (used by the wizard's Observe stage).

All routes use the same session-cookie auth dependency as `api/jobs.py`.

### Dependency additions (`backend/pyproject.toml`)

- `llama-cpp-python` (new, with `[server]` extras unnecessary; we only need the `Llama` class). Build with `CMAKE_ARGS="-DGGML_CUDA=on"` documented in README.
- `numpy` (already implicit via transformers, pin explicitly if not).

`transformers`, `torch`, `accelerate` are already implied per `services/inference_run.py:186-189`.

---

## Frontend design

### New top-level page

`frontend/src/App.tsx`:
- Add `"capability-transfer"` to the `Page` union (line 50).
- Add `{ page: "capability-transfer", label: "Transfer", icon: <Wand2 size={17} /> }` to `navItems` (line 110), placed **between Training and Quantize** (it's the same conceptual layer: post-training model surgery).
- Add a `pageDescriptions` entry (line 129).
- Wire the route at lines 360-402: `{page === "capability-transfer" && <CapabilityTransferWizard {...wizardProps} />}`.

### New component: `frontend/src/components/CapabilityTransferWizard.tsx`

Mirrors `FineTuneWizard.tsx`'s 6-stage cyberpunk pattern (lines 48-54 there). Stages:

1. **Source** — `.thx-cards` grid of source candidates: HF base models (filtered to `supports_bf16_inference`) + GGUF artifacts (with the "last-layer mode" pill). Selection populates `sourceModelSlug` / `sourceRuntime`.
2. **Target** — same grid, scoped to models smaller than source by `model_registry.parameter_count` so the picker reflects the paper's intent. Cards show capability badges via `.thx-cap`.
3. **Calibration set** — `ArtifactPicker` (new reusable, see below) over approved datasets with format hint `"jsonl with prompt_present + prompt_absent fields"`. Inline help via `<FieldNote link="#calibration-pairs">`.
4. **Extract** — `.thx-params` form with: contrast mode (`.thx-seg` for `prompt_pair` / `system_pair`), layer targets (`.thx-seg` for `all` / `every-4` / `last` / custom), launch button. On submit: `POST /api/capability-transfers` then `POST /api/capability-transfers/{id}/extract`. Live progress via `JobLogPanel`.
5. **Align** — appears once extraction finishes. `.thx-params` for `rank` (slider 4-64, default 16) and `layer_pairs` (auto-suggested as proportional mapping `target_idx = round(source_idx * target_layers / source_layers)`, editable). Launch → `POST /{id}/align`. Live `JobLogPanel`.
6. **Deploy** — Activate toggle (`.thx-btn--primary` "Activate transfer on inference"), `alpha` slider (0–4, default 1.0), per-layer mute checkboxes. Calls `POST /{id}/activate`. Shows the active-transfer pill that will appear in chat / models pages.

Job submission + SSE tail uses the same pattern as `FineTuneWizard.tsx:409-431` and `JobLogPanel.tsx:27-37` — no new SSE plumbing.

### New reusable: `frontend/src/components/ArtifactPicker.tsx`

`.thx-cards`-based picker filtered by artifact type. Used by the wizard's calibration step now, and by Quantize / Benchmarks / Generate later (already promised in `fix_plan.md` Phase 5). Props: `{ artifactTypes: string[], value, onChange, emptyHint }`. Pulls from `/api/artifacts`.

### Active-transfer pill

Reusable component `ActiveTransferPill.tsx` rendered on:
- The Models page's Active-Inference-Bar (`fix_plan.md` Phase 2 already plans this row — add the pill next to the inference target).
- The Chat page topbar (Phase 3 of `fix_plan.md`).

Shows `<source> → <target> · α=<alpha> · L:<layers>` with an "Off" button. Color-codes `.thx-cap--ok` when active and `.thx-cap--w` for last-layer (degraded) mode.

### KnowledgeBase entries (`App.tsx:4293-4364`)

Add five entries (used by `<FieldNote link="#…" />` in the wizard):

- `id="capability-vectors"` — what a per-layer direction is, why we contrast.
- `id="calibration-pairs"` — JSONL schema, sizing guidance (≥256 pairs, ≥1024 for stable alignment).
- `id="linear-alignment"` — the rank-r SVD step in plain language; what `rank` controls (lower = smoother but lossier).
- `id="alpha-tuning"` — `alpha` is the strength knob; 0.5–1.5 is the safe band; >2 destabilizes.
- `id="gguf-degraded-mode"` — explains the last-layer-only constraint and how to escape it (use HF source).

### API client additions (`frontend/src/api/client.ts`)

Typed wrappers: `capabilityTransfers.list()`, `.create(body)`, `.align(id)`, `.activate(id, {alpha, layer_targets})`, `.deactivate(id)`, `.delete(id)`.

---

## Files modified / added

**Backend — new files**
- `backend/traininghub/api/capability_transfers.py`
- `backend/traininghub/workers/run_extract_capability.py`
- `backend/traininghub/workers/run_align_capability.py`
- `backend/traininghub/services/model_introspection.py` (architecture-aware layer list lookup, shared by both workers and the inference hook attach)
- `backend/traininghub/services/capability_transfers.py` (CRUD + status transitions; mirrors how `services/inference.py` wraps DB)

**Backend — modified**
- `backend/traininghub/core/database.py` — add `capability_transfers` table to `init_database()` (around line 144).
- `backend/traininghub/services/jobs.py` — register two new entries in `JOB_PREFIXES` (line 20) and `WORKER_MODULES` (line 34); extend `_validate_job_request` (line 123).
- `backend/traininghub/services/inference_run.py` — replace `_run_llama_cli_prompt` with `_run_llama_cpp_prompt` (CLI kept as env-gated fallback); add steering-hook plumbing to `_run_transformers_prompt` (lines 70-122) and to the new `_run_llama_cpp_prompt` via `logits_processor`.
- `backend/traininghub/services/inference.py` — accept optional `capability_transfer_id` in `validate_inference_target` (line 63) and carry it through.
- `backend/traininghub/main.py` — `app.include_router(capability_transfers.router)`.
- `backend/pyproject.toml` — add `llama-cpp-python`, document CUDA build flag.

**Frontend — new files**
- `frontend/src/components/CapabilityTransferWizard.tsx`
- `frontend/src/components/ArtifactPicker.tsx`
- `frontend/src/components/ActiveTransferPill.tsx`

**Frontend — modified**
- `frontend/src/App.tsx` — add page to union (line 50), nav item (line 110), pageDescriptions (line 129), route render (lines 360-402); add 5 KnowledgeBase entries (lines 4293-4364); render `<ActiveTransferPill />` in the topbar dashboard area near the inference selector.
- `frontend/src/styles/cyberpunk.css` — add a `.thx-xfer-*` module: source→target arrow visualizer, layer-pair grid (small bipartite diagram), alpha slider styling. Reuse `.thx-cards`, `.thx-params`, `.thx-seg`, `.thx-btn--primary` unchanged.
- `frontend/src/api/client.ts` — `capabilityTransfers` typed namespace.

**Reused without changes**
- `frontend/src/components/JobLogPanel.tsx` (live SSE tail in stages 4 + 5).
- `backend/traininghub/workers/common.py` (`run_worker`, `WorkerContext`, `register_artifact`).
- `backend/traininghub/services/datasets.py` (`get_approved_version` for calibration-set validation).
- `backend/traininghub/api/jobs.py` (`/api/jobs/{id}/events` SSE — used by the wizard verbatim).

---

## Verification

End-to-end smoke (run against the dev backend with `TRAININGHUB_INFERENCE_RUNTIME=transformers` and `LLAMA_CPP_N_GPU_LAYERS=999`):

1. **Build hygiene** — `npx tsc --noEmit && npx vite build` clean; `pytest backend/tests` green; `ruff check backend/` clean.
2. **DB migration** — restart backend on an existing dev DB; `capability_transfers` table created without dropping data.
3. **Extract (HF → HF)** — pick a small open model as source (e.g. `Qwen2-1.5B-Instruct`) and a *smaller* model as target (e.g. `Qwen2-0.5B-Instruct`) so it fits on one GPU. Use a 256-pair toy CoT calibration set (provided in `backend/tests/fixtures/calibration_cot.jsonl`). Confirm: progress events stream, `.npz` artifact registered, `capability_transfers.status='extracted'`.
4. **Align (HF → HF)** — same pair. Confirm `alignment_map` artifact written, status `ready`, residual reconstruction error metric emitted.
5. **Activate + chat** — open Chat page (per `fix_plan.md` Phase 3 — pre-req), pick the target model as inference target, click Activate on the transfer in the wizard. Send a multi-step math prompt. With `alpha=0` confirm baseline behavior; with `alpha=1.0` confirm visibly different (more stepwise) output. Toggle off → reverts.
6. **GGUF degraded path** — pick a quantized GGUF as target. Wizard refuses align (per-layer needs hooks on target) — verify the error is friendly. Pick GGUF as *source* with HF target — confirm extraction runs in `last`-layer mode with the warning surfaced in the wizard log panel.
7. **llama-cpp-python integration** — even with no transfer active, send a chat prompt at a GGUF target. Streaming works; `_LLAMA_CACHE` reuses the loaded model on the second prompt (TTL test).
8. **Cleanup** — delete a transfer; confirm both artifacts and the row are gone, and that activating the deleted transfer returns 404.
9. **Cross-page consistency** — `ActiveTransferPill` appears identically on Models and Chat. Switching active inference target preserves `capability_transfer_id` only when the new target is compatible; otherwise auto-clears with a toast.
10. **Knowledge anchors** — every `<FieldNote link="#…" />` in the wizard navigates to the correct KnowledgeBase card.

Where a real GPU isn't available, all three workers honor `dry_run=true` (mirrors `run_generate_examples.py`'s convention) and emit canned events / a stub `.npz` so smoke tests run on CI.

---

## Sequencing

| Phase | Surface | Lines BE / FE | Ships standalone? |
|---|---|---|---|
| 1 — Backend skeleton | DB table, services, two empty workers, API stubs (no inference hooks yet) | ~600 / 0 | No (admin-only) |
| 2 — Extract worker (HF) | `run_extract_capability.py` full impl + `model_introspection` | ~400 / 0 | No |
| 3 — Align worker (HF) | `run_align_capability.py` full impl with SVD | ~300 / 0 | No |
| 4 — Inference hooks (HF) | Steering plumbing in `inference_run.py` + active-target schema bump | ~250 / 0 | No |
| 5 — Wizard UI | `CapabilityTransferWizard`, `ArtifactPicker`, `ActiveTransferPill`, KB entries, nav wiring | 0 / ~1200 | **Yes — full HF-only feature** |
| 6 — llama-cpp-python runtime swap | Replace CLI with python binding; add `logits_processor` last-layer steering | ~250 / 0 | Yes (chat speedup + GGUF degraded steering) |
| 7 — GGUF extraction (`embedding=True`) | Last-layer extraction path in `run_extract_capability.py` | ~80 / ~30 (UI flag) | Yes |

Recommended order **1 → 2 → 3 → 4 → 5 → 6 → 7**. Phase 5 is the first user-visible release; phases 6-7 incrementally extend GGUF support without retroactively changing earlier code.
