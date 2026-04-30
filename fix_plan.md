# TrainingHub Frontend — Audit & Remediation Plan

## Context

You asked for a frontend review and a remediation plan. The product target is a **novice user** who needs to:
fine-tune a model, quantize it, download models and training data, benchmark on popular suites, and (newly added) **chat with the active model to assess output quality**.

The frontend was recently re-skinned on three pages (Datasets, Training, Dashboard) to a cyberpunk console aesthetic.
The other six pages still use the legacy panel styling, several novice-hostile freeform inputs, and one critical filter bug
silently breaks the new dashboard's Live Training panel.

The backend exposes most of what's needed for fine-tune / quantize / download / dataset-import / 6 math benchmarks, but **does not** expose a runtime inference endpoint (chat) or any way to query benchmark scores back from the FE. This plan covers both FE and the minimum BE work required.

---

# PART A — AUDIT REPORT

Severity legend: **C** = Critical (broken / silently wrong), **H** = High (missing must-have), **M** = Medium (UX hostile to novices), **L** = Low (cosmetic).

## A1. Broken / inaccurate behavior

| # | Sev | Where | Issue |
|---|-----|-------|-------|
| 1 | **C** | [App.tsx:433](frontend/src/App.tsx#L433) | `TRAINING_JOB_TYPES = new Set(["fine_tune", "training", "qlora", "lora", "full_finetune"])` does **not** match what the backend emits. Backend creates `train_lora` / `train_qlora` / `train_full` (see [backend/traininghub/api/jobs.py:116](backend/traininghub/api/jobs.py#L116) and [backend/traininghub/services/jobs.py:29-30](backend/traininghub/services/jobs.py#L29)). Result: the new dashboard's "Live Training" panel and the Active/Training counters report **0** for every real training run. FineTuneWizard's own filter (`j.job_type.startsWith("train_")` at [FineTuneWizard.tsx:359](frontend/src/components/FineTuneWizard.tsx#L359)) is correct — only the dashboard is wrong. |
| 2 | **H** | [App.tsx:434](frontend/src/App.tsx#L434) | `DOWNLOAD_JOB_TYPES = new Set(["model_download", "dataset_import"])` lumps dataset imports into the "Model Downloads" panel. They are different things and should be split, or dataset_import should move under a "Data" panel. |
| 3 | **M** | [App.tsx ~932](frontend/src/App.tsx#L932) | `DashLiveMetricsPanel` plots only `loss / train_loss / eval_loss`. None of the existing workers emit those keys consistently — `run_train_lora` emits `step / epoch / loss` but other workers don't. Loss canvas stays empty for any non-loss metric job and the user has no idea why. |
| 4 | **M** | [App.tsx:533](frontend/src/App.tsx#L533) | The dashboard subscribes SSE only to the *selected* job. As soon as you click another row the previous stream disconnects, so live training metrics are not actually "always live" — just "live for the one row you most recently clicked". |
| 5 | **M** | [App.tsx:3370-3371](frontend/src/App.tsx#L3370) | Models page filters upload-eligible artifacts on `["gguf_fp16", "gguf_quantized", ...]` but the worker actually registers `gguf_quantized` and `downloaded_model` (no `gguf_fp16`). String drift. |
| 6 | **L** | Topbar [App.tsx:331-347](frontend/src/App.tsx#L331) | `Pause Live` / `Refresh` buttons render in legacy `.button` style above the cyberpunk dashboard. Visual seam. |

## A2. Missing functionality (novice cannot complete intended journeys)

| # | Sev | Gap |
|---|-----|-----|
| 7 | **H** | **No chat with the active model.** User can't sanity-check fine-tune quality. Backend has no `/api/inference/run`; the only generation path is the batch dataset worker (confirmed in `services/inference.py` — only persistence). |
| 8 | **H** | **Benchmark results are invisible.** Backend writes scores into `benchmark_results` table ([core/database.py:127](backend/traininghub/core/database.py#L127)) but no GET endpoint and no FE view. The user has to download a JSON artifact to read scores. |
| 9 | **H** | **Only 6 math benchmarks**. You explicitly want "various popular and highly regarded benchmarks" — currently no MMLU / HellaSwag / ARC / IFEval / HumanEval. List in [run_benchmark_math.py:16-22](backend/traininghub/workers/run_benchmark_math.py#L16). |
| 10 | **H** | **Quantize page accepts only freeform paths.** `source_gguf` and `source_checkpoint` are bare text inputs ([App.tsx:3212, 3205](frontend/src/App.tsx#L3205)). A novice has no way to discover what to type. There is no artifact picker even though `/api/artifacts` already returns everything needed. |
| 11 | **M** | **Benchmarks page accepts a freeform `checkpoint_path`** ([App.tsx:3079-3084](frontend/src/App.tsx#L3079)). Same problem — no artifact picker. |
| 12 | **M** | **No benchmark history.** Past scores aren't queryable; novice can't compare runs. |
| 13 | **M** | **Generate page is one-knob.** Hardcodes `output_schema=math_sft`, `validation_strictness=normal`, `temperature=0.7`, `top_p=0.9`, `max_tokens=256` ([App.tsx:2980-2985](frontend/src/App.tsx#L2980)). Backend accepts category_mix / difficulty_mix / dry_run but FE never exposes them. |
| 14 | **M** | **No "what should I do next" cues** — novice opens the app, lands on Dashboard, and there's no journey hint pointing to Models → Datasets → Training → Benchmarks → Chat. |
| 15 | **L** | **Knowledge anchors don't link.** FieldNote `link="#training"` etc. don't navigate to the Knowledge page; the Knowledge page is on a separate route, so `#training` either does nothing or breaks the hashroute. |

## A3. Visual inconsistency

These pages still use legacy `app.css` classes (`.panel`, `.formGrid`, `.list`, `.tableWrap`, etc.) and look completely different from Datasets / Training / Dashboard:

- **Generate** ([App.tsx:2954](frontend/src/App.tsx#L2954))
- **Benchmarks** ([App.tsx:3053](frontend/src/App.tsx#L3053))
- **Quantize** ([App.tsx:3156](frontend/src/App.tsx#L3156))
- **Models** ([App.tsx:3249](frontend/src/App.tsx#L3249)) — including the `InferenceModelSelector` and the registry table
- **Cleanup** ([App.tsx:3519](frontend/src/App.tsx#L3519))
- **Knowledge** ([App.tsx:3571](frontend/src/App.tsx#L3571))

The `LoginScreen` is also legacy. Acceptable since it's pre-auth, but it's the very first impression.

## A4. Novice-hostile UX

- Free-form path / repo / URL inputs everywhere instead of typeahead pickers backed by `/api/models`, `/api/artifacts`, `/api/datasets`.
- No "recommended preset" buttons on Quantize (`Q4_K_M for laptops`, `Q5_K_M for desktops`, `Q8_0 for servers`).
- Benchmarks: `limit=10` default is unlabeled — novices won't know it's a smoke value.
- Models page is a 2×2 grid of dense forms; the InferenceModelSelector (the most important novice control: "what model am I currently using?") is squeezed into a corner panel.
- Cleanup page shows raw `action` strings and paths, no human-readable size totals next to each item.
- Generate page's "Review Queue" only shows generated artifacts — no link back to the source job, no preview of generated rows.

## A5. Backend gaps (required to deliver A2 features)

1. **No chat/completion endpoint.** Need new `POST /api/inference/run` that loads the active inference target and returns tokens (SSE for streaming). Reuse loaders from `run_generate_examples.py`.
2. **No benchmark-results query endpoint.** Need `GET /api/benchmarks/results?model_slug=…&benchmark=…`.
3. **No benchmark catalog endpoint.** Need `GET /api/benchmarks/catalog` returning the list of supported benchmark IDs + suite metadata.
4. **No general-purpose benchmark workers.** Need MMLU, HellaSwag, ARC, IFEval, HumanEval evaluators (lm-eval-harness wrappers; backend already calls lm-eval for the math suites).

---

# PART B — REMEDIATION PLAN

Phased so each phase ships value standalone. Each phase ends shippable; nothing waits on the next.

## Phase 1 — Critical fixes (small, fast, unblock the dashboard)

| Change | File | Notes |
|---|---|---|
| Fix `TRAINING_JOB_TYPES` to `{"train_lora", "train_qlora", "train_full"}` (or use `j.job_type.startsWith("train_")` for parity with FineTuneWizard) | [App.tsx:433](frontend/src/App.tsx#L433) | Bug #1 |
| Split downloads vs imports: rename `DOWNLOAD_JOB_TYPES` to `MODEL_DOWNLOAD_JOB_TYPES = {"model_download"}`; add `DATASET_IMPORT_JOB_TYPES = {"dataset_import"}`. Render them in two separate panels under the Ops column. | [App.tsx:434, 555-575, 1100](frontend/src/App.tsx#L434) | Bug #2 |
| Restyle topbar Pause/Refresh buttons to `.thx-btn` when on dashboard | [App.tsx:336-347](frontend/src/App.tsx#L336), `cyberpunk.css` | Bug #6 |
| Loss-curve fallback: when no loss key is present, plot the first numeric metric the worker emits, with the metric name on the panel header. Update `DashLiveMetricsPanel` accordingly. | [App.tsx ~932](frontend/src/App.tsx#L932) | Bug #3 |
| Multi-job event aggregation: replace single SSE stream with one stream per *running* job + a small ring buffer; show a global event ticker at the bottom of the Event Stream panel that lets the user toggle between "selected" and "all running". | [App.tsx:540-569](frontend/src/App.tsx#L540) | Bug #4 |
| Fix artifact-type filter on Models upload form to match worker output (`gguf_quantized`, `downloaded_model`, `training_*`). Drop `gguf_fp16`. | [App.tsx:3370](frontend/src/App.tsx#L3370) | Bug #5 |

## Phase 2 — Cyberpunk re-skin of the remaining pages

Re-skin Generate, Benchmarks, Quantize, Models, Cleanup, Knowledge to the `.thx` language. Reuse the existing components from `cyberpunk.css`:

- Page wrapper: `<div className="thx">` with `.thx-stage-h` (crumb + h2 with optional `.thx-glitch` + lede + stamp).
- Cards: `.thx-panel` + `.thx-panel-h` + `.thx-tag`.
- Forms: `.thx-params` / `.thx-field` / `.thx-field-label` (already used by FineTuneWizard — see e.g. [FineTuneWizard.tsx:1130-1180](frontend/src/components/FineTuneWizard.tsx#L1130)).
- Lists: `.thx-runs` + `.thx-run-*` for jobs/artifacts; `.thx-cards` + `.thx-card-*` for selectable models / datasets.
- Action buttons: `.thx-btn` / `.thx-btn--primary` / `.thx-btn--danger`.
- Status pills: `.thx-cap` / `.thx-cap--ok|--no|--w|--c`.
- Empty / dropzone: `.thx-empty`, `.thx-drop`.

Per-page page-skeleton plan (replacing each function body, no component renames so routing in [App.tsx:362-392](frontend/src/App.tsx#L362) doesn't change):

- **Models page** — restructure into a 3-row layout:
  1. *Active Inference Bar* (full-width, prominent): currently active target + button to swap; this is the most important novice control.
  2. *Acquire* row: HF search panel (already cyberpunk-friendly via `HubAcquirePanel`) + URL transfer + HF upload.
  3. *Registry* table re-skinned with `.thx-runs` rows showing capability badges (`.thx-cap`).
- **Quantize page** — pipeline of `.thx-pipe-node`: Source → Convert → Quantize → Done. Source artifact picker uses `.thx-cards`. Quant type chooser uses `.thx-seg`.
- **Benchmarks page** — split into Submit + History, both cyberpunk panels (covered by Phase 4).
- **Generate page** — proper `.thx-params` grid with all backend knobs + dry-run toggle.
- **Cleanup page** — `.thx-runs` rows with size totals.
- **Knowledge page** — magazine-layout `.thx-panel` cards; fix in-app `#anchor` navigation to actually scroll to the right card.

The `LoginScreen` will get a subtle cyberpunk skin (Chakra Petch heading + `.thx-deep` background) — minimal because it's pre-app.

## Phase 3 — Chat with the active model

**Backend** (minimum viable):

1. Add `traininghub/services/inference_run.py` with `run_prompt(target, prompt, sampling)` that:
   - resolves `target_type`,
   - for `base_model`: lazy-loads HF transformers in BF16 (gated by `TRAININGHUB_INFERENCE_RUNTIME=transformers`), calls `model.generate` with `streamer=TextIteratorStreamer`,
   - for `gguf_artifact`: shells out to `llama-cli` (already a dep used by `run_generate_examples.py:143-165`) with `--n-predict` and reads stdout token-by-token.
   - returns an async generator of token strings.
2. Add `traininghub/api/inference.py` route `POST /api/inference/run` that returns SSE: `event: token`, `event: done`, `event: error`. Body shape: `{ prompt: string, system?: string, temperature?: number, top_p?: number, max_tokens?: number, stop?: string[] }`. Auth via existing session.
3. Cache the loaded model in-process for a short TTL (e.g. 5 min) keyed by `target_id` to avoid reloading every prompt.
4. Tests: a `dry_run=true` mode that returns canned tokens so smoke tests don't need GPUs (mirrors existing dry-run convention).

**Frontend**:

5. Add a new top-level page `chat` to the navigation in [App.tsx:101-111](frontend/src/App.tsx#L101) with icon `MessageSquare` (lucide-react) — placed between `Models` and `Cleanup`.
6. New file `frontend/src/components/ChatConsole.tsx` (cyberpunk-styled) with:
   - **Top bar**: active inference target pill + button "Change in Models page" (deep-link via `#/models`).
   - **Conversation pane**: message bubbles using `.thx-row`-style frames; assistant messages stream token-by-token; user messages render instantly.
   - **System prompt editor** in a collapsible aside (`.thx-aside`).
   - **Sampling controls** under the input (temperature / top_p / max tokens / stop) using `.thx-field`.
   - **Latency / tokens-per-second HUD** in the topbar mirroring the wizard's HUD.
7. EventSource client for the SSE stream; cancel on Esc or "Stop".
8. Minimal markdown rendering for assistant output (code blocks + bold/italic) — implement with a tiny in-house renderer (no extra dependencies); fenced code blocks get the existing `.thx-log` mono treatment.
9. Persist last 50 messages in `localStorage` keyed by target id so toggling targets keeps history per model.

## Phase 4 — Benchmarks: catalog + submit + history + scoreboard

**Backend**:

1. Add `GET /api/benchmarks/catalog` returning `[{id, family, label, description, smoke_default, full_default}]`.
2. Add `GET /api/benchmarks/results?model_slug=&benchmark=&limit=` reading from `benchmark_results`. Response shape mirrors the table.
3. Add new evaluators (separate worker file each) and register in `services/jobs.py`:
   - `mmlu` → `traininghub.workers.run_benchmark_mmlu` (lm-eval-harness task `mmlu`).
   - `hellaswag` → `run_benchmark_hellaswag` (lm-eval task `hellaswag`).
   - `arc` → `run_benchmark_arc` (lm-eval tasks `arc_challenge` + `arc_easy`).
   - `ifeval` → `run_benchmark_ifeval` (lm-eval task `ifeval`).
   - `humaneval` → `run_benchmark_code` (`bigcode-eval-harness` if available, else lm-eval `humaneval`).
   - Common helper extracted to `traininghub/workers/_lm_eval_runner.py` so the math worker also delegates to it.
4. Make `POST /api/jobs/benchmark` accept *any* benchmark id from the catalog (currently any non-math id is rejected at [run_benchmark_math.py:113-115](backend/traininghub/workers/run_benchmark_math.py#L113)). Routing now: math benchmarks go to math worker; others go to the new workers.

**Frontend** — replace the current Benchmarks page with three cyberpunk panels:

5. **Catalog & Submit**:
   - `.thx-cards` grid grouped by family (Math / Knowledge / Reasoning / Instruction-following / Code).
   - Multi-select with smoke / full preset buttons.
   - Model picker = `.thx-cards` filtered by `supports_benchmark`; checkpoint picker = artifact `.thx-cards` (replaces freeform `checkpoint_path`).
   - Submit uses `.thx-btn--primary` and pipes the resulting job into the dashboard's selected job.
6. **History** panel: scoreboard table powered by `/api/benchmarks/results`. Filter by model + benchmark; rows clickable to view the artifact JSON.
7. **Compare** view: pick 2–3 models, render a small bar chart per benchmark using `.thx-spark`.
8. Update `KnowledgeBase` to include a "Benchmarks" entry explaining each new suite.

## Phase 5 — Novice UX upgrades

(Mostly FE; some are wired with new backend endpoints from Phases 3–4.)

- **Onboarding ribbon** on the dashboard if `models.length === 0` ("→ Download a base model from Models") and again if `datasets.length === 0` ("→ Acquire a dataset"), and again if `activeInferenceTarget == null` ("→ Pick a runtime in Models").
- **Artifact pickers** everywhere paths are needed today (Quantize source, Benchmarks checkpoint_path, Generate teacher_model). Backed by `/api/artifacts` filtered by the right artifact_type set.
- **Recommended-preset chips** on Quantize (`Q4_K_M for 8 GB consumer`, `Q5_K_M for desktops`, `Q8_0 for servers`).
- **Inline help** unification: every `FieldNote` link target gets a real anchor in `KnowledgeBase` and a working scroll behavior using `useEffect` on hashchange.
- **Toast layer** for non-modal success/error messages so deletes / approvals don't clutter inline `validationBox`.

---

## File-by-file summary

### Existing files modified

- `frontend/src/App.tsx` — fix bugs (Phase 1), wire new `chat` route + nav item (Phase 3), replace bodies of `Generate` / `Benchmarks` / `Quantize` / `Models` / `Cleanup` / `KnowledgeBase` (Phases 2 + 4 + 5). Update `TRAINING_JOB_TYPES`, split download/import sets, fix multi-SSE.
- `frontend/src/styles/cyberpunk.css` — add modules: `.thx-chat-*`, `.thx-bench-*`, `.thx-quant-*`, `.thx-models-*`, `.thx-cleanup-*`, `.thx-kb-*`. Add a wrapper `.thx-topbar-action` for the topbar buttons.
- `frontend/src/components/FineTuneWizard.tsx` — no behavioral change; only adopt the same multi-SSE helper if extracted into `useJobEventStream`.
- `frontend/src/api/client.ts` — add typed clients for `inferenceRun` (SSE), `benchmarksCatalog`, `benchmarksResults`.

### New frontend files

- `frontend/src/components/ChatConsole.tsx`
- `frontend/src/components/ArtifactPicker.tsx` — reusable cyberpunk picker used by Quantize / Benchmarks / Generate.
- `frontend/src/components/BenchmarkScoreboard.tsx`
- `frontend/src/components/OnboardingRibbon.tsx`
- `frontend/src/api/sse.ts` — small SSE helper with cancel + reconnect + auth.

### New backend files

- `backend/traininghub/services/inference_run.py`
- `backend/traininghub/workers/_lm_eval_runner.py`
- `backend/traininghub/workers/run_benchmark_mmlu.py`
- `backend/traininghub/workers/run_benchmark_hellaswag.py`
- `backend/traininghub/workers/run_benchmark_arc.py`
- `backend/traininghub/workers/run_benchmark_ifeval.py`
- `backend/traininghub/workers/run_benchmark_code.py`

### Backend files modified

- `backend/traininghub/api/inference.py` — add `POST /run` (SSE).
- `backend/traininghub/api/benchmarks.py` (new module mounted on `/api/benchmarks`) — `GET /catalog`, `GET /results`.
- `backend/traininghub/api/jobs.py` — let `POST /jobs/benchmark` accept the broader catalog and pick the worker based on family.
- `backend/traininghub/services/jobs.py` — register the new job types and worker modules.
- `backend/traininghub/workers/run_benchmark_math.py` — delegate the lm-eval shell-out into `_lm_eval_runner.py`.
- `backend/pyproject.toml` — add `transformers`, `accelerate`, `torch` (already implied for training), and document optional `llama-cpp-python` dep for the chat path.

### Reused without changes

- `frontend/src/components/JobLogPanel.tsx` (its `JobEvent` type stays canonical).
- `services/inference.py` for active-target persistence.
- `core/database.py:127` `benchmark_results` schema — already adequate.

---

## Verification

End-to-end checks (executed against dev backend with one small base model + one GGUF artifact loaded):

1. **Critical fixes (Phase 1)**
   - `npx tsc --noEmit && npx vite build` → clean.
   - Trigger a real LoRA job via FineTuneWizard; dashboard's *Active*, *Live Training* counters and pipeline reflect it within one refresh interval.
   - Trigger a `model_download` and a `dataset_import` simultaneously; they appear in *separate* panels.
   - Dashboard event stream stays attached when switching the selected job and shows the running-job ticker continuing.

2. **Re-skin (Phase 2)**
   - Visual diff (Playwright screenshot) of every page before/after; confirm `.thx` wrapper present, no leftover `.panel`/`.formGrid` selectors on the rebuilt pages.
   - All FieldNote `link="#…"` anchors scroll to the matching Knowledge entry.

3. **Chat (Phase 3)**
   - With `dry_run=true` runtime, send 5 messages, confirm SSE token stream renders incrementally and Stop cancels mid-generation.
   - Switch active inference target (base_model ↔ gguf_artifact); confirm the loaded model is swapped and `localStorage` preserves history per target.
   - Verify a smoke prompt against a real BF16 model fits inside one GPU and returns under 30s for `max_tokens=128`.

4. **Benchmarks + scoreboard (Phase 4)**
   - `GET /api/benchmarks/catalog` returns 11 entries (6 math + 5 new families).
   - Run smoke MMLU (limit=10) and HumanEval (limit=5); both succeed; results visible on the scoreboard.
   - Compare view: pick two models, confirm bar chart renders.

5. **Novice UX (Phase 5)**
   - Fresh DB: Onboarding ribbon shows the right next step at each empty state.
   - Quantize: select a `gguf_fp16`-tagged artifact via the picker (no typing); preset chip auto-selects `Q4_K_M`; job submits.

6. **Build hygiene**
   - `npx tsc --noEmit --noUnusedLocals --noUnusedParameters` passes for the new files.
   - Pre-existing legacy unused-locals warnings outside the changed files are not regressed.
   - Backend: `pytest backend/tests` green; `ruff check` clean.

---

## Sequencing & rollout

| Phase | Lines of FE/BE diff (rough) | Ships standalone? |
|---|---|---|
| 1 — Critical fixes | ~120 / 0 | Yes |
| 2 — Re-skin pages | ~1500 / 0 | Yes (cosmetic-only) |
| 3 — Chat (FE + BE) | ~600 / ~400 | Yes |
| 4 — Benchmarks (FE + BE) | ~700 / ~700 | Yes |
| 5 — Novice UX | ~400 / 0 | Yes |

Recommended order: **1 → 2 → 3 → 4 → 5**. Each phase is mergeable on its own; nothing in a later phase undoes anything from an earlier phase.
