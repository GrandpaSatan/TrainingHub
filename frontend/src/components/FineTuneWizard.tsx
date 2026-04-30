import {
  useEffect,
  useMemo,
  useState,
  ReactNode,
  MouseEvent,
} from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle,
  ChevronLeft,
  ChevronRight,
  Cpu,
  Database,
  GitBranch,
  Layers,
  Play,
  Rocket,
  Sliders,
  Target,
  Terminal,
  Zap,
} from "lucide-react";
import {
  api,
  ArtifactRecord,
  DatasetRecord,
  JobRecord,
  ModelRecord,
} from "../api/client";
import "../styles/cyberpunk.css";

type Mode = "lora" | "qlora" | "full";
type Preset = "smoke" | "standard" | "custom";

type StepKey = "base" | "data" | "calibrate" | "allocate" | "deploy" | "observe";
type TrainingJobEvent = {
  id?: number;
  created_at?: number;
  event_type: string;
  level: string;
  message: string;
  data?: Record<string, unknown>;
};
type TrainingLossPoint = {
  step: number;
  value: number;
  runtimeSeconds?: number;
};
type TrainingLossSeries = {
  jobId: string;
  label: string;
  status: string;
  points: TrainingLossPoint[];
};

type StepDef = {
  key: StepKey;
  num: string;
  label: string;
  meta: string;
  icon: ReactNode;
};

const STEPS: StepDef[] = [
  { key: "base",      num: "01", label: "Select Base",   meta: "MODEL",     icon: <Layers size={14} /> },
  { key: "data",      num: "02", label: "Ingest Data",   meta: "DATASET",   icon: <Database size={14} /> },
  { key: "calibrate", num: "03", label: "Calibrate",     meta: "METHOD",    icon: <Sliders size={14} /> },
  { key: "allocate",  num: "04", label: "Allocate",      meta: "RESOURCE",  icon: <Cpu size={14} /> },
  { key: "deploy",    num: "05", label: "Deploy",        meta: "LAUNCH",    icon: <Rocket size={14} /> },
  { key: "observe",   num: "06", label: "Observe",       meta: "TELEMETRY", icon: <Activity size={14} /> },
];
const TRAINING_EVENT_BUFFER_LIMIT = 300;
const TRAINING_EVENT_NAMES = [
  "queued",
  "started",
  "inference_shutdown",
  "worker_start",
  "training_start",
  "training_complete",
  "metric",
  "artifact",
  "succeeded",
  "failed",
  "cancelled",
  "worker_error",
];

type AsideStat = { k: string; v: string };
type AsideContent = {
  meta: string;
  title: string;
  desc: string;
  stats?: AsideStat[];
  tags?: { label: string; tone?: "ok" | "no" | "w" | "c" }[];
  note?: string;
};

type Props = {
  models: ModelRecord[];
  datasets: DatasetRecord[];
  jobs: JobRecord[];
  artifacts: ArtifactRecord[];
  refresh: () => void;
  setSelectedJob: (job: JobRecord) => void;
};

function trainingModeSupported(model: ModelRecord, mode: Mode) {
  if (mode === "lora") return model.supports_lora;
  if (mode === "qlora") return model.supports_qlora;
  return model.supports_full_finetune;
}

function paramCountReadable(raw: string): string {
  if (!raw) return "—";
  const n = Number(raw);
  if (!Number.isFinite(n) || n <= 0) return raw;
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  return n.toLocaleString();
}

function fmtRows(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

const MORRIGAN_GPU_BUDGET_GB = 12;
const LARGE_LORA_MODEL_BILLIONS = 4;
const SAFE_LORA_SEQUENCE_LENGTH = 1024;

function estimateVramGb(model: ModelRecord, mode: Mode, batch: number, maxSequenceLength: number): string {
  const estimate = estimatePlannedVramGbNumber(model, mode, batch, maxSequenceLength);
  return estimate === null ? "—" : `${estimate.toFixed(1)} GB`;
}

function estimateVramGbNumber(model: ModelRecord, mode: Mode, batch: number, maxSequenceLength: number): number | null {
  const billions = modelParameterBillions(model);
  if (billions <= 0) return null;
  const sequenceScale = Math.max(0.25, maxSequenceLength / 2048);
  const batchOverhead = Math.max(0, batch - 1) * 0.55;
  if (mode === "qlora") return billions * 0.9 + billions * 0.25 * sequenceScale + batchOverhead + 0.8;
  if (mode === "lora") return billions * 2.1 + billions * 0.6 * sequenceScale + batchOverhead + 0.7;
  return billions * 14 + billions * 1.4 * sequenceScale + batchOverhead + 1.5;
}

function estimatePlannedVramGbNumber(model: ModelRecord, mode: Mode, batch: number, maxSequenceLength: number): number | null {
  const estimate = estimateVramGbNumber(model, mode, batch, maxSequenceLength);
  if (estimate === null) return null;
  return mode === "full" ? estimate / plannedGpuCount(mode) : estimate;
}

function plannedGpuCount(mode: Mode): number {
  return mode === "full" ? 2 : 1;
}

function allocationModeLabel(mode: Mode): string {
  return mode === "full" ? "2 GPU balanced" : "1 GPU most-free";
}

function estimateTrainingRuntime(model: ModelRecord, mode: Mode, maxSteps: number, batch: number, ga: number, maxSequenceLength: number): string {
  const seconds = maxSteps * trainingSecondsPerStep(model, mode, batch, ga, maxSequenceLength);
  return formatEstimatedRuntime(seconds);
}

function trainingSecondsPerStep(model: ModelRecord, mode: Mode, batch: number, ga: number, maxSequenceLength: number): number {
  const params = modelParameterBillions(model) || 1.2;
  const paramScale = Math.max(0.35, Math.pow(params / 1.2, 0.85));
  const sequenceScale = Math.max(0.5, maxSequenceLength / 3072);
  const batchScale = 0.8 + Math.max(0, batch - 1) * 0.2;
  const gradientScale = Math.max(1, ga);
  const modeScale = mode === "full" ? 1 : mode === "qlora" ? 0.8 : 0.55;
  return 0.312 * paramScale * sequenceScale * batchScale * gradientScale * modeScale;
}

function formatEstimatedRuntime(seconds: number): string {
  const minutes = Math.max(1, Math.round(seconds / 60));
  if (minutes < 60) return `~${minutes} min`;
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return m === 0 ? `~${h}h` : `~${h}h ${m}m`;
}

function modelParameterBillions(model: ModelRecord): number {
  const sources = [model.parameter_count, model.provider_id, model.display_name, model.slug];
  for (const source of sources) {
    const parsed = parseParameterBillions(String(source || ""));
    if (parsed > 0) return parsed;
  }
  return 0;
}

function parseParameterBillions(raw: string): number {
  const numeric = Number(raw);
  if (Number.isFinite(numeric) && numeric > 0) {
    return numeric >= 1e6 ? numeric / 1e9 : numeric;
  }
  const match = raw.match(/(\d+(?:\.\d+)?)\s*B/i);
  return match ? Number(match[1]) : 0;
}

function defaultTrainingSequenceLength(model: ModelRecord, mode: Mode): number {
  const modelMax = Math.max(128, model.max_sequence_length || 2048);
  if (mode === "lora" && modelParameterBillions(model) >= LARGE_LORA_MODEL_BILLIONS) {
    return Math.min(modelMax, SAFE_LORA_SEQUENCE_LENGTH);
  }
  return modelMax;
}

function trainingMemoryRisk(model: ModelRecord, mode: Mode, maxSequenceLength: number): { blocking: boolean; message: string } | null {
  const billions = modelParameterBillions(model);
  if (mode === "lora" && billions >= LARGE_LORA_MODEL_BILLIONS && maxSequenceLength > SAFE_LORA_SEQUENCE_LENGTH) {
    return {
      blocking: true,
      message: `${model.display_name} LoRA at ${maxSequenceLength} tokens exceeds the ${MORRIGAN_GPU_BUDGET_GB} GB GPU budget. Use QLoRA or ${SAFE_LORA_SEQUENCE_LENGTH} tokens.`,
    };
  }
  const estimate = estimatePlannedVramGbNumber(model, mode, 1, maxSequenceLength);
  if (estimate !== null && estimate >= MORRIGAN_GPU_BUDGET_GB) {
    return {
      blocking: false,
      message: `Estimated VRAM is ${estimate.toFixed(1)} GB per device on a ${MORRIGAN_GPU_BUDGET_GB} GB GPU. QLoRA is safer for this model.`,
    };
  }
  return null;
}

const MODE_INFO: Record<Mode, AsideContent> = {
  lora: {
    meta: "METHOD · 03A",
    title: "LoRA",
    desc:
      "Low-Rank Adaptation. Train two small matrices that decompose weight updates instead of touching the base parameters. ~1% trainable params, big speed win, no quality cliff for most tasks.",
    stats: [
      { k: "TRAINABLE", v: "~1% of base" },
      { k: "VRAM CLASS", v: "MED" },
      { k: "OUTPUT", v: "Adapter or merged" },
      { k: "BEST FOR", v: "Style, format, narrow tasks" },
    ],
    note: "Default for first attempts. Predictable, reversible, cheap.",
  },
  qlora: {
    meta: "METHOD · 03B",
    title: "QLoRA",
    desc:
      "LoRA over a 4-bit NF4 quantized base. Same quality envelope as LoRA at roughly a quarter of the VRAM. The compromise lives in load-time precision, not in the trained adapter.",
    stats: [
      { k: "TRAINABLE", v: "~1% of base" },
      { k: "VRAM CLASS", v: "LOW" },
      { k: "OUTPUT", v: "Adapter (merge separately)" },
      { k: "BEST FOR", v: "Big models on small GPUs" },
    ],
    note: "Pick this when the base model otherwise won't fit.",
  },
  full: {
    meta: "METHOD · 03C",
    title: "Full Fine-tune",
    desc:
      "Update every parameter. Highest fidelity, highest blast radius. Expect multi-GPU. Use only when LoRA can't reach the quality ceiling you need.",
    stats: [
      { k: "TRAINABLE", v: "100%" },
      { k: "VRAM CLASS", v: "HIGH" },
      { k: "OUTPUT", v: "Full checkpoint" },
      { k: "BEST FOR", v: "Domain shift, deep behavior change" },
    ],
    note: "Verify the model exposes supports_full_finetune before queuing.",
  },
};

const PARAM_INFO: Record<string, AsideContent> = {
  max_steps: {
    meta: "PARAM · max_steps",
    title: "Max Steps",
    desc:
      "Total optimizer updates the run will perform. Each step processes one effective batch. The trainer stops here regardless of epoch count — set it from your dataset size and target epochs.",
    stats: [
      { k: "SMOKE", v: "1–5" },
      { k: "STANDARD", v: "200–2000" },
      { k: "OVERFIT RISK", v: "rises past ~3 epochs" },
    ],
    note: "Effective epoch ≈ (max_steps × batch × gradient_accum) ÷ rows.",
  },
  learning_rate: {
    meta: "PARAM · learning_rate",
    title: "Learning Rate",
    desc:
      "Step magnitude per update. The single most sensitive knob. Too high — loss diverges or oscillates. Too low — the run never lands.",
    stats: [
      { k: "LoRA / QLoRA", v: "1e-4 to 3e-4" },
      { k: "Full FT", v: "1e-5 to 5e-5" },
      { k: "WARMUP", v: "Recommended" },
    ],
    note: "If you change mode, re-check this value — full FT and LoRA live an order of magnitude apart.",
  },
  batch: {
    meta: "PARAM · per_device_batch",
    title: "Batch Size",
    desc:
      "Examples per device per forward pass. Larger batch = smoother gradient, less noise, more VRAM. If memory pressure shows up, lower this and raise gradient accumulation instead.",
    stats: [
      { k: "MIN", v: "1" },
      { k: "TYPICAL", v: "1 – 8" },
      { k: "VRAM IMPACT", v: "Linear" },
    ],
    note: "Effective batch = batch × gradient_accumulation × num_devices.",
  },
  ga: {
    meta: "PARAM · gradient_accumulation",
    title: "Gradient Accumulation",
    desc:
      "Multiply effective batch without paying VRAM cost. The trainer accumulates gradients across N micro-batches before stepping the optimizer.",
    stats: [
      { k: "EFFECT", v: "Multiplies batch" },
      { k: "VRAM COST", v: "≈ 0" },
      { k: "WALLTIME COST", v: "Linear" },
    ],
    note: "Use this when batch=1 already saturates the GPU.",
  },
  max_sequence_length: {
    meta: "PARAM · max_sequence_length",
    title: "Sequence Length",
    desc:
      "Maximum tokens kept per training example. Longer sequences increase activation memory sharply. On 12 GB GPUs, 4B-class LoRA runs should stay near 1024 tokens or use QLoRA.",
    stats: [
      { k: "SAFE 4B LoRA", v: "1024" },
      { k: "QLoRA", v: "2048 viable" },
      { k: "VRAM IMPACT", v: "High" },
    ],
    note: "Lower this before lowering quality-critical parameters. It is the fastest way to recover GPU headroom.",
  },
  lora_rank: {
    meta: "PARAM · lora_rank",
    title: "LoRA Rank",
    desc:
      "Inner dimension of the adapter. Higher rank carries more capacity at cost of more trainable parameters and more VRAM. Diminishing returns past 32 for most tasks.",
    stats: [
      { k: "DEFAULT", v: "8" },
      { k: "STANDARD RANGE", v: "8 – 32" },
      { k: "ALPHA RULE", v: "alpha ≈ 2 × rank" },
    ],
    note: "Rank 8 is a strong default. Raise only if validation loss plateaus high.",
  },
  merge: {
    meta: "PARAM · merge_adapter",
    title: "Merge Adapter",
    desc:
      "After training, fold the trained low-rank matrices back into the base weights. Produces one self-contained checkpoint. Loses the ability to detach the adapter or stack others on top.",
    stats: [
      { k: "OUTPUT", v: "Merged checkpoint" },
      { k: "REVERSIBLE", v: "No" },
      { k: "REQUIRED FOR", v: "GGUF conversion" },
    ],
    note: "Disable to keep just the adapter file.",
  },
  output_name: {
    meta: "PARAM · output_name",
    title: "Output Name",
    desc:
      "Slug for the artifact written to disk and registered in the artifact store. Defaults to `<model>-<mode>` if left blank. Use a name you'll still recognize a month from now.",
  },
};

const PRESET_INFO: Record<Preset, AsideContent> = {
  smoke: {
    meta: "PRESET · 04A",
    title: "Smoke",
    desc:
      "1–5 steps, no real learning. Ships every config through the pipeline once to confirm tokenizer, dataloader, optimizer, checkpointing all wire up. Run this first whenever something material changes.",
    stats: [{ k: "STEPS", v: "1 – 5" }, { k: "WALLTIME", v: "< 5 min" }, { k: "GOAL", v: "Validate plumbing" }],
    note: "If smoke fails, the real run will fail too — just slower.",
  },
  standard: {
    meta: "PRESET · 04B",
    title: "Standard",
    desc:
      "Defaults tuned for ~10K example datasets. Hyperparams chosen to balance training stability and VRAM. Use this for the first real attempt before reaching for Custom.",
    stats: [
      { k: "BATCH", v: "1" },
      { k: "GRAD ACCUM", v: "1" },
      { k: "LR", v: "Auto by mode" },
    ],
  },
  custom: {
    meta: "PRESET · 04C",
    title: "Custom",
    desc:
      "Manual override. Every value below is now in your hands. The wizard estimates VRAM and runtime from your choices but performs no clamping.",
    note: "Use after a Standard run that didn't converge — never on first try.",
  },
};

export function FineTuneWizard({
  models,
  datasets,
  jobs,
  artifacts,
  refresh,
  setSelectedJob,
}: Props) {
  const approvedDatasets = useMemo(
    () => datasets.filter((d) => d.approved),
    [datasets]
  );

  const [step, setStep] = useState<StepKey>("base");
  const [completed, setCompleted] = useState<Record<StepKey, boolean>>({
    base: false,
    data: false,
    calibrate: false,
    allocate: false,
    deploy: false,
    observe: false,
  });

  const [modelSlug, setModelSlug] = useState<string>("");
  const [datasetId, setDatasetId] = useState<string>("");
  const [mode, setMode] = useState<Mode>("lora");
  const [preset, setPreset] = useState<Preset>("standard");
  const [outputName, setOutputName] = useState<string>("");
  const [maxSteps, setMaxSteps] = useState<number>(200);
  const [maxSequenceLength, setMaxSequenceLength] = useState<number>(1024);
  const [learningRate, setLearningRate] = useState<number>(0.0002);
  const [batchSize, setBatchSize] = useState<number>(1);
  const [gradientAccumulation, setGradientAccumulation] = useState<number>(1);
  const [loraRank, setLoraRank] = useState<number>(8);
  const [mergeAdapter, setMergeAdapter] = useState<boolean>(true);

  const [aside, setAside] = useState<AsideContent | null>(null);
  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState<string>("");
  const [now, setNow] = useState(new Date());

  useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    if (!modelSlug && models.length > 0) setModelSlug(models[0].slug);
  }, [models, modelSlug]);

  useEffect(() => {
    if (!datasetId && approvedDatasets.length > 0) {
      setDatasetId(approvedDatasets[0].dataset_id);
    }
  }, [approvedDatasets, datasetId]);

  useEffect(() => {
    if (preset === "smoke") setMaxSteps(5);
    else if (preset === "standard") setMaxSteps(200);
  }, [preset]);

  const selectedModel = useMemo(
    () => models.find((m) => m.slug === modelSlug),
    [models, modelSlug]
  );
  const selectedDataset = useMemo(
    () => approvedDatasets.find((d) => d.dataset_id === datasetId),
    [approvedDatasets, datasetId]
  );

  useEffect(() => {
    if (!selectedModel) return;
    setMaxSequenceLength(defaultTrainingSequenceLength(selectedModel, mode));
    setMergeAdapter(mode === "lora" && modelParameterBillions(selectedModel) < LARGE_LORA_MODEL_BILLIONS);
    if (mode === "full") setLearningRate(0.00005);
    else setLearningRate(0.0002);
  }, [selectedModel, mode]);

  useEffect(() => {
    setCompleted((prev) => ({
      ...prev,
      base: !!selectedModel,
      data: !!selectedDataset,
      calibrate: !!selectedModel && trainingModeSupported(selectedModel, mode),
      allocate: !!preset && maxSteps > 0,
    }));
  }, [selectedModel, selectedDataset, mode, preset, maxSteps]);

  const trainingJobs = useMemo(
    () =>
      jobs
        .filter((j) => j.job_type.startsWith("train_"))
        .slice(0, 12),
    [jobs]
  );

  const trainingArtifacts = useMemo(
    () =>
      artifacts.filter((a) =>
        [
          "training_adapter",
          "training_checkpoint",
          "training_merged_checkpoint",
          "training_report",
        ].includes(a.artifact_type)
      ),
    [artifacts]
  );

  const stepIndex = STEPS.findIndex((s) => s.key === step);
  const progressPct = Math.round(((stepIndex + 1) / STEPS.length) * 100);

  function go(target: StepKey) {
    setStep(target);
    setAside(null);
  }

  function goNext() {
    const idx = STEPS.findIndex((s) => s.key === step);
    if (idx < STEPS.length - 1) go(STEPS[idx + 1].key);
  }
  function goPrev() {
    const idx = STEPS.findIndex((s) => s.key === step);
    if (idx > 0) go(STEPS[idx - 1].key);
  }

  const modeOk = !!selectedModel && trainingModeSupported(selectedModel, mode);
  const memoryRisk = selectedModel ? trainingMemoryRisk(selectedModel, mode, maxSequenceLength) : null;
  const canLaunch = !!selectedModel && !!selectedDataset && modeOk && !memoryRisk?.blocking;
  const launchBlockReason = !selectedModel
    ? "No base model"
    : !selectedDataset
    ? "No approved dataset"
    : !modeOk
    ? `${selectedModel.display_name} does not support ${mode.toUpperCase()}`
    : memoryRisk?.blocking
    ? memoryRisk.message
    : "";

  async function handleLaunch() {
    if (!canLaunch || !selectedModel || !selectedDataset) return;
    setLaunching(true);
    setLaunchError("");
    try {
      const job = await api.post<JobRecord>("/api/jobs/fine-tune", {
        model_slug: modelSlug,
        dataset_id: datasetId,
        mode,
        preset,
        output_name: outputName || undefined,
        max_steps: maxSteps,
        learning_rate: learningRate,
        per_device_train_batch_size: batchSize,
        gradient_accumulation_steps: gradientAccumulation,
        max_sequence_length: maxSequenceLength,
        requested_gpu_count: plannedGpuCount(mode),
        lora_rank: mode === "full" ? undefined : loraRank,
        merge_adapter: mode === "full" ? undefined : mergeAdapter,
      });
      setSelectedJob(job);
      setCompleted((prev) => ({ ...prev, deploy: true }));
      go("observe");
      refresh();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown failure";
      setLaunchError(msg);
    } finally {
      setLaunching(false);
    }
  }

  // ---- Aside (right rail) default content per step
  const defaultAside: AsideContent = useMemo(() => {
    switch (step) {
      case "base":
        return {
          meta: "STAGE · 01 OF 06",
          title: "Base Model",
          desc:
            "Choose the foundation. Every other decision in this flow inherits from this one — VRAM budget, supported modes, even sane defaults for learning rate. Hover any model to inspect.",
          stats: [
            { k: "AVAILABLE", v: String(models.length) },
            {
              k: "SUPPORTING LoRA",
              v: String(models.filter((m) => m.supports_lora).length),
            },
            {
              k: "SUPPORTING QLoRA",
              v: String(models.filter((m) => m.supports_qlora).length),
            },
          ],
          note: "Models register through the Models page. If a row is missing, register or download it there first.",
        };
      case "data":
        return {
          meta: "STAGE · 02 OF 06",
          title: "Dataset",
          desc:
            "Only approved versions can be queued. Approval is a guarantee that the schema parses, the row count matches, and a human inspected a sample. Use the Datasets page to acquire and approve new sources.",
          stats: [
            { k: "ALL DATASETS", v: String(datasets.length) },
            { k: "APPROVED", v: String(approvedDatasets.length) },
          ],
          note: approvedDatasets.length === 0
            ? "Nothing approved yet. The wizard will refuse to launch until at least one approved dataset exists."
            : undefined,
        };
      case "calibrate":
        return {
          meta: "STAGE · 03 OF 06",
          title: "Calibration",
          desc:
            "Pick the training method, then tune the hyperparameters. Hover any field on the right to inspect it before changing it. The estimator below the form recomputes live.",
          note: "Defaults are chosen so a Standard run on this model in this mode will land. Diverge with intent.",
        };
      case "allocate":
        return {
          meta: "STAGE · 04 OF 06",
          title: "Allocation",
          desc:
            "Pick a preset and name the artifact this run will produce. Smoke is a 1–5 step plumbing check. Standard is the everyday training run. Custom puts every parameter under your control.",
          stats: selectedModel
            ? [
                { k: "EST. VRAM", v: estimateVramGb(selectedModel, mode, batchSize, maxSequenceLength) },
                { k: "EST. WALLTIME", v: estimateTrainingRuntime(selectedModel, mode, maxSteps, batchSize, gradientAccumulation, maxSequenceLength) },
              ]
            : undefined,
        };
      case "deploy":
        return {
          meta: "STAGE · 05 OF 06",
          title: "Pre-Flight",
          desc:
            "Final review. Every field below comes from the prior stages. Launching enqueues a fine-tune job — no GPU is reserved until the worker picks it up.",
          note: canLaunch
            ? "Pre-flight checks passed. Cleared for launch."
            : `Blocked: ${launchBlockReason}`,
        };
      case "observe":
        return {
          meta: "STAGE · 06 OF 06",
          title: "Telemetry",
          desc:
            "Live training jobs and the artifacts they have produced. Click any job to load its full log stream into the panel below.",
          stats: [
            { k: "TRAINING JOBS", v: String(trainingJobs.length) },
            { k: "ARTIFACTS", v: String(trainingArtifacts.length) },
          ],
        };
    }
  }, [
    step,
    models,
    datasets.length,
    approvedDatasets.length,
    selectedModel,
    mode,
    batchSize,
    maxSequenceLength,
    maxSteps,
    gradientAccumulation,
    canLaunch,
    launchBlockReason,
    trainingJobs.length,
    trainingArtifacts.length,
  ]);

  const activeAside = aside ?? defaultAside;

  // ---- Hover helpers
  function hover(content: AsideContent) {
    return {
      onMouseEnter: () => setAside(content),
      onMouseLeave: () => setAside(null),
      onFocus: () => setAside(content),
      onBlur: () => setAside(null),
    };
  }

  return (
    <div className="thx">
      {/* TOPBAR */}
      <div className="thx-shell">
        <div className="thx-topbar">
          <div className="thx-topbar-brand">
            <span className="thx-topbar-brand-mark" />
            <div className="thx-topbar-title">
              <span className="t1">TRAININGHUB · CONSOLE</span>
              <span className="t2">FINE-TUNE WIZARD</span>
            </div>
          </div>
          <div className="thx-topbar-spacer" />
          <div className="thx-hud">
            <div>
              <span className="thx-dot ok" style={{ color: "var(--thx-green)" }} />
              <span>NETLINK&nbsp;<b>OK</b></span>
            </div>
            <div>
              <span>USR&nbsp;<b>operator</b></span>
            </div>
            <div>
              <span>STG&nbsp;<b>{stepIndex + 1}/{STEPS.length}</b></span>
            </div>
            <div>
              <span>UTC&nbsp;<b>{now.toISOString().slice(11, 19)}</b></span>
            </div>
          </div>
        </div>

        {/* RAIL */}
        <aside className="thx-panel thx-rail">
          <div className="thx-rail-h">[ STAGES ]</div>
          {STEPS.map((s, i) => {
            const isActive = s.key === step;
            const isDone = completed[s.key] && !isActive;
            return (
              <button
                key={s.key}
                type="button"
                className={
                  "thx-step" +
                  (isActive ? " is-active" : "") +
                  (isDone ? " is-done" : "")
                }
                onClick={() => go(s.key)}
              >
                <span className="thx-step-num">{s.num}</span>
                <span className="thx-step-body">
                  <span className="label">{s.label}</span>
                  <span className="meta">[ {s.meta} ]</span>
                </span>
              </button>
            );
          })}
        </aside>

        {/* STAGE */}
        <main className="thx-stage">
          <div className="thx-stage-h">
            <div>
              <div className="crumb">
                STG_{STEPS[stepIndex].num} · {STEPS[stepIndex].meta}
              </div>
              <h2>
                <span className="thx-glitch" data-text={STEPS[stepIndex].label.toUpperCase()}>
                  {STEPS[stepIndex].label.toUpperCase()}
                </span>
              </h2>
              <p className="lede">{stageLede(step)}</p>
            </div>
            <div className="stamp">
              SESSION&nbsp;//&nbsp;{now.toISOString().slice(0, 10)}
              <span>OPERATOR · GUEST</span>
            </div>
          </div>

          {step === "base" && (
            <SectionBase
              models={models}
              modelSlug={modelSlug}
              setModelSlug={setModelSlug}
              hover={hover}
            />
          )}

          {step === "data" && (
            <SectionData
              approvedDatasets={approvedDatasets}
              datasetId={datasetId}
              setDatasetId={setDatasetId}
              hover={hover}
            />
          )}

          {step === "calibrate" && (
            <SectionCalibrate
              selectedModel={selectedModel}
              mode={mode}
              setMode={setMode}
              maxSteps={maxSteps}
              setMaxSteps={setMaxSteps}
              maxSequenceLength={maxSequenceLength}
              setMaxSequenceLength={setMaxSequenceLength}
              learningRate={learningRate}
              setLearningRate={setLearningRate}
              batchSize={batchSize}
              setBatchSize={setBatchSize}
              gradientAccumulation={gradientAccumulation}
              setGradientAccumulation={setGradientAccumulation}
              loraRank={loraRank}
              setLoraRank={setLoraRank}
              mergeAdapter={mergeAdapter}
              setMergeAdapter={setMergeAdapter}
              hover={hover}
            />
          )}

          {step === "allocate" && (
            <SectionAllocate
              preset={preset}
              setPreset={setPreset}
              outputName={outputName}
              setOutputName={setOutputName}
              maxSteps={maxSteps}
              setMaxSteps={setMaxSteps}
              selectedModel={selectedModel}
              mode={mode}
              batchSize={batchSize}
              gradientAccumulation={gradientAccumulation}
              maxSequenceLength={maxSequenceLength}
              hover={hover}
            />
          )}

          {step === "deploy" && (
            <SectionDeploy
              selectedModel={selectedModel}
              selectedDataset={selectedDataset}
              mode={mode}
              preset={preset}
              outputName={outputName}
              maxSteps={maxSteps}
              maxSequenceLength={maxSequenceLength}
              learningRate={learningRate}
              batchSize={batchSize}
              gradientAccumulation={gradientAccumulation}
              loraRank={loraRank}
              mergeAdapter={mergeAdapter}
              canLaunch={canLaunch}
              launchBlockReason={launchBlockReason}
              launching={launching}
              launchError={launchError}
              onLaunch={handleLaunch}
            />
          )}

          {step === "observe" && (
            <SectionObserve
              jobs={trainingJobs}
              artifacts={trainingArtifacts}
              setSelectedJob={setSelectedJob}
            />
          )}
        </main>

        {/* ASIDE */}
        <aside className="thx-panel thx-aside">
          <div className="thx-aside-h">
            <span>[ CONTEXT FEED ]</span>
            <span className="ping">LIVE</span>
          </div>
          <div className="thx-aside-meta">{activeAside.meta}</div>
          <h3 className="thx-aside-title">{activeAside.title}</h3>
          <p className="thx-aside-desc">{activeAside.desc}</p>
          {activeAside.stats && activeAside.stats.length > 0 && (
            <div className="thx-aside-stats">
              {activeAside.stats.map((s) => (
                <div className="thx-aside-stat" key={s.k}>
                  <span className="k">{s.k}</span>
                  <span className="v">{s.v}</span>
                </div>
              ))}
            </div>
          )}
          {activeAside.tags && activeAside.tags.length > 0 && (
            <div className="thx-aside-tags">
              {activeAside.tags.map((t) => (
                <span
                  key={t.label}
                  className={"thx-cap" + (t.tone ? ` thx-cap--${t.tone}` : "")}
                >
                  {t.label}
                </span>
              ))}
            </div>
          )}
          {activeAside.note && (
            <div className="thx-aside-note">// {activeAside.note}</div>
          )}
        </aside>

        {/* ACTION BAR */}
        <div className="thx-action">
          <button
            type="button"
            className="thx-btn"
            onClick={goPrev}
            disabled={stepIndex === 0}
          >
            <ChevronLeft size={14} />
            PREV
          </button>
          <div className="thx-progress">
            <div className="thx-progress-meta">
              <span>SEQUENCE</span>
              <span>
                <span className="v">{progressPct}%</span> · STG_{STEPS[stepIndex].num}
              </span>
            </div>
            <div
              className="thx-progress-bar"
              style={{ ["--p" as string]: `${progressPct}%` }}
            />
          </div>
          {step === "deploy" ? (
            <button
              type="button"
              className="thx-btn thx-btn--primary"
              onClick={handleLaunch}
              disabled={!canLaunch || launching}
            >
              <Rocket size={14} />
              {launching ? "LAUNCHING…" : "LAUNCH"}
            </button>
          ) : (
            <button
              type="button"
              className="thx-btn thx-btn--primary"
              onClick={goNext}
              disabled={stepIndex === STEPS.length - 1}
            >
              NEXT
              <ChevronRight size={14} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function stageLede(step: StepKey): string {
  switch (step) {
    case "base":
      return "Pick the foundation model. Every later decision inherits from this. Hover any tile to inspect specs in the right pane.";
    case "data":
      return "Bind an approved dataset to this run. Only datasets that passed validation can be selected — others must be reviewed in the Datasets page first.";
    case "calibrate":
      return "Choose the fine-tuning method, then dial in the hyperparameters. Hover any field to read what it does before changing it.";
    case "allocate":
      return "Pick a training preset and name the artifact this run will produce. The estimator below recomputes VRAM and walltime from your choices.";
    case "deploy":
      return "Final review. Confirm the configuration. Launch enqueues the job — workers pick it up as GPUs free.";
    case "observe":
      return "Live training jobs and their artifacts. Click any job to wire it into the log panel below.";
  }
}

/* ============================================================
   SECTIONS
   ============================================================ */

function SectionBase({
  models,
  modelSlug,
  setModelSlug,
  hover,
}: {
  models: ModelRecord[];
  modelSlug: string;
  setModelSlug: (slug: string) => void;
  hover: (a: AsideContent) => Record<string, (e: any) => void>;
}) {
  return (
    <section className="thx-section is-active">
      <div className="thx-instructions">
        <span>
          <strong>Instruction</strong>
          Each tile is a registered base model. Selected tiles glow yellow. Capability badges
          tell you which fine-tuning modes will be allowed in stage 03.
        </span>
      </div>

      {models.length === 0 ? (
        <div className="thx-empty">
          NO MODELS REGISTERED · OPEN THE MODELS PAGE TO REGISTER OR DOWNLOAD ONE
        </div>
      ) : (
        <div className="thx-cards">
          {models.map((m) => {
            const selected = m.slug === modelSlug;
            const aside: AsideContent = {
              meta: `MODEL · ${m.family.toUpperCase()}`,
              title: m.display_name.toUpperCase(),
              desc:
                m.hardware_note ||
                `Registered ${m.family} family base model (${paramCountReadable(m.parameter_count)} params). Default dtype: ${m.default_dtype}. Max sequence: ${m.max_sequence_length} tokens.`,
              stats: [
                { k: "PARAMETERS", v: paramCountReadable(m.parameter_count) },
                { k: "DTYPE", v: m.default_dtype },
                { k: "CTX LEN", v: String(m.max_sequence_length) },
                { k: "PROVIDER", v: m.provider_id },
              ],
              tags: [
                { label: "LoRA", tone: m.supports_lora ? "ok" : "no" },
                { label: "QLoRA", tone: m.supports_qlora ? "ok" : "no" },
                { label: "FULL", tone: m.supports_full_finetune ? "ok" : "no" },
                { label: "BENCH", tone: m.supports_benchmark ? "c" : "no" },
                { label: "GGUF", tone: m.supports_quantization ? "c" : "no" },
              ],
              note: m.hardware_note || undefined,
            };
            return (
              <button
                key={m.slug}
                type="button"
                className={"thx-card" + (selected ? " is-selected" : "")}
                onClick={() => setModelSlug(m.slug)}
                {...hover(aside)}
              >
                <div className="thx-card-row">
                  <div>
                    <div className="thx-card-title">{m.display_name}</div>
                    <div className="thx-card-sub">
                      {m.family} · {paramCountReadable(m.parameter_count)}
                    </div>
                  </div>
                  <div className="thx-card-status">
                    {selected ? "// SELECTED" : "READY"}
                  </div>
                </div>
                <div className="thx-card-stats">
                  <div className="thx-card-stat">
                    <span className="k">DTYPE</span>
                    <span className="v">{m.default_dtype}</span>
                  </div>
                  <div className="thx-card-stat">
                    <span className="k">CTX</span>
                    <span className="v">{m.max_sequence_length}</span>
                  </div>
                </div>
                <div className="thx-aside-tags" style={{ marginTop: 10 }}>
                  <span className={"thx-cap " + (m.supports_lora ? "thx-cap--ok" : "thx-cap--no")}>LoRA</span>
                  <span className={"thx-cap " + (m.supports_qlora ? "thx-cap--ok" : "thx-cap--no")}>QLoRA</span>
                  <span className={"thx-cap " + (m.supports_full_finetune ? "thx-cap--ok" : "thx-cap--no")}>FULL</span>
                </div>
              </button>
            );
          })}
        </div>
      )}
    </section>
  );
}

function SectionData({
  approvedDatasets,
  datasetId,
  setDatasetId,
  hover,
}: {
  approvedDatasets: DatasetRecord[];
  datasetId: string;
  setDatasetId: (id: string) => void;
  hover: (a: AsideContent) => Record<string, (e: any) => void>;
}) {
  return (
    <section className="thx-section is-active">
      <div className="thx-instructions">
        <span>
          <strong>Instruction</strong>
          Only datasets marked APPROVED on the Datasets page show here. Approval gates training —
          this is intentional. Hover any tile for full validation summary.
        </span>
      </div>
      {approvedDatasets.length === 0 ? (
        <div className="thx-empty">
          NO APPROVED DATASETS · OPEN THE DATASETS PAGE TO ACQUIRE, VALIDATE, AND APPROVE
        </div>
      ) : (
        <div className="thx-cards">
          {approvedDatasets.map((d) => {
            const selected = d.dataset_id === datasetId;
            const validation = d.validation;
            const errCount = validation?.errors?.length ?? 0;
            const warnCount = validation?.warnings?.length ?? 0;
            const aside: AsideContent = {
              meta: `DATASET · ${d.dataset_type.toUpperCase()}`,
              title: d.title.toUpperCase(),
              desc: `Approved version ${d.version_id}. ${fmtRows(d.row_count)} rows across ${
                Object.keys(d.split_counts || {}).length
              } split(s). Validation accepted ${validation?.accepted_count ?? 0} rows.`,
              stats: [
                { k: "ROWS", v: fmtRows(d.row_count) },
                { k: "TYPE", v: d.dataset_type },
                { k: "VERSION", v: d.version_id },
                { k: "SPLITS", v: Object.keys(d.split_counts || {}).join(", ") || "—" },
                { k: "ERRORS", v: String(errCount) },
                { k: "WARNINGS", v: String(warnCount) },
              ],
              tags: [
                { label: "APPROVED", tone: "ok" },
                ...(errCount === 0 ? [{ label: "CLEAN", tone: "ok" as const }] : [{ label: `${errCount} ERR`, tone: "no" as const }]),
                ...(warnCount === 0 ? [] : [{ label: `${warnCount} WARN`, tone: "w" as const }]),
              ],
              note: errCount > 0 ? "Validation errors exist. Re-review on the Datasets page before training." : undefined,
            };
            return (
              <button
                key={d.dataset_id}
                type="button"
                className={"thx-card" + (selected ? " is-selected" : "")}
                onClick={() => setDatasetId(d.dataset_id)}
                {...hover(aside)}
              >
                <div className="thx-card-row">
                  <div>
                    <div className="thx-card-title">{d.title}</div>
                    <div className="thx-card-sub">
                      {d.dataset_type} · v{d.version_id}
                    </div>
                  </div>
                  <div className="thx-card-status">
                    {selected ? "// BOUND" : "APPROVED"}
                  </div>
                </div>
                <div className="thx-card-stats">
                  <div className="thx-card-stat">
                    <span className="k">ROWS</span>
                    <span className="v">{fmtRows(d.row_count)}</span>
                  </div>
                  <div className="thx-card-stat">
                    <span className="k">SPLITS</span>
                    <span className="v">{Object.keys(d.split_counts || {}).length}</span>
                  </div>
                  <div className="thx-card-stat">
                    <span className="k">ERR</span>
                    <span className="v">{errCount}</span>
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      )}
    </section>
  );
}

function SectionCalibrate({
  selectedModel,
  mode,
  setMode,
  maxSteps,
  setMaxSteps,
  maxSequenceLength,
  setMaxSequenceLength,
  learningRate,
  setLearningRate,
  batchSize,
  setBatchSize,
  gradientAccumulation,
  setGradientAccumulation,
  loraRank,
  setLoraRank,
  mergeAdapter,
  setMergeAdapter,
  hover,
}: {
  selectedModel?: ModelRecord;
  mode: Mode;
  setMode: (m: Mode) => void;
  maxSteps: number;
  setMaxSteps: (n: number) => void;
  maxSequenceLength: number;
  setMaxSequenceLength: (n: number) => void;
  learningRate: number;
  setLearningRate: (n: number) => void;
  batchSize: number;
  setBatchSize: (n: number) => void;
  gradientAccumulation: number;
  setGradientAccumulation: (n: number) => void;
  loraRank: number;
  setLoraRank: (n: number) => void;
  mergeAdapter: boolean;
  setMergeAdapter: (b: boolean) => void;
  hover: (a: AsideContent) => Record<string, (e: any) => void>;
}) {
  const memoryRisk = selectedModel ? trainingMemoryRisk(selectedModel, mode, maxSequenceLength) : null;
  const maxSequenceLimit = selectedModel?.max_sequence_length || 4096;
  const modes: { key: Mode; label: string; sub: string; supported: boolean }[] = [
    {
      key: "lora",
      label: "LoRA",
      sub: "Adapter · low VRAM",
      supported: !!selectedModel?.supports_lora,
    },
    {
      key: "qlora",
      label: "QLoRA",
      sub: "4-bit base · lowest VRAM",
      supported: !!selectedModel?.supports_qlora,
    },
    {
      key: "full",
      label: "Full",
      sub: "All weights · multi-GPU",
      supported: !!selectedModel?.supports_full_finetune,
    },
  ];

  return (
    <section className="thx-section is-active">
      <div className="thx-instructions">
        <span>
          <strong>Instruction</strong>
          Pick a method, then tune the parameters. Greyed tiles aren't supported by the
          current base model. Hover any field below to see what it does before changing it.
        </span>
      </div>

      <div className="thx-panel">
        <div className="thx-panel-h">
          <h3>Method</h3>
          <span className="thx-tag">[ 03A · MODE ]</span>
        </div>
        <div className="thx-seg">
          {modes.map((m) => (
            <button
              key={m.key}
              type="button"
              className={"thx-seg-item" + (mode === m.key ? " is-active" : "")}
              onClick={() => m.supported && setMode(m.key)}
              disabled={!m.supported}
              {...hover(MODE_INFO[m.key])}
            >
              {m.label}
              <span className="sub">{m.sub}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="thx-panel">
        <div className="thx-panel-h">
          <h3>Hyperparameters</h3>
          <span className="thx-tag">[ 03B · CONFIG ]</span>
        </div>
        <div className="thx-params">
          <div className="thx-field" {...hover(PARAM_INFO.max_steps)}>
            <div className="thx-field-label">
              <span>max_steps</span>
              <span className="v">{maxSteps}</span>
            </div>
            <input
              type="number"
              min={1}
              value={maxSteps}
              onChange={(e) => setMaxSteps(Math.max(1, Number(e.target.value)))}
            />
          </div>

          <div className="thx-field" {...hover(PARAM_INFO.learning_rate)}>
            <div className="thx-field-label">
              <span>learning_rate</span>
              <span className="v">{learningRate.toExponential(2)}</span>
            </div>
            <input
              type="number"
              step={0.00001}
              value={learningRate}
              onChange={(e) => setLearningRate(Number(e.target.value))}
            />
          </div>

          <div className="thx-field" {...hover(PARAM_INFO.batch)}>
            <div className="thx-field-label">
              <span>per_device_batch</span>
              <span className="v">{batchSize}</span>
            </div>
            <input
              type="number"
              min={1}
              value={batchSize}
              onChange={(e) => setBatchSize(Math.max(1, Number(e.target.value)))}
            />
          </div>

          <div className="thx-field" {...hover(PARAM_INFO.ga)}>
            <div className="thx-field-label">
              <span>gradient_accumulation</span>
              <span className="v">{gradientAccumulation}</span>
            </div>
            <input
              type="number"
              min={1}
              value={gradientAccumulation}
              onChange={(e) =>
                setGradientAccumulation(Math.max(1, Number(e.target.value)))
              }
            />
          </div>

          <div className="thx-field" {...hover(PARAM_INFO.max_sequence_length)}>
            <div className="thx-field-label">
              <span>max_sequence_length</span>
              <span className="v">{maxSequenceLength}</span>
            </div>
            <input
              type="number"
              min={128}
              max={maxSequenceLimit}
              step={128}
              value={maxSequenceLength}
              onChange={(e) =>
                setMaxSequenceLength(
                  Math.min(maxSequenceLimit, Math.max(128, Number(e.target.value)))
                )
              }
            />
          </div>

          {mode !== "full" && (
            <>
              <div className="thx-field" {...hover(PARAM_INFO.lora_rank)}>
                <div className="thx-field-label">
                  <span>lora_rank</span>
                  <span className="v">{loraRank}</span>
                </div>
                <input
                  type="range"
                  min={4}
                  max={64}
                  step={2}
                  value={loraRank}
                  onChange={(e) => setLoraRank(Number(e.target.value))}
                />
                <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "var(--thx-font-mono)", fontSize: 9, color: "var(--thx-text-dim)", letterSpacing: "0.12em" }}>
                  <span>4</span><span>16</span><span>32</span><span>64</span>
                </div>
              </div>

              <div className="thx-field" {...hover(PARAM_INFO.merge)}>
                <div className="thx-field-label">
                  <span>merge_adapter</span>
                  <span className="v">{mergeAdapter ? "ON" : "OFF"}</span>
                </div>
                <label className="thx-toggle">
                  <input
                    type="checkbox"
                    checked={mergeAdapter}
                    onChange={(e) => setMergeAdapter(e.target.checked)}
                  />
                  <span className="thx-toggle-track" />
                  <span style={{ fontFamily: "var(--thx-font-mono)", fontSize: 11, color: "var(--thx-text-mid)" }}>
                    Fold adapter into base after training
                  </span>
                </label>
              </div>
            </>
          )}
        </div>
      </div>

      {memoryRisk && (
        <div
          className="thx-aside-note"
          style={{ borderColor: memoryRisk.blocking ? "var(--thx-red)" : "var(--thx-yellow)" }}
        >
          // GPU MEMORY · {memoryRisk.message}
        </div>
      )}

      {selectedModel && (
        <div className="thx-panel thx-panel--accent">
          <div className="thx-panel-h">
            <h3>Estimator</h3>
            <span className="thx-tag">[ 03C · DERIVED ]</span>
          </div>
          <div className="thx-summary">
            <div className="thx-summary-item">
              <span className="k">EST. VRAM</span>
              <span className="v">{estimateVramGb(selectedModel, mode, batchSize, maxSequenceLength)}</span>
              <span className="vmono">{mode.toUpperCase()} · BS={batchSize} · CTX={maxSequenceLength}</span>
            </div>
            <div className="thx-summary-item">
              <span className="k">EST. WALLTIME</span>
              <span className="v">{estimateTrainingRuntime(selectedModel, mode, maxSteps, batchSize, gradientAccumulation, maxSequenceLength)}</span>
              <span className="vmono">{maxSteps} STEPS · GA={gradientAccumulation}</span>
            </div>
            <div className="thx-summary-item">
              <span className="k">EFFECTIVE BATCH</span>
              <span className="v">{batchSize * gradientAccumulation}</span>
              <span className="vmono">{batchSize} × {gradientAccumulation}</span>
            </div>
            <div className="thx-summary-item">
              <span className="k">METHOD</span>
              <span className="v">{mode.toUpperCase()}</span>
              <span className="vmono">{mode === "full" ? "ALL PARAMS" : `RANK ${loraRank}`}</span>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

function SectionAllocate({
  preset,
  setPreset,
  outputName,
  setOutputName,
  maxSteps,
  setMaxSteps,
  selectedModel,
  mode,
  batchSize,
  gradientAccumulation,
  maxSequenceLength,
  hover,
}: {
  preset: Preset;
  setPreset: (p: Preset) => void;
  outputName: string;
  setOutputName: (s: string) => void;
  maxSteps: number;
  setMaxSteps: (n: number) => void;
  selectedModel?: ModelRecord;
  mode: Mode;
  batchSize: number;
  gradientAccumulation: number;
  maxSequenceLength: number;
  hover: (a: AsideContent) => Record<string, (e: any) => void>;
}) {
  const presets: { key: Preset; label: string; sub: string }[] = [
    { key: "smoke",    label: "Smoke",    sub: "5 steps · validate plumbing" },
    { key: "standard", label: "Standard", sub: "200 steps · everyday run" },
    { key: "custom",   label: "Custom",   sub: "Manual · use prior values" },
  ];

  return (
    <section className="thx-section is-active">
      <div className="thx-instructions">
        <span>
          <strong>Instruction</strong>
          Choose a preset and name the artifact. Smoke runs verify the pipeline before you
          burn real GPU hours. Output name defaults to <code style={{ fontFamily: "var(--thx-font-mono)" }}>&lt;model&gt;-&lt;mode&gt;</code>.
        </span>
      </div>

      <div className="thx-panel">
        <div className="thx-panel-h">
          <h3>Preset</h3>
          <span className="thx-tag">[ 04A · PROFILE ]</span>
        </div>
        <div className="thx-seg">
          {presets.map((p) => (
            <button
              key={p.key}
              type="button"
              className={"thx-seg-item" + (preset === p.key ? " is-active" : "")}
              onClick={() => setPreset(p.key)}
              {...hover(PRESET_INFO[p.key])}
            >
              {p.label}
              <span className="sub">{p.sub}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="thx-panel">
        <div className="thx-panel-h">
          <h3>Output</h3>
          <span className="thx-tag">[ 04B · ARTIFACT ]</span>
        </div>
        <div className="thx-params">
          <div className="thx-field" {...hover(PARAM_INFO.output_name)}>
            <div className="thx-field-label">
              <span>output_name</span>
              <span className="v">{outputName ? "SET" : "AUTO"}</span>
            </div>
            <input
              type="text"
              placeholder={
                selectedModel ? `${selectedModel.slug}-${mode}` : "trained-model"
              }
              value={outputName}
              onChange={(e) => setOutputName(e.target.value)}
            />
          </div>
          <div className="thx-field" {...hover(PARAM_INFO.max_steps)}>
            <div className="thx-field-label">
              <span>max_steps</span>
              <span className="v">{maxSteps}</span>
            </div>
            <input
              type="number"
              min={1}
              value={maxSteps}
              onChange={(e) => setMaxSteps(Math.max(1, Number(e.target.value)))}
            />
          </div>
        </div>
      </div>

      {selectedModel && (
        <div className="thx-panel thx-panel--accent">
          <div className="thx-panel-h">
            <h3>Resource Forecast</h3>
            <span className="thx-tag">[ 04C · ESTIMATE ]</span>
          </div>
          <div className="thx-summary">
            <div className="thx-summary-item">
              <span className="k">EST. VRAM</span>
              <span className="v">{estimateVramGb(selectedModel, mode, batchSize, maxSequenceLength)}</span>
              <span className="vmono">// per device · ctx {maxSequenceLength}</span>
            </div>
            <div className="thx-summary-item">
              <span className="k">GPU PLAN</span>
              <span className="v">{plannedGpuCount(mode)}</span>
              <span className="vmono">// {allocationModeLabel(mode)}</span>
            </div>
            <div className="thx-summary-item">
              <span className="k">EST. WALLTIME</span>
              <span className="v">{estimateTrainingRuntime(selectedModel, mode, maxSteps, batchSize, gradientAccumulation, maxSequenceLength)}</span>
              <span className="vmono">// {maxSteps} steps</span>
            </div>
            <div className="thx-summary-item">
              <span className="k">PROFILE</span>
              <span className="v">{preset.toUpperCase()}</span>
              <span className="vmono">// {mode.toUpperCase()}</span>
            </div>
          </div>
          <div className="thx-aside-note" style={{ marginTop: 12 }}>
            // FIGURES ARE HEURISTICS · ACTUAL VRAM/WALLTIME DEPENDS ON SEQUENCE LENGTH AND DATASET COMPOSITION
          </div>
        </div>
      )}
    </section>
  );
}

function SectionDeploy({
  selectedModel,
  selectedDataset,
  mode,
  preset,
  outputName,
  maxSteps,
  maxSequenceLength,
  learningRate,
  batchSize,
  gradientAccumulation,
  loraRank,
  mergeAdapter,
  canLaunch,
  launchBlockReason,
  launching,
  launchError,
  onLaunch,
}: {
  selectedModel?: ModelRecord;
  selectedDataset?: DatasetRecord;
  mode: Mode;
  preset: Preset;
  outputName: string;
  maxSteps: number;
  maxSequenceLength: number;
  learningRate: number;
  batchSize: number;
  gradientAccumulation: number;
  loraRank: number;
  mergeAdapter: boolean;
  canLaunch: boolean;
  launchBlockReason: string;
  launching: boolean;
  launchError: string;
  onLaunch: () => void;
}) {
  const memoryRisk = selectedModel ? trainingMemoryRisk(selectedModel, mode, maxSequenceLength) : null;
  const checks = [
    {
      label: "BASE_MODEL_RESOLVED",
      ok: !!selectedModel,
      detail: selectedModel?.display_name ?? "no model selected",
    },
    {
      label: "DATASET_APPROVED",
      ok: !!selectedDataset,
      detail: selectedDataset
        ? `${selectedDataset.title} · ${fmtRows(selectedDataset.row_count)} rows`
        : "no approved dataset bound",
    },
    {
      label: "METHOD_SUPPORTED",
      ok:
        !!selectedModel &&
        ((mode === "lora" && selectedModel.supports_lora) ||
          (mode === "qlora" && selectedModel.supports_qlora) ||
          (mode === "full" && selectedModel.supports_full_finetune)),
      detail: `${mode.toUpperCase()} on ${selectedModel?.display_name ?? "?"}`,
    },
    {
      label: "STEPS_NONZERO",
      ok: maxSteps > 0,
      detail: `${maxSteps} step${maxSteps === 1 ? "" : "s"} requested`,
    },
    {
      label: "GPU_MEMORY_BUDGET",
      ok: !memoryRisk?.blocking,
      detail: memoryRisk?.message || (selectedModel ? `${estimateVramGb(selectedModel, mode, batchSize, maxSequenceLength)} estimated at ${maxSequenceLength} tokens` : "no model selected"),
    },
    {
      label: "GPU_ALLOCATION",
      ok: true,
      detail: `${plannedGpuCount(mode)} requested · ${allocationModeLabel(mode)}`,
    },
    {
      label: "OUTPUT_NAME",
      ok: true,
      detail: outputName || `auto: ${selectedModel?.slug ?? "model"}-${mode}`,
    },
  ];

  return (
    <section className="thx-section is-active">
      <div className="thx-instructions">
        <span>
          <strong>Instruction</strong>
          Final review. The grid below mirrors what will be sent to <code style={{ fontFamily: "var(--thx-font-mono)" }}>/api/jobs/fine-tune</code>.
          Press LAUNCH to enqueue. Monitoring jumps to stage 06 automatically.
        </span>
      </div>

      <div className="thx-panel thx-panel--accent">
        <div className="thx-panel-h">
          <h3>Configuration</h3>
          <span className="thx-tag">[ 05A · PAYLOAD ]</span>
        </div>
        <div className="thx-summary">
          <div className="thx-summary-item">
            <span className="k">BASE MODEL</span>
            <span className="v">{selectedModel?.display_name || "—"}</span>
            <span className="vmono">{selectedModel?.slug || ""}</span>
          </div>
          <div className="thx-summary-item">
            <span className="k">DATASET</span>
            <span className="v">{selectedDataset?.title || "—"}</span>
            <span className="vmono">v{selectedDataset?.version_id || "—"}</span>
          </div>
          <div className="thx-summary-item">
            <span className="k">METHOD</span>
            <span className="v">{mode.toUpperCase()}</span>
            <span className="vmono">{preset.toUpperCase()} preset</span>
          </div>
          <div className="thx-summary-item">
            <span className="k">OUTPUT</span>
            <span className="v">{outputName || `${selectedModel?.slug ?? "model"}-${mode}`}</span>
            <span className="vmono">{mergeAdapter && mode !== "full" ? "MERGED" : mode === "full" ? "FULL CKPT" : "ADAPTER"}</span>
          </div>
          <div className="thx-summary-item">
            <span className="k">MAX STEPS</span>
            <span className="v">{maxSteps}</span>
            <span className="vmono">EFF. BATCH = {batchSize * gradientAccumulation}</span>
          </div>
          <div className="thx-summary-item">
            <span className="k">SEQ LENGTH</span>
            <span className="v">{maxSequenceLength}</span>
            <span className="vmono">tokens per row</span>
          </div>
          <div className="thx-summary-item">
            <span className="k">LEARNING RATE</span>
            <span className="v">{learningRate.toExponential(2)}</span>
            <span className="vmono">{mode === "full" ? "full FT band" : "LoRA band"}</span>
          </div>
          {mode !== "full" && (
            <div className="thx-summary-item">
              <span className="k">LORA RANK</span>
              <span className="v">{loraRank}</span>
              <span className="vmono">alpha guideline = {loraRank * 2}</span>
            </div>
          )}
          {selectedModel && (
            <div className="thx-summary-item">
              <span className="k">EST. VRAM</span>
              <span className="v">{estimateVramGb(selectedModel, mode, batchSize, maxSequenceLength)}</span>
              <span className="vmono">heuristic</span>
            </div>
          )}
        </div>
      </div>
      {memoryRisk && !memoryRisk.blocking && (
        <div className="thx-aside-note" style={{ borderColor: "var(--thx-yellow)" }}>
          // GPU MEMORY · {memoryRisk.message}
        </div>
      )}

      <div className="thx-panel">
        <div className="thx-panel-h">
          <h3>Pre-Flight</h3>
          <span className="thx-tag">[ 05B · CHECKLIST ]</span>
        </div>
        <div className="thx-checklist">
          {checks.map((c) => (
            <div
              key={c.label}
              className={"thx-check " + (c.ok ? "is-ok" : "is-bad")}
            >
              <span className="marker">{c.ok ? "[ OK ]" : "[ NO ]"}</span>
              <span style={{ color: "var(--thx-text)", letterSpacing: "0.12em" }}>{c.label}</span>
              <span style={{ color: "var(--thx-text-dim)", flex: 1, marginLeft: 14 }}>· {c.detail}</span>
              {c.ok ? <CheckCircle size={13} color="var(--thx-green)" /> : <AlertTriangle size={13} color="var(--thx-red)" />}
            </div>
          ))}
        </div>
        {launchError && (
          <div
            style={{
              marginTop: 12,
              padding: "10px 12px",
              borderLeft: "2px solid var(--thx-red)",
              background: "var(--thx-red-soft)",
              fontFamily: "var(--thx-font-mono)",
              fontSize: 11,
              color: "var(--thx-red)",
            }}
          >
            // LAUNCH FAILED · {launchError}
          </div>
        )}
        <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 14 }}>
          <button
            type="button"
            className="thx-btn thx-btn--primary"
            onClick={onLaunch}
            disabled={!canLaunch || launching}
          >
            <Play size={14} />
            {launching ? "TRANSMITTING…" : "LAUNCH FINE-TUNE"}
          </button>
        </div>
        {!canLaunch && (
          <div className="thx-aside-note" style={{ marginTop: 10, borderColor: "var(--thx-red)" }}>
            // BLOCKED · {launchBlockReason}
          </div>
        )}
      </div>
    </section>
  );
}

function SectionObserve({
  jobs,
  artifacts,
  setSelectedJob,
}: {
  jobs: JobRecord[];
  artifacts: ArtifactRecord[];
  setSelectedJob: (job: JobRecord) => void;
}) {
  const eventsByJobId = useTrainingJobEvents(jobs);
  const lossSeries = useMemo(() => buildLossSeries(jobs, eventsByJobId), [jobs, eventsByJobId]);

  const running = jobs.filter((j) => j.status === "running").length;
  const queued = jobs.filter((j) => j.status === "queued" || j.status === "pending").length;
  const done = jobs.filter((j) => j.status === "succeeded" || j.status === "completed").length;
  const failed = jobs.filter((j) => j.status === "failed" || j.status === "error").length;

  return (
    <section className="thx-section is-active">
      <div className="thx-instructions">
        <span>
          <strong>Instruction</strong>
          Live training jobs and the artifacts they have produced. Click any row to wire it
          into the log panel below.
        </span>
      </div>

      <div className="thx-mons">
        <div className="thx-mon">
          <div className="k">[ ACTIVE ]</div>
          <div className="v yl">{running.toString().padStart(2, "0")}</div>
        </div>
        <div className="thx-mon">
          <div className="k">[ QUEUED ]</div>
          <div className="v cy">{queued.toString().padStart(2, "0")}</div>
        </div>
        <div className="thx-mon">
          <div className="k">[ COMPLETED ]</div>
          <div className="v gr">{done.toString().padStart(2, "0")}</div>
        </div>
        <div className="thx-mon">
          <div className="k">[ FAILED ]</div>
          <div className="v" style={{ color: "var(--thx-red)" }}>{failed.toString().padStart(2, "0")}</div>
        </div>
      </div>

      <div className="thx-panel">
        <div className="thx-panel-h">
          <h3>Loss · Live Metrics</h3>
          <span className="thx-tag">[ 06A · REAL · {lossSeries.length.toString().padStart(2, "0")} ]</span>
        </div>
        <TrainingLossChart series={lossSeries} />
      </div>

      <div className="thx-panel">
        <div className="thx-panel-h">
          <h3>Training Jobs</h3>
          <span className="thx-tag">[ 06B · JOBS ]</span>
        </div>
        {jobs.length === 0 ? (
          <div className="thx-empty">
            NO TRAINING JOBS · LAUNCH ONE FROM STAGE 05
          </div>
        ) : (
          <div className="thx-runs">
            {jobs.map((j) => {
              const cls = jobStatusClass(j.status);
              return (
                <button
                  key={j.job_id}
                  type="button"
                  className={"thx-run " + cls}
                  onClick={() => setSelectedJob(j)}
                >
                  <span className="thx-run-dot" />
                  <span className="thx-run-id">{j.job_id}</span>
                  <span className="thx-run-meta">{j.job_type} · {jobGpuPlan(j)}</span>
                  <span className="thx-run-status">{j.status}</span>
                </button>
              );
            })}
          </div>
        )}
      </div>

      <div className="thx-panel">
        <div className="thx-panel-h">
          <h3>Artifacts</h3>
          <span className="thx-tag">[ 06C · OUTPUT ]</span>
        </div>
        {artifacts.length === 0 ? (
          <div className="thx-empty">
            NO ARTIFACTS YET · ADAPTERS, CHECKPOINTS, AND REPORTS LAND HERE
          </div>
        ) : (
          <div className="thx-runs">
            {artifacts.map((a) => (
              <div className="thx-run" key={a.artifact_id} style={{ cursor: "default" }}>
                <span className="thx-run-dot" style={{ background: "var(--thx-cyan)" }} />
                <span className="thx-run-id">{a.display_name}</span>
                <span className="thx-run-meta">{a.artifact_type}</span>
                <span className="thx-run-status">{(a.size_bytes / 1e9).toFixed(2)} GB</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

function jobGpuPlan(job: JobRecord): string {
  const allocation = job.payload.gpu_allocation as { strategy?: string } | undefined;
  const gpuText = job.gpu_ids.length ? job.gpu_ids.map((id) => `GPU${id}`).join("+") : "no GPU";
  return allocation?.strategy ? `${gpuText} · ${allocation.strategy}` : gpuText;
}

function useTrainingJobEvents(jobs: JobRecord[]): Record<string, TrainingJobEvent[]> {
  const [eventsByJobId, setEventsByJobId] = useState<Record<string, TrainingJobEvent[]>>({});
  const jobIds = useMemo(() => jobs.map((job) => job.job_id).sort(), [jobs]);
  const jobIdsKey = jobIds.join("|");

  useEffect(() => {
    if (jobIds.length === 0) {
      return;
    }
    const sources = jobIds.map((jobId) => {
      const source = new EventSource(`/api/jobs/${jobId}/events`);
      const appendEvent = (event: MessageEvent) => {
        try {
          const parsed = JSON.parse(event.data) as TrainingJobEvent;
          setEventsByJobId((current) => {
            const currentEvents = current[jobId] || [];
            if (eventAlreadySeen(currentEvents, parsed)) {
              return current;
            }
            return {
              ...current,
              [jobId]: [...currentEvents, parsed].slice(-TRAINING_EVENT_BUFFER_LIMIT),
            };
          });
        } catch {
          // Event streams stay open; ignore malformed frames.
        }
      };
      source.onmessage = appendEvent;
      TRAINING_EVENT_NAMES.forEach((name) => {
        source.addEventListener(name, (event) => appendEvent(event as MessageEvent));
      });
      source.addEventListener("close", () => source.close());
      return source;
    });

    return () => {
      sources.forEach((source) => source.close());
    };
  }, [jobIdsKey]);

  return eventsByJobId;
}

function eventAlreadySeen(events: TrainingJobEvent[], nextEvent: TrainingJobEvent): boolean {
  if (nextEvent.id !== undefined) {
    return events.some((event) => event.id === nextEvent.id);
  }
  return events.some(
    (event) =>
      event.event_type === nextEvent.event_type &&
      event.created_at === nextEvent.created_at &&
      event.message === nextEvent.message
  );
}

function buildLossSeries(jobs: JobRecord[], eventsByJobId: Record<string, TrainingJobEvent[]>): TrainingLossSeries[] {
  return jobs
    .map((job) => {
      const events = eventsByJobId[job.job_id] || [];
      const points = events
        .filter((event) => event.event_type === "metric" && event.data)
        .map((event, index) => lossPointFromEvent(event, index))
        .filter((point): point is TrainingLossPoint => point !== null)
        .slice(-120);
      return {
        jobId: job.job_id,
        label: trainingJobLabel(job),
        status: job.status,
        points,
      };
    })
    .filter((series) => series.points.length > 0);
}

function lossPointFromEvent(event: TrainingJobEvent, index: number): TrainingLossPoint | null {
  const data = event.data || {};
  const value = firstFiniteNumber(data.loss, data.train_loss, data.eval_loss);
  if (value === null) {
    return null;
  }
  return {
    step: firstFiniteNumber(data.step) ?? index + 1,
    value,
    runtimeSeconds: firstFiniteNumber(data.runtime_seconds) ?? undefined,
  };
}

function TrainingLossChart({ series }: { series: TrainingLossSeries[] }) {
  if (series.length === 0) {
    return (
      <div className="thx-spark" style={{ minHeight: 128, display: "grid", placeItems: "center" }}>
        <div className="thx-empty">NO REAL LOSS METRICS AVAILABLE YET</div>
      </div>
    );
  }
  const allPoints = series.flatMap((item) => item.points);
  const minStep = Math.min(...allPoints.map((point) => point.step));
  const maxStep = Math.max(...allPoints.map((point) => point.step));
  const minLoss = Math.min(...allPoints.map((point) => point.value));
  const maxLoss = Math.max(...allPoints.map((point) => point.value));
  const stepSpan = Math.max(1, maxStep - minStep);
  const lossPadding = Math.max(0.0001, (maxLoss - minLoss) * 0.12);
  const yMin = minLoss - lossPadding;
  const yMax = maxLoss + lossPadding;
  const ySpan = Math.max(0.0001, yMax - yMin);
  const colors = ["var(--thx-yellow)", "var(--thx-cyan)", "var(--thx-green)", "var(--thx-magenta)", "var(--thx-red)"];
  const width = 100;
  const height = 44;

  function pointToCoord(point: TrainingLossPoint) {
    const x = ((point.step - minStep) / stepSpan) * width;
    const y = height - ((point.value - yMin) / ySpan) * height;
    return { x, y };
  }

  return (
    <>
      <div className="thx-spark" style={{ height: 148 }}>
        <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-label="Live training loss metrics">
          <line x1="0" y1={height} x2={width} y2={height} stroke="rgba(255,255,255,0.12)" strokeWidth="0.4" />
          <line x1="0" y1="0" x2="0" y2={height} stroke="rgba(255,255,255,0.12)" strokeWidth="0.4" />
          {series.map((item, index) => {
            const color = colors[index % colors.length];
            const coords = item.points.map(pointToCoord);
            const path = coords.map((coord, coordIndex) => `${coordIndex === 0 ? "M" : "L"} ${coord.x.toFixed(2)} ${coord.y.toFixed(2)}`).join(" ");
            const latest = coords[coords.length - 1];
            return (
              <g key={item.jobId}>
                <path d={path} fill="none" stroke={color} strokeWidth="1.2" vectorEffect="non-scaling-stroke" />
                {latest && <circle cx={latest.x} cy={latest.y} r="1.4" fill={color} />}
              </g>
            );
          })}
        </svg>
      </div>
      <div className="thx-runs" style={{ marginTop: 8 }}>
        {series.map((item, index) => {
          const latest = item.points[item.points.length - 1];
          const color = colors[index % colors.length];
          return (
            <div className={"thx-run " + jobStatusClass(item.status)} key={item.jobId} style={{ cursor: "default" }}>
              <span className="thx-run-dot" style={{ background: color }} />
              <span className="thx-run-id">{item.label}</span>
              <span className="thx-run-meta">step {latest.step}</span>
              <span className="thx-run-status">loss {formatLossValue(latest.value)}</span>
            </div>
          );
        })}
      </div>
    </>
  );
}

function trainingJobLabel(job: JobRecord): string {
  const payload = job.payload || {};
  return String(payload.output_name || payload.model_slug || job.slug || job.job_id);
}

function firstFiniteNumber(...values: unknown[]): number | null {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
    if (typeof value === "string" && value.trim()) {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) {
        return parsed;
      }
    }
  }
  return null;
}

function formatLossValue(value: number): string {
  if (Math.abs(value) < 0.001 && value !== 0) {
    return value.toExponential(2);
  }
  return value.toFixed(value < 1 ? 4 : 3);
}

function jobStatusClass(status: string): string {
  if (status === "running") return "is-running";
  if (status === "succeeded" || status === "completed") return "is-done";
  if (status === "failed" || status === "error") return "is-failed";
  return "is-queued";
}
