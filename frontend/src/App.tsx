import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  Archive,
  BookOpen,
  ChevronDown,
  ChevronRight,
  CheckCircle,
  Database,
  Download,
  Eye,
  FileText,
  Filter,
  Gauge,
  GitBranch,
  Layers,
  ListChecks,
  LogOut,
  MessageSquare,
  Pause,
  Play,
  RefreshCw,
  Search,
  Shield,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  Upload,
  Wand2,
  XCircle
} from "lucide-react";
import {
  api,
  ArtifactRecord,
  BenchmarkCatalogItem,
  BenchmarkResultRecord,
  CapabilityTransferRecord,
  DatasetRecordsResponse,
  DatasetRecord,
  HubResolvedResource,
  InferenceOption,
  InferenceTarget,
  JobRecord,
  ModelRecord,
  ValidationResult
} from "./api/client";
import { FieldNote } from "./components/FieldNote";
import { JobEvent } from "./components/JobLogPanel";
import { FineTuneWizard } from "./components/FineTuneWizard";
import { DatasetsWizard } from "./components/DatasetsWizard";
import { ChatConsole } from "./components/ChatConsole";
import { CapabilityTransferWizard } from "./components/CapabilityTransferWizard";
import { ActiveTransferPill } from "./components/ActiveTransferPill";
import { ArtifactPicker } from "./components/ArtifactPicker";
import { OnboardingRibbon } from "./components/OnboardingRibbon";
import { ToastLayer, ToastMessage, ToastTone } from "./components/ToastLayer";
import "./styles/cyberpunk.css";

type Page =
  | "dashboard"
  | "datasets"
  | "generate"
  | "benchmarks"
  | "training"
  | "capability-transfer"
  | "quantize"
  | "models"
  | "chat"
  | "cleanup"
  | "knowledge";

type DatasetSource = "huggingface" | "csv" | "url";
type DatasetRouteKey = "overview" | "acquire" | "review" | "detail";
type FormErrors = Record<string, string>;

type AppRoute = {
  page: Page;
  datasetRoute: DatasetRouteKey;
  datasetId?: string;
};

type GpuRecord = {
  index: number;
  name: string;
  memory_total_mb: number;
  memory_used_mb: number;
  utilization_gpu_percent: number;
  temperature_c: number;
};

type DiskRecord = {
  total_bytes: number;
  used_bytes: number;
  free_bytes: number;
};

type GpuSample = {
  timestamp: number;
  utilization: number;
  memoryPercent: number;
  temperature: number;
};

type MetricSummary = {
  key: string;
  label: string;
  value: string;
  series: number[];
};

type JobLifecycle = "queued" | "running" | "terminal";
type JobEventScope = "selected" | "all";
type DashboardJobEvent = JobEvent & { job_id: string };
type LiveMetricTrace = {
  key: string;
  label: string;
  series: number[];
};
type ToastHandler = (message: string, tone?: ToastTone, title?: string) => void;

const navItems: { page: Page; label: string; icon: JSX.Element }[] = [
  { page: "dashboard", label: "Dashboard", icon: <Gauge size={17} /> },
  { page: "datasets", label: "Datasets", icon: <Database size={17} /> },
  { page: "generate", label: "Generate", icon: <Sparkles size={17} /> },
  { page: "benchmarks", label: "Benchmarks", icon: <Activity size={17} /> },
  { page: "training", label: "Training", icon: <GitBranch size={17} /> },
  { page: "capability-transfer", label: "Transfer", icon: <Wand2 size={17} /> },
  { page: "quantize", label: "Quantize", icon: <Archive size={17} /> },
  { page: "models", label: "Models", icon: <Layers size={17} /> },
  { page: "chat", label: "Chat", icon: <MessageSquare size={17} /> },
  { page: "cleanup", label: "Cleanup", icon: <Trash2 size={17} /> },
  { page: "knowledge", label: "Knowledge", icon: <BookOpen size={17} /> }
];

const datasetRouteItems: { route: DatasetRouteKey; label: string; icon: JSX.Element }[] = [
  { route: "overview", label: "Overview", icon: <Gauge size={15} /> },
  { route: "acquire", label: "Acquire", icon: <Download size={15} /> },
  { route: "review", label: "Review", icon: <ListChecks size={15} /> }
];

const pageDescriptions: Record<Page, string> = {
  dashboard: "System status, active jobs, GPU pressure, and recent artifacts.",
  datasets: "Source, validate, approve, and hand off canonical data for training.",
  generate: "Create local review examples with the active inference target.",
  benchmarks: "Run math, knowledge, reasoning, instruction-following, and code suites.",
  training: "Queue LoRA, QLoRA, and full fine-tuning jobs from approved datasets.",
  "capability-transfer": "Transfer latent capabilities at inference time without changing model weights.",
  quantize: "Convert checkpoints to GGUF and prepare quantized artifacts.",
  models: "Download, register, upload, and select local model artifacts.",
  chat: "Prompt the active inference target and inspect streamed model output.",
  cleanup: "Review cleanup manifests before deleting local artifacts.",
  knowledge: "Reference notes for dataset, model, benchmark, and quantization controls."
};

function parseHashRoute(): AppRoute {
  const clean = window.location.hash.replace(/^#\/?/, "").trim();
  const parts = clean.split("/").filter(Boolean);
  const page = navItems.some((item) => item.page === parts[0]) ? (parts[0] as Page) : "dashboard";
  if (page !== "datasets") {
    return { page, datasetRoute: "overview" };
  }
  const route = ["overview", "acquire", "review", "detail"].includes(parts[1]) ? (parts[1] as DatasetRouteKey) : "overview";
  return { page, datasetRoute: route, datasetId: parts[2] };
}

function routePath(page: Page, datasetRoute: DatasetRouteKey = "overview", datasetId?: string) {
  if (page !== "datasets") {
    return `#/${page}`;
  }
  const suffix = datasetRoute === "detail" && datasetId ? `/${datasetId}` : "";
  return `#/datasets/${datasetRoute}${suffix}`;
}

function datasetRouteTitle(route: AppRoute) {
  if (route.page !== "datasets") {
    return navItems.find((item) => item.page === route.page)?.label || "Dashboard";
  }
  if (route.datasetRoute === "acquire") {
    return "Datasets / Acquire";
  }
  if (route.datasetRoute === "review") {
    return "Datasets / Review";
  }
  if (route.datasetRoute === "detail") {
    return "Datasets / Detail";
  }
  return "Datasets / Overview";
}

function datasetRouteDescription(route: AppRoute) {
  if (route.page !== "datasets") {
    return pageDescriptions[route.page];
  }
  if (route.datasetRoute === "acquire") {
    return "Download, upload, or import sources with validation before a job starts.";
  }
  if (route.datasetRoute === "review") {
    return "Filter records, inspect canonical rows, and approve reviewed versions.";
  }
  if (route.datasetRoute === "detail") {
    return "Audit validation, lineage, and review readiness for one dataset version.";
  }
  return pageDescriptions.datasets;
}

export function App() {
  const [user, setUser] = useState<string | null>(null);
  const [route, setRoute] = useState<AppRoute>(() => parseHashRoute());
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [artifacts, setArtifacts] = useState<ArtifactRecord[]>([]);
  const [capabilityTransfers, setCapabilityTransfers] = useState<CapabilityTransferRecord[]>([]);
  const [activeInferenceTarget, setActiveInferenceTarget] = useState<InferenceTarget | null>(null);
  const [inferenceOptions, setInferenceOptions] = useState<InferenceOption[]>([]);
  const [gpus, setGpus] = useState<GpuRecord[]>([]);
  const [disk, setDisk] = useState<DiskRecord | null>(null);
  const [selectedJob, setSelectedJob] = useState<JobRecord | undefined>();
  const [error, setError] = useState<string>("");
  const [liveRefreshEnabled, setLiveRefreshEnabled] = useState(true);
  const [lastUpdatedAt, setLastUpdatedAt] = useState<number | null>(null);
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const page = route.page;

  function pushToast(message: string, tone: ToastTone = "info", title?: string) {
    const id = Date.now() + Math.floor(Math.random() * 1000);
    const nextTitle = title || (tone === "success" ? "Done" : tone === "error" ? "Action failed" : "Status");
    setToasts((current) => [...current, { id, tone, title: nextTitle, message }].slice(-4));
    window.setTimeout(() => {
      setToasts((current) => current.filter((toast) => toast.id !== id));
    }, tone === "error" ? 9000 : 6000);
  }

  function dismissToast(id: number) {
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }

  async function refreshAll() {
    try {
      const [
        me,
        nextModels,
        nextDatasets,
        nextJobs,
        nextArtifacts,
        nextCapabilityTransfers,
        nextInferenceTarget,
        nextInferenceOptions,
        nextGpus,
        nextDisk
      ] = await Promise.all([
        api.get<{ username: string }>("/api/me"),
        api.get<ModelRecord[]>("/api/models"),
        api.get<DatasetRecord[]>("/api/datasets"),
        api.get<JobRecord[]>("/api/jobs"),
        api.get<ArtifactRecord[]>("/api/artifacts"),
        api.capabilityTransfers.list(),
        api.get<InferenceTarget | null>("/api/inference/target"),
        api.get<InferenceOption[]>("/api/inference/options"),
        api.get<GpuRecord[]>("/api/system/gpus"),
        api.get<DiskRecord>("/api/system/disk")
      ]);
      setUser(me.username);
      setModels(nextModels);
      setDatasets(nextDatasets);
      setJobs(nextJobs);
      setArtifacts(nextArtifacts);
      setCapabilityTransfers(nextCapabilityTransfers);
      setActiveInferenceTarget(nextInferenceTarget);
      setInferenceOptions(nextInferenceOptions);
      setGpus(nextGpus);
      setDisk(nextDisk);
      setSelectedJob((current) => nextJobs.find((job) => job.job_id === current?.job_id) || nextJobs[0]);
      setLastUpdatedAt(Date.now());
      setError("");
    } catch (err) {
      setUser(null);
      if (err instanceof Error && !err.message.includes("Login required")) {
        setError(err.message);
      }
    }
  }

  useEffect(() => {
    refreshAll();
    const id = window.setInterval(() => {
      if (user && liveRefreshEnabled) {
        refreshAll();
      }
    }, 5000);
    return () => window.clearInterval(id);
  }, [user, liveRefreshEnabled]);

  useEffect(() => {
    if (!window.location.hash) {
      window.history.replaceState(null, "", routePath("dashboard"));
    }
    const syncRoute = () => setRoute(parseHashRoute());
    window.addEventListener("hashchange", syncRoute);
    return () => window.removeEventListener("hashchange", syncRoute);
  }, []);

  async function logout() {
    await api.post("/api/auth/logout", {});
    setUser(null);
  }

  function navigate(nextPage: Page, datasetRoute: DatasetRouteKey = "overview", datasetId?: string) {
    const nextHash = routePath(nextPage, datasetRoute, datasetId);
    if (window.location.hash === nextHash) {
      setRoute(parseHashRoute());
      return;
    }
    window.location.hash = nextHash;
  }

  if (!user) {
    return <LoginScreen onLogin={refreshAll} error={error} setError={setError} />;
  }

  return (
    <div className="appShell">
      <aside className="sidebar">
        <div className="brand">
          <Shield size={22} />
          <div>
            <strong>TrainingHub</strong>
            <span>Morrigan</span>
          </div>
        </div>
        <nav>
          {navItems.map((item) =>
            item.page === "datasets" ? (
              <div className="navGroup" key={item.page}>
                <button
                  className={page === item.page ? "active" : ""}
                  aria-expanded={page === "datasets"}
                  onClick={() => navigate("datasets", "overview")}
                >
                  {item.icon}
                  <span>{item.label}</span>
                  {page === "datasets" ? <ChevronDown className="navChevron" size={15} /> : <ChevronRight className="navChevron" size={15} />}
                </button>
                {page === "datasets" && (
                  <div className="navNested" aria-label="Dataset navigation">
                    {datasetRouteItems.map((datasetItem) => (
                      <button
                        key={datasetItem.route}
                        className={route.datasetRoute === datasetItem.route ? "active" : ""}
                        onClick={() => navigate("datasets", datasetItem.route)}
                      >
                        {datasetItem.icon}
                        <span>{datasetItem.label}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <button key={item.page} className={page === item.page ? "active" : ""} onClick={() => navigate(item.page)}>
                {item.icon}
                <span>{item.label}</span>
              </button>
            )
          )}
        </nav>
        <button className="logout" onClick={logout}>
          <LogOut size={16} />
          <span>{user}</span>
        </button>
      </aside>
      <main>
        <header className="topbar">
          <div>
            <h1>{datasetRouteTitle(route)}</h1>
            <p>{datasetRouteDescription(route)}</p>
          </div>
          <div className="topbarActions">
            {page === "dashboard" && (
              <button className="thx-btn thx-topbar-action" onClick={() => setLiveRefreshEnabled((enabled) => !enabled)}>
                {liveRefreshEnabled ? <Pause size={16} /> : <Play size={16} />}
                {liveRefreshEnabled ? "Pause Live" : "Resume Live"}
              </button>
            )}
            <button className={page === "dashboard" ? "thx-btn thx-topbar-action" : "secondary"} onClick={refreshAll}>
              <RefreshCw size={16} /> Refresh
            </button>
            <ActiveTransferPill activeInferenceTarget={activeInferenceTarget} transfers={capabilityTransfers} refresh={refreshAll} onToast={pushToast} />
          </div>
        </header>
        {error && <div className="alert">{error}</div>}
        {page === "dashboard" && (
          <Dashboard
            gpus={gpus}
            disk={disk}
            jobs={jobs}
            models={models}
            datasets={datasets}
            artifacts={artifacts}
            activeInferenceTarget={activeInferenceTarget}
            selectedJob={selectedJob}
            setSelectedJob={setSelectedJob}
            liveRefreshEnabled={liveRefreshEnabled}
            lastUpdatedAt={lastUpdatedAt}
          />
        )}
        {page === "datasets" && (
          <DatasetsWizard
            datasets={datasets}
            jobs={jobs}
            activeInferenceTarget={activeInferenceTarget}
            refresh={refreshAll}
            setSelectedJob={setSelectedJob}
            onToast={pushToast}
          />
        )}
        {page === "generate" && (
          <GenerateData artifacts={artifacts} activeInferenceTarget={activeInferenceTarget} refresh={refreshAll} setSelectedJob={setSelectedJob} onToast={pushToast} />
        )}
        {page === "benchmarks" && (
          <Benchmarks
            models={models}
            artifacts={artifacts}
            activeInferenceTarget={activeInferenceTarget}
            refresh={refreshAll}
            setSelectedJob={setSelectedJob}
            onToast={pushToast}
          />
        )}
        {page === "training" && <FineTuneWizard models={models} datasets={datasets} jobs={jobs} artifacts={artifacts} refresh={refreshAll} setSelectedJob={setSelectedJob} />}
        {page === "capability-transfer" && (
          <CapabilityTransferWizard
            models={models}
            datasets={datasets}
            jobs={jobs}
            artifacts={artifacts}
            transfers={capabilityTransfers}
            activeInferenceTarget={activeInferenceTarget}
            refresh={refreshAll}
            setSelectedJob={setSelectedJob}
            onToast={pushToast}
          />
        )}
        {page === "quantize" && <Quantize artifacts={artifacts} refresh={refreshAll} setSelectedJob={setSelectedJob} onToast={pushToast} />}
        {page === "models" && (
          <Models
            models={models}
            artifacts={artifacts}
            activeInferenceTarget={activeInferenceTarget}
            capabilityTransfers={capabilityTransfers}
            inferenceOptions={inferenceOptions}
            refresh={refreshAll}
            setSelectedJob={setSelectedJob}
            onToast={pushToast}
          />
        )}
        {page === "chat" && <ChatConsole activeInferenceTarget={activeInferenceTarget} capabilityTransfers={capabilityTransfers} refresh={refreshAll} onToast={pushToast} />}
        {page === "cleanup" && <Cleanup refresh={refreshAll} setSelectedJob={setSelectedJob} onToast={pushToast} />}
        {page === "knowledge" && <KnowledgeBase />}
      </main>
      <ToastLayer toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}

function LoginScreen({ onLogin, error, setError }: { onLogin: () => void; error: string; setError: (value: string) => void }) {
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");

  async function submit(event: FormEvent) {
    event.preventDefault();
    try {
      await api.post("/api/auth/login", { username, password });
      await onLogin();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed.");
    }
  }

  return (
    <main className="loginScreen thx thx-login">
      <form onSubmit={submit} className="loginPanel thx-panel thx-login-panel">
        <div className="thx-login-mark">
          <Shield size={28} />
        </div>
        <div>
          <div className="crumb">MORRIGAN · AUTH</div>
          <h1>TrainingHub</h1>
        </div>
        <label className="thx-field">
          <span className="thx-field-label">
            <span>Username</span>
            <span className="v">{username}</span>
          </span>
          <input value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="username" />
        </label>
        <label className="thx-field">
          <span className="thx-field-label">
            <span>Password</span>
            <span className="v">{password ? "set" : "empty"}</span>
          </span>
          <input type="password" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete="current-password" />
        </label>
        {error && <div className="alert">{error}</div>}
        <button type="submit" className="thx-btn thx-btn--primary">
          <Shield size={16} /> Login
        </button>
      </form>
    </main>
  );
}

const MODEL_DOWNLOAD_JOB_TYPES = new Set(["model_download"]);
const DATASET_IMPORT_JOB_TYPES = new Set(["dataset_import"]);
const UPLOADABLE_ARTIFACT_TYPES = new Set([
  "gguf_quantized",
  "downloaded_model",
  "training_checkpoint",
  "training_merged_checkpoint",
  "training_adapter"
]);
const GENERATE_TEACHER_ARTIFACT_TYPES = new Set([
  "gguf_quantized",
  "downloaded_model",
  "training_checkpoint",
  "training_merged_checkpoint"
]);
const EVENT_BUFFER_LIMIT = 200;
const JOB_EVENT_NAMES = [
  "queued",
  "started",
  "inference_shutdown",
  "worker_start",
  "command",
  "command_complete",
  "metric",
  "artifact",
  "succeeded",
  "failed",
  "cancelled",
  "worker_error",
  "fallback",
  "download",
  "model_downloaded",
  "dataset_imported",
  "cleaning",
  "raw_cleanup",
  "training_start",
  "training_complete",
  "cleanup",
  "cleanup_skip",
  "hf_repo",
  "hf_upload",
  "model_uploaded"
];

function isTrainingJob(job: JobRecord): boolean {
  return job.job_type.startsWith("train_");
}

function isModelDownloadJob(job: JobRecord): boolean {
  return MODEL_DOWNLOAD_JOB_TYPES.has(job.job_type);
}

function isDatasetImportJob(job: JobRecord): boolean {
  return DATASET_IMPORT_JOB_TYPES.has(job.job_type);
}

function isDataTransferJob(job: JobRecord): boolean {
  return isModelDownloadJob(job) || isDatasetImportJob(job);
}

function Dashboard({
  gpus,
  disk,
  jobs,
  models,
  datasets,
  artifacts,
  activeInferenceTarget,
  selectedJob,
  setSelectedJob,
  liveRefreshEnabled,
  lastUpdatedAt
}: {
  gpus: GpuRecord[];
  disk: DiskRecord | null;
  jobs: JobRecord[];
  models: ModelRecord[];
  datasets: DatasetRecord[];
  artifacts: ArtifactRecord[];
  activeInferenceTarget: InferenceTarget | null;
  selectedJob?: JobRecord;
  setSelectedJob: (job: JobRecord) => void;
  liveRefreshEnabled: boolean;
  lastUpdatedAt: number | null;
}) {
  const [gpuSamples, setGpuSamples] = useState<Record<number, GpuSample[]>>({});
  const [now, setNow] = useState(() => new Date());
  const [eventScope, setEventScope] = useState<JobEventScope>("selected");
  const queuedJobs = jobs.filter((job) => job.status === "queued");
  const runningJobs = jobs.filter((job) => job.status === "running");
  const { selectedEvents: jobEvents, allRunningEvents } = useJobEventStreams(jobs, selectedJob, liveRefreshEnabled);
  const metricSummary = useMemo(() => extractMetricSummary(jobEvents), [jobEvents]);
  const terminalJobs = jobs.filter((job) => ["succeeded", "failed", "cancelled"].includes(job.status));
  const trainingRuns = useMemo(() => jobs.filter(isTrainingJob).slice(0, 12), [jobs]);
  const modelDownloads = useMemo(
    () => jobs.filter(isModelDownloadJob).slice(0, 8),
    [jobs]
  );
  const datasetImports = useMemo(
    () => jobs.filter(isDatasetImportJob).slice(0, 8),
    [jobs]
  );
  const liveTransfers = [...modelDownloads, ...datasetImports].filter(
    (job) => job.status === "running" || job.status === "queued"
  );
  const gpuPressure = gpus.length ? Math.round(gpus.reduce((total, gpu) => total + gpu.utilization_gpu_percent, 0) / gpus.length) : 0;
  const vramPressure = useMemo(() => {
    if (gpus.length === 0) return 0;
    const total = gpus.reduce((sum, gpu) => sum + gpu.memory_total_mb, 0);
    const used = gpus.reduce((sum, gpu) => sum + gpu.memory_used_mb, 0);
    return total > 0 ? Math.round((used / total) * 100) : 0;
  }, [gpus]);
  const diskUsedPercent = disk ? Math.round((disk.used_bytes / disk.total_bytes) * 100) : 0;

  useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    if (gpus.length === 0) {
      return;
    }
    const timestamp = Date.now();
    setGpuSamples((current) => {
      const next = { ...current };
      gpus.forEach((gpu) => {
        const memoryPercent = gpu.memory_total_mb > 0 ? Math.round((gpu.memory_used_mb / gpu.memory_total_mb) * 100) : 0;
        const sample = {
          timestamp,
          utilization: gpu.utilization_gpu_percent,
          memoryPercent,
          temperature: gpu.temperature_c
        };
        next[gpu.index] = [...(next[gpu.index] || []), sample].slice(-36);
      });
      return next;
    });
  }, [gpus]);

  const inferenceLabel = activeInferenceTarget?.display_name || "—";
  const inferenceSubtitle = activeInferenceTarget ? inferenceTargetSubtitle(activeInferenceTarget) : "no local target";

  return (
    <div className="thx thx-dash">
      <div className="thx-dash-shell">
        <div className="thx-stage-h thx-dash-h">
          <div>
            <div className="crumb">OPS · DECK · 00 · TELEMETRY</div>
            <h2>
              <span className="thx-glitch" data-text="MISSION CONTROL">
                MISSION CONTROL
              </span>
            </h2>
            <p className="lede">
              Server pulse, model and dataset transfers, and live training telemetry — every active worker
              piped through one console.
            </p>
          </div>
          <div className="stamp">
            UTC&nbsp;//&nbsp;{now.toISOString().slice(0, 10)}
            <span>{now.toISOString().slice(11, 19)} · OPERATOR</span>
          </div>
        </div>

        <div className={`thx-dash-link ${liveRefreshEnabled ? "is-live" : "is-paused"}`}>
          <span className="thx-dot" style={{ color: liveRefreshEnabled ? "var(--thx-green)" : "var(--thx-text-dim)" }} />
          <span className="k">{liveRefreshEnabled ? "[ NETLINK · STREAMING ]" : "[ NETLINK · PAUSED ]"}</span>
          <span className="v">{formatRefreshTime(lastUpdatedAt)}</span>
          <span className="thx-dash-link-bar" aria-hidden="true">
            <span className="thx-dash-link-pulse" />
          </span>
        </div>

        <OnboardingRibbon models={models} datasets={datasets} activeInferenceTarget={activeInferenceTarget} />

        <div className="thx-mons thx-dash-mons">
          <DashMon
            label="ACTIVE"
            value={pad2(runningJobs.length)}
            sub={`${queuedJobs.length} queued · ${terminalJobs.length} terminal`}
            tone="yl"
            pulse={runningJobs.length > 0}
          />
          <DashMon
            label="TRANSFERS"
            value={pad2(liveTransfers.length)}
            sub={`${modelDownloads.length} model · ${datasetImports.length} data`}
            tone="cy"
            pulse={liveTransfers.length > 0}
          />
          <DashMon
            label="GPU LOAD"
            value={gpus.length ? `${gpuPressure}%` : "—"}
            sub={gpus.length ? `${gpus.length} device${gpus.length === 1 ? "" : "s"}` : "no telemetry"}
            tone={pressureToneTone(gpuPressure)}
            pulse={gpuPressure > 60}
          />
          <DashMon
            label="VRAM"
            value={gpus.length ? `${vramPressure}%` : "—"}
            sub={gpus.length ? `${formatVram(gpus)}` : "no telemetry"}
            tone={pressureToneTone(vramPressure)}
          />
          <DashMon
            label="DISK FREE"
            value={disk ? formatBytes(disk.free_bytes) : "—"}
            sub={disk ? `${diskUsedPercent}% used` : "unavailable"}
            tone={diskUsedPercent >= 90 ? "rd" : diskUsedPercent >= 75 ? "yl" : "gr"}
          />
          <DashMon
            label="INFERENCE"
            value={inferenceLabel}
            sub={inferenceSubtitle}
            tone={activeInferenceTarget ? "gr" : "rd"}
            mono
          />
        </div>

        <div className="thx-dash-grid">
          <section className="thx-dash-col thx-dash-col--server">
            <DashHardwarePanel gpus={gpus} samples={gpuSamples} jobs={jobs} />
            <DashSystemPanel
              disk={disk}
              diskUsedPercent={diskUsedPercent}
              activeInferenceTarget={activeInferenceTarget}
              gpus={gpus}
            />
          </section>

          <section className="thx-dash-col thx-dash-col--training">
            <DashTrainingPanel
              jobs={trainingRuns}
              selectedJob={selectedJob}
              setSelectedJob={setSelectedJob}
            />
            <DashSelectedRunPanel job={selectedJob} events={jobEvents} now={now} />
            <DashLiveMetricsPanel metrics={metricSummary} events={jobEvents} />
            <DashEventStreamPanel
              job={selectedJob}
              selectedEvents={jobEvents}
              allRunningEvents={allRunningEvents}
              runningJobCount={runningJobs.length}
              eventScope={eventScope}
              setEventScope={setEventScope}
              liveRefreshEnabled={liveRefreshEnabled}
            />
          </section>

          <section className="thx-dash-col thx-dash-col--ops">
            <DashTransferPanel
              title="Model Downloads"
              tag="[ 03A · MODEL ]"
              emptyMessage="NO MODEL DOWNLOAD WORKERS · QUEUE FROM MODELS PAGE"
              jobs={modelDownloads}
              selectedJob={selectedJob}
              setSelectedJob={setSelectedJob}
              events={jobEvents}
              now={now}
            />
            <DashTransferPanel
              title="Dataset Imports"
              tag="[ 03B · DATA ]"
              emptyMessage="NO DATASET IMPORT WORKERS · QUEUE FROM DATASETS PAGE"
              jobs={datasetImports}
              selectedJob={selectedJob}
              setSelectedJob={setSelectedJob}
              events={jobEvents}
              now={now}
            />
            <DashArtifactsPanel artifacts={artifacts} />
          </section>
        </div>
      </div>
    </div>
  );
}

function DashMon({
  label,
  value,
  sub,
  tone,
  pulse,
  mono
}: {
  label: string;
  value: string;
  sub: string;
  tone: "yl" | "cy" | "gr" | "rd";
  pulse?: boolean;
  mono?: boolean;
}) {
  return (
    <div className={`thx-mon thx-dash-mon ${pulse ? "is-pulse" : ""}`}>
      <div className="k">[ {label} ]</div>
      <div className={`v ${tone} ${mono ? "thx-dash-mon-mono" : ""}`}>{value}</div>
      <div className="thx-dash-mon-sub">{sub}</div>
    </div>
  );
}

function DashHardwarePanel({
  gpus,
  samples,
  jobs
}: {
  gpus: GpuRecord[];
  samples: Record<number, GpuSample[]>;
  jobs: JobRecord[];
}) {
  const runningJobs = jobs.filter((job) => job.status === "running");
  return (
    <div className="thx-panel thx-dash-panel">
      <div className="thx-panel-h">
        <h3>Hardware Telemetry</h3>
        <span className="thx-tag">[ 01A · GPU SOCKETS ]</span>
      </div>
      {gpus.length === 0 ? (
        <div className="thx-empty">NO NVIDIA GPU TELEMETRY · NVIDIA-SMI UNREACHABLE</div>
      ) : (
        <div className="thx-dash-gpus">
          {gpus.map((gpu) => {
            const assignedJob = runningJobs.find((job) => job.gpu_ids.includes(gpu.index));
            return <DashGpuCard gpu={gpu} samples={samples[gpu.index] || []} assignedJob={assignedJob} key={gpu.index} />;
          })}
        </div>
      )}
    </div>
  );
}

function DashGpuCard({ gpu, samples, assignedJob }: { gpu: GpuRecord; samples: GpuSample[]; assignedJob?: JobRecord }) {
  const memoryPercent = gpu.memory_total_mb > 0 ? Math.round((gpu.memory_used_mb / gpu.memory_total_mb) * 100) : 0;
  const tempPercent = clampPercent(gpu.temperature_c);
  return (
    <article className={`thx-dash-gpu ${assignedJob ? "is-leased" : ""}`}>
      <div className="thx-dash-gpu-h">
        <div className="thx-dash-gpu-id">
          <span>GPU</span>
          <strong>{String(gpu.index).padStart(2, "0")}</strong>
        </div>
        <div className="thx-dash-gpu-name">
          <span className="k">// SOCKET</span>
          <span className="v">{gpu.name}</span>
        </div>
        <span className={`thx-cap ${assignedJob ? "thx-cap--w" : "thx-cap--c"}`}>
          {assignedJob ? "LEASED" : "IDLE"}
        </span>
      </div>
      <DashSparkline samples={samples} />
      <div className="thx-dash-meters">
        <DashMeter label="UTIL" value={gpu.utilization_gpu_percent} detail={`${gpu.utilization_gpu_percent}%`} tone={pressureToneTone(gpu.utilization_gpu_percent)} />
        <DashMeter label="VRAM" value={memoryPercent} detail={`${gpu.memory_used_mb}/${gpu.memory_total_mb}MB`} tone={pressureToneTone(memoryPercent)} />
        <DashMeter label="TEMP" value={tempPercent} detail={`${gpu.temperature_c}°C`} tone={tempTone(gpu.temperature_c)} />
      </div>
      <div className="thx-dash-gpu-lease">
        <span className="k">// ASSIGNED</span>
        <span className="v">{assignedJob?.job_id || "—"}</span>
      </div>
    </article>
  );
}

function DashMeter({ label, value, detail, tone }: { label: string; value: number; detail: string; tone: "yl" | "cy" | "gr" | "rd" }) {
  const cappedValue = clampPercent(value);
  return (
    <div className="thx-dash-meter">
      <div className="thx-dash-meter-h">
        <span>[ {label} ]</span>
        <strong>{detail}</strong>
      </div>
      <div className={`thx-dash-meter-bar tone-${tone}`} aria-hidden="true">
        <span style={{ width: `${cappedValue}%` }} />
      </div>
    </div>
  );
}

function DashSparkline({ samples }: { samples: GpuSample[] }) {
  if (samples.length < 2) {
    return <div className="thx-dash-spark thx-dash-spark--empty">// COLLECTING SAMPLES</div>;
  }
  const utilPoints = samples
    .map((sample, index) => {
      const x = (index / (samples.length - 1)) * 100;
      const y = 32 - (clampPercent(sample.utilization) / 100) * 28;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
  const memPoints = samples
    .map((sample, index) => {
      const x = (index / (samples.length - 1)) * 100;
      const y = 32 - (clampPercent(sample.memoryPercent) / 100) * 28;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
  return (
    <div className="thx-dash-spark">
      <svg viewBox="0 0 100 36" preserveAspectRatio="none" aria-hidden="true">
        <polyline className="mem" points={memPoints} />
        <polyline className="util" points={utilPoints} />
      </svg>
      <div className="thx-dash-spark-axis">
        <span>UTIL</span>
        <span>VRAM</span>
      </div>
    </div>
  );
}

function DashSystemPanel({
  disk,
  diskUsedPercent,
  activeInferenceTarget,
  gpus
}: {
  disk: DiskRecord | null;
  diskUsedPercent: number;
  activeInferenceTarget: InferenceTarget | null;
  gpus: GpuRecord[];
}) {
  const totalVram = gpus.reduce((sum, g) => sum + g.memory_total_mb, 0);
  const usedVram = gpus.reduce((sum, g) => sum + g.memory_used_mb, 0);
  return (
    <div className="thx-panel thx-dash-panel">
      <div className="thx-panel-h">
        <h3>System Pulse</h3>
        <span className="thx-tag">[ 01B · STORAGE / RUNTIME ]</span>
      </div>
      <div className="thx-dash-sys">
        <div className="thx-dash-sys-block">
          <div className="thx-dash-sys-label">[ DATA ROOT ]</div>
          {disk ? (
            <>
              <div className="thx-dash-sys-num">{formatBytes(disk.free_bytes)}<span> free</span></div>
              <div className={`thx-dash-meter-bar tone-${diskUsedPercent >= 90 ? "rd" : diskUsedPercent >= 75 ? "yl" : "gr"}`} aria-hidden="true">
                <span style={{ width: `${diskUsedPercent}%` }} />
              </div>
              <div className="thx-dash-sys-meta">
                <span>USED · {formatBytes(disk.used_bytes)}</span>
                <span>TOTAL · {formatBytes(disk.total_bytes)}</span>
              </div>
            </>
          ) : (
            <div className="thx-dash-sys-meta">// telemetry unavailable</div>
          )}
        </div>
        <div className="thx-dash-sys-block">
          <div className="thx-dash-sys-label">[ VRAM POOL ]</div>
          {gpus.length ? (
            <>
              <div className="thx-dash-sys-num">
                {(usedVram / 1024).toFixed(1)}<span>/{(totalVram / 1024).toFixed(1)} GB</span>
              </div>
              <div className="thx-dash-sys-meta">
                <span>{gpus.length} DEVICE{gpus.length === 1 ? "" : "S"}</span>
                <span>UNIFIED · POOL</span>
              </div>
            </>
          ) : (
            <div className="thx-dash-sys-meta">// no GPU telemetry</div>
          )}
        </div>
        <div className="thx-dash-sys-block thx-dash-sys-block--wide">
          <div className="thx-dash-sys-label">[ ACTIVE INFERENCE TARGET ]</div>
          {activeInferenceTarget ? (
            <>
              <div className="thx-dash-sys-num thx-dash-sys-num--name">{activeInferenceTarget.display_name}</div>
              <div className="thx-dash-sys-meta">
                <span className="thx-cap thx-cap--c">{activeInferenceTarget.target_type === "base_model" ? "BASE" : "GGUF"}</span>
                <span>{inferenceTargetSubtitle(activeInferenceTarget)}</span>
              </div>
            </>
          ) : (
            <div className="thx-dash-sys-meta">// no local target selected · MODELS PAGE</div>
          )}
        </div>
      </div>
    </div>
  );
}

function DashTrainingPanel({
  jobs,
  selectedJob,
  setSelectedJob
}: {
  jobs: JobRecord[];
  selectedJob?: JobRecord;
  setSelectedJob: (job: JobRecord) => void;
}) {
  const queued = jobs.filter((j) => getJobLifecycle(j) === "queued");
  const running = jobs.filter((j) => getJobLifecycle(j) === "running");
  const terminal = jobs.filter((j) => getJobLifecycle(j) === "terminal");
  return (
    <div className="thx-panel thx-dash-panel">
      <div className="thx-panel-h">
        <h3>Live Training</h3>
        <span className="thx-tag">[ 02A · WORKER LANES ]</span>
      </div>
      <div className="thx-dash-pipe">
        <DashPipeNode label="QUEUED" count={queued.length} state="queued" />
        <span className={`thx-dash-pipe-edge ${running.length ? "is-flow" : ""}`} />
        <DashPipeNode label="RUNNING" count={running.length} state="running" />
        <span className={`thx-dash-pipe-edge ${terminal.length ? "is-flow" : ""}`} />
        <DashPipeNode label="TERMINAL" count={terminal.length} state="terminal" />
      </div>
      {jobs.length === 0 ? (
        <div className="thx-empty">NO TRAINING WORKERS REGISTERED</div>
      ) : (
        <div className="thx-dash-runs">
          {jobs.map((job) => (
            <DashRunRow
              key={job.job_id}
              job={job}
              selected={selectedJob?.job_id === job.job_id}
              onSelect={() => setSelectedJob(job)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function DashPipeNode({ label, count, state }: { label: string; count: number; state: "queued" | "running" | "terminal" }) {
  const stateClass = count === 0 ? "" : state === "running" ? "is-active" : state === "terminal" ? "is-done" : "is-queue";
  return (
    <div className={`thx-dash-pipe-node ${stateClass}`}>
      <div className="thx-dash-pipe-ring">
        <strong>{pad2(count)}</strong>
      </div>
      <span className="thx-dash-pipe-label">{label}</span>
    </div>
  );
}

function DashRunRow({ job, selected, onSelect }: { job: JobRecord; selected: boolean; onSelect: () => void }) {
  const lifecycle = getJobLifecycle(job);
  const cls = lifecycle === "running" ? "is-running" : lifecycle === "queued" ? "is-queued" : job.status === "succeeded" ? "is-done" : job.status === "failed" ? "is-failed" : "is-terminal";
  return (
    <button
      type="button"
      className={`thx-run thx-dash-run ${cls} ${selected ? "is-selected" : ""}`}
      onClick={onSelect}
    >
      <span className="thx-run-dot" />
      <span className="thx-run-id">{job.job_id}</span>
      <span className="thx-run-meta">{job.job_type}</span>
      <span className="thx-run-status">{job.status} · {formatElapsedTime(job)}</span>
    </button>
  );
}

function DashSelectedRunPanel({ job, events, now }: { job?: JobRecord; events: JobEvent[]; now: Date }) {
  if (!job) {
    return (
      <div className="thx-panel thx-dash-panel">
        <div className="thx-panel-h">
          <h3>Selected Run</h3>
          <span className="thx-tag">[ 02B · FOCUS ]</span>
        </div>
        <div className="thx-empty">SELECT A RUN ABOVE TO PIPE INTO TELEMETRY</div>
      </div>
    );
  }
  const payloadItems = payloadHighlights(job.payload).slice(0, 6);
  const lastEvent = events[events.length - 1];
  const elapsed = formatElapsedTimeAt(job, now);
  return (
    <div className="thx-panel thx-dash-panel">
      <div className="thx-panel-h">
        <h3>Selected Run</h3>
        <span className="thx-tag">[ 02B · FOCUS ]</span>
      </div>
      <div className="thx-dash-focus">
        <div className="thx-dash-focus-h">
          <span className={`thx-cap ${capToneFor(job.status)}`}>{job.status.toUpperCase()}</span>
          <strong>{job.job_id}</strong>
          <span className="thx-dash-focus-type">{job.job_type}</span>
        </div>
        <p className="thx-dash-focus-msg">
          {job.terminal_message || lastEvent?.message || "// awaiting worker handshake"}
        </p>
        <div className="thx-dash-focus-grid">
          <DashFact k="ELAPSED" v={elapsed} />
          <DashFact k="WORKER" v={job.worker_pid ? `PID ${job.worker_pid}` : "pending"} />
          <DashFact k="GPU" v={job.gpu_ids.length ? job.gpu_ids.map((g) => `#${g}`).join(" ") : "—"} />
          <DashFact k="CREATED" v={formatEpochTime(job.created_at)} />
          {payloadItems.map((item) => (
            <DashFact k={labelizeKey(item.key).toUpperCase()} v={item.value} key={item.key} />
          ))}
        </div>
      </div>
    </div>
  );
}

function DashFact({ k, v }: { k: string; v: string }) {
  return (
    <div className="thx-dash-fact">
      <span className="k">[ {k} ]</span>
      <span className="v">{v}</span>
    </div>
  );
}

function DashLiveMetricsPanel({ metrics, events }: { metrics: MetricSummary[]; events: JobEvent[] }) {
  const sparkRef = useRef<HTMLCanvasElement | null>(null);
  const liveTrace = useMemo(() => selectLiveMetricTrace(events), [events]);
  const metricCount = events.filter((e) => e.event_type === "metric").length;

  useEffect(() => {
    const canvas = sparkRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const series = liveTrace?.series || [];
    if (series.length < 2) return;
    const min = Math.min(...series);
    const max = Math.max(...series);
    const range = max - min || 1;
    ctx.strokeStyle = "rgba(252, 238, 10, 0.85)";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    series.forEach((v, i) => {
      const x = (i / (series.length - 1)) * w;
      const y = h - ((v - min) / range) * (h - 4) - 2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.shadowColor = "rgba(252, 238, 10, 0.55)";
    ctx.shadowBlur = 6;
    ctx.stroke();
    ctx.shadowBlur = 0;
  }, [liveTrace]);

  return (
    <div className="thx-panel thx-dash-panel">
      <div className="thx-panel-h">
        <h3>Live Metrics</h3>
        <span className="thx-tag">[ 02C · TELEMETRY · {pad2(metricCount)} ]</span>
      </div>
      <div className="thx-dash-loss">
        <div className="thx-dash-loss-h">
          <span>[ {liveTrace ? liveTrace.label.toUpperCase() : "METRIC"} · LIVE TRACE ]</span>
          <span className="v">
            {liveTrace?.series.length ? formatMetricValue(liveTrace.series[liveTrace.series.length - 1]) : "—"}
          </span>
        </div>
        <div className="thx-spark thx-dash-loss-canvas">
          <canvas ref={sparkRef} style={{ width: "100%", height: "100%" }} />
        </div>
      </div>
      {metrics.length === 0 ? (
        <div className="thx-empty">NO METRIC EVENTS · STREAM STARTS WHEN WORKER EMITS</div>
      ) : (
        <div className="thx-dash-metrics">
          {metrics.map((metric) => (
            <article className="thx-dash-metric" key={metric.key}>
              <div className="k">[ {metric.label.toUpperCase()} ]</div>
              <div className="v">{metric.value}</div>
              <DashMetricSpark values={metric.series} />
            </article>
          ))}
        </div>
      )}
    </div>
  );
}

function DashMetricSpark({ values }: { values: number[] }) {
  if (values.length < 2) {
    return <div className="thx-dash-metric-spark thx-dash-metric-spark--empty">// awaiting trend</div>;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const points = values
    .map((value, index) => {
      const x = (index / (values.length - 1)) * 100;
      const y = 28 - ((value - min) / range) * 24;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
  return (
    <svg className="thx-dash-metric-spark" viewBox="0 0 100 32" preserveAspectRatio="none" aria-hidden="true">
      <polyline points={points} />
    </svg>
  );
}

function DashEventStreamPanel({
  job,
  selectedEvents,
  allRunningEvents,
  runningJobCount,
  eventScope,
  setEventScope,
  liveRefreshEnabled
}: {
  job?: JobRecord;
  selectedEvents: DashboardJobEvent[];
  allRunningEvents: DashboardJobEvent[];
  runningJobCount: number;
  eventScope: JobEventScope;
  setEventScope: (scope: JobEventScope) => void;
  liveRefreshEnabled: boolean;
}) {
  const events = eventScope === "all" ? allRunningEvents : selectedEvents;
  const visibleEvents = events.slice(-60);
  return (
    <div className="thx-panel thx-dash-panel">
      <div className="thx-panel-h">
        <h3>Event Stream</h3>
        <span className="thx-tag">
          [ 02D · {liveRefreshEnabled ? "STREAMING" : "PAUSED"} · {pad2(visibleEvents.length)} ]
        </span>
      </div>
      <div className="thx-dash-event-toggle" role="group" aria-label="Event stream scope">
        <button
          type="button"
          className={eventScope === "selected" ? "is-active" : ""}
          onClick={() => setEventScope("selected")}
          disabled={!job}
        >
          selected
        </button>
        <button
          type="button"
          className={eventScope === "all" ? "is-active" : ""}
          onClick={() => setEventScope("all")}
          disabled={runningJobCount === 0}
        >
          all running
        </button>
      </div>
      {!job && eventScope === "selected" ? (
        <div className="thx-empty">SELECT A RUN TO ATTACH SSE STREAM</div>
      ) : (
        <div className="thx-dash-log">
          {visibleEvents.length === 0 ? (
            <div className="thx-dash-log-line l-i">
              <span className="ts">--:--:--</span> <span className="thx-dash-log-type">[ idle ]</span> awaiting worker events
            </div>
          ) : (
            visibleEvents.map((event, index) => (
              <div key={`${event.job_id}-${event.id || event.created_at || index}-${event.event_type}`} className={`thx-dash-log-line l-${eventLevelClass(event.level)}`}>
                <span className="ts">{event.created_at ? new Date(event.created_at * 1000).toISOString().slice(11, 19) : "--:--:--"}</span>{" "}
                {eventScope === "all" && <span className="thx-dash-log-job">[ {event.job_id} ]</span>}{" "}
                <span className="thx-dash-log-type">[ {event.event_type} ]</span> {event.message}
              </div>
            ))
          )}
        </div>
      )}
      <div className="thx-dash-event-footer">
        <span>[ SELECTED · {pad2(selectedEvents.length)} ]</span>
        <span>[ RUNNING · {pad2(runningJobCount)} ]</span>
        <span>[ GLOBAL · {pad2(allRunningEvents.length)} ]</span>
      </div>
    </div>
  );
}

function DashTransferPanel({
  title,
  tag,
  emptyMessage,
  jobs,
  selectedJob,
  setSelectedJob,
  events,
  now
}: {
  title: string;
  tag: string;
  emptyMessage: string;
  jobs: JobRecord[];
  selectedJob?: JobRecord;
  setSelectedJob: (job: JobRecord) => void;
  events: JobEvent[];
  now: Date;
}) {
  const selectedTransferBytes = useMemo(() => {
    if (!selectedJob || !isDataTransferJob(selectedJob)) return undefined;
    for (let i = events.length - 1; i >= 0; i--) {
      const ev = events[i];
      if (ev.event_type === "metric" && ev.data && typeof (ev.data as Record<string, unknown>).bytes_downloaded === "number") {
        return (ev.data as Record<string, unknown>).bytes_downloaded as number;
      }
    }
    return undefined;
  }, [selectedJob, events]);
  return (
    <div className="thx-panel thx-dash-panel">
      <div className="thx-panel-h">
        <h3>{title}</h3>
        <span className="thx-tag">{tag}</span>
      </div>
      {jobs.length === 0 ? (
        <div className="thx-empty">{emptyMessage}</div>
      ) : (
        <div className="thx-dash-dl">
          {jobs.map((job) => {
            const selected = selectedJob?.job_id === job.job_id;
            const bytes = selected ? selectedTransferBytes : undefined;
            return (
              <DashTransferRow
                key={job.job_id}
                job={job}
                selected={selected}
                onSelect={() => setSelectedJob(job)}
                bytes={bytes}
                now={now}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}

function DashTransferRow({
  job,
  selected,
  onSelect,
  bytes,
  now
}: {
  job: JobRecord;
  selected: boolean;
  onSelect: () => void;
  bytes?: number;
  now: Date;
}) {
  const lifecycle = getJobLifecycle(job);
  const cls = lifecycle === "running" ? "is-running" : lifecycle === "queued" ? "is-queued" : job.status === "succeeded" ? "is-done" : job.status === "failed" ? "is-failed" : "is-terminal";
  const sourceLabel = transferSourceLabel(job);
  const elapsed = formatElapsedTimeAt(job, now);
  return (
    <button
      type="button"
      className={`thx-dash-dl-row ${cls} ${selected ? "is-selected" : ""}`}
      onClick={onSelect}
    >
      <div className="thx-dash-dl-h">
        <span className="thx-run-dot" />
        <strong className="thx-dash-dl-name">{transferDisplayName(job)}</strong>
        <span className={`thx-cap ${transferCapTone(job)}`}>{sourceLabel}</span>
      </div>
      <div className={`thx-dash-dl-bar ${cls}`} aria-hidden="true">
        <span className="thx-dash-dl-fill" />
      </div>
      <div className="thx-dash-dl-meta">
        <span className="k">[ STATUS ]</span>
        <span className="v">{job.status}</span>
        <span className="k">[ ELAPSED ]</span>
        <span className="v">{elapsed}</span>
        {typeof bytes === "number" && (
          <>
            <span className="k">[ FETCHED ]</span>
            <span className="v thx-yellow">{formatBytes(bytes)}</span>
          </>
        )}
      </div>
    </button>
  );
}

function DashArtifactsPanel({ artifacts }: { artifacts: ArtifactRecord[] }) {
  const recent = artifacts.slice(0, 8);
  return (
    <div className="thx-panel thx-dash-panel">
      <div className="thx-panel-h">
        <h3>Recent Artifacts</h3>
        <span className="thx-tag">[ 03C · OUTPUTS ]</span>
      </div>
      {recent.length === 0 ? (
        <div className="thx-empty">NO ARTIFACTS REGISTERED</div>
      ) : (
        <div className="thx-dash-artifacts">
          {recent.map((artifact) => (
            <div className="thx-dash-artifact" key={artifact.artifact_id}>
              <div className="thx-dash-artifact-h">
                <strong>{artifact.display_name}</strong>
                <span className="thx-cap thx-cap--c">{artifact.artifact_type}</span>
              </div>
              <div className="thx-dash-artifact-path">{artifact.path}</div>
              <div className="thx-dash-artifact-meta">
                <span>{formatBytes(artifact.size_bytes)}</span>
                <span>· {formatEpochTime(artifact.created_at)}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function pad2(n: number) {
  return String(n).padStart(2, "0");
}

function pressureToneTone(value: number): "yl" | "cy" | "gr" | "rd" {
  if (value >= 90) return "rd";
  if (value >= 70) return "yl";
  if (value > 0) return "gr";
  return "cy";
}

function tempTone(value: number): "yl" | "cy" | "gr" | "rd" {
  if (value >= 85) return "rd";
  if (value >= 75) return "yl";
  return "gr";
}

function capToneFor(status: string): string {
  if (status === "running") return "thx-cap--w";
  if (status === "succeeded") return "thx-cap--ok";
  if (status === "failed") return "thx-cap--no";
  if (status === "queued") return "thx-cap--c";
  return "";
}

function eventLevelClass(level: string): string {
  if (level === "error") return "e";
  if (level === "warning") return "w";
  if (level === "success") return "ok";
  return "i";
}

function formatVram(gpus: GpuRecord[]): string {
  const total = gpus.reduce((sum, g) => sum + g.memory_total_mb, 0);
  const used = gpus.reduce((sum, g) => sum + g.memory_used_mb, 0);
  return `${(used / 1024).toFixed(1)}/${(total / 1024).toFixed(1)} GB`;
}

function transferDisplayName(job: JobRecord): string {
  const payload = job.payload || {};
  const candidate =
    (payload.display_name as string) ||
    (payload.repo_id as string) ||
    (payload.url as string) ||
    (payload.slug as string) ||
    job.slug ||
    job.job_id;
  return String(candidate);
}

function transferSourceLabel(job: JobRecord): string {
  const payload = job.payload || {};
  if (job.job_type === "dataset_import") return "DATASET";
  const sourceType = (payload.source_type as string) || "";
  if (sourceType === "hf") return "HF";
  if (sourceType === "url") return "URL";
  return "MODEL";
}

function transferCapTone(job: JobRecord): string {
  const label = transferSourceLabel(job);
  if (label === "DATASET") return "thx-cap--c";
  if (label === "HF") return "thx-cap--w";
  if (label === "URL") return "thx-cap--ok";
  return "";
}

function useJobEventStreams(jobs: JobRecord[], selectedJob: JobRecord | undefined, enabled: boolean) {
  const [eventsByJobId, setEventsByJobId] = useState<Record<string, DashboardJobEvent[]>>({});
  const runningJobIds = useMemo(() => jobs.filter((job) => job.status === "running").map((job) => job.job_id).sort(), [jobs]);
  const streamJobIds = useMemo(() => {
    const ids = new Set(runningJobIds);
    if (selectedJob) {
      ids.add(selectedJob.job_id);
    }
    return Array.from(ids).sort();
  }, [runningJobIds, selectedJob?.job_id]);
  const streamJobIdsKey = streamJobIds.join("|");

  useEffect(() => {
    if (!enabled || streamJobIds.length === 0) {
      return;
    }

    const sources = streamJobIds.map((jobId) => {
      const source = new EventSource(`/api/jobs/${jobId}/events`);
      const appendEvent = (event: MessageEvent) => {
        try {
          const parsed = JSON.parse(event.data) as JobEvent & { job_id?: string };
          const nextEvent: DashboardJobEvent = { ...parsed, job_id: parsed.job_id || jobId };
          setEventsByJobId((current) => {
            const currentEvents = current[jobId] || [];
            if (nextEvent.id && currentEvents.some((item) => item.id === nextEvent.id)) {
              return current;
            }
            return {
              ...current,
              [jobId]: [...currentEvents, nextEvent].slice(-EVENT_BUFFER_LIMIT)
            };
          });
        } catch {
          // Ignore malformed event frames; the stream will continue.
        }
      };
      source.onmessage = appendEvent;
      JOB_EVENT_NAMES.forEach((name) => {
        source.addEventListener(name, (event) => appendEvent(event as MessageEvent));
      });
      source.addEventListener("close", () => source.close());
      return source;
    });

    return () => {
      sources.forEach((source) => source.close());
    };
  }, [enabled, streamJobIdsKey]);

  const selectedEvents = selectedJob ? eventsByJobId[selectedJob.job_id] || [] : [];
  const allRunningEvents = useMemo(() => {
    return runningJobIds
      .flatMap((jobId) => eventsByJobId[jobId] || [])
      .sort((a, b) => (a.created_at || 0) - (b.created_at || 0) || (a.id || 0) - (b.id || 0));
  }, [eventsByJobId, runningJobIds]);

  return { selectedEvents, allRunningEvents };
}

function selectLiveMetricTrace(events: JobEvent[]): LiveMetricTrace | null {
  const metricEvents = events.filter((event) => event.event_type === "metric" && event.data);
  const preferredKeys = ["loss", "train_loss", "eval_loss"];
  for (const key of preferredKeys) {
    const series = numericMetricSeries(metricEvents, key);
    if (series.length > 0) {
      return { key, label: labelizeKey(key), series: series.slice(-80) };
    }
  }
  for (const event of metricEvents) {
    for (const [key, value] of Object.entries(event.data || {})) {
      if (typeof value === "number" && Number.isFinite(value)) {
        const series = numericMetricSeries(metricEvents, key);
        return { key, label: labelizeKey(key), series: series.slice(-80) };
      }
    }
  }
  return null;
}

function numericMetricSeries(metricEvents: JobEvent[], key: string): number[] {
  return metricEvents
    .map((event) => event.data?.[key])
    .filter((item): item is number => typeof item === "number" && Number.isFinite(item));
}

function extractMetricSummary(events: JobEvent[]): MetricSummary[] {
  const metricEvents = events.filter((event) => event.event_type === "metric" && event.data);
  const latestMetric = metricEvents[metricEvents.length - 1]?.data || {};
  return Object.entries(latestMetric)
    .filter(([, value]) => typeof value === "number" || typeof value === "string" || typeof value === "boolean")
    .slice(0, 6)
    .map(([key, value]) => {
      const series = metricEvents
        .map((event) => event.data?.[key])
        .filter((item): item is number => typeof item === "number" && Number.isFinite(item));
      return {
        key,
        label: labelizeKey(key),
        value: formatMetricValue(value),
        series
      };
    });
}

function payloadHighlights(payload: Record<string, unknown>) {
  return Object.entries(payload)
    .filter(([, value]) => typeof value === "string" || typeof value === "number" || typeof value === "boolean" || Array.isArray(value))
    .slice(0, 6)
    .map(([key, value]) => ({ key, value: formatPayloadValue(value) }));
}

function formatMetricValue(value: unknown) {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      return "unknown";
    }
    if (Math.abs(value) >= 1000) {
      return new Intl.NumberFormat().format(Math.round(value));
    }
    if (Math.abs(value) > 0 && Math.abs(value) < 1) {
      return value.toFixed(4);
    }
    return Number.isInteger(value) ? String(value) : value.toFixed(2);
  }
  if (typeof value === "boolean") {
    return value ? "yes" : "no";
  }
  return String(value || "unknown");
}

function formatPayloadValue(value: unknown) {
  if (Array.isArray(value)) {
    return value.length ? value.map((item) => String(item)).join(", ") : "none";
  }
  return formatMetricValue(value);
}

function labelizeKey(value: string) {
  return value
    .replace(/[_-]+/g, " ")
    .replace(/\b(gpu|gpus|id|ids|hf|gguf|url|api)\b/gi, (match) => match.toUpperCase())
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function getJobLifecycle(job: JobRecord): JobLifecycle {
  if (job.status === "queued") {
    return "queued";
  }
  if (job.status === "running") {
    return "running";
  }
  return "terminal";
}

function statusTone(status: string): "ok" | "warn" | "danger" | "neutral" {
  if (status === "succeeded" || status === "running") {
    return "ok";
  }
  if (status === "failed") {
    return "danger";
  }
  if (status === "cancelled") {
    return "warn";
  }
  return "neutral";
}

function pressureTone(value: number): "ok" | "warn" | "danger" | "neutral" {
  if (value >= 90) {
    return "danger";
  }
  if (value >= 70) {
    return "warn";
  }
  if (value > 0) {
    return "ok";
  }
  return "neutral";
}

function clampPercent(value: number) {
  return Math.max(0, Math.min(100, value));
}

function formatElapsedTime(job: JobRecord) {
  const start = job.started_at || job.created_at;
  const end = job.finished_at || Math.floor(Date.now() / 1000);
  const seconds = Math.max(0, end - start);
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m ${seconds % 60}s`;
  }
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

function formatElapsedTimeAt(job: JobRecord, now: Date) {
  const start = job.started_at || job.created_at;
  const end = job.finished_at || Math.floor(now.getTime() / 1000);
  const seconds = Math.max(0, end - start);
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m ${seconds % 60}s`;
  }
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

function formatEpochTime(value?: number) {
  if (!value) {
    return "unknown";
  }
  return new Date(value * 1000).toLocaleString();
}

function formatRefreshTime(value: number | null) {
  if (!value) {
    return "Waiting for first refresh";
  }
  return `Last update ${new Date(value).toLocaleTimeString()}`;
}

function Datasets({
  datasets,
  jobs,
  route,
  navigateDataset,
  activeInferenceTarget,
  refresh,
  setSelectedJob
}: {
  datasets: DatasetRecord[];
  jobs: JobRecord[];
  route: AppRoute;
  navigateDataset: (datasetRoute: DatasetRouteKey, datasetId?: string) => void;
  activeInferenceTarget: InferenceTarget | null;
  refresh: () => void;
  setSelectedJob: (job: JobRecord) => void;
}) {
  const [datasetType, setDatasetType] = useState("math_sft");
  const [title, setTitle] = useState("Math SFT dataset");
  const [slug, setSlug] = useState("math-sft");
  const [maxSequenceLength, setMaxSequenceLength] = useState("2048");
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [uploadErrors, setUploadErrors] = useState<FormErrors>({});
  const [url, setUrl] = useState("");
  const [urlTitle, setUrlTitle] = useState("URL review dataset");
  const [urlSlug, setUrlSlug] = useState("url-review");
  const [urlMaxRows, setUrlMaxRows] = useState("100");
  const [urlErrors, setUrlErrors] = useState<FormErrors>({});
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [reviewSample, setReviewSample] = useState<DatasetRecordsResponse | null>(null);
  const [records, setRecords] = useState<DatasetRecordsResponse | null>(null);
  const [recordQuery, setRecordQuery] = useState("");
  const [recordSplit, setRecordSplit] = useState("");
  const [recordLimit, setRecordLimit] = useState(50);
  const [recordOffset, setRecordOffset] = useState(0);
  const [selectedSource, setSelectedSource] = useState<DatasetSource>("huggingface");

  useEffect(() => {
    setSelectedDatasetId((current) => {
      if (route.datasetId && datasets.some((dataset) => dataset.dataset_id === route.datasetId)) {
        return route.datasetId;
      }
      if (datasets.length === 0) {
        return "";
      }
      if (datasets.some((dataset) => dataset.dataset_id === current)) {
        return current;
      }
      return datasets[0].dataset_id;
    });
  }, [datasets, route.datasetId]);

  useEffect(() => {
    if (!selectedDatasetId) {
      setReviewSample(null);
      setRecords(null);
      return;
    }
    loadDatasetPreview(selectedDatasetId, recordQuery, recordSplit, 0);
  }, [selectedDatasetId]);

  async function upload(event: FormEvent) {
    event.preventDefault();
    const errors = validateCsvUploadForm(title, slug, maxSequenceLength, file);
    setUploadErrors(errors);
    if (Object.keys(errors).length > 0) {
      return;
    }
    const body = new FormData();
    body.append("file", file as File);
    body.append("dataset_type", datasetType);
    body.append("title", title.trim());
    body.append("slug", slug.trim());
    body.append("max_sequence_length", maxSequenceLength.trim());
    try {
      const result = await api.postForm<{ created: boolean; validation: ValidationResult }>("/api/datasets/upload", body);
      setValidation(result.validation);
      await refresh();
      if (result.created) {
        navigateDataset("review");
      }
    } catch (err) {
      setUploadErrors({ form: err instanceof Error ? err.message : "Unable to upload dataset." });
    }
  }

  async function approve(datasetId: string) {
    await api.post(`/api/datasets/${datasetId}/approve`, {});
    await refresh();
    await loadDatasetPreview(datasetId);
  }

  async function reject(datasetId: string) {
    await api.post(`/api/datasets/${datasetId}/reject`, {});
    await refresh();
    setReviewSample(null);
    setRecords(null);
  }

  async function importUrl(event: FormEvent) {
    event.preventDefault();
    const errors = validateUrlImportForm(url, urlTitle, urlSlug, urlMaxRows);
    setUrlErrors(errors);
    if (Object.keys(errors).length > 0) {
      return;
    }
    try {
      const job = await api.post<JobRecord>("/api/datasets/import-url", {
        url: url.trim(),
        title: urlTitle.trim(),
        slug: urlSlug.trim(),
        dataset_type: datasetType,
        max_rows: Number(urlMaxRows) || undefined,
        default_split: "holdout"
      });
      setSelectedJob(job);
      await refresh();
    } catch (err) {
      setUrlErrors({ form: err instanceof Error ? err.message : "Unable to import URL dataset." });
    }
  }

  async function loadDatasetPreview(datasetId = selectedDatasetId, query = recordQuery, split = recordSplit, offset = recordOffset) {
    if (!datasetId) {
      return;
    }
    const [sample, page] = await Promise.all([
      api.get<DatasetRecordsResponse>(`/api/datasets/${datasetId}/review-sample?sample_size=100`),
      api.get<DatasetRecordsResponse>(datasetRecordsPath(datasetId, query, split, offset, recordLimit))
    ]);
    setReviewSample(sample);
    setRecords(page);
    setRecordOffset(offset);
  }

  async function applyRecordFilters() {
    await loadDatasetPreview(selectedDatasetId, recordQuery, recordSplit, 0);
  }

  async function clearRecordFilters() {
    setRecordQuery("");
    setRecordSplit("");
    await loadDatasetPreview(selectedDatasetId, "", "", 0);
  }

  async function pageRecords(direction: "next" | "previous") {
    const nextOffset = direction === "next" ? recordOffset + recordLimit : Math.max(0, recordOffset - recordLimit);
    await loadDatasetPreview(selectedDatasetId, recordQuery, recordSplit, nextOffset);
  }

  const selectedDataset = datasets.find((dataset) => dataset.dataset_id === selectedDatasetId);
  const recentDatasetJobs = jobs.filter((job) => job.job_type === "dataset_import").slice(0, 5);

  return (
    <div className="datasetPage">
      <DatasetRouteTabs currentRoute={route.datasetRoute} navigateDataset={navigateDataset} selectedDatasetId={selectedDatasetId} />
      {route.datasetRoute === "overview" && (
        <DatasetOverviewPage
          datasets={datasets}
          selectedDataset={selectedDataset}
          selectedDatasetId={selectedDatasetId}
          recentDatasetJobs={recentDatasetJobs}
          reviewSample={reviewSample}
          activeInferenceTarget={activeInferenceTarget}
          setSelectedDatasetId={setSelectedDatasetId}
          navigateDataset={navigateDataset}
          approve={approve}
          reject={reject}
        />
      )}
      {route.datasetRoute === "acquire" && (
        <DatasetAcquirePage
          selectedSource={selectedSource}
          setSelectedSource={setSelectedSource}
          activeInferenceTarget={activeInferenceTarget}
          refresh={refresh}
          setSelectedJob={setSelectedJob}
          datasetType={datasetType}
          setDatasetType={setDatasetType}
          title={title}
          setTitle={setTitle}
          slug={slug}
          setSlug={setSlug}
          maxSequenceLength={maxSequenceLength}
          setMaxSequenceLength={setMaxSequenceLength}
          file={file}
          setFile={setFile}
          upload={upload}
          uploadErrors={uploadErrors}
          validation={validation}
          url={url}
          setUrl={setUrl}
          urlTitle={urlTitle}
          setUrlTitle={setUrlTitle}
          urlSlug={urlSlug}
          setUrlSlug={setUrlSlug}
          urlMaxRows={urlMaxRows}
          setUrlMaxRows={setUrlMaxRows}
          importUrl={importUrl}
          urlErrors={urlErrors}
        />
      )}
      {route.datasetRoute === "review" && (
        <DatasetReviewPage
          datasets={datasets}
          selectedDataset={selectedDataset}
          selectedDatasetId={selectedDatasetId}
          setSelectedDatasetId={setSelectedDatasetId}
          records={records}
          reviewSample={reviewSample}
          recordQuery={recordQuery}
          setRecordQuery={setRecordQuery}
          recordSplit={recordSplit}
          setRecordSplit={setRecordSplit}
          recordLimit={recordLimit}
          setRecordLimit={setRecordLimit}
          recordOffset={recordOffset}
          applyRecordFilters={applyRecordFilters}
          clearRecordFilters={clearRecordFilters}
          pageRecords={pageRecords}
          navigateDataset={navigateDataset}
          approve={approve}
          reject={reject}
        />
      )}
      {route.datasetRoute === "detail" && (
        <DatasetDetailPage
          selectedDataset={selectedDataset}
          reviewSample={reviewSample}
          activeInferenceTarget={activeInferenceTarget}
          navigateDataset={navigateDataset}
          approve={approve}
          reject={reject}
        />
      )}
    </div>
  );
}

function DatasetRouteTabs({
  currentRoute,
  navigateDataset,
  selectedDatasetId
}: {
  currentRoute: DatasetRouteKey;
  navigateDataset: (datasetRoute: DatasetRouteKey, datasetId?: string) => void;
  selectedDatasetId: string;
}) {
  return (
    <div className="datasetRouteTabs" aria-label="Dataset workflow pages">
      {datasetRouteItems.map((item) => (
        <button
          key={item.route}
          type="button"
          className={currentRoute === item.route ? "active" : ""}
          onClick={() => navigateDataset(item.route)}
        >
          {item.icon}
          <span>{item.label}</span>
        </button>
      ))}
      <button
        type="button"
        className={currentRoute === "detail" ? "active" : ""}
        disabled={!selectedDatasetId}
        onClick={() => navigateDataset("detail", selectedDatasetId)}
      >
        <FileText size={15} />
        <span>Detail</span>
      </button>
    </div>
  );
}

function DatasetOverviewPage({
  datasets,
  selectedDataset,
  selectedDatasetId,
  recentDatasetJobs,
  reviewSample,
  activeInferenceTarget,
  setSelectedDatasetId,
  navigateDataset,
  approve,
  reject
}: {
  datasets: DatasetRecord[];
  selectedDataset?: DatasetRecord;
  selectedDatasetId: string;
  recentDatasetJobs: JobRecord[];
  reviewSample: DatasetRecordsResponse | null;
  activeInferenceTarget: InferenceTarget | null;
  setSelectedDatasetId: (datasetId: string) => void;
  navigateDataset: (datasetRoute: DatasetRouteKey, datasetId?: string) => void;
  approve: (datasetId: string) => void;
  reject: (datasetId: string) => void;
}) {
  const approvedCount = datasets.filter((dataset) => dataset.approved).length;
  const totalRows = datasets.reduce((sum, dataset) => sum + dataset.row_count, 0);
  const issueCount = datasets.reduce((sum, dataset) => sum + dataset.validation.errors.length + dataset.validation.warnings.length, 0);

  return (
    <div className="datasetRoutePage">
      <DatasetWorkflowHeader datasets={datasets} selectedDataset={selectedDataset} />
      <section className="datasetMetricGrid" aria-label="Dataset status metrics">
        <MetricTile label="Versions" value={formatCount(datasets.length)} detail={`${approvedCount} approved`} />
        <MetricTile label="Canonical rows" value={formatCount(totalRows)} detail={`${issueCount} validation signals`} />
        <MetricTile label="Active target" value={activeInferenceTarget?.display_name || "Unset"} detail={activeInferenceTarget ? inferenceTargetSubtitle(activeInferenceTarget) : "Select on Models"} />
        <MetricTile label="Import jobs" value={formatCount(recentDatasetJobs.length)} detail="Recent dataset imports" />
      </section>
      <div className="datasetOverviewGrid">
        <section className="panel datasetVersionsPanel">
          <div className="panelHeader">
            <div>
              <h2>Local Versions</h2>
              <p>{datasets.length ? "Select a version, then inspect or approve it." : "No local versions yet."}</p>
            </div>
            <Database size={18} />
          </div>
          <DatasetVersionList
            datasets={datasets}
            selectedDatasetId={selectedDatasetId}
            setSelectedDatasetId={setSelectedDatasetId}
            navigateDataset={navigateDataset}
            approve={approve}
            reject={reject}
          />
        </section>
        <DatasetReadinessRail selectedDataset={selectedDataset} reviewSample={reviewSample} activeInferenceTarget={activeInferenceTarget} />
        <section className="panel datasetJobsPanel">
          <div className="panelHeader">
            <div>
              <h2>Recent Imports</h2>
              <p>Jobs that produced or are producing review datasets.</p>
            </div>
            <GitBranch size={18} />
          </div>
          <div className="list">
            {recentDatasetJobs.map((job) => (
              <div className="listItem" key={job.job_id}>
                <strong>{job.job_id}</strong>
                <span>{job.slug} - {job.status}</span>
              </div>
            ))}
            {recentDatasetJobs.length === 0 && <div className="empty datasetEmpty">Dataset import jobs appear here after an acquisition starts.</div>}
          </div>
        </section>
      </div>
    </div>
  );
}

function DatasetAcquirePage({
  selectedSource,
  setSelectedSource,
  activeInferenceTarget,
  refresh,
  setSelectedJob,
  datasetType,
  setDatasetType,
  title,
  setTitle,
  slug,
  setSlug,
  maxSequenceLength,
  setMaxSequenceLength,
  file,
  setFile,
  upload,
  uploadErrors,
  validation,
  url,
  setUrl,
  urlTitle,
  setUrlTitle,
  urlSlug,
  setUrlSlug,
  urlMaxRows,
  setUrlMaxRows,
  importUrl,
  urlErrors
}: {
  selectedSource: DatasetSource;
  setSelectedSource: (source: DatasetSource) => void;
  activeInferenceTarget: InferenceTarget | null;
  refresh: () => void;
  setSelectedJob: (job: JobRecord) => void;
  datasetType: string;
  setDatasetType: (value: string) => void;
  title: string;
  setTitle: (value: string) => void;
  slug: string;
  setSlug: (value: string) => void;
  maxSequenceLength: string;
  setMaxSequenceLength: (value: string) => void;
  file: File | null;
  setFile: (value: File | null) => void;
  upload: (event: FormEvent) => void;
  uploadErrors: FormErrors;
  validation: ValidationResult | null;
  url: string;
  setUrl: (value: string) => void;
  urlTitle: string;
  setUrlTitle: (value: string) => void;
  urlSlug: string;
  setUrlSlug: (value: string) => void;
  urlMaxRows: string;
  setUrlMaxRows: (value: string) => void;
  importUrl: (event: FormEvent) => void;
  urlErrors: FormErrors;
}) {
  return (
    <section className="datasetRoutePage datasetSourceDeck" aria-labelledby="dataset-source-heading">
      <div className="datasetSectionIntro">
        <div>
          <span>Acquire</span>
          <h2 id="dataset-source-heading">Choose a source path</h2>
        </div>
        <p>Pick the route that matches the data source. Each form validates required fields before a job or upload starts.</p>
      </div>
      <div className="datasetSourceTabs" role="tablist" aria-label="Dataset source">
        <button
          type="button"
          className={selectedSource === "huggingface" ? "active" : ""}
          role="tab"
          aria-selected={selectedSource === "huggingface"}
          onClick={() => setSelectedSource("huggingface")}
        >
          <Search size={16} />
          <span>Hugging Face</span>
        </button>
        <button
          type="button"
          className={selectedSource === "csv" ? "active" : ""}
          role="tab"
          aria-selected={selectedSource === "csv"}
          onClick={() => setSelectedSource("csv")}
        >
          <Upload size={16} />
          <span>Upload CSV</span>
        </button>
        <button
          type="button"
          className={selectedSource === "url" ? "active" : ""}
          role="tab"
          aria-selected={selectedSource === "url"}
          onClick={() => setSelectedSource("url")}
        >
          <Download size={16} />
          <span>Import URL</span>
        </button>
      </div>

      {selectedSource === "huggingface" && (
        <HubAcquirePanel
          resourceType="dataset"
          defaultInput="AI-MO/NuminaMath-CoT"
          activeInferenceTarget={activeInferenceTarget}
          refresh={refresh}
          setSelectedJob={setSelectedJob}
        />
      )}
      {selectedSource === "csv" && (
        <section className="panel datasetSourceCard">
          <div className="panelHeader">
            <div>
              <h2>Upload CSV</h2>
              <p>Validate local prompt, response, split, and metadata columns.</p>
            </div>
            <Upload size={18} />
          </div>
          <form className="formGrid datasetSourceForm" onSubmit={upload} noValidate>
            {uploadErrors.form && <div className="alert wideField">{uploadErrors.form}</div>}
            <label>
              Template
              <select value={datasetType} onChange={(event) => setDatasetType(event.target.value)}>
                <option value="chat_sft">Chat SFT</option>
                <option value="math_sft">Math SFT</option>
                <option value="holdout">Benchmark holdout</option>
              </select>
              <FieldNote note="Templates use the canonical prompt, response, split, and metadata columns." link="#datasets" />
            </label>
            <button type="button" className="secondary" onClick={() => api.template(datasetType)}>
              <Database size={16} /> Download Template
            </button>
            <label>
              Title
              <input value={title} onChange={(event) => setTitle(event.target.value)} aria-invalid={Boolean(uploadErrors.title)} />
              <FieldError message={uploadErrors.title} />
            </label>
            <label>
              Slug
              <input value={slug} onChange={(event) => setSlug(slugInput(event.target.value))} aria-invalid={Boolean(uploadErrors.slug)} />
              <FieldError message={uploadErrors.slug} />
            </label>
            <label>
              Max sequence length
              <input
                type="number"
                min="128"
                max="32768"
                value={maxSequenceLength}
                onChange={(event) => setMaxSequenceLength(event.target.value)}
                aria-invalid={Boolean(uploadErrors.maxSequenceLength)}
              />
              <FieldNote note="Rows longer than this approximate limit are rejected before benchmark or generation use." link="#sequence-length" />
              <FieldError message={uploadErrors.maxSequenceLength} />
            </label>
            <label>
              CSV file
              <input
                type="file"
                accept=".csv,text/csv"
                onChange={(event) => setFile(event.target.files?.[0] || null)}
                aria-invalid={Boolean(uploadErrors.file)}
              />
              {file && <span className="inputMeta">{file.name}</span>}
              <FieldError message={uploadErrors.file} />
            </label>
            <button type="submit">
              <Upload size={16} /> Validate Upload
            </button>
          </form>
          {validation && <ValidationTable validation={validation} />}
        </section>
      )}
      {selectedSource === "url" && (
        <section className="panel datasetSourceCard">
          <div className="panelHeader">
            <div>
              <h2>Import URL Dataset</h2>
              <p>Pull a remote file into a review dataset with a row limit.</p>
            </div>
            <Download size={18} />
          </div>
          <form className="formGrid datasetSourceForm" onSubmit={importUrl} noValidate>
            {urlErrors.form && <div className="alert wideField">{urlErrors.form}</div>}
            <label className="wideField">
              Dataset URL
              <input value={url} onChange={(event) => setUrl(event.target.value)} aria-invalid={Boolean(urlErrors.url)} />
              <FieldError message={urlErrors.url} />
            </label>
            <label>
              Title
              <input value={urlTitle} onChange={(event) => setUrlTitle(event.target.value)} aria-invalid={Boolean(urlErrors.title)} />
              <FieldError message={urlErrors.title} />
            </label>
            <label>
              Slug
              <input value={urlSlug} onChange={(event) => setUrlSlug(slugInput(event.target.value))} aria-invalid={Boolean(urlErrors.slug)} />
              <FieldError message={urlErrors.slug} />
            </label>
            <label>
              Max rows
              <input
                type="number"
                min="1"
                max="100000"
                value={urlMaxRows}
                onChange={(event) => setUrlMaxRows(event.target.value)}
                aria-invalid={Boolean(urlErrors.maxRows)}
              />
              <FieldError message={urlErrors.maxRows} />
            </label>
            <button type="submit">
              <Download size={16} /> Import From URL
            </button>
          </form>
        </section>
      )}
    </section>
  );
}

function DatasetReviewPage({
  datasets,
  selectedDataset,
  selectedDatasetId,
  setSelectedDatasetId,
  records,
  reviewSample,
  recordQuery,
  setRecordQuery,
  recordSplit,
  setRecordSplit,
  recordLimit,
  setRecordLimit,
  recordOffset,
  applyRecordFilters,
  clearRecordFilters,
  pageRecords,
  navigateDataset,
  approve,
  reject
}: {
  datasets: DatasetRecord[];
  selectedDataset?: DatasetRecord;
  selectedDatasetId: string;
  setSelectedDatasetId: (datasetId: string) => void;
  records: DatasetRecordsResponse | null;
  reviewSample: DatasetRecordsResponse | null;
  recordQuery: string;
  setRecordQuery: (value: string) => void;
  recordSplit: string;
  setRecordSplit: (value: string) => void;
  recordLimit: number;
  setRecordLimit: (value: number) => void;
  recordOffset: number;
  applyRecordFilters: () => void;
  clearRecordFilters: () => void;
  pageRecords: (direction: "next" | "previous") => void;
  navigateDataset: (datasetRoute: DatasetRouteKey, datasetId?: string) => void;
  approve: (datasetId: string) => void;
  reject: (datasetId: string) => void;
}) {
  return (
    <div className="datasetRoutePage datasetReviewLayout">
      <section className="panel datasetVersionsPanel">
        <div className="panelHeader">
          <div>
            <h2>Versions</h2>
            <p>{datasets.length ? "Choose a dataset to inspect." : "Import or upload a dataset first."}</p>
          </div>
          <Database size={18} />
        </div>
        <DatasetVersionList
          datasets={datasets}
          selectedDatasetId={selectedDatasetId}
          setSelectedDatasetId={setSelectedDatasetId}
          navigateDataset={navigateDataset}
          approve={approve}
          reject={reject}
        />
      </section>
      <DatasetRecordBrowser
        selectedDataset={selectedDataset}
        records={records}
        reviewSample={reviewSample}
        recordQuery={recordQuery}
        setRecordQuery={setRecordQuery}
        recordSplit={recordSplit}
        setRecordSplit={setRecordSplit}
        recordLimit={recordLimit}
        setRecordLimit={setRecordLimit}
        recordOffset={recordOffset}
        applyRecordFilters={applyRecordFilters}
        clearRecordFilters={clearRecordFilters}
        pageRecords={pageRecords}
        navigateDataset={navigateDataset}
        approve={approve}
        reject={reject}
      />
    </div>
  );
}

function DatasetDetailPage({
  selectedDataset,
  reviewSample,
  activeInferenceTarget,
  navigateDataset,
  approve,
  reject
}: {
  selectedDataset?: DatasetRecord;
  reviewSample: DatasetRecordsResponse | null;
  activeInferenceTarget: InferenceTarget | null;
  navigateDataset: (datasetRoute: DatasetRouteKey, datasetId?: string) => void;
  approve: (datasetId: string) => void;
  reject: (datasetId: string) => void;
}) {
  if (!selectedDataset) {
    return (
      <section className="datasetRoutePage panel">
        <div className="panelHeader">
          <div>
            <h2>No dataset selected</h2>
            <p>Acquire a dataset before opening a detail view.</p>
          </div>
          <FileText size={18} />
        </div>
        <button type="button" onClick={() => navigateDataset("acquire")}>
          <Download size={16} /> Acquire Dataset
        </button>
      </section>
    );
  }

  return (
    <div className="datasetRoutePage datasetDetailLayout">
      <section className="panel datasetPreviewPanel">
        <div className="panelHeader">
          <div>
            <h2>{selectedDataset.title}</h2>
            <p>{selectedDataset.version_id}</p>
          </div>
          <DatasetStatusPill dataset={selectedDataset} />
        </div>
        <DatasetSummaryStrip selectedDataset={selectedDataset} />
        <ValidationTable validation={selectedDataset.validation} />
        <div className="datasetActions detailActions">
          <button type="button" className="secondary" onClick={() => navigateDataset("review")}>
            <Eye size={14} /> Review Records
          </button>
          <button type="button" disabled={selectedDataset.approved} onClick={() => approve(selectedDataset.dataset_id)}>
            <CheckCircle size={14} /> Approve
          </button>
          <button type="button" className="danger" disabled={selectedDataset.approved} onClick={() => reject(selectedDataset.dataset_id)}>
            <XCircle size={14} /> Reject & Delete
          </button>
        </div>
      </section>
      <DatasetReadinessRail selectedDataset={selectedDataset} reviewSample={reviewSample} activeInferenceTarget={activeInferenceTarget} />
      <section className="panel datasetPreviewPanel">
        <div className="panelHeader">
          <div>
            <h2>Review Sample</h2>
            <p>{reviewSample ? `${reviewSample.sample_size || 0} of ${reviewSample.required_review_sample_size || 0} required rows loaded` : "Sample not loaded."}</p>
          </div>
          <Eye size={18} />
        </div>
        {reviewSample ? <DatasetRowsTable records={reviewSample.records} /> : <div className="empty datasetEmpty">Review rows load after selecting a dataset.</div>}
      </section>
    </div>
  );
}

function DatasetVersionList({
  datasets,
  selectedDatasetId,
  setSelectedDatasetId,
  navigateDataset,
  approve,
  reject
}: {
  datasets: DatasetRecord[];
  selectedDatasetId: string;
  setSelectedDatasetId: (datasetId: string) => void;
  navigateDataset: (datasetRoute: DatasetRouteKey, datasetId?: string) => void;
  approve: (datasetId: string) => void;
  reject: (datasetId: string) => void;
}) {
  if (datasets.length === 0) {
    return <div className="empty datasetEmpty">Imported and uploaded datasets appear here.</div>;
  }
  return (
    <div className="datasetVersionList">
      {datasets.map((dataset) => (
        <div className={`datasetVersionCard ${dataset.dataset_id === selectedDatasetId ? "selected" : ""}`} key={dataset.dataset_id}>
          <div className="datasetVersionTitle">
            <strong>{dataset.title}</strong>
            <span>{dataset.slug}</span>
          </div>
          <div className="datasetVersionMeta">
            <span>
              <small>Rows</small>
              <strong>{formatCount(dataset.row_count)}</strong>
            </span>
            <span>
              <small>Split</small>
              <strong>{splitSummary(dataset.split_counts)}</strong>
            </span>
            <DatasetStatusPill dataset={dataset} />
          </div>
          <div className="datasetActions">
            <button className="small secondary" type="button" onClick={() => setSelectedDatasetId(dataset.dataset_id)}>
              <Eye size={14} /> Select
            </button>
            <button className="small secondary" type="button" onClick={() => navigateDataset("detail", dataset.dataset_id)}>
              <FileText size={14} /> Detail
            </button>
            <button className="small" type="button" disabled={dataset.approved} onClick={() => approve(dataset.dataset_id)}>
              <CheckCircle size={14} /> Approve
            </button>
            <button className="small danger" type="button" disabled={dataset.approved} onClick={() => reject(dataset.dataset_id)}>
              <XCircle size={14} /> Reject & Delete
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

function DatasetRecordBrowser({
  selectedDataset,
  records,
  reviewSample,
  recordQuery,
  setRecordQuery,
  recordSplit,
  setRecordSplit,
  recordLimit,
  setRecordLimit,
  recordOffset,
  applyRecordFilters,
  clearRecordFilters,
  pageRecords,
  navigateDataset,
  approve,
  reject
}: {
  selectedDataset?: DatasetRecord;
  records: DatasetRecordsResponse | null;
  reviewSample: DatasetRecordsResponse | null;
  recordQuery: string;
  setRecordQuery: (value: string) => void;
  recordSplit: string;
  setRecordSplit: (value: string) => void;
  recordLimit: number;
  setRecordLimit: (value: number) => void;
  recordOffset: number;
  applyRecordFilters: () => void;
  clearRecordFilters: () => void;
  pageRecords: (direction: "next" | "previous") => void;
  navigateDataset: (datasetRoute: DatasetRouteKey, datasetId?: string) => void;
  approve: (datasetId: string) => void;
  reject: (datasetId: string) => void;
}) {
  const rows = records?.records || [];
  const totalMatching = records?.total_matching || 0;
  const canPageBackward = recordOffset > 0;
  const canPageForward = recordOffset + recordLimit < totalMatching;

  return (
    <section className="panel datasetPreviewPanel">
      <div className="panelHeader">
        <div>
          <h2>Record Browser</h2>
          <p>
            {selectedDataset
              ? `${formatCount(totalMatching)} matching rows in ${selectedDataset.title}`
              : "Select a dataset to inspect rows."}
          </p>
        </div>
        <Filter size={18} />
      </div>
      {selectedDataset && <DatasetSummaryStrip selectedDataset={selectedDataset} />}
      <div className="formGrid inlineForm datasetFilterBar">
        <label>
          Search
          <input value={recordQuery} onChange={(event) => setRecordQuery(event.target.value)} placeholder="prompt, response, metadata" />
        </label>
        <label>
          Split
          <select value={recordSplit} onChange={(event) => setRecordSplit(event.target.value)}>
            <option value="">All</option>
            <option value="train">train</option>
            <option value="validation">validation</option>
            <option value="holdout">holdout</option>
          </select>
        </label>
        <label>
          Page size
          <select value={recordLimit} onChange={(event) => setRecordLimit(Number(event.target.value))}>
            <option value={25}>25</option>
            <option value={50}>50</option>
            <option value={100}>100</option>
          </select>
        </label>
        <button type="button" onClick={applyRecordFilters} disabled={!selectedDataset}>
          <SlidersHorizontal size={16} /> Apply Filters
        </button>
        <button type="button" className="secondary" onClick={clearRecordFilters} disabled={!selectedDataset || (!recordQuery && !recordSplit)}>
          <RefreshCw size={16} /> Clear
        </button>
      </div>
      <div className="activeFilterRow" aria-live="polite">
        <span>{recordQuery ? `query: ${recordQuery}` : "query: none"}</span>
        <span>{recordSplit ? `split: ${recordSplit}` : "split: all"}</span>
        <span>{records ? `rows ${recordOffset + 1}-${Math.min(recordOffset + recordLimit, totalMatching)} of ${totalMatching}` : "rows not loaded"}</span>
      </div>
      {records ? <DatasetRowsTable records={rows} /> : <div className="empty datasetEmpty">Choose a dataset version to preview canonical rows.</div>}
      <div className="datasetPager">
        <button type="button" className="secondary" disabled={!canPageBackward} onClick={() => pageRecords("previous")}>
          Previous
        </button>
        <span>{records ? `Page ${Math.floor(recordOffset / recordLimit) + 1}` : "No page"}</span>
        <button type="button" className="secondary" disabled={!canPageForward} onClick={() => pageRecords("next")}>
          Next
        </button>
      </div>
      {selectedDataset && (
        <div className="datasetActions detailActions">
          <button type="button" className="secondary" onClick={() => navigateDataset("detail", selectedDataset.dataset_id)}>
            <FileText size={14} /> Detail
          </button>
          <button type="button" disabled={selectedDataset.approved} onClick={() => approve(selectedDataset.dataset_id)}>
            <CheckCircle size={14} /> Approve
          </button>
          <button type="button" className="danger" disabled={selectedDataset.approved} onClick={() => reject(selectedDataset.dataset_id)}>
            <XCircle size={14} /> Reject & Delete
          </button>
        </div>
      )}
      {reviewSample && records && reviewSample.required_review_sample_size !== undefined && (
        <div className="statusLine">
          Review sample loaded: {reviewSample.sample_size || 0} of {reviewSample.required_review_sample_size} required rows.
        </div>
      )}
    </section>
  );
}

function DatasetSummaryStrip({ selectedDataset }: { selectedDataset: DatasetRecord }) {
  return (
    <div className="datasetSummaryStrip">
      <div>
        <span>Selected version</span>
        <strong>{selectedDataset.version_id}</strong>
      </div>
      <div>
        <span>Validation</span>
        <strong>{selectedDataset.validation.valid ? "passed" : "needs fixes"}</strong>
      </div>
      <div>
        <span>Rows</span>
        <strong>{formatCount(selectedDataset.row_count)}</strong>
      </div>
      <div className="wideSummary">
        <span>JSONL</span>
        <strong>{selectedDataset.jsonl_path}</strong>
      </div>
    </div>
  );
}

function MetricTile({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="datasetMetricTile">
      <span>{label}</span>
      <strong>{value}</strong>
      <p>{detail}</p>
    </div>
  );
}

function DatasetWorkflowHeader({ datasets, selectedDataset }: { datasets: DatasetRecord[]; selectedDataset?: DatasetRecord }) {
  const approvedCount = datasets.filter((dataset) => dataset.approved).length;
  const validationPassed = Boolean(selectedDataset?.validation.valid);
  const workflowSteps = [
    { label: "Model", detail: "Download or select base model", state: "done" },
    { label: "Dataset", detail: datasets.length ? `${datasets.length} version${datasets.length === 1 ? "" : "s"} ready to review` : "Acquire source", state: datasets.length ? "done" : "active" },
    { label: "Validate", detail: validationPassed ? "Rows are canonical" : "Inspect rows", state: validationPassed ? "done" : selectedDataset ? "active" : "blocked" },
    { label: "Approve", detail: approvedCount ? `${approvedCount} approved` : "Gate training", state: approvedCount ? "done" : selectedDataset ? "active" : "blocked" },
    { label: "Train", detail: "Queue TrainingHub job", state: approvedCount ? "active" : "blocked" }
  ];

  return (
    <section className="datasetFlowHeader">
      <div className="datasetFlowCopy">
        <span>Dataset pipeline</span>
        <h2>Build clean data for fine-tuning</h2>
        <p>Resolve the source, normalize it into TrainingHub JSONL, validate real rows, approve the version, then use that version in a training job.</p>
      </div>
      <div className="datasetFlowSteps" aria-label="Fine-tuning dataset workflow">
        {workflowSteps.map((step, index) => (
          <div className={`datasetFlowStep ${step.state}`} key={step.label}>
            <strong>{index + 1}</strong>
            <span>{step.label}</span>
            <small>{step.detail}</small>
          </div>
        ))}
      </div>
    </section>
  );
}

function DatasetReadinessRail({
  selectedDataset,
  reviewSample,
  activeInferenceTarget
}: {
  selectedDataset?: DatasetRecord;
  reviewSample: DatasetRecordsResponse | null;
  activeInferenceTarget: InferenceTarget | null;
}) {
  const issues = selectedDataset ? selectedDataset.validation.errors.length + selectedDataset.validation.warnings.length : 0;
  const readinessItems = [
    { label: "Cleaner model", value: activeInferenceTarget?.display_name || "Morrigan local inference", state: "done" },
    { label: "Dataset version", value: selectedDataset?.version_id || "Not selected", state: selectedDataset ? "done" : "blocked" },
    { label: "Validation", value: selectedDataset ? `${selectedDataset.validation.accepted_count} accepted / ${issues} issues` : "Waiting for source", state: selectedDataset?.validation.valid ? "done" : selectedDataset ? "active" : "blocked" },
    { label: "Review sample", value: reviewSample ? `${reviewSample.sample_size || 0} of ${reviewSample.required_review_sample_size || 0} rows` : "Not loaded", state: reviewSample ? "done" : "blocked" },
    { label: "Training", value: selectedDataset?.approved ? "Ready for TrainingHub" : "Approval required", state: selectedDataset?.approved ? "done" : "blocked" }
  ];

  return (
    <aside className="datasetReadinessRail" aria-label="Dataset readiness">
      <div>
        <span>Next action</span>
        <h2>{selectedDataset?.approved ? "Use this dataset in training" : selectedDataset ? "Inspect and approve" : "Acquire a dataset"}</h2>
      </div>
      <div className="datasetReadinessList">
        {readinessItems.map((item) => (
          <div className={`datasetReadinessItem ${item.state}`} key={item.label}>
            <span>{item.label}</span>
            <strong>{item.value}</strong>
          </div>
        ))}
      </div>
      <div className="datasetLineage">
        <span>Lineage</span>
        <strong>{"Source -> Canonical JSONL -> Approved version -> TrainingHub job -> registered artifacts"}</strong>
      </div>
    </aside>
  );
}

function DatasetStatusPill({ dataset }: { dataset: DatasetRecord }) {
  return <span className={`datasetStatus ${datasetStatusKind(dataset)}`}>{datasetStatus(dataset)}</span>;
}

function datasetRecordsPath(datasetId: string, query: string, split: string, offset = 0, limit = 50) {
  const params = new URLSearchParams({ offset: String(offset), limit: String(limit) });
  if (query) {
    params.set("query", query);
  }
  if (split) {
    params.set("split", split);
  }
  return `/api/datasets/${datasetId}/records?${params.toString()}`;
}

function validateCsvUploadForm(title: string, slug: string, maxSequenceLength: string, file: File | null) {
  const errors: FormErrors = {};
  addError(errors, "title", validateRequired(title, "Title"));
  addError(errors, "slug", validateSlug(slug));
  addError(errors, "maxSequenceLength", validatePositiveInteger(maxSequenceLength, "Max sequence length", 128, 32768));
  if (!file) {
    errors.file = "Choose a CSV file before validating upload.";
  } else if (!file.name.toLowerCase().endsWith(".csv") && !file.type.includes("csv")) {
    errors.file = "Use a CSV file with a .csv extension.";
  }
  return errors;
}

function validateUrlImportForm(url: string, title: string, slug: string, maxRows: string) {
  const errors: FormErrors = {};
  addError(errors, "url", validateDatasetUrl(url));
  addError(errors, "title", validateRequired(title, "Title"));
  addError(errors, "slug", validateSlug(slug));
  addError(errors, "maxRows", validatePositiveInteger(maxRows, "Max rows", 1, 100000));
  return errors;
}

function validateHubFindForm(input: string) {
  const errors: FormErrors = {};
  addError(errors, "input", validateHubInput(input));
  return errors;
}

function validateHubConfirmForm(resourceType: "model" | "dataset", values: Record<string, string>) {
  const errors: FormErrors = {};
  if (resourceType === "model") {
    addError(errors, "modelSlug", values.modelSlug ? validateSlug(values.modelSlug) : undefined);
    addError(errors, "modelDisplayName", values.modelDisplayName ? undefined : "Display name is required.");
    return errors;
  }
  addError(errors, "datasetTitle", validateRequired(values.datasetTitle, "Title"));
  addError(errors, "datasetSlug", validateSlug(values.datasetSlug));
  addError(errors, "datasetSplit", validateRequired(values.datasetSplit, "Split"));
  addError(errors, "datasetMaxRows", validatePositiveInteger(values.datasetMaxRows, "Max rows", 1, 100000));
  return errors;
}

function validateRequired(value: string, label: string) {
  return value.trim() ? undefined : `${label} is required.`;
}

function validateSlug(value: string) {
  const clean = value.trim();
  if (!clean) {
    return "Slug is required.";
  }
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(clean)) {
    return "Use lowercase letters, numbers, and single hyphens.";
  }
  return undefined;
}

function validatePositiveInteger(value: string, label: string, min: number, max: number) {
  if (!value.trim()) {
    return `${label} is required.`;
  }
  if (!/^\d+$/.test(value.trim())) {
    return `${label} must be a whole number.`;
  }
  const parsed = Number(value);
  if (parsed < min || parsed > max) {
    return `${label} must be between ${formatCount(min)} and ${formatCount(max)}.`;
  }
  return undefined;
}

function validateDatasetUrl(value: string) {
  if (!value.trim()) {
    return "Dataset URL is required.";
  }
  try {
    const parsed = new URL(value.trim());
    return ["http:", "https:", "file:"].includes(parsed.protocol) ? undefined : "Use an http, https, or file URL.";
  } catch {
    return "Enter a valid URL.";
  }
}

function validateHubInput(value: string) {
  const clean = value.trim();
  if (!clean) {
    return "Enter a Hugging Face URL or repo id.";
  }
  try {
    const parsed = new URL(clean);
    if (["huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co"].includes(parsed.hostname)) {
      return undefined;
    }
    return "Use a huggingface.co or hf.co URL.";
  } catch {
    return /^(datasets\/|models\/)?[A-Za-z0-9][A-Za-z0-9_.-]*\/[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(clean)
      ? undefined
      : "Use org/name, datasets/org/name, or a Hugging Face URL.";
  }
}

function addError(errors: FormErrors, key: string, message?: string) {
  if (message) {
    errors[key] = message;
  }
}

function slugInput(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9-]+/g, "-").replace(/-{2,}/g, "-");
}

function datasetStatus(dataset: DatasetRecord) {
  if (dataset.approved) {
    return "approved";
  }
  if (dataset.reviewed_at) {
    return "rejected";
  }
  return "pending review";
}

function datasetStatusKind(dataset: DatasetRecord) {
  if (dataset.approved) {
    return "approved";
  }
  if (dataset.reviewed_at) {
    return "rejected";
  }
  return "pending";
}

function splitSummary(splitCounts: Record<string, number>) {
  const entries = Object.entries(splitCounts || {}).filter(([, count]) => count > 0);
  if (entries.length === 0) {
    return "none";
  }
  return entries.map(([split, count]) => `${split} ${formatCount(count)}`).join(" / ");
}

function DatasetRowsTable({ records }: { records: DatasetRecordsResponse["records"] }) {
  if (records.length === 0) {
    return <div className="empty">No records loaded.</div>;
  }
  return (
    <div className="datasetPreview datasetRowsList">
      {records.map((record) => (
        <article className="datasetRowCard" key={record.index}>
          <div className="datasetRowIndex">#{record.index + 1}</div>
          <div className="datasetRowFields">
            <div>
              <span>Prompt</span>
              <p>{record.prompt}</p>
            </div>
            <div>
              <span>Response</span>
              <p>{record.response}</p>
            </div>
            <div>
              <span>Metadata</span>
              <code>{JSON.stringify(record.metadata)}</code>
            </div>
          </div>
        </article>
      ))}
    </div>
  );
}

function splitPatterns(value: string) {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseMixInput(value: string): Record<string, number> {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .reduce<Record<string, number>>((mix, item) => {
      const [rawKey, rawValue] = item.split(/[:=]/);
      const key = rawKey?.trim();
      const weight = Number(rawValue);
      if (key && Number.isFinite(weight)) {
        mix[key] = weight;
      }
      return mix;
    }, {});
}

function FieldError({ message }: { message?: string }) {
  if (!message) {
    return null;
  }
  return <span className="fieldError">{message}</span>;
}

function ValidationTable({ validation }: { validation: ValidationResult }) {
  const issues = [...validation.errors, ...validation.warnings];
  return (
    <div className="validationBox">
      <strong>{validation.valid ? "Validation passed" : "Validation failed"}</strong>
      <span>{validation.accepted_count} accepted rows</span>
      {issues.map((issue, index) => (
        <div className={issue.code.includes("leakage") ? "warningLine" : "errorLine"} key={`${issue.code}-${index}`}>
          Row {issue.row_number} - {issue.field}: {issue.message}
        </div>
      ))}
    </div>
  );
}

function HubAcquirePanel({
  resourceType,
  defaultInput,
  activeInferenceTarget,
  refresh,
  setSelectedJob,
  onToast = () => undefined
}: {
  resourceType: "model" | "dataset";
  defaultInput: string;
  activeInferenceTarget: InferenceTarget | null;
  refresh: () => void;
  setSelectedJob: (job: JobRecord) => void;
  onToast?: ToastHandler;
}) {
  const [input, setInput] = useState(defaultInput);
  const [revision, setRevision] = useState("");
  const [resolved, setResolved] = useState<HubResolvedResource | null>(null);
  const [statusMessage, setStatusMessage] = useState("");
  const [error, setError] = useState("");
  const [isResolving, setIsResolving] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [lastJob, setLastJob] = useState<JobRecord | null>(null);
  const [modelSlug, setModelSlug] = useState("");
  const [modelDisplayName, setModelDisplayName] = useState("");
  const [includePatterns, setIncludePatterns] = useState("*.gguf,*.json,tokenizer*");
  const [datasetType, setDatasetType] = useState("math_sft");
  const [datasetConfig, setDatasetConfig] = useState("");
  const [datasetSplit, setDatasetSplit] = useState("train");
  const [datasetTitle, setDatasetTitle] = useState("");
  const [datasetSlug, setDatasetSlug] = useState("");
  const [datasetMaxRows, setDatasetMaxRows] = useState("100");
  const [formErrors, setFormErrors] = useState<FormErrors>({});
  const title = resourceType === "model" ? "Find Hugging Face Model" : "Find Hugging Face Dataset";
  const confirmLabel = resourceType === "model" ? "Confirm Download" : "Confirm Clean Import";
  const stepIndex = lastJob ? 3 : isSubmitting ? 2 : resolved ? 1 : 0;
  const cleaningModel = activeInferenceTarget?.display_name || "Morrigan local inference";

  async function findHubResource(event: FormEvent) {
    event.preventDefault();
    const errors = validateHubFindForm(input);
    setFormErrors(errors);
    if (Object.keys(errors).length > 0) {
      return;
    }
    setIsResolving(true);
    setError("");
    setStatusMessage("");
    setLastJob(null);
    try {
      const result = await api.post<HubResolvedResource>("/api/hub/resolve", {
        input,
        resource_type: resourceType,
        revision: revision || undefined
      });
      setResolved(result);
      setInput(result.repo_id);
      const defaultSlug = slugFromRepo(result.repo_id);
      if (resourceType === "model") {
        setModelSlug(defaultSlug);
        setModelDisplayName(nameFromRepo(result.repo_id));
        setIncludePatterns(summaryPatternList(result.summary, "default_include_patterns") || (summaryBool(result.summary, "has_gguf") ? "*.gguf,*.json,tokenizer*" : "*.safetensors,*.json,tokenizer*,*.txt,*.jinja"));
      } else {
        setDatasetSlug(defaultSlug);
        setDatasetTitle(`${nameFromRepo(result.repo_id)} review dataset`);
        setDatasetConfig(firstConfigName(result.summary.configs));
        setDatasetSplit(firstSummaryText(result.summary.splits) || "train");
      }
      setStatusMessage(`Found ${result.resource_type} ${result.repo_id}. Review the details before starting the job.`);
      onToast(`Found ${result.resource_type} ${result.repo_id}.`, "info", "Hub resource resolved");
    } catch (err) {
      setResolved(null);
      const message = err instanceof Error ? err.message : "Unable to resolve Hugging Face repository.";
      setError(message);
      onToast(message, "error", "Hub lookup failed");
    } finally {
      setIsResolving(false);
    }
  }

  async function confirmHubResource() {
    if (!resolved) {
      setError("Find a Hugging Face repository before confirming.");
      return;
    }
    const errors = validateHubConfirmForm(resourceType, {
      modelSlug,
      modelDisplayName,
      datasetTitle,
      datasetSlug,
      datasetSplit,
      datasetMaxRows
    });
    setFormErrors(errors);
    if (Object.keys(errors).length > 0) {
      return;
    }
    setIsSubmitting(true);
    setError("");
    try {
      const job =
        resourceType === "model"
          ? await api.post<JobRecord>("/api/models/download-hf", {
              repo_id: resolved.repo_id,
              revision: revision || undefined,
              confirmed_sha: resolved.sha,
              slug: modelSlug || undefined,
              display_name: modelDisplayName || undefined,
              family: "downloaded",
              include_patterns: splitPatterns(includePatterns)
            })
          : await api.post<JobRecord>("/api/datasets/import-hf", {
              repo_id: resolved.repo_id,
              config_name: datasetConfig || undefined,
              revision: revision || undefined,
              confirmed_sha: resolved.sha,
              split: datasetSplit || "train",
              title: datasetTitle || nameFromRepo(resolved.repo_id),
              slug: datasetSlug || slugFromRepo(resolved.repo_id),
              dataset_type: datasetType,
              max_rows: Number(datasetMaxRows) || undefined,
              default_split: "holdout",
              clean_with_inference: true,
              delete_raw_after_clean: true,
              cleaning_model: cleaningModel
            });
      setLastJob(job);
      setSelectedJob(job);
      setStatusMessage(`${job.job_id} queued. The selected job now tracks download, processing, and review status.`);
      onToast(`${job.job_id} queued for ${resolved.repo_id}.`, "success", resourceType === "model" ? "Model download queued" : "Dataset import queued");
      await Promise.resolve(refresh());
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unable to start Hugging Face acquisition.";
      setError(message);
      onToast(message, "error", "Hub acquisition failed");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <section className="thx-panel thx-hub-panel">
      <div className="thx-panel-h">
        <h3>{title}</h3>
        <span className="thx-tag">[ HF · {resourceType.toUpperCase()} ]</span>
      </div>
      <div className="thx-hub-steps" aria-label="Hugging Face acquisition steps">
        {["Find", "Confirm", "Process", "Review"].map((step, index) => (
          <span className={index < stepIndex ? "is-done" : index === stepIndex ? "is-active" : ""} key={step}>
            {step}
          </span>
        ))}
      </div>
      <form className="thx-params thx-hub-find" onSubmit={findHubResource}>
        <label className="thx-field thx-field--wide">
          <span className="thx-field-label">
            <span>Hugging Face URL or repo</span>
            <span className="v">{input ? "ready" : "required"}</span>
          </span>
          <input type="text" value={input} onChange={(event) => setInput(event.target.value)} aria-invalid={Boolean(formErrors.input)} />
          <FieldNote note="Use a full Hugging Face URL or a repo id such as org/name." link="#model-support" />
          <FieldError message={formErrors.input} />
        </label>
        <label className="thx-field">
          <span className="thx-field-label">
            <span>Revision</span>
            <span className="v">{revision || "main"}</span>
          </span>
          <input type="text" value={revision} onChange={(event) => setRevision(event.target.value)} placeholder="main" />
          <FieldNote note="Leave blank to use the repository default branch." link="#datasets" />
        </label>
        <div className="thx-form-actions">
          <button type="submit" className="thx-btn" disabled={!input || isResolving}>
            <Search size={16} /> {isResolving ? "Finding" : "Find"}
          </button>
        </div>
      </form>
      {error && <div className="alert" role="alert">{error}</div>}
      {statusMessage && <div className="thx-status-line" aria-live="polite">{statusMessage}</div>}
      {resolved && (
        <div className="thx-hub-confirm" aria-live="polite">
          <div className="thx-summary">
            <div className="thx-summary-item">
              <span className="k">Repository</span>
              <span className="v">{resolved.repo_id}</span>
            </div>
            <div className="thx-summary-item">
              <span className="k">Revision SHA</span>
              <span className="v">{shortSha(resolved.sha)}</span>
            </div>
            <div className="thx-summary-item">
              <span className="k">Visibility</span>
              <span className="v">{resolved.private ? "private" : resolved.gated ? `gated ${resolved.gated}` : "public"}</span>
            </div>
            <div className="thx-summary-item">
              <span className="k">Activity</span>
              <span className="v">{formatCount(resolved.downloads)} / {formatCount(resolved.likes)}</span>
              <span className="vmono">downloads / likes</span>
            </div>
            <div className="thx-summary-item">
              <span className="k">Files</span>
              <span className="v">{summaryText(resolved.summary, "file_count")}</span>
            </div>
            {resourceType === "model" && (
              <div className="thx-summary-item">
                <span className="k">Download est.</span>
                <span className="v">{formatBytes(summaryNumber(resolved.summary, "estimated_download_size_bytes"))}</span>
                <span className="vmono">{formatCount(summaryNumber(resolved.summary, "matched_file_count"))} matched files</span>
              </div>
            )}
            <div className="thx-summary-item">
              <span className="k">{resourceType === "model" ? "Model hints" : "Dataset hints"}</span>
              <span className="v">{resourceType === "model" ? modelHint(resolved.summary) : datasetHint(resolved.summary)}</span>
            </div>
          </div>
          <div className="thx-tag-row">
            {resolved.tags.slice(0, 8).map((tag) => (
              <span className="thx-cap thx-cap--c" key={tag}>{tag}</span>
            ))}
            {resolved.tags.length === 0 && <span className="thx-cap">no tags</span>}
          </div>
          {resourceType === "model" ? (
            <div className="thx-params thx-hub-confirm-form">
              <label className="thx-field">
                <span className="thx-field-label">
                  <span>Local slug</span>
                  <span className="v">{modelSlug}</span>
                </span>
                <input type="text" value={modelSlug} onChange={(event) => setModelSlug(slugInput(event.target.value))} aria-invalid={Boolean(formErrors.modelSlug)} />
                <FieldError message={formErrors.modelSlug} />
              </label>
              <label className="thx-field">
                <span className="thx-field-label">
                  <span>Display name</span>
                  <span className="v">registry</span>
                </span>
                <input type="text" value={modelDisplayName} onChange={(event) => setModelDisplayName(event.target.value)} aria-invalid={Boolean(formErrors.modelDisplayName)} />
                <FieldError message={formErrors.modelDisplayName} />
              </label>
              <label className="thx-field thx-field--wide">
                <span className="thx-field-label">
                  <span>Include patterns</span>
                  <span className="v">files</span>
                </span>
                <input type="text" value={includePatterns} onChange={(event) => setIncludePatterns(event.target.value)} />
                <FieldNote note="Patterns keep large downloads targeted. GGUF repos default to GGUF and config files." link="#model-support" />
              </label>
            </div>
          ) : (
            <div className="thx-params thx-hub-confirm-form">
              <label className="thx-field">
                <span className="thx-field-label">
                  <span>Dataset type</span>
                  <span className="v">{datasetType}</span>
                </span>
                <select value={datasetType} onChange={(event) => setDatasetType(event.target.value)}>
                  <option value="chat_sft">Chat SFT</option>
                  <option value="math_sft">Math SFT</option>
                  <option value="holdout">Benchmark holdout</option>
                </select>
              </label>
              <label className="thx-field">
                <span className="thx-field-label">
                  <span>Config</span>
                  <span className="v">{datasetConfig || "default"}</span>
                </span>
                <input type="text" value={datasetConfig} onChange={(event) => setDatasetConfig(event.target.value)} placeholder="default" />
              </label>
              <label className="thx-field">
                <span className="thx-field-label">
                  <span>Split</span>
                  <span className="v">{datasetSplit}</span>
                </span>
                <input type="text" value={datasetSplit} onChange={(event) => setDatasetSplit(event.target.value)} aria-invalid={Boolean(formErrors.datasetSplit)} />
                <FieldError message={formErrors.datasetSplit} />
              </label>
              <label className="thx-field">
                <span className="thx-field-label">
                  <span>Title</span>
                  <span className="v">dataset</span>
                </span>
                <input type="text" value={datasetTitle} onChange={(event) => setDatasetTitle(event.target.value)} aria-invalid={Boolean(formErrors.datasetTitle)} />
                <FieldError message={formErrors.datasetTitle} />
              </label>
              <label className="thx-field">
                <span className="thx-field-label">
                  <span>Slug</span>
                  <span className="v">{datasetSlug}</span>
                </span>
                <input type="text" value={datasetSlug} onChange={(event) => setDatasetSlug(slugInput(event.target.value))} aria-invalid={Boolean(formErrors.datasetSlug)} />
                <FieldError message={formErrors.datasetSlug} />
              </label>
              <label className="thx-field">
                <span className="thx-field-label">
                  <span>Max rows</span>
                  <span className="v">{datasetMaxRows}</span>
                </span>
                <input
                  type="number"
                  min="1"
                  max="100000"
                  value={datasetMaxRows}
                  onChange={(event) => setDatasetMaxRows(event.target.value)}
                  aria-invalid={Boolean(formErrors.datasetMaxRows)}
                />
                <FieldError message={formErrors.datasetMaxRows} />
              </label>
              <div className="thx-status-line thx-field--wide">
                Cleaned with {cleaningModel}. Raw download cache is deleted after canonical JSONL is written.
              </div>
            </div>
          )}
          <div className="thx-form-actions">
            <button type="button" className="thx-btn thx-btn--primary" onClick={confirmHubResource} disabled={isSubmitting}>
              <Download size={16} /> {isSubmitting ? "Starting" : confirmLabel}
            </button>
          </div>
        </div>
      )}
      {lastJob && (
        <div className="thx-status-line">
          <strong>{lastJob.job_id}</strong>
          <span>{lastJob.job_type} - {lastJob.status}</span>
        </div>
      )}
    </section>
  );
}

function GenerateData({
  artifacts,
  activeInferenceTarget,
  refresh,
  setSelectedJob,
  onToast
}: {
  artifacts: ArtifactRecord[];
  activeInferenceTarget: InferenceTarget | null;
  refresh: () => void;
  setSelectedJob: (job: JobRecord) => void;
  onToast: ToastHandler;
}) {
  const [seedPrompt, setSeedPrompt] = useState("Create grade-school arithmetic problems with worked solutions.");
  const [teacherModel, setTeacherModel] = useState("local");
  const [targetCount, setTargetCount] = useState(100);
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [maxTokens, setMaxTokens] = useState(256);
  const [outputSchema, setOutputSchema] = useState("math_sft");
  const [validationStrictness, setValidationStrictness] = useState("normal");
  const [categoryMix, setCategoryMix] = useState("arithmetic:0.34, algebra:0.33, word-problem:0.33");
  const [difficultyMix, setDifficultyMix] = useState("easy:0.34, medium:0.33, hard:0.33");
  const [dryRun, setDryRun] = useState(false);
  const teacherArtifacts = useMemo(
    () => artifacts.filter((artifact) => GENERATE_TEACHER_ARTIFACT_TYPES.has(artifact.artifact_type)),
    [artifacts]
  );

  useEffect(() => {
    if (activeInferenceTarget) {
      setTeacherModel(inferenceTargetRuntimeValue(activeInferenceTarget));
    }
  }, [activeInferenceTarget?.target_type, activeInferenceTarget?.model_slug, activeInferenceTarget?.artifact_id]);

  async function start() {
    try {
      const job = await api.post<JobRecord>("/api/jobs/generate", {
        teacher_model: teacherModel,
        seed_prompt: seedPrompt,
        target_count: targetCount,
        category_mix: parseMixInput(categoryMix),
        difficulty_mix: parseMixInput(difficultyMix),
        temperature,
        top_p: topP,
        max_tokens: maxTokens,
        output_schema: outputSchema,
        validation_strictness: validationStrictness,
        use_teacher_model: teacherModel !== "local",
        dry_run: dryRun
      });
      setSelectedJob(job);
      onToast(`${job.job_id} queued for ${targetCount} generated rows.`, "success", "Generation queued");
      await refresh();
    } catch (err) {
      onToast(err instanceof Error ? err.message : "Generation job could not be queued.", "error", "Generation failed");
    }
  }

  async function importGenerated(artifact: ArtifactRecord) {
    try {
      await api.post("/api/datasets/import-generated", {
        artifact_id: artifact.artifact_id,
        title: artifact.display_name,
        slug: "generated-review",
        dataset_type: "math_sft"
      });
      onToast(`${artifact.display_name} imported into the dataset review queue.`, "success", "Dataset created");
      await refresh();
    } catch (err) {
      onToast(err instanceof Error ? err.message : "Generated dataset could not be imported.", "error", "Import failed");
    }
  }

  const generatedArtifacts = artifacts.filter((artifact) => artifact.artifact_type === "generated_dataset");
  const activeTeacher = activeInferenceTarget ? activeInferenceTarget.display_name : "local deterministic generator";

  return (
    <div className="thx thx-page thx-generate">
      <div className="thx-stage-h">
        <div>
          <div className="crumb">DATA · SYNTH · 01 · GENERATE</div>
          <h2>
            <span className="thx-glitch" data-text="LOCAL EXAMPLE GENERATION">
              LOCAL EXAMPLE GENERATION
            </span>
          </h2>
          <p className="lede">Create candidate review rows with the active runtime or deterministic smoke path.</p>
        </div>
        <div className="stamp">
          ACTIVE
          <span>{activeTeacher}</span>
        </div>
      </div>

      <div className="thx-page-grid thx-page-grid--wide">
        <section className="thx-panel thx-panel--accent">
          <div className="thx-panel-h">
            <h3>Generation Parameters</h3>
            <span className="thx-tag">[ JOB · GENERATE ]</span>
          </div>
          <div className="thx-params">
            <div className="thx-field thx-field--wide">
              <span className="thx-field-label">
                <span>Teacher model</span>
                <span className="v">{teacherModel === "local" ? "LOCAL" : "SELECTED"}</span>
              </span>
              <div className="thx-seg thx-teacher-seg">
                <button type="button" className={`thx-seg-item ${teacherModel === "local" ? "is-active" : ""}`} onClick={() => setTeacherModel("local")}>
                  Local smoke
                  <span className="sub">deterministic generator</span>
                </button>
                <button
                  type="button"
                  className={`thx-seg-item ${activeInferenceTarget && teacherModel === inferenceTargetRuntimeValue(activeInferenceTarget) ? "is-active" : ""}`}
                  onClick={() => activeInferenceTarget && setTeacherModel(inferenceTargetRuntimeValue(activeInferenceTarget))}
                  disabled={!activeInferenceTarget}
                >
                  Active runtime
                  <span className="sub">{activeInferenceTarget?.display_name || "set in Models"}</span>
                </button>
              </div>
              <ArtifactPicker
                artifacts={teacherArtifacts}
                selectedValue={teacherModel}
                valueMode="path"
                onSelect={(artifact) => setTeacherModel(artifact?.path || "local")}
                emptyMessage="NO TEACHER ARTIFACTS AVAILABLE"
                className="thx-teacher-artifacts"
              />
              <FieldNote note="The worker starts a job-scoped teacher only when real generation is enabled." link="#teacher-model" />
            </div>
            <label className="thx-field thx-field--wide">
              <span className="thx-field-label">
                <span>Seed prompt</span>
                <span className="v">{seedPrompt.length} chars</span>
              </span>
              <textarea value={seedPrompt} onChange={(event) => setSeedPrompt(event.target.value)} />
            </label>
            <label className="thx-field">
              <span className="thx-field-label">
                <span>Target count</span>
                <span className="v">{targetCount}</span>
              </span>
              <input type="number" min={1} max={5000} value={targetCount} onChange={(event) => setTargetCount(Number(event.target.value))} />
            </label>
            <label className="thx-field">
              <span className="thx-field-label">
                <span>Temperature</span>
                <span className="v">{temperature.toFixed(2)}</span>
              </span>
              <input type="number" min={0} max={2} step={0.05} value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} />
            </label>
            <label className="thx-field">
              <span className="thx-field-label">
                <span>Top P</span>
                <span className="v">{topP.toFixed(2)}</span>
              </span>
              <input type="number" min={0.05} max={1} step={0.05} value={topP} onChange={(event) => setTopP(Number(event.target.value))} />
            </label>
            <label className="thx-field">
              <span className="thx-field-label">
                <span>Max tokens</span>
                <span className="v">{maxTokens}</span>
              </span>
              <input type="number" min={16} max={4096} value={maxTokens} onChange={(event) => setMaxTokens(Number(event.target.value))} />
            </label>
            <label className="thx-field">
              <span className="thx-field-label">
                <span>Output schema</span>
                <span className="v">{outputSchema}</span>
              </span>
              <select value={outputSchema} onChange={(event) => setOutputSchema(event.target.value)}>
                <option value="math_sft">Math SFT</option>
                <option value="chat_sft">Chat SFT</option>
              </select>
            </label>
            <label className="thx-field">
              <span className="thx-field-label">
                <span>Validation</span>
                <span className="v">{validationStrictness}</span>
              </span>
              <select value={validationStrictness} onChange={(event) => setValidationStrictness(event.target.value)}>
                <option value="normal">Normal</option>
                <option value="strict">Strict</option>
              </select>
            </label>
            <label className="thx-field">
              <span className="thx-field-label">
                <span>Category mix</span>
                <span className="v">weighted</span>
              </span>
              <input type="text" value={categoryMix} onChange={(event) => setCategoryMix(event.target.value)} />
            </label>
            <label className="thx-field">
              <span className="thx-field-label">
                <span>Difficulty mix</span>
                <span className="v">weighted</span>
              </span>
              <input type="text" value={difficultyMix} onChange={(event) => setDifficultyMix(event.target.value)} />
            </label>
            <label className="thx-field thx-field--toggle">
              <span className="thx-field-label">
                <span>Dry run</span>
                <span className="v">{dryRun ? "on" : "off"}</span>
              </span>
              <span className="thx-toggle">
                <input type="checkbox" checked={dryRun} onChange={(event) => setDryRun(event.target.checked)} />
                <span className="thx-toggle-track" />
                <span className="thx-toggle-copy">Use deterministic smoke output</span>
              </span>
            </label>
          </div>
          <div className="thx-form-actions">
            <button type="button" className="thx-btn thx-btn--primary" onClick={start} disabled={!seedPrompt.trim() || targetCount < 1}>
              <Play size={16} /> Start Generation
            </button>
          </div>
        </section>

        <section className="thx-panel">
          <div className="thx-panel-h">
            <h3>Review Queue</h3>
            <span className="thx-tag">[ ARTIFACTS · {pad2(generatedArtifacts.length)} ]</span>
          </div>
          {generatedArtifacts.length === 0 ? (
            <div className="thx-empty">NO GENERATED DATASET ARTIFACTS</div>
          ) : (
            <div className="thx-runs thx-artifacts">
              {generatedArtifacts.map((artifact) => (
                <article className="thx-run thx-artifact-run" key={artifact.artifact_id}>
                  <span className="thx-run-dot" />
                  <span className="thx-run-id">{artifact.display_name}</span>
                  <span className="thx-run-meta">{formatBytes(artifact.size_bytes)}</span>
                  <button type="button" className="thx-btn" onClick={() => importGenerated(artifact)}>
                    <Database size={15} /> Create Review Dataset
                  </button>
                  <span className="thx-artifact-path">{artifact.path}</span>
                </article>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

function Benchmarks({
  models,
  artifacts,
  activeInferenceTarget,
  refresh,
  setSelectedJob,
  onToast
}: {
  models: ModelRecord[];
  artifacts: ArtifactRecord[];
  activeInferenceTarget: InferenceTarget | null;
  refresh: () => void;
  setSelectedJob: (job: JobRecord) => void;
  onToast: ToastHandler;
}) {
  const [modelSlug, setModelSlug] = useState("lfm25-12b-base");
  const [checkpointArtifactId, setCheckpointArtifactId] = useState("");
  const [selectedBenchmarkIds, setSelectedBenchmarkIds] = useState<string[]>([]);
  const [catalog, setCatalog] = useState<BenchmarkCatalogItem[]>([]);
  const [historyResults, setHistoryResults] = useState<BenchmarkResultRecord[]>([]);
  const [scoreboardResults, setScoreboardResults] = useState<BenchmarkResultRecord[]>([]);
  const [historyModelSlug, setHistoryModelSlug] = useState("");
  const [historyBenchmark, setHistoryBenchmark] = useState("");
  const [compareModelSlugs, setCompareModelSlugs] = useState<string[]>([]);
  const [limit, setLimit] = useState(10);
  const [majK, setMajK] = useState(1);
  const [dryRun, setDryRun] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const benchmarkModels = useMemo(() => models.filter((model) => model.supports_benchmark), [models]);
  const checkpointArtifacts = useMemo(
    () => artifacts.filter((artifact) => BENCHMARK_CHECKPOINT_ARTIFACT_TYPES.has(artifact.artifact_type)),
    [artifacts]
  );
  const selectedCheckpoint = checkpointArtifacts.find((artifact) => artifact.artifact_id === checkpointArtifactId);
  const checkpointPath = selectedCheckpoint?.path || "";
  const selectedCatalogItems = useMemo(
    () => catalog.filter((item) => selectedBenchmarkIds.includes(item.id)),
    [catalog, selectedBenchmarkIds]
  );
  const catalogGroups = useMemo(() => groupBenchmarkCatalog(catalog), [catalog]);
  const catalogById = useMemo(() => new Map(catalog.map((item) => [item.id, item])), [catalog]);
  const selectedModel = benchmarkModels.find((model) => model.slug === modelSlug);
  const latestScores = useMemo(() => latestBenchmarkScores(scoreboardResults), [scoreboardResults]);
  const compareBenchmarkIds = selectedBenchmarkIds.length > 0 ? selectedBenchmarkIds : catalog.map((item) => item.id);

  useEffect(() => {
    if (!activeInferenceTarget) {
      return;
    }
    if (activeInferenceTarget.target_type === "base_model" && activeInferenceTarget.model_slug) {
      setModelSlug(activeInferenceTarget.model_slug);
      setCheckpointArtifactId("");
      return;
    }
    if (activeInferenceTarget.model_slug) {
      setModelSlug(activeInferenceTarget.model_slug);
    }
    if (activeInferenceTarget.artifact_id) {
      setCheckpointArtifactId(activeInferenceTarget.artifact_id);
    }
  }, [activeInferenceTarget?.target_type, activeInferenceTarget?.model_slug, activeInferenceTarget?.artifact_id]);

  useEffect(() => {
    if (benchmarkModels.length > 0 && !benchmarkModels.some((model) => model.slug === modelSlug)) {
      setModelSlug(benchmarkModels[0].slug);
    }
  }, [benchmarkModels, modelSlug]);

  useEffect(() => {
    setCompareModelSlugs((current) => {
      const valid = current.filter((slug) => benchmarkModels.some((model) => model.slug === slug)).slice(0, 3);
      return valid.length > 0 ? valid : benchmarkModels.slice(0, 2).map((model) => model.slug);
    });
  }, [benchmarkModels]);

  useEffect(() => {
    let cancelled = false;
    async function loadCatalog() {
      try {
        const nextCatalog = await api.get<BenchmarkCatalogItem[]>("/api/benchmarks/catalog");
        if (cancelled) {
          return;
        }
        setCatalog(nextCatalog);
        setSelectedBenchmarkIds((current) => {
          if (current.length > 0) {
            return current.filter((id) => nextCatalog.some((item) => item.id === id));
          }
          const defaults = nextCatalog.filter((item) => item.family === "Math").slice(0, 2).map((item) => item.id);
          return defaults.length > 0 ? defaults : nextCatalog.slice(0, 1).map((item) => item.id);
        });
      } catch (err) {
        if (!cancelled) {
          const message = err instanceof Error ? err.message : "Unable to load benchmark catalog.";
          setStatusMessage(message);
          onToast(message, "error", "Catalog unavailable");
        }
      }
    }
    void loadCatalog();
    void loadScoreboardResults();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function loadFilteredResults() {
      try {
        const nextResults = await api.get<BenchmarkResultRecord[]>(
          buildBenchmarkResultsPath({ modelSlug: historyModelSlug, benchmark: historyBenchmark, limit: 60 })
        );
        if (!cancelled) {
          setHistoryResults(nextResults);
        }
      } catch (err) {
        if (!cancelled) {
          const message = err instanceof Error ? err.message : "Unable to load benchmark results.";
          setStatusMessage(message);
          onToast(message, "error", "Benchmark history unavailable");
        }
      }
    }
    void loadFilteredResults();
    return () => {
      cancelled = true;
    };
  }, [historyModelSlug, historyBenchmark]);

  async function loadScoreboardResults() {
    try {
      const nextResults = await api.get<BenchmarkResultRecord[]>(buildBenchmarkResultsPath({ limit: 200 }));
      setScoreboardResults(nextResults);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unable to load comparison results.";
      setStatusMessage(message);
      onToast(message, "error", "Comparison unavailable");
    }
  }

  async function refreshHistoryResults() {
    try {
      const nextResults = await api.get<BenchmarkResultRecord[]>(
        buildBenchmarkResultsPath({ modelSlug: historyModelSlug, benchmark: historyBenchmark, limit: 60 })
      );
      setHistoryResults(nextResults);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unable to refresh benchmark history.";
      setStatusMessage(message);
      onToast(message, "error", "Refresh failed");
    }
  }

  async function start() {
    if (!modelSlug || selectedBenchmarkIds.length === 0) {
      return;
    }
    try {
      const job = await api.post<JobRecord>("/api/jobs/benchmark", {
        model_slug: modelSlug,
        checkpoint_path: checkpointPath || undefined,
        benchmarks: selectedBenchmarkIds,
        limit,
        maj_k: majK,
        prompt_template: "default_cot",
        max_new_tokens: 512,
        dry_run: dryRun
      });
      setSelectedJob(job);
      setStatusMessage(`${job.job_id} queued for ${selectedBenchmarkIds.length} benchmark suite${selectedBenchmarkIds.length === 1 ? "" : "s"}.`);
      onToast(`${job.job_id} queued for ${selectedBenchmarkIds.length} benchmark suite${selectedBenchmarkIds.length === 1 ? "" : "s"}.`, "success", "Benchmark queued");
      await refresh();
      await Promise.all([refreshHistoryResults(), loadScoreboardResults()]);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Benchmark job could not be queued.";
      setStatusMessage(message);
      onToast(message, "error", "Benchmark failed");
    }
  }

  function toggleBenchmark(benchmarkId: string) {
    setSelectedBenchmarkIds((current) =>
      current.includes(benchmarkId) ? current.filter((item) => item !== benchmarkId) : [...current, benchmarkId]
    );
  }

  function applyPreset(kind: "smoke" | "full") {
    const items = selectedCatalogItems.length > 0 ? selectedCatalogItems : catalog;
    const values = items.map((item) => (kind === "smoke" ? item.smoke_default : item.full_default));
    if (values.length > 0) {
      setLimit(Math.max(...values));
    }
  }

  function toggleCompareModel(slug: string) {
    setCompareModelSlugs((current) => {
      if (current.includes(slug)) {
        return current.filter((item) => item !== slug);
      }
      return current.length >= 3 ? current : [...current, slug];
    });
  }

  return (
    <div className="thx thx-page thx-bench">
      <div className="thx-stage-h">
        <div>
          <div className="crumb">EVAL · SCOREBOARD · 04 · BENCHMARKS</div>
          <h2>
            <span className="thx-glitch" data-text="BENCHMARKS">
              BENCHMARKS
            </span>
          </h2>
          <p className="lede">Queue popular evaluation suites, inspect result history, and compare model scores.</p>
        </div>
        <div className="stamp">
          SELECTED
          <span>{pad2(selectedBenchmarkIds.length)} suites</span>
        </div>
      </div>

      <div className="thx-bench-grid">
        <section className="thx-panel thx-panel--accent">
          <div className="thx-panel-h">
            <h3>Catalog & Submit</h3>
            <span className="thx-tag">[ CATALOG · {pad2(catalog.length)} ]</span>
          </div>
          <div className="thx-section-stack">
            <div className="thx-bench-subhead">
              <span>Model target</span>
              <span>{selectedModel?.display_name || "unselected"}</span>
            </div>
            <div className="thx-cards thx-card-grid--compact">
              {benchmarkModels.map((model) => (
                <button
                  type="button"
                  className={`thx-card ${model.slug === modelSlug ? "is-selected" : ""}`}
                  onClick={() => setModelSlug(model.slug)}
                  key={model.slug}
                >
                  <span className="thx-card-row">
                    <span className="thx-card-title">{model.display_name}</span>
                    <span className="thx-card-status">{model.family}</span>
                  </span>
                  <span className="thx-card-sub">{model.provider_id}</span>
                  <span className="thx-card-stats">
                    <span className="thx-card-stat">
                      <span className="k">Params</span>
                      <span className="v">{model.parameter_count || "unknown"}</span>
                    </span>
                    <span className="thx-card-stat">
                      <span className="k">Context</span>
                      <span className="v">{formatCount(model.max_sequence_length)}</span>
                    </span>
                  </span>
                </button>
              ))}
              {benchmarkModels.length === 0 && <div className="thx-empty">NO BENCHMARK-CAPABLE MODELS REGISTERED</div>}
            </div>

            <div className="thx-bench-subhead">
              <span>Checkpoint artifact</span>
              <span>{checkpointPath ? "artifact override" : "model default"}</span>
            </div>
            <ArtifactPicker
              artifacts={checkpointArtifacts}
              selectedValue={checkpointArtifactId}
              valueMode="artifact_id"
              onSelect={(artifact) => setCheckpointArtifactId(artifact?.artifact_id || "")}
              emptyMessage="NO CHECKPOINT ARTIFACTS AVAILABLE"
              defaultOption={{
                label: "Model default",
                value: "",
                detail: selectedModel?.provider_id || "select a model"
              }}
              className="thx-card-grid--compact"
            />

            <div className="thx-bench-catalog">
              {catalogGroups.map((group) => (
                <div className="thx-bench-family" key={group.family}>
                  <div className="thx-bench-family-h">
                    <span>{group.family}</span>
                    <span>{pad2(group.items.length)}</span>
                  </div>
                  <div className="thx-seg thx-bench-suites">
                    {group.items.map((benchmark) => (
                      <button
                        type="button"
                        className={`thx-seg-item ${selectedBenchmarkIds.includes(benchmark.id) ? "is-active" : ""}`}
                        onClick={() => toggleBenchmark(benchmark.id)}
                        key={benchmark.id}
                      >
                        {benchmark.label}
                        <span className="sub">{benchmark.description}</span>
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            <div className="thx-params">
              <label className="thx-field thx-field--wide">
                <span className="thx-field-label">
                  <span>Preset</span>
                  <span className="v">{limit}</span>
                </span>
                <div className="thx-bench-preset-row">
                  <button type="button" className="thx-btn" onClick={() => applyPreset("smoke")}>
                    Smoke
                  </button>
                  <button type="button" className="thx-btn" onClick={() => applyPreset("full")}>
                    Full
                  </button>
                </div>
                <FieldNote
                  note="Smoke uses the catalog's quick defaults. Full raises the per-suite cap to the selected suite defaults."
                  link="#benchmark-presets"
                />
              </label>
              <label className="thx-field">
                <span className="thx-field-label">
                  <span>Limit</span>
                  <span className="v">{limit}</span>
                </span>
                <input type="number" min={1} max={10000} value={limit} onChange={(event) => setLimit(Number(event.target.value))} />
                <FieldNote note="Use small limits for acceptance smoke, then raise for final comparisons." link="#benchmark-limit" />
              </label>
              <label className="thx-field">
                <span className="thx-field-label">
                  <span>Maj@k</span>
                  <span className="v">{majK}</span>
                </span>
                <input type="number" min={1} max={64} value={majK} onChange={(event) => setMajK(Number(event.target.value))} />
                <FieldNote note="Majority vote generates multiple answers and compares the most common result." link="#maj-k" />
              </label>
              <label className="thx-field thx-field--toggle">
                <span className="thx-field-label">
                  <span>Dry run</span>
                  <span className="v">{dryRun ? "on" : "off"}</span>
                </span>
                <span className="thx-toggle">
                  <input type="checkbox" checked={dryRun} onChange={(event) => setDryRun(event.target.checked)} />
                  <span className="thx-toggle-track" />
                  <span className="thx-toggle-copy">Force deterministic smoke results</span>
                </span>
              </label>
            </div>
            {statusMessage && <div className="thx-status-line">{statusMessage}</div>}
            <div className="thx-form-actions">
              <button type="button" className="thx-btn thx-btn--primary" onClick={start} disabled={!modelSlug || selectedBenchmarkIds.length === 0}>
                <Play size={16} /> Run Benchmark
              </button>
            </div>
          </div>
        </section>

        <section className="thx-panel">
          <div className="thx-panel-h">
            <h3>History</h3>
            <span className="thx-tag">[ RESULTS · {pad2(historyResults.length)} ]</span>
          </div>
          <div className="thx-params thx-bench-filters">
            <label className="thx-field">
              <span className="thx-field-label">
                <span>Model filter</span>
                <span className="v">{historyModelSlug || "all"}</span>
              </span>
              <select value={historyModelSlug} onChange={(event) => setHistoryModelSlug(event.target.value)}>
                <option value="">All models</option>
                {benchmarkModels.map((model) => (
                  <option key={model.slug} value={model.slug}>
                    {model.display_name}
                  </option>
                ))}
              </select>
            </label>
            <label className="thx-field">
              <span className="thx-field-label">
                <span>Benchmark</span>
                <span className="v">{historyBenchmark || "all"}</span>
              </span>
              <select value={historyBenchmark} onChange={(event) => setHistoryBenchmark(event.target.value)}>
                <option value="">All suites</option>
                {catalog.map((benchmark) => (
                  <option key={benchmark.id} value={benchmark.id}>
                    {benchmark.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          {historyResults.length === 0 ? (
            <div className="thx-empty thx-history-empty">NO BENCHMARK RESULTS MATCH THE CURRENT FILTER</div>
          ) : (
            <div className="thx-runs thx-bench-history">
              {historyResults.map((result) => {
                const score = benchmarkMetricScore(result.metrics);
                const content = (
                  <>
                    <span className="thx-run-dot" />
                    <span className="thx-run-id">
                      {catalogById.get(result.benchmark_name)?.label || result.benchmark_name}
                      <span className="thx-bench-result-sub">{result.model_slug}</span>
                    </span>
                    <span className="thx-run-meta">{formatEpochTime(result.created_at)}</span>
                    <span className="thx-run-status">{formatBenchmarkScore(score)}</span>
                  </>
                );
                return result.artifact_id ? (
                  <a
                    className="thx-run thx-bench-result"
                    href={`/api/artifacts/${result.artifact_id}/download`}
                    target="_blank"
                    rel="noreferrer"
                    key={result.result_id}
                  >
                    {content}
                  </a>
                ) : (
                  <article className="thx-run thx-bench-result" key={result.result_id}>
                    {content}
                  </article>
                );
              })}
            </div>
          )}
        </section>

        <section className="thx-panel thx-bench-compare-panel">
          <div className="thx-panel-h">
            <h3>Compare</h3>
            <span className="thx-tag">[ MODELS · {pad2(compareModelSlugs.length)} ]</span>
          </div>
          <div className="thx-bench-compare-pickers">
            {benchmarkModels.map((model) => (
              <button
                type="button"
                className={`thx-cap ${compareModelSlugs.includes(model.slug) ? "thx-cap--ok" : "thx-cap--c"}`}
                onClick={() => toggleCompareModel(model.slug)}
                disabled={!compareModelSlugs.includes(model.slug) && compareModelSlugs.length >= 3}
                key={model.slug}
              >
                {model.display_name}
              </button>
            ))}
          </div>
          {compareModelSlugs.length === 0 ? (
            <div className="thx-empty">SELECT 2-3 MODELS TO COMPARE</div>
          ) : (
            <div className="thx-bench-bars">
              {compareBenchmarkIds.map((benchmarkId) => (
                <div className="thx-bench-bar-group" key={benchmarkId}>
                  <div className="thx-bench-bar-label">{catalogById.get(benchmarkId)?.label || benchmarkId}</div>
                  {compareModelSlugs.map((slug) => {
                    const result = latestScores.get(`${slug}:${benchmarkId}`);
                    const score = result ? benchmarkMetricScore(result.metrics) : null;
                    return (
                      <div className="thx-bench-bar-row" key={`${slug}:${benchmarkId}`}>
                        <span>{models.find((model) => model.slug === slug)?.display_name || slug}</span>
                        <div className="thx-bench-bar-track">
                          <span className="thx-bench-bar-fill" style={{ width: `${Math.max((score || 0) * 100, score === null ? 0 : 2)}%` }} />
                        </div>
                        <strong>{formatBenchmarkScore(score)}</strong>
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

const BENCHMARK_CHECKPOINT_ARTIFACT_TYPES = new Set([
  "training_checkpoint",
  "training_merged_checkpoint",
  "downloaded_model",
  "gguf_quantized"
]);

function groupBenchmarkCatalog(catalog: BenchmarkCatalogItem[]) {
  const groups: { family: string; items: BenchmarkCatalogItem[] }[] = [];
  const groupByFamily = new Map<string, BenchmarkCatalogItem[]>();
  for (const item of catalog) {
    const group = groupByFamily.get(item.family);
    if (group) {
      group.push(item);
    } else {
      const items = [item];
      groupByFamily.set(item.family, items);
      groups.push({ family: item.family, items });
    }
  }
  return groups;
}

function buildBenchmarkResultsPath({
  modelSlug,
  benchmark,
  limit
}: {
  modelSlug?: string;
  benchmark?: string;
  limit: number;
}) {
  const params = new URLSearchParams();
  if (modelSlug) {
    params.set("model_slug", modelSlug);
  }
  if (benchmark) {
    params.set("benchmark", benchmark);
  }
  params.set("limit", String(limit));
  return `/api/benchmarks/results?${params.toString()}`;
}

function latestBenchmarkScores(results: BenchmarkResultRecord[]) {
  const latest = new Map<string, BenchmarkResultRecord>();
  for (const result of results) {
    const key = `${result.model_slug}:${result.benchmark_name}`;
    if (!latest.has(key)) {
      latest.set(key, result);
    }
  }
  return latest;
}

function benchmarkMetricScore(metrics: Record<string, unknown>): number | null {
  for (const key of ["pass_at_1", "accuracy", "acc", "exact_match", "flexible_match"]) {
    const value = metrics[key];
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
  }
  return null;
}

function formatBenchmarkScore(score: number | null) {
  return score === null ? "pending" : `${Math.round(score * 1000) / 10}%`;
}

function Quantize({
  artifacts,
  refresh,
  setSelectedJob,
  onToast
}: {
  artifacts: ArtifactRecord[];
  refresh: () => void;
  setSelectedJob: (job: JobRecord) => void;
  onToast: ToastHandler;
}) {
  const [operation, setOperation] = useState("quantize");
  const [sourceGguf, setSourceGguf] = useState("");
  const [sourceCheckpoint, setSourceCheckpoint] = useState("");
  const [quantType, setQuantType] = useState("Q4_K_M");
  const checkpointArtifacts = useMemo(
    () => artifacts.filter((artifact) => ["training_checkpoint", "training_merged_checkpoint", "downloaded_model"].includes(artifact.artifact_type)),
    [artifacts]
  );
  const ggufArtifacts = useMemo(
    () => artifacts.filter((artifact) => artifact.artifact_type === "gguf_fp16" || (artifact.path.toLowerCase().endsWith(".gguf") && artifact.artifact_type !== "gguf_quantized")),
    [artifacts]
  );
  const quantPresets = [
    { id: "Q4_K_M", label: "Q4_K_M", detail: "8 GB consumer" },
    { id: "Q5_K_M", label: "Q5_K_M", detail: "desktop" },
    { id: "Q8_0", label: "Q8_0", detail: "server" }
  ];

  async function start() {
    try {
      const job =
        operation === "convert_gguf"
          ? await api.post<JobRecord>("/api/jobs/convert-gguf", {
              source_checkpoint: sourceCheckpoint,
              output_name: "traininghub-output",
              outtype: "f16"
            })
          : await api.post<JobRecord>("/api/jobs/quantize", {
              source_gguf: sourceGguf,
              quant_type: quantType
            });
      setSelectedJob(job);
      onToast(`${job.job_id} queued for ${operation === "convert_gguf" ? "GGUF conversion" : quantType}.`, "success", "Quantize job queued");
      await refresh();
    } catch (err) {
      onToast(err instanceof Error ? err.message : "Quantize pipeline job could not be queued.", "error", "Quantize failed");
    }
  }

  const sourceReady = operation === "convert_gguf" ? Boolean(sourceCheckpoint) : Boolean(sourceGguf);
  const activeArtifacts = operation === "convert_gguf" ? checkpointArtifacts : ggufArtifacts;
  const activeSourcePath = operation === "convert_gguf" ? sourceCheckpoint : sourceGguf;

  return (
    <div className="thx thx-page thx-quant">
      <div className="thx-stage-h">
        <div>
          <div className="crumb">MODEL · PACK · 03 · QUANTIZE</div>
          <h2>
            <span className="thx-glitch" data-text="GGUF PIPELINE">
              GGUF PIPELINE
            </span>
          </h2>
          <p className="lede">Select a source artifact, convert when needed, then emit a runnable quantized GGUF.</p>
        </div>
        <div className="stamp">
          MODE
          <span>{operation === "convert_gguf" ? "CONVERT" : "QUANTIZE"}</span>
        </div>
      </div>

      <section className="thx-panel thx-panel--accent">
        <div className="thx-panel-h">
          <h3>Pipeline</h3>
          <span className="thx-tag">[ SOURCE · CONVERT · QUANTIZE · DONE ]</span>
        </div>
        <div className="thx-pipe">
          <div className={`thx-pipe-node ${sourceReady ? "is-done" : "is-active"}`}>
            <span className="thx-pipe-node-ring" />
            <span className="thx-pipe-node-label">Source</span>
            <span className="thx-pipe-node-hint">{sourceReady ? "selected" : "required"}</span>
          </div>
          <span className={`thx-pipe-edge ${sourceReady ? "is-flow" : "is-active"}`} />
          <div className={`thx-pipe-node ${operation === "convert_gguf" && sourceReady ? "is-active" : operation === "quantize" ? "is-done" : ""}`}>
            <span className="thx-pipe-node-ring" />
            <span className="thx-pipe-node-label">Convert</span>
            <span className="thx-pipe-node-hint">HF to GGUF</span>
          </div>
          <span className={`thx-pipe-edge ${sourceReady ? "is-active" : ""}`} />
          <div className={`thx-pipe-node ${operation === "quantize" && sourceReady ? "is-active" : ""}`}>
            <span className="thx-pipe-node-ring" />
            <span className="thx-pipe-node-label">Quantize</span>
            <span className="thx-pipe-node-hint">{quantType}</span>
          </div>
          <span className="thx-pipe-edge" />
          <div className="thx-pipe-node">
            <span className="thx-pipe-node-ring" />
            <span className="thx-pipe-node-label">Done</span>
            <span className="thx-pipe-node-hint">artifact</span>
          </div>
        </div>
      </section>

      <div className="thx-page-grid">
        <section className="thx-panel">
          <div className="thx-panel-h">
            <h3>Source</h3>
            <span className="thx-tag">[ ARTIFACT PICKER · {pad2(activeArtifacts.length)} ]</span>
          </div>
          <div className="thx-seg thx-operation-seg">
            <button type="button" className={`thx-seg-item ${operation === "convert_gguf" ? "is-active" : ""}`} onClick={() => setOperation("convert_gguf")}>
              Convert HF to GGUF
              <span className="sub">checkpoint source</span>
            </button>
            <button type="button" className={`thx-seg-item ${operation === "quantize" ? "is-active" : ""}`} onClick={() => setOperation("quantize")}>
              Quantize GGUF
              <span className="sub">FP16/BF16 source</span>
            </button>
          </div>
          <ArtifactPicker
            artifacts={activeArtifacts}
            selectedValue={activeSourcePath}
            valueMode="path"
            onSelect={(artifact) => {
              if (operation === "convert_gguf") {
                setSourceCheckpoint(artifact?.path || "");
              } else {
                setSourceGguf(artifact?.path || "");
              }
            }}
            emptyMessage="NO MATCHING SOURCE ARTIFACTS"
            className="thx-source-cards"
          />
          <div className="thx-field thx-field--wide">
            <span className="thx-field-label">
              <span>{operation === "convert_gguf" ? "Source HF checkpoint" : "Source FP16/BF16 GGUF"}</span>
              <span className="v">{sourceReady ? "ready" : "empty"}</span>
            </span>
            <div className="thx-path-readout">{activeSourcePath || "Pick an artifact above."}</div>
            <FieldNote
              note={
                operation === "convert_gguf"
                  ? "Merged or full training checkpoints can be converted directly; adapters must be merged during training first."
                  : "TrainingHub rejects re-quantizing files already marked with a quant type."
              }
              link={operation === "convert_gguf" ? "#training" : "#quantization"}
            />
          </div>
        </section>

        <section className="thx-panel">
          <div className="thx-panel-h">
            <h3>Quant Type</h3>
            <span className="thx-tag">[ PRESET · {quantType} ]</span>
          </div>
          <div className="thx-seg">
            {quantPresets.map((preset) => (
              <button
                type="button"
                className={`thx-seg-item ${quantType === preset.id ? "is-active" : ""}`}
                disabled={operation === "convert_gguf"}
                onClick={() => setQuantType(preset.id)}
                key={preset.id}
              >
                {preset.label}
                <span className="sub">{preset.detail}</span>
              </button>
            ))}
          </div>
          <div className="thx-form-actions">
            <button type="button" className="thx-btn thx-btn--primary" onClick={start} disabled={!sourceReady}>
              <Play size={16} /> Start Pipeline Job
            </button>
          </div>
        </section>

        <section className="thx-panel thx-page-span">
          <div className="thx-panel-h">
            <h3>Artifacts</h3>
            <span className="thx-tag">[ REGISTRY · {pad2(artifacts.length)} ]</span>
          </div>
          {artifacts.length === 0 ? (
            <div className="thx-empty">NO ARTIFACTS REGISTERED</div>
          ) : (
            <div className="thx-runs thx-artifacts">
              {artifacts.map((artifact) => (
                <article className="thx-run thx-artifact-run" key={artifact.artifact_id}>
                  <span className="thx-run-dot" />
                  <span className="thx-run-id">{artifact.display_name}</span>
                  <span className="thx-run-meta">{artifact.artifact_type}</span>
                  <span className="thx-run-status">{formatBytes(artifact.size_bytes)}</span>
                  <span className="thx-artifact-path">{artifact.path}</span>
                </article>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

function Models({
  models,
  artifacts,
  activeInferenceTarget,
  capabilityTransfers,
  inferenceOptions,
  refresh,
  setSelectedJob,
  onToast
}: {
  models: ModelRecord[];
  artifacts: ArtifactRecord[];
  activeInferenceTarget: InferenceTarget | null;
  capabilityTransfers: CapabilityTransferRecord[];
  inferenceOptions: InferenceOption[];
  refresh: () => void;
  setSelectedJob: (job: JobRecord) => void;
  onToast: ToastHandler;
}) {
  const defaultModelRepoId = "Alex2790/LFM2-8B-A1B-GGUF";
  const [modelUrl, setModelUrl] = useState("");
  const [urlSlug, setUrlSlug] = useState("url-model");
  const [urlDisplayName, setUrlDisplayName] = useState("URL model");
  const [uploadRepoId, setUploadRepoId] = useState("");
  const [uploadModelSlug, setUploadModelSlug] = useState("");
  const [uploadArtifactId, setUploadArtifactId] = useState("");
  const [uploadSourcePath, setUploadSourcePath] = useState("");
  const [uploadPrivate, setUploadPrivate] = useState(true);
  const [deleteMessage, setDeleteMessage] = useState("");
  const [deleteCandidate, setDeleteCandidate] = useState<ModelRecord | null>(null);

  useEffect(() => {
    if (!uploadModelSlug && models.length > 0) {
      setUploadModelSlug(models[0].slug);
    }
  }, [models, uploadModelSlug]);

  async function downloadFromUrl(event: FormEvent) {
    event.preventDefault();
    try {
      const job = await api.post<JobRecord>("/api/models/download-url", {
        url: modelUrl,
        slug: urlSlug || undefined,
        display_name: urlDisplayName || undefined,
        family: "downloaded"
      });
      setSelectedJob(job);
      onToast(`${job.job_id} queued for URL model transfer.`, "success", "Model download queued");
      await refresh();
    } catch (err) {
      onToast(err instanceof Error ? err.message : "URL model download could not be queued.", "error", "Download failed");
    }
  }

  async function uploadToHf(event: FormEvent) {
    event.preventDefault();
    try {
      const job = await api.post<JobRecord>("/api/models/upload-hf", {
        repo_id: uploadRepoId,
        model_slug: uploadArtifactId || uploadSourcePath ? undefined : uploadModelSlug,
        artifact_id: uploadArtifactId || undefined,
        source_path: uploadSourcePath || undefined,
        private: uploadPrivate,
        large_folder: true,
        commit_message: "Upload TrainingHub model artifact"
      });
      setSelectedJob(job);
      onToast(`${job.job_id} queued for ${uploadRepoId}.`, "success", "Upload queued");
      await refresh();
    } catch (err) {
      onToast(err instanceof Error ? err.message : "HF upload could not be queued.", "error", "Upload failed");
    }
  }

  async function deleteModel(modelSlug: string) {
    try {
      const result = await api.delete<{ deleted: boolean; deleted_paths: string[] }>(`/api/models/${modelSlug}`);
      const message = result.deleted ? `${modelSlug} deleted. Removed ${result.deleted_paths.length} local path${result.deleted_paths.length === 1 ? "" : "s"}.` : `${modelSlug} was not deleted.`;
      setDeleteMessage(message);
      setDeleteCandidate(null);
      onToast(message, result.deleted ? "success" : "info", result.deleted ? "Model deleted" : "Model unchanged");
      await refresh();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unable to delete model.";
      setDeleteMessage(message);
      onToast(message, "error", "Delete failed");
    }
  }

  const uploadableArtifacts = artifacts.filter((artifact) => UPLOADABLE_ARTIFACT_TYPES.has(artifact.artifact_type));

  return (
    <div className="thx thx-page thx-models">
      <div className="thx-stage-h">
        <div>
          <div className="crumb">MODEL · BAY · 04 · REGISTRY</div>
          <h2>
            <span className="thx-glitch" data-text="MODEL OPERATIONS">
              MODEL OPERATIONS
            </span>
          </h2>
          <p className="lede">Select the active local runtime, acquire model files, and publish approved artifacts.</p>
        </div>
        <div className="stamp">
          REGISTERED
          <span>{pad2(models.length)} models</span>
        </div>
      </div>

      <div className="thx-models-stack">
        <InferenceModelSelector
          activeInferenceTarget={activeInferenceTarget}
          capabilityTransfers={capabilityTransfers}
          inferenceOptions={inferenceOptions}
          refresh={refresh}
          onToast={onToast}
        />

        <div className="thx-models-acquire">
          <HubAcquirePanel
            resourceType="model"
            defaultInput={defaultModelRepoId}
            activeInferenceTarget={activeInferenceTarget}
            refresh={refresh}
            setSelectedJob={setSelectedJob}
            onToast={onToast}
          />

          <section className="thx-panel">
            <div className="thx-panel-h">
              <h3>URL Transfer</h3>
              <span className="thx-tag">[ ACQUIRE · URL ]</span>
            </div>
            <form className="thx-params" onSubmit={downloadFromUrl}>
              <label className="thx-field thx-field--wide">
                <span className="thx-field-label">
                  <span>Model URL</span>
                  <span className="v">{modelUrl ? "ready" : "required"}</span>
                </span>
                <input type="text" value={modelUrl} onChange={(event) => setModelUrl(event.target.value)} />
              </label>
              <label className="thx-field">
                <span className="thx-field-label">
                  <span>Local slug</span>
                  <span className="v">{urlSlug}</span>
                </span>
                <input type="text" value={urlSlug} onChange={(event) => setUrlSlug(slugInput(event.target.value))} />
              </label>
              <label className="thx-field">
                <span className="thx-field-label">
                  <span>Display name</span>
                  <span className="v">registry</span>
                </span>
                <input type="text" value={urlDisplayName} onChange={(event) => setUrlDisplayName(event.target.value)} />
              </label>
              <div className="thx-form-actions">
                <button type="submit" className="thx-btn thx-btn--primary" disabled={!modelUrl}>
                  <Download size={16} /> Download URL Model
                </button>
              </div>
            </form>
          </section>

          <section className="thx-panel">
            <div className="thx-panel-h">
              <h3>HF Upload</h3>
              <span className="thx-tag">[ PUBLISH · ARTIFACT ]</span>
            </div>
            <form className="thx-params" onSubmit={uploadToHf}>
              <label className="thx-field thx-field--wide">
                <span className="thx-field-label">
                  <span>Target HF repo</span>
                  <span className="v">{uploadRepoId ? "ready" : "required"}</span>
                </span>
                <input type="text" value={uploadRepoId} onChange={(event) => setUploadRepoId(event.target.value)} />
              </label>
              <label className="thx-field">
                <span className="thx-field-label">
                  <span>Model</span>
                  <span className="v">{uploadModelSlug || "none"}</span>
                </span>
                <select value={uploadModelSlug} onChange={(event) => setUploadModelSlug(event.target.value)} disabled={Boolean(uploadArtifactId || uploadSourcePath)}>
                  {models.map((model) => (
                    <option key={model.slug} value={model.slug}>
                      {model.display_name}
                    </option>
                  ))}
                </select>
              </label>
              <label className="thx-field">
                <span className="thx-field-label">
                  <span>Artifact</span>
                  <span className="v">{uploadableArtifacts.length}</span>
                </span>
                <select value={uploadArtifactId} onChange={(event) => setUploadArtifactId(event.target.value)} disabled={Boolean(uploadSourcePath)}>
                  <option value="">Use selected model</option>
                  {uploadableArtifacts.map((artifact) => (
                    <option key={artifact.artifact_id} value={artifact.artifact_id}>
                      {artifact.display_name}
                    </option>
                  ))}
                </select>
              </label>
              <label className="thx-field thx-field--wide">
                <span className="thx-field-label">
                  <span>Source path</span>
                  <span className="v">{uploadSourcePath ? "override" : "optional"}</span>
                </span>
                <input type="text" value={uploadSourcePath} onChange={(event) => setUploadSourcePath(event.target.value)} />
              </label>
              <label className="thx-field thx-field--toggle">
                <span className="thx-field-label">
                  <span>Private repo</span>
                  <span className="v">{uploadPrivate ? "yes" : "no"}</span>
                </span>
                <span className="thx-toggle">
                  <input type="checkbox" checked={uploadPrivate} onChange={(event) => setUploadPrivate(event.target.checked)} />
                  <span className="thx-toggle-track" />
                  <span className="thx-toggle-copy">Create or update as private</span>
                </span>
              </label>
              <div className="thx-form-actions">
                <button type="submit" className="thx-btn thx-btn--primary" disabled={!uploadRepoId}>
                  <Upload size={16} /> Upload To HF
                </button>
              </div>
            </form>
          </section>
        </div>

        <section className="thx-panel thx-models-registry">
          <div className="thx-panel-h">
            <h3>Registry</h3>
            <span className="thx-tag">[ MODELS · {pad2(models.length)} ]</span>
          </div>
          {deleteMessage && <div className="thx-status-line">{deleteMessage}</div>}
          {models.length === 0 ? (
            <div className="thx-empty">NO MODELS REGISTERED</div>
          ) : (
            <div className="thx-runs thx-model-runs">
              {models.map((model) => (
                <article className="thx-model-row" key={model.slug}>
                  <div className="thx-model-main">
                    <strong>{model.display_name}</strong>
                    <span>{model.slug} · {model.provider_id}</span>
                    <p>{model.hardware_note}</p>
                  </div>
                  <div className="thx-model-caps">
                    <span className={`thx-cap ${model.supports_bf16_inference ? "thx-cap--ok" : "thx-cap--no"}`}>BF16</span>
                    <span className={`thx-cap ${model.supports_benchmark ? "thx-cap--ok" : "thx-cap--no"}`}>BENCH</span>
                    <span className={`thx-cap ${model.supports_gguf_path ? "thx-cap--c" : "thx-cap--no"}`}>GGUF</span>
                    <span className={`thx-cap ${model.supports_quantization ? "thx-cap--w" : "thx-cap--no"}`}>QUANT</span>
                    <span className={`thx-cap ${model.supports_lora ? "thx-cap--ok" : "thx-cap--no"}`}>LORA</span>
                    <span className={`thx-cap ${model.supports_qlora ? "thx-cap--ok" : "thx-cap--no"}`}>QLORA</span>
                    <span className={`thx-cap ${model.supports_full_finetune ? "thx-cap--ok" : "thx-cap--no"}`}>FULL</span>
                    <span className="thx-cap">{formatBytes(model.local_size_bytes || 0)}</span>
                    {model.seeded && <span className="thx-cap thx-cap--w">SEEDED</span>}
                  </div>
                  <button className="thx-btn thx-btn--danger" type="button" disabled={!model.deletable} onClick={() => setDeleteCandidate(model)}>
                    <Trash2 size={14} /> Delete
                  </button>
                </article>
              ))}
            </div>
          )}
        </section>
      </div>
      {deleteCandidate && (
        <div className="thx-confirm-back" onClick={() => setDeleteCandidate(null)}>
          <div className="thx-confirm" onClick={(event) => event.stopPropagation()}>
            <div className="thx-confirm-icon">
              <Trash2 size={26} />
            </div>
            <div className="thx-confirm-meta">DESTRUCTIVE · MODEL CLEANUP</div>
            <h3>Delete {deleteCandidate.display_name}?</h3>
            <p>
              This removes owned local files and drops the model from the registry. Seeded models are tombstoned so they stay deleted after restart.
            </p>
            <div className="thx-confirm-stats">
              <div><span>SLUG</span><strong>{deleteCandidate.slug.slice(0, 14)}</strong></div>
              <div><span>DISK</span><strong>{formatBytes(deleteCandidate.local_size_bytes || 0)}</strong></div>
              <div><span>KIND</span><strong>{deleteCandidate.seeded ? "SEEDED" : "LOCAL"}</strong></div>
            </div>
            <div className="thx-confirm-actions">
              <button className="thx-btn" type="button" onClick={() => setDeleteCandidate(null)}>
                Cancel
              </button>
              <button className="thx-btn thx-btn--primary thx-btn--danger-fill" type="button" onClick={() => deleteModel(deleteCandidate.slug)}>
                <Trash2 size={14} /> Delete Forever
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function InferenceModelSelector({
  activeInferenceTarget,
  capabilityTransfers,
  inferenceOptions,
  refresh,
  onToast
}: {
  activeInferenceTarget: InferenceTarget | null;
  capabilityTransfers: CapabilityTransferRecord[];
  inferenceOptions: InferenceOption[];
  refresh: () => void;
  onToast: ToastHandler;
}) {
  const [selectedValue, setSelectedValue] = useState("");
  const [message, setMessage] = useState("");

  useEffect(() => {
    if (activeInferenceTarget) {
      setSelectedValue(inferenceTargetValue(activeInferenceTarget));
    }
  }, [activeInferenceTarget?.target_type, activeInferenceTarget?.model_slug, activeInferenceTarget?.artifact_id]);

  async function save() {
    const option = inferenceOptions.find((item) => inferenceOptionValue(item) === selectedValue);
    if (!option) {
      setMessage("Select an inference target.");
      return;
    }
    try {
      await api.post<InferenceTarget>("/api/inference/target", {
        target_type: option.target_type,
        model_slug: option.model_slug,
        artifact_id: option.artifact_id
      });
      setMessage("Active inference model updated.");
      onToast("Active inference model updated.", "success", "Runtime selected");
      await refresh();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unable to update inference model.";
      setMessage(message);
      onToast(message, "error", "Runtime update failed");
    }
  }

  const selectedOption = inferenceOptions.find((item) => inferenceOptionValue(item) === selectedValue);

  return (
    <section className="thx-panel thx-panel--accent thx-models-active">
      <div className="thx-panel-h">
        <h3>Active Inference</h3>
        <span className="thx-tag">[ RUNTIME · {activeInferenceTarget ? "ONLINE" : "UNSET"} ]</span>
      </div>
      <div className="thx-models-active-grid">
        <div className="thx-summary">
          <div className="thx-summary-item">
            <span className="k">Current target</span>
            <span className="v">{activeInferenceTarget?.display_name || "Unset"}</span>
            <span className="vmono">{activeInferenceTarget ? inferenceTargetSubtitle(activeInferenceTarget) : "select a local target"}</span>
          </div>
          <div className="thx-summary-item">
            <span className="k">Runtime kind</span>
            <span className="v">{activeInferenceTarget ? inferenceTypeLabel(activeInferenceTarget.target_type) : "none"}</span>
            <span className="vmono">{pad2(inferenceOptions.length)} available options</span>
          </div>
        </div>
        <ActiveTransferPill activeInferenceTarget={activeInferenceTarget} transfers={capabilityTransfers} refresh={refresh} onToast={onToast} />
        <div className="thx-cards thx-inference-cards">
          {inferenceOptions.map((option) => {
            const value = inferenceOptionValue(option);
            return (
              <button
                type="button"
                className={`thx-card ${value === selectedValue ? "is-selected" : ""} ${option.enabled ? "" : "is-disabled"}`}
                disabled={!option.enabled}
                onClick={() => setSelectedValue(value)}
                key={value}
              >
                <span className="thx-card-row">
                  <span className="thx-card-title">{option.display_name}</span>
                  <span className="thx-card-status">{inferenceTypeLabel(option.target_type)}</span>
                </span>
                <span className="thx-card-sub">{option.enabled ? option.description : option.disabled_reason}</span>
              </button>
            );
          })}
          {inferenceOptions.length === 0 && <div className="thx-empty">NO INFERENCE OPTIONS REGISTERED</div>}
        </div>
      </div>
      <FieldNote
        note={
          selectedOption
            ? selectedOption.enabled
              ? selectedOption.description
              : selectedOption.disabled_reason
            : "Large models require a GGUF artifact for local inference."
        }
        link="#model-support"
      />
      <div className="thx-form-actions">
        <button type="button" className="thx-btn thx-btn--primary" onClick={save} disabled={!selectedOption || !selectedOption.enabled}>
          <RefreshCw size={16} /> Set Active Model
        </button>
        {message && <div className="thx-status-line">{message}</div>}
      </div>
    </section>
  );
}

function Cleanup({ refresh, setSelectedJob, onToast }: { refresh: () => void; setSelectedJob: (job: JobRecord) => void; onToast: ToastHandler }) {
  const [manifest, setManifest] = useState<Record<string, any> | null>(null);
  const [approvedPaths, setApprovedPaths] = useState<string[]>([]);

  async function scan() {
    try {
      const result = await api.post<Record<string, any>>("/api/cleanup/scan", { include_immediate: true });
      setManifest(result);
      setApprovedPaths([]);
      const count = Array.isArray(result.items) ? result.items.length : 0;
      onToast(`${count} cleanup candidate${count === 1 ? "" : "s"} found.`, "info", "Cleanup scan complete");
    } catch (err) {
      onToast(err instanceof Error ? err.message : "Cleanup scan failed.", "error", "Cleanup scan failed");
    }
  }

  async function apply() {
    if (!manifest) {
      return;
    }
    try {
      const job = await api.post<JobRecord>("/api/cleanup/apply", { manifest_id: manifest.manifest_id, approved_paths: approvedPaths });
      setSelectedJob(job);
      onToast(`${job.job_id} queued for ${approvedPaths.length} approved path${approvedPaths.length === 1 ? "" : "s"}.`, "success", "Cleanup queued");
      await refresh();
    } catch (err) {
      onToast(err instanceof Error ? err.message : "Cleanup job could not be queued.", "error", "Cleanup failed");
    }
  }

  function toggle(path: string) {
    setApprovedPaths((current) => (current.includes(path) ? current.filter((item) => item !== path) : [...current, path]));
  }

  const items = ((manifest?.items as Record<string, any>[] | undefined) || []);
  const totalBytes = items.reduce((sum, item) => sum + (typeof item.size_bytes === "number" ? item.size_bytes : 0), 0);
  const approvedBytes = items.reduce(
    (sum, item) => sum + (approvedPaths.includes(String(item.path || "")) && typeof item.size_bytes === "number" ? item.size_bytes : 0),
    0
  );

  return (
    <div className="thx thx-page thx-cleanup">
      <div className="thx-stage-h">
        <div>
          <div className="crumb">OPS · CLEAN · 05 · MANIFEST</div>
          <h2>
            <span className="thx-glitch" data-text="CLEANUP REVIEW">
              CLEANUP REVIEW
            </span>
          </h2>
          <p className="lede">Scan candidate local artifacts, approve exact paths, then queue a cleanup job.</p>
        </div>
        <div className="stamp">
          APPROVED
          <span>{pad2(approvedPaths.length)} items</span>
        </div>
      </div>

      <section className="thx-panel thx-panel--accent">
        <div className="thx-panel-h">
          <h3>Manifest Controls</h3>
          <span className="thx-tag">[ SCAN · APPROVE · APPLY ]</span>
        </div>
        <div className="thx-summary">
          <div className="thx-summary-item">
            <span className="k">Manifest</span>
            <span className="v">{manifest?.manifest_id || "none"}</span>
            <span className="vmono">{manifest ? "draft loaded" : "scan required"}</span>
          </div>
          <div className="thx-summary-item">
            <span className="k">Candidates</span>
            <span className="v">{pad2(items.length)}</span>
            <span className="vmono">{formatBytes(totalBytes)}</span>
          </div>
          <div className="thx-summary-item">
            <span className="k">Approved</span>
            <span className="v">{pad2(approvedPaths.length)}</span>
            <span className="vmono">{formatBytes(approvedBytes)}</span>
          </div>
        </div>
        <div className="thx-form-actions">
          <button type="button" className="thx-btn" onClick={scan}>
            <RefreshCw size={16} /> Scan Cleanup Candidates
          </button>
          <button type="button" className="thx-btn thx-btn--danger" onClick={apply} disabled={approvedPaths.length === 0}>
            <Trash2 size={16} /> Apply Approved Items
          </button>
        </div>
      </section>

      <section className="thx-panel">
        <div className="thx-panel-h">
          <h3>Candidates</h3>
          <span className="thx-tag">[ PATHS · {pad2(items.length)} ]</span>
        </div>
        {!manifest ? (
          <div className="thx-empty">NO CLEANUP MANIFEST LOADED</div>
        ) : items.length === 0 ? (
          <div className="thx-empty">SCAN FOUND NO CLEANUP CANDIDATES</div>
        ) : (
          <div className="thx-runs thx-cleanup-runs">
            {items.map((item, index) => {
              const path = String(item.path || "");
              const checked = approvedPaths.includes(path);
              const size = typeof item.size_bytes === "number" ? item.size_bytes : 0;
              return (
                <label className={`thx-cleanup-row ${checked ? "is-selected" : ""}`} key={`${path || item.action}-${index}`}>
                  <input type="checkbox" disabled={!path} checked={checked} onChange={() => toggle(path)} />
                  <span className="thx-run-dot" />
                  <span className="thx-cleanup-main">
                    <strong>{labelizeKey(String(item.action || "candidate"))}</strong>
                    <span>{path || String(item.reason || "No path supplied")}</span>
                  </span>
                  <span className="thx-cap thx-cap--w">{item.action || "scan"}</span>
                  <span className="thx-run-status">{size ? formatBytes(size) : "process"}</span>
                </label>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}

function KnowledgeBase() {
  const entries = [
    { id: "datasets", title: "Dataset validation", family: "Data", text: "Prompt, response, and split are required. Math rows also require final_answer." },
    { id: "sequence-length", title: "Max sequence length", family: "Data", text: "Approximate row length guard used before tokenizer-specific checks in workers." },
    { id: "teacher-model", title: "Teacher model", family: "Generate", text: "Generation jobs can use a local model when real generation is enabled, otherwise smoke candidates are produced." },
    { id: "model-support", title: "Model support", family: "Models", text: "Large 35B A3B and large Gemma variants are routed to quantized GGUF on two 12 GB GPUs." },
    { id: "training", title: "Training", family: "Fine-tune", text: "Approved datasets can queue LoRA, QLoRA, or explicitly supported full fine-tuning jobs." },
    { id: "benchmark-limit", title: "Benchmark limit", family: "Benchmarks", text: "Small limits are for smoke validation; full reports should use the complete benchmark." },
    { id: "benchmark-presets", title: "Benchmark presets", family: "Benchmarks", text: "Smoke presets queue a quick per-suite sample; full presets raise the cap to each selected catalog default." },
    { id: "maj-k", title: "Maj@k", family: "Benchmarks", text: "Majority voting samples multiple answers and scores the most common final answer." },
    { id: "mmlu", title: "MMLU", family: "Benchmarks", text: "MMLU probes broad academic knowledge and reasoning across many subject areas." },
    { id: "hellaswag", title: "HellaSwag", family: "Benchmarks", text: "HellaSwag checks commonsense continuation selection and everyday reasoning." },
    { id: "arc", title: "ARC", family: "Benchmarks", text: "ARC combines easy and challenge science-question suites for reasoning comparisons." },
    { id: "ifeval", title: "IFEval", family: "Benchmarks", text: "IFEval measures whether responses follow concrete prompt instructions and constraints." },
    { id: "humaneval", title: "HumanEval", family: "Benchmarks", text: "HumanEval evaluates Python function synthesis and code-generation correctness." },
    { id: "quantization", title: "Quantization", family: "Quantize", text: "Quantize only from FP16/BF16 GGUF sources and store checksums and commands." },
    { id: "capability-vectors", title: "Capability vectors", family: "Transfer", text: "A vector is the averaged activation difference between capability-present and capability-absent prompts at selected source layers." },
    { id: "calibration-pairs", title: "Calibration pairs", family: "Transfer", text: "Use approved JSONL with prompt_present and prompt_absent fields, or system_present and system_absent for system-pair contrast." },
    { id: "linear-alignment", title: "Linear alignment", family: "Transfer", text: "Alignment fits a low-rank map from source activations to target activations using closed-form linear algebra, not SGD." },
    { id: "alpha-tuning", title: "Alpha tuning", family: "Transfer", text: "Alpha controls steering strength. Start near 1.0; values above 2 can destabilize generation." },
    { id: "gguf-degraded-mode", title: "GGUF degraded mode", family: "Transfer", text: "GGUF support is last-layer only through llama-cpp-python. Use HF Transformers targets for full per-layer steering." }
  ];

  useEffect(() => {
    function scrollToHashAnchor() {
      const clean = window.location.hash.replace(/^#\/?/, "");
      const [, anchor] = clean.split("/");
      if (!anchor) {
        return;
      }
      window.setTimeout(() => {
        document.getElementById(anchor)?.scrollIntoView({ block: "start", behavior: "smooth" });
      }, 0);
    }

    scrollToHashAnchor();
    window.addEventListener("hashchange", scrollToHashAnchor);
    return () => window.removeEventListener("hashchange", scrollToHashAnchor);
  }, []);

  return (
    <div className="thx thx-page thx-kb">
      <div className="thx-stage-h">
        <div>
          <div className="crumb">REF · KB · 06 · ANCHORS</div>
          <h2>
            <span className="thx-glitch" data-text="KNOWLEDGE BASE">
              KNOWLEDGE BASE
            </span>
          </h2>
          <p className="lede">Reference cards for the controls used across generation, training, benchmarking, and cleanup.</p>
        </div>
        <div className="stamp">
          ANCHORS
          <span>{pad2(entries.length)} cards</span>
        </div>
      </div>
      <section className="thx-kb-grid">
        {entries.map((entry) => (
          <article className="thx-panel thx-kb-card" id={entry.id} key={entry.id}>
            <div className="thx-panel-h">
              <h3>{entry.title}</h3>
              <span className="thx-tag">[ {entry.family} ]</span>
            </div>
            <p>{entry.text}</p>
            <a className="thx-kb-anchor" href={`#/knowledge/${entry.id}`}>#{entry.id}</a>
          </article>
        ))}
        <article className="thx-panel thx-kb-card thx-kb-card--refs">
          <div className="thx-panel-h">
            <h3>External references</h3>
            <span className="thx-tag">[ DOCS ]</span>
          </div>
          <p>Primary documentation used by benchmark and fine-tuning workflows.</p>
          <div className="thx-kb-links">
            <a href="https://lm-evaluation-harness.readthedocs.io/">LM Evaluation Harness</a>
            <a href="https://huggingface.co/docs/peft/developer_guides/lora">PEFT LoRA</a>
          </div>
        </article>
      </section>
    </div>
  );
}

function formatBytes(value: number) {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let next = value;
  let index = 0;
  while (next >= 1024 && index < units.length - 1) {
    next /= 1024;
    index += 1;
  }
  return `${next.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function formatCount(value?: number) {
  if (typeof value !== "number") {
    return "unknown";
  }
  return new Intl.NumberFormat().format(value);
}

function shortSha(value: string) {
  return value ? value.slice(0, 12) : "unknown";
}

function slugFromRepo(repoId: string) {
  const name = repoId.split("/").filter(Boolean).pop() || "hub-resource";
  return name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "") || "hub-resource";
}

function nameFromRepo(repoId: string) {
  return repoId.split("/").filter(Boolean).pop()?.replace(/[-_]+/g, " ") || repoId;
}

function summaryText(summary: Record<string, unknown>, key: string) {
  const value = summary[key];
  if (Array.isArray(value)) {
    return value.length ? value.map((item) => summaryItemText(item)).join(", ") : "unknown";
  }
  if (value && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    return entries.length ? entries.map(([entryKey, entryValue]) => `${entryKey} ${entryValue}`).join(", ") : "unknown";
  }
  if (typeof value === "boolean") {
    return value ? "yes" : "no";
  }
  if (value === undefined || value === null || value === "") {
    return "unknown";
  }
  return String(value);
}

function summaryBool(summary: Record<string, unknown>, key: string) {
  return summary[key] === true;
}

function summaryNumber(summary: Record<string, unknown>, key: string) {
  const value = summary[key];
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function summaryPatternList(summary: Record<string, unknown>, key: string) {
  const value = summary[key];
  if (!Array.isArray(value)) {
    return "";
  }
  return value.map((item) => String(item).trim()).filter(Boolean).join(",");
}

function firstSummaryText(value: unknown) {
  if (!Array.isArray(value) || value.length === 0) {
    return "";
  }
  return summaryItemText(value[0]);
}

function firstConfigName(value: unknown) {
  if (!Array.isArray(value) || value.length === 0) {
    return "";
  }
  const first = value[0];
  if (first && typeof first === "object") {
    const record = first as Record<string, unknown>;
    return String(record.config_name || record.name || "");
  }
  return String(first || "");
}

function summaryItemText(value: unknown) {
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return String(record.config_name || record.name || record.split || JSON.stringify(record));
  }
  return String(value);
}

function modelHint(summary: Record<string, unknown>) {
  const pipeline = summaryText(summary, "pipeline");
  const library = summaryText(summary, "library");
  const gguf = summaryBool(summary, "has_gguf") ? "GGUF available" : "checkpoint files";
  return [pipeline, library, gguf].filter((item) => item !== "unknown").join(" / ") || gguf;
}

function datasetHint(summary: Record<string, unknown>) {
  const splits = summaryText(summary, "splits");
  const fields = summaryText(summary, "sample_fields");
  if (splits !== "unknown" && fields !== "unknown") {
    return `${splits} / ${fields}`;
  }
  return splits !== "unknown" ? splits : fields;
}

function inferenceTargetRuntimeValue(target: InferenceTarget) {
  if (target.target_type === "base_model") {
    return target.provider_id || target.model_slug;
  }
  return target.path || target.display_name;
}

function inferenceTargetSubtitle(target: InferenceTarget) {
  if (target.target_type === "base_model") {
    return target.provider_id;
  }
  return target.path || inferenceTypeLabel(target.target_type);
}

function inferenceTargetValue(target: InferenceTarget) {
  if (target.target_type === "base_model") {
    return `base_model:${target.model_slug}`;
  }
  return `gguf_artifact:${target.artifact_id}`;
}

function inferenceOptionValue(option: InferenceOption) {
  if (option.target_type === "base_model") {
    return `base_model:${option.model_slug}`;
  }
  return `gguf_artifact:${option.artifact_id}`;
}

function inferenceTypeLabel(value: string) {
  if (value === "base_model") {
    return "base";
  }
  if (value === "gguf_artifact") {
    return "GGUF";
  }
  return value;
}

function yesNo(value: boolean) {
  return value ? "yes" : "no";
}
