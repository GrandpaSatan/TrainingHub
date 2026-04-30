import {
  FormEvent,
  ReactNode,
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  Activity,
  AlertTriangle,
  Archive,
  CheckCircle,
  ChevronLeft,
  ChevronRight,
  Download,
  Eye,
  Globe,
  Heart,
  Library,
  Link as LinkIcon,
  Search,
  Shuffle,
  Sparkles,
  Trash2,
  Upload,
  Zap,
} from "lucide-react";
import {
  api,
  DatasetRecord,
  DatasetRecordsResponse,
  DatasetRecordView,
  HubResolvedResource,
  InferenceTarget,
  JobRecord,
  ValidationResult,
} from "../api/client";
import type { ToastTone } from "./ToastLayer";
import "../styles/cyberpunk.css";

type Source = "huggingface" | "csv" | "url";
type StepKey = "source" | "ingest" | "process" | "inspect" | "approve" | "library";
type DatasetType = "math_sft" | "chat_sft" | "holdout";
type FormErrors = Record<string, string>;

type HubSearchHit = {
  id: string;
  downloads?: number;
  likes?: number;
  tags?: string[];
  author?: string;
  description?: string;
  lastModified?: string;
};

type StepDef = {
  key: StepKey;
  num: string;
  label: string;
  meta: string;
  icon: ReactNode;
};

const STEPS: StepDef[] = [
  { key: "source",  num: "01", label: "Source",   meta: "ORIGIN",   icon: <Library size={14} /> },
  { key: "ingest",  num: "02", label: "Ingest",   meta: "FETCH",    icon: <Download size={14} /> },
  { key: "process", num: "03", label: "Process",  meta: "PIPELINE", icon: <Sparkles size={14} /> },
  { key: "inspect", num: "04", label: "Inspect",  meta: "SAMPLE",   icon: <Eye size={14} /> },
  { key: "approve", num: "05", label: "Approve",  meta: "GATE",     icon: <CheckCircle size={14} /> },
  { key: "library", num: "06", label: "Library",  meta: "MANIFEST", icon: <Archive size={14} /> },
];

type AsideStat = { k: string; v: string };
type AsideContent = {
  meta: string;
  title: string;
  desc: string;
  stats?: AsideStat[];
  note?: string;
};

const SOURCE_INFO: Record<Source, AsideContent> = {
  huggingface: {
    meta: "ORIGIN · 01A",
    title: "Hugging Face",
    desc:
      "Pull a dataset directly from the Hub. The pipeline downloads, normalizes to canonical JSONL, and (optionally) cleans every row through the active local inference target before any human ever sees it.",
    stats: [
      { k: "AUTH", v: "Token from .env" },
      { k: "RATE", v: "HF default" },
      { k: "CLEAN", v: "Local inference" },
    ],
    note: "Use this for any well-known public dataset. Search before you guess the repo id.",
  },
  csv: {
    meta: "ORIGIN · 01B",
    title: "CSV Upload",
    desc:
      "Drop a local CSV. The cleaner converts it to canonical prompt/response/split records, runs validation, and routes any malformed rows to the warnings panel before approval.",
    stats: [
      { k: "MAX SIZE", v: "Browser limit" },
      { k: "ENCODING", v: "UTF-8" },
      { k: "TEMPLATE", v: "Three preset shapes" },
    ],
    note: "Templates exist for math_sft, chat_sft, and benchmark holdout. Download one if your CSV is from scratch.",
  },
  url: {
    meta: "ORIGIN · 01C",
    title: "Direct URL",
    desc:
      "Pull a JSONL, CSV, or Parquet file from any HTTP(S) endpoint or a local file:// path. The fetcher caps at the row limit you set so dry-run imports stay cheap.",
    stats: [
      { k: "PROTOCOLS", v: "http · https · file" },
      { k: "FORMATS", v: "csv · jsonl · parquet" },
      { k: "ROW CAP", v: "1 – 100,000" },
    ],
    note: "Use this for curated datasets that aren't on the Hub yet, or for small canary subsets.",
  },
};

function fmtCount(n: number | undefined): string {
  if (n === undefined || n === null || !Number.isFinite(n)) return "—";
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

function fmtBytes(n: number | undefined): string {
  if (!n || !Number.isFinite(n)) return "—";
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KB`;
  return `${n} B`;
}

function slugify(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9-]+/g, "-").replace(/-{2,}/g, "-").replace(/^-|-$/g, "");
}

function nameFromRepo(repoId: string): string {
  const parts = repoId.split("/");
  return (parts[1] || repoId).replace(/[_-]+/g, " ");
}

function shuffleIndices(n: number): number[] {
  const arr = Array.from({ length: n }, (_, i) => i);
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

type Props = {
  datasets: DatasetRecord[];
  jobs: JobRecord[];
  activeInferenceTarget: InferenceTarget | null;
  refresh: () => void;
  setSelectedJob: (job: JobRecord) => void;
  onToast?: (message: string, tone?: ToastTone, title?: string) => void;
};

export function DatasetsWizard({
  datasets,
  jobs,
  activeInferenceTarget,
  refresh,
  setSelectedJob,
  onToast = () => undefined,
}: Props) {
  const [step, setStep] = useState<StepKey>("source");
  const [completed, setCompleted] = useState<Record<StepKey, boolean>>({
    source: false,
    ingest: false,
    process: false,
    inspect: false,
    approve: false,
    library: false,
  });

  const [source, setSource] = useState<Source>("huggingface");
  const [datasetType, setDatasetType] = useState<DatasetType>("math_sft");
  const [aside, setAside] = useState<AsideContent | null>(null);
  const [now, setNow] = useState(new Date());
  const [error, setError] = useState<string>("");
  const [statusMessage, setStatusMessage] = useState<string>("");

  // ---- HF state
  const [hfQuery, setHfQuery] = useState<string>("math reasoning");
  const [hfResults, setHfResults] = useState<HubSearchHit[]>([]);
  const [hfSearching, setHfSearching] = useState(false);
  const [hfSearchError, setHfSearchError] = useState<string>("");
  const [hfSelected, setHfSelected] = useState<HubSearchHit | null>(null);
  const [hfResolved, setHfResolved] = useState<HubResolvedResource | null>(null);
  const [hfResolving, setHfResolving] = useState(false);
  const [hfTitle, setHfTitle] = useState("");
  const [hfSlug, setHfSlug] = useState("");
  const [hfSplit, setHfSplit] = useState("train");
  const [hfMaxRows, setHfMaxRows] = useState("2000");
  const [hfClean, setHfClean] = useState(true);

  // ---- CSV state
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [csvTitle, setCsvTitle] = useState("Local CSV");
  const [csvSlug, setCsvSlug] = useState("local-csv");
  const [csvMaxSeq, setCsvMaxSeq] = useState("2048");
  const [csvDragActive, setCsvDragActive] = useState(false);
  const [csvValidation, setCsvValidation] = useState<ValidationResult | null>(null);
  const [csvErrors, setCsvErrors] = useState<FormErrors>({});

  // ---- URL state
  const [urlValue, setUrlValue] = useState("");
  const [urlTitle, setUrlTitle] = useState("Remote dataset");
  const [urlSlug, setUrlSlug] = useState("remote-dataset");
  const [urlMaxRows, setUrlMaxRows] = useState("1000");
  const [urlErrors, setUrlErrors] = useState<FormErrors>({});

  // ---- Process / job tracking
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // ---- Inspect / sample
  const [inspectId, setInspectId] = useState<string | null>(null);
  const [sampleSize, setSampleSize] = useState<number>(20);
  const [sample, setSample] = useState<DatasetRecordsResponse | null>(null);
  const [sampleLoading, setSampleLoading] = useState(false);
  const [shuffleSeed, setShuffleSeed] = useState(0);

  // ---- Library / delete
  const [pendingDelete, setPendingDelete] = useState<DatasetRecord | null>(null);

  useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(id);
  }, []);

  // Default selection for inspect step
  useEffect(() => {
    if (!inspectId && datasets.length > 0) {
      const fav = datasets.find((d) => !d.approved) ?? datasets[0];
      setInspectId(fav.dataset_id);
    }
  }, [datasets, inspectId]);

  // Load sample when dataset or sample size changes
  useEffect(() => {
    if (!inspectId) {
      setSample(null);
      return;
    }
    let cancelled = false;
    setSampleLoading(true);
    api
      .get<DatasetRecordsResponse>(`/api/datasets/${inspectId}/review-sample?sample_size=${sampleSize}`)
      .then((res) => {
        if (!cancelled) setSample(res);
      })
      .catch(() => {
        if (!cancelled) setSample(null);
      })
      .finally(() => {
        if (!cancelled) setSampleLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [inspectId, sampleSize, shuffleSeed]);

  // Track active job for the process step
  const activeJob = useMemo(
    () => jobs.find((j) => j.job_id === activeJobId) ?? null,
    [jobs, activeJobId]
  );

  // Library data
  const approvedCount = useMemo(() => datasets.filter((d) => d.approved).length, [datasets]);
  const totalRows = useMemo(() => datasets.reduce((acc, d) => acc + d.row_count, 0), [datasets]);

  // Mark steps complete
  useEffect(() => {
    setCompleted((prev) => ({
      ...prev,
      source: true,
      ingest:
        (source === "huggingface" && !!hfResolved) ||
        (source === "csv" && !!csvFile) ||
        (source === "url" && urlValue.trim().length > 0),
      process: !!activeJob && (activeJob.status === "succeeded" || activeJob.status === "completed"),
      inspect: !!sample && (sample.records?.length ?? 0) > 0,
      approve: !!datasets.find((d) => d.dataset_id === inspectId)?.approved,
      library: datasets.length > 0,
    }));
  }, [source, hfResolved, csvFile, urlValue, activeJob, sample, datasets, inspectId]);

  const stepIndex = STEPS.findIndex((s) => s.key === step);
  const progressPct = Math.round(((stepIndex + 1) / STEPS.length) * 100);

  function go(target: StepKey) {
    setStep(target);
    setAside(null);
    setError("");
  }
  function goNext() {
    const i = STEPS.findIndex((s) => s.key === step);
    if (i < STEPS.length - 1) go(STEPS[i + 1].key);
  }
  function goPrev() {
    const i = STEPS.findIndex((s) => s.key === step);
    if (i > 0) go(STEPS[i - 1].key);
  }

  // ---- HF search (calls public HF api directly)
  async function searchHub(query: string) {
    setHfSearching(true);
    setHfSearchError("");
    try {
      const url = `https://huggingface.co/api/datasets?search=${encodeURIComponent(query)}&limit=20&full=false`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HF API returned ${res.status}`);
      const json: HubSearchHit[] = await res.json();
      setHfResults(json || []);
    } catch (err) {
      setHfSearchError(err instanceof Error ? err.message : "Search failed");
      setHfResults([]);
    } finally {
      setHfSearching(false);
    }
  }

  async function selectHub(hit: HubSearchHit) {
    setHfSelected(hit);
    setHfTitle(`${nameFromRepo(hit.id)} (${datasetType})`);
    setHfSlug(slugify(`${hit.id.split("/").pop() || hit.id}`));
    setHfResolving(true);
    setHfResolved(null);
    setError("");
    try {
      const result = await api.post<HubResolvedResource>("/api/hub/resolve", {
        input: hit.id,
        resource_type: "dataset",
      });
      setHfResolved(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to resolve repo.");
    } finally {
      setHfResolving(false);
    }
  }

  async function startHubImport() {
    if (!hfResolved) return;
    setBusy(true);
    setError("");
    try {
      const job = await api.post<JobRecord>("/api/datasets/import-hf", {
        repo_id: hfResolved.repo_id,
        confirmed_sha: hfResolved.sha,
        split: hfSplit || "train",
        title: hfTitle,
        slug: hfSlug,
        dataset_type: datasetType,
        max_rows: Number(hfMaxRows) || undefined,
        default_split: "holdout",
        clean_with_inference: hfClean,
        delete_raw_after_clean: true,
      });
      setActiveJobId(job.job_id);
      setSelectedJob(job);
      setStatusMessage(`Queued ${job.job_id} — ${hfResolved.repo_id}`);
      onToast(`${job.job_id} queued for ${hfResolved.repo_id}.`, "success", "Dataset import queued");
      go("process");
      refresh();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Hub import failed.";
      setError(message);
      onToast(message, "error", "Hub import failed");
    } finally {
      setBusy(false);
    }
  }

  // ---- CSV upload
  async function uploadCsv(event?: FormEvent) {
    if (event) event.preventDefault();
    const errs: FormErrors = {};
    if (!csvFile) errs.file = "Pick a CSV file.";
    if (!csvTitle.trim()) errs.title = "Title is required.";
    if (!csvSlug.trim() || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(csvSlug)) errs.slug = "Lowercase letters, numbers, hyphens.";
    if (!/^\d+$/.test(csvMaxSeq) || Number(csvMaxSeq) < 128) errs.maxSeq = "≥ 128.";
    setCsvErrors(errs);
    if (Object.keys(errs).length) return;

    setBusy(true);
    setError("");
    try {
      const body = new FormData();
      body.append("file", csvFile as File);
      body.append("dataset_type", datasetType);
      body.append("title", csvTitle.trim());
      body.append("slug", csvSlug.trim());
      body.append("max_sequence_length", csvMaxSeq.trim());
      const result = await api.postForm<{ created: boolean; validation: ValidationResult }>(
        "/api/datasets/upload",
        body
      );
      setCsvValidation(result.validation);
      setStatusMessage(`Uploaded ${csvFile?.name}. ${result.validation.accepted_count} rows accepted.`);
      onToast(`${result.validation.accepted_count} rows accepted from ${csvFile?.name}.`, "success", "CSV uploaded");
      refresh();
      go("inspect");
    } catch (err) {
      const message = err instanceof Error ? err.message : "CSV upload failed.";
      setError(message);
      onToast(message, "error", "CSV upload failed");
    } finally {
      setBusy(false);
    }
  }

  // ---- URL import
  async function importUrl(event?: FormEvent) {
    if (event) event.preventDefault();
    const errs: FormErrors = {};
    try {
      const u = new URL(urlValue.trim());
      if (!["http:", "https:", "file:"].includes(u.protocol)) errs.url = "http, https, or file URL only.";
    } catch {
      errs.url = "Invalid URL.";
    }
    if (!urlTitle.trim()) errs.title = "Title required.";
    if (!urlSlug.trim() || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(urlSlug)) errs.slug = "Lowercase letters, numbers, hyphens.";
    if (!/^\d+$/.test(urlMaxRows) || Number(urlMaxRows) < 1) errs.maxRows = "≥ 1.";
    setUrlErrors(errs);
    if (Object.keys(errs).length) return;

    setBusy(true);
    setError("");
    try {
      const job = await api.post<JobRecord>("/api/datasets/import-url", {
        url: urlValue.trim(),
        title: urlTitle.trim(),
        slug: urlSlug.trim(),
        dataset_type: datasetType,
        max_rows: Number(urlMaxRows) || undefined,
        default_split: "holdout",
      });
      setActiveJobId(job.job_id);
      setSelectedJob(job);
      setStatusMessage(`Queued ${job.job_id} — ${urlValue}`);
      onToast(`${job.job_id} queued for ${urlValue}.`, "success", "URL import queued");
      go("process");
      refresh();
    } catch (err) {
      const message = err instanceof Error ? err.message : "URL import failed.";
      setError(message);
      onToast(message, "error", "URL import failed");
    } finally {
      setBusy(false);
    }
  }

  async function approveDataset(id: string) {
    setBusy(true);
    setError("");
    try {
      await api.post(`/api/datasets/${id}/approve`, {});
      setStatusMessage("Approved. Cleared for training.");
      onToast("Dataset approved and cleared for training.", "success", "Dataset approved");
      refresh();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Approval failed.";
      setError(message);
      onToast(message, "error", "Approval failed");
    } finally {
      setBusy(false);
    }
  }

  async function deleteDataset(id: string) {
    setBusy(true);
    setError("");
    try {
      await api.post(`/api/datasets/${id}/reject`, {});
      setStatusMessage("Deleted.");
      if (inspectId === id) setInspectId(null);
      setPendingDelete(null);
      onToast("Dataset was rejected and deleted.", "success", "Dataset deleted");
      refresh();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Delete failed.";
      setError(message);
      onToast(message, "error", "Delete failed");
    } finally {
      setBusy(false);
    }
  }

  // ---- ASIDE defaults
  const defaultAside: AsideContent = useMemo(() => {
    switch (step) {
      case "source":
        return {
          meta: "STAGE · 01 OF 06",
          title: "Origin",
          desc:
            "Pick where the data lives. Each path runs through the same canonical pipeline downstream — the only difference is the fetcher and how cleanup is invoked.",
          stats: [
            { k: "VERSIONS LOCAL", v: String(datasets.length) },
            { k: "APPROVED", v: String(approvedCount) },
            { k: "TOTAL ROWS", v: fmtCount(totalRows) },
          ],
          note: activeInferenceTarget
            ? `Active cleaner: ${activeInferenceTarget.display_name}`
            : "No active cleaner. Set one on the Models page before HF or URL imports run.",
        };
      case "ingest":
        return source === "huggingface"
          ? SOURCE_INFO.huggingface
          : source === "csv"
          ? SOURCE_INFO.csv
          : SOURCE_INFO.url;
      case "process":
        return {
          meta: "STAGE · 03 OF 06",
          title: "Pipeline",
          desc:
            "The job moves through fetch → normalize → clean → write. Cleaning runs every row through the active inference target so malformed prompts don't reach reviewers.",
          stats: activeJob
            ? [
                { k: "JOB", v: activeJob.job_id.slice(0, 12) },
                { k: "STATUS", v: activeJob.status.toUpperCase() },
                { k: "WORKER", v: String(activeJob.worker_pid ?? "—") },
              ]
            : [{ k: "JOB", v: "Idle" }],
          note: "Raw downloads are deleted after cleanup unless disk usage gets weird — toggle delete_raw_after_clean if you need forensic copies.",
        };
      case "inspect":
        return {
          meta: "STAGE · 04 OF 06",
          title: "Sample",
          desc:
            "Pull a random N rows for visual inspection. Reviewers hold the only veto on what trains. The sample is reshuffled on demand — don't approve before you've seen at least three different draws.",
          stats: [
            { k: "SAMPLE SIZE", v: String(sampleSize) },
            { k: "MAX", v: "100" },
            { k: "RESHUFFLE", v: "Free" },
          ],
          note: "Inspect prompt, response, split, and metadata. If anything looks templated, leaked, or empty — reject.",
        };
      case "approve":
        const ds = datasets.find((d) => d.dataset_id === inspectId);
        return {
          meta: "STAGE · 05 OF 06",
          title: "Gate",
          desc:
            "Approval is a guarantee for the trainer that a human read at least N rows and signed off. Once approved, the version becomes selectable on the Training wizard. Rejected versions are deleted from disk.",
          stats: ds
            ? [
                { k: "VERSION", v: ds.version_id },
                { k: "ROWS", v: fmtCount(ds.row_count) },
                { k: "VALIDATION", v: ds.validation.valid ? "PASS" : "ISSUES" },
              ]
            : [],
          note: ds?.approved
            ? "Already approved. Re-run a new ingest if the source has updated."
            : "Reject is destructive — the JSONL is removed from disk and the registry entry is dropped.",
        };
      case "library":
        return {
          meta: "STAGE · 06 OF 06",
          title: "Manifest",
          desc:
            "Every local version, with its validation tally and approval state. Delete versions you won't need to keep the artifact directory predictable. Deletion is final.",
          stats: [
            { k: "TOTAL", v: String(datasets.length) },
            { k: "APPROVED", v: String(approvedCount) },
            { k: "PENDING", v: String(datasets.length - approvedCount) },
          ],
        };
    }
  }, [step, source, datasets, approvedCount, totalRows, activeInferenceTarget, activeJob, sampleSize, inspectId]);

  const activeAside = aside ?? defaultAside;

  function hover(content: AsideContent) {
    return {
      onMouseEnter: () => setAside(content),
      onMouseLeave: () => setAside(null),
      onFocus: () => setAside(content),
      onBlur: () => setAside(null),
    };
  }

  const inspectDataset = datasets.find((d) => d.dataset_id === inspectId);
  const recentImports = jobs
    .filter((j) => j.job_type === "dataset_import" || j.job_type.startsWith("dataset_"))
    .slice(0, 8);

  return (
    <div className="thx">
      <div className="thx-shell">
        {/* TOPBAR */}
        <div className="thx-topbar">
          <div className="thx-topbar-brand">
            <span className="thx-topbar-brand-mark" />
            <div className="thx-topbar-title">
              <span className="t1">TRAININGHUB · CONSOLE</span>
              <span className="t2">DATA ACQUISITION</span>
            </div>
          </div>
          <div className="thx-topbar-spacer" />
          <div className="thx-hud">
            <div>
              <span className="thx-dot" style={{ color: activeInferenceTarget ? "var(--thx-green)" : "var(--thx-yellow)" }} />
              <span>CLEANER&nbsp;<b>{activeInferenceTarget ? "READY" : "UNSET"}</b></span>
            </div>
            <div>
              <span>VERS&nbsp;<b>{datasets.length}</b></span>
            </div>
            <div>
              <span>APR&nbsp;<b className="ok" style={{ color: "var(--thx-green)" }}>{approvedCount}</b></span>
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
          {STEPS.map((s) => {
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
              <div className="crumb">DATASETS / {STEPS[stepIndex].label.toUpperCase()}</div>
              <h2>
                {step === "source"  && "Choose a Source"}
                {step === "ingest"  && (source === "huggingface" ? "Search the Hub" : source === "csv" ? "Upload a CSV" : "Pull from URL")}
                {step === "process" && "Pipeline Telemetry"}
                {step === "inspect" && "Inspect a Sample"}
                {step === "approve" && "Approval Gate"}
                {step === "library" && "Local Library"}
              </h2>
              <p className="lede">
                {step === "source"  && "Three intake routes, one canonical pipeline. Pick the one that matches the data, then go."}
                {step === "ingest"  && (source === "huggingface"
                  ? "Search the Hugging Face Hub, resolve the repo, name your version, then queue the import."
                  : source === "csv"
                  ? "Drop a CSV. The cleaner converts it to canonical JSONL and validates every row."
                  : "Paste a URL to any CSV, JSONL, or Parquet file. The fetcher caps at your row limit.")}
                {step === "process" && "Live job state. The pipeline is fetch → normalize → clean → write. Cleanup deletes the raw download once canonical rows are persisted."}
                {step === "inspect" && "Pull a random slice between 1 and 100 rows. Read prompt, response, split, and metadata. Reshuffle freely."}
                {step === "approve" && "Approve only after you've reviewed at least one full sample. Reject is destructive."}
                {step === "library" && "All local versions. Approve, delete, or just confirm what's on disk before a training cut."}
              </p>
            </div>
            <div className="stamp">
              {now.toLocaleDateString(undefined, { month: "short", day: "2-digit" })}
              <span>{now.toISOString().slice(11, 19)} UTC</span>
            </div>
          </div>

          {error && (
            <div className="thx-instructions" style={{ borderLeftColor: "var(--thx-red)", color: "var(--thx-red)" }}>
              <strong style={{ color: "var(--thx-red)" }}>FAULT</strong>
              {error}
            </div>
          )}
          {!error && statusMessage && (
            <div className="thx-instructions">
              <strong>SIGNAL</strong>
              {statusMessage}
            </div>
          )}

          {/* === SOURCE STEP === */}
          {step === "source" && (
            <section className="thx-section is-active">
              <div className="thx-cards">
                {(["huggingface", "csv", "url"] as Source[]).map((s) => {
                  const info = SOURCE_INFO[s];
                  const icon = s === "huggingface" ? <Globe size={20} /> : s === "csv" ? <Upload size={20} /> : <LinkIcon size={20} />;
                  return (
                    <button
                      key={s}
                      type="button"
                      className={"thx-card" + (source === s ? " is-selected" : "")}
                      onClick={() => setSource(s)}
                      {...hover(SOURCE_INFO[s])}
                    >
                      <div className="thx-card-row">
                        <span className="thx-card-sub">{info.meta}</span>
                        <span style={{ color: "var(--thx-yellow)" }}>{icon}</span>
                      </div>
                      <div className="thx-card-title">{info.title}</div>
                      <div style={{ fontSize: 12.5, color: "var(--thx-text-mid)", lineHeight: 1.5 }}>
                        {info.desc}
                      </div>
                      <div className="thx-card-stats">
                        {info.stats?.map((st) => (
                          <div className="thx-card-stat" key={st.k}>
                            <span className="k">{st.k}</span>
                            <span className="v">{st.v}</span>
                          </div>
                        ))}
                      </div>
                    </button>
                  );
                })}
              </div>

              <div className="thx-panel">
                <div className="thx-panel-h">
                  <h3>Dataset Template</h3>
                  <span className="thx-tag">[ SCHEMA · CANONICAL ]</span>
                </div>
                <div className="thx-seg">
                  {(["math_sft", "chat_sft", "holdout"] as DatasetType[]).map((t) => (
                    <button
                      key={t}
                      type="button"
                      className={"thx-seg-item" + (datasetType === t ? " is-active" : "")}
                      onClick={() => setDatasetType(t)}
                    >
                      {t === "math_sft" ? "Math SFT" : t === "chat_sft" ? "Chat SFT" : "Holdout"}
                      <span className="sub">
                        {t === "math_sft" ? "Reasoning + answer" : t === "chat_sft" ? "Multi-turn chat" : "Benchmark only"}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            </section>
          )}

          {/* === INGEST STEP === */}
          {step === "ingest" && (
            <section className="thx-section is-active">
              {source === "huggingface" && (
                <>
                  <div className="thx-panel">
                    <div className="thx-panel-h">
                      <h3>Hugging Face · Search</h3>
                      <span className="thx-tag">GET /api/datasets</span>
                    </div>
                    <form
                      className="thx-hub-bar"
                      onSubmit={(e) => {
                        e.preventDefault();
                        searchHub(hfQuery);
                      }}
                    >
                      <span className="thx-hub-bar-icon">
                        <Search size={14} />
                      </span>
                      <input
                        value={hfQuery}
                        onChange={(e) => setHfQuery(e.target.value)}
                        placeholder="search the hub — e.g. math, code, instruction tuning"
                      />
                      <button type="submit" className="thx-btn" disabled={hfSearching || !hfQuery.trim()}>
                        {hfSearching ? <Activity size={14} /> : <Zap size={14} />}
                        {hfSearching ? "Searching" : "Search"}
                      </button>
                    </form>
                    {hfSearchError && (
                      <div className="thx-empty" style={{ marginTop: 10, color: "var(--thx-red)", borderColor: "var(--thx-red)" }}>
                        {hfSearchError}
                      </div>
                    )}

                    <div className="thx-hub-results">
                      {hfResults.length === 0 && !hfSearching && (
                        <div className="thx-empty">[ EMPTY · RUN A SEARCH ]</div>
                      )}
                      {hfResults.map((hit) => {
                        const isSelected = hfSelected?.id === hit.id;
                        return (
                          <button
                            key={hit.id}
                            type="button"
                            className={"thx-hub-row" + (isSelected ? " is-selected" : "")}
                            onClick={() => selectHub(hit)}
                          >
                            <div className="thx-hub-row-main">
                              <div className="thx-hub-row-id">{hit.id}</div>
                              <div className="thx-hub-row-meta">
                                <span><Download size={11} /> {fmtCount(hit.downloads)}</span>
                                <span><Heart size={11} /> {fmtCount(hit.likes)}</span>
                                {hit.tags?.slice(0, 3).map((t) => (
                                  <span className="thx-cap thx-cap--c" key={t}>{t}</span>
                                ))}
                              </div>
                            </div>
                            {isSelected && (
                              <span className="thx-hub-row-mark">▸ SELECTED</span>
                            )}
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  {hfResolving && (
                    <div className="thx-empty" style={{ borderColor: "var(--thx-cyan)" }}>
                      [ RESOLVING REPO · {hfSelected?.id} ]
                    </div>
                  )}

                  {hfResolved && (
                    <div className="thx-panel thx-panel--accent">
                      <div className="thx-panel-h">
                        <h3>Confirm · {hfResolved.repo_id}</h3>
                        <span className="thx-tag">SHA · {hfResolved.sha.slice(0, 10)}</span>
                      </div>
                      <div className="thx-summary" style={{ marginBottom: 14 }}>
                        <div className="thx-summary-item">
                          <span className="k">Repo</span>
                          <span className="v">{hfResolved.repo_id}</span>
                        </div>
                        <div className="thx-summary-item">
                          <span className="k">Visibility</span>
                          <span className="v">{hfResolved.private ? "PRIVATE" : "PUBLIC"}</span>
                        </div>
                        <div className="thx-summary-item">
                          <span className="k">Downloads</span>
                          <span className="v">{fmtCount(hfResolved.downloads)}</span>
                        </div>
                        <div className="thx-summary-item">
                          <span className="k">Last modified</span>
                          <span className="v">{hfResolved.last_modified?.slice(0, 10) ?? "—"}</span>
                        </div>
                      </div>

                      <div className="thx-params">
                        <label className="thx-field">
                          <div className="thx-field-label"><span>title</span></div>
                          <input value={hfTitle} onChange={(e) => setHfTitle(e.target.value)} />
                        </label>
                        <label className="thx-field">
                          <div className="thx-field-label"><span>slug</span></div>
                          <input value={hfSlug} onChange={(e) => setHfSlug(slugify(e.target.value))} />
                        </label>
                        <label className="thx-field">
                          <div className="thx-field-label"><span>split</span></div>
                          <input value={hfSplit} onChange={(e) => setHfSplit(e.target.value)} />
                        </label>
                        <label className="thx-field">
                          <div className="thx-field-label"><span>max_rows</span><span className="v">{hfMaxRows}</span></div>
                          <input type="number" min={1} max={100000} value={hfMaxRows} onChange={(e) => setHfMaxRows(e.target.value)} />
                        </label>
                        <div className="thx-field">
                          <div className="thx-field-label"><span>clean_with_inference</span></div>
                          <label className="thx-toggle">
                            <input type="checkbox" checked={hfClean} onChange={(e) => setHfClean(e.target.checked)} />
                            <span className="thx-toggle-track" />
                            <span style={{ fontFamily: "var(--thx-font-mono)", fontSize: 11, color: "var(--thx-text-mid)" }}>
                              {hfClean ? "ON · local cleaner" : "OFF · raw rows"}
                            </span>
                          </label>
                        </div>
                      </div>
                    </div>
                  )}
                </>
              )}

              {source === "csv" && (
                <>
                  <form
                    className={"thx-drop" + (csvDragActive ? " is-active" : "") + (csvFile ? " has-file" : "")}
                    onSubmit={uploadCsv}
                    onDragOver={(e) => {
                      e.preventDefault();
                      setCsvDragActive(true);
                    }}
                    onDragLeave={() => setCsvDragActive(false)}
                    onDrop={(e) => {
                      e.preventDefault();
                      setCsvDragActive(false);
                      const f = e.dataTransfer.files?.[0];
                      if (f) setCsvFile(f);
                    }}
                  >
                    <div className="thx-drop-icon">
                      <Upload size={28} />
                    </div>
                    <div className="thx-drop-prompt">
                      {csvFile ? csvFile.name : "DROP CSV · OR CLICK TO BROWSE"}
                    </div>
                    <div className="thx-drop-meta">
                      {csvFile
                        ? `${fmtBytes(csvFile.size)} · ${csvFile.type || "text/csv"}`
                        : "UTF-8 · headers required · prompt + response + split"}
                    </div>
                    <input
                      type="file"
                      accept=".csv,text/csv"
                      onChange={(e) => setCsvFile(e.target.files?.[0] || null)}
                    />
                    {csvErrors.file && <div className="thx-drop-error">{csvErrors.file}</div>}
                  </form>

                  <div className="thx-panel">
                    <div className="thx-panel-h">
                      <h3>Cleaning Profile</h3>
                      <span className="thx-tag">[ {datasetType.toUpperCase()} ]</span>
                    </div>
                    <div className="thx-params">
                      <label className="thx-field">
                        <div className="thx-field-label"><span>title</span></div>
                        <input value={csvTitle} onChange={(e) => setCsvTitle(e.target.value)} />
                        {csvErrors.title && <div className="thx-field-err">{csvErrors.title}</div>}
                      </label>
                      <label className="thx-field">
                        <div className="thx-field-label"><span>slug</span></div>
                        <input value={csvSlug} onChange={(e) => setCsvSlug(slugify(e.target.value))} />
                        {csvErrors.slug && <div className="thx-field-err">{csvErrors.slug}</div>}
                      </label>
                      <label className="thx-field">
                        <div className="thx-field-label"><span>max_sequence_length</span><span className="v">{csvMaxSeq}</span></div>
                        <input
                          type="number"
                          min={128}
                          max={32768}
                          value={csvMaxSeq}
                          onChange={(e) => setCsvMaxSeq(e.target.value)}
                        />
                        {csvErrors.maxSeq && <div className="thx-field-err">{csvErrors.maxSeq}</div>}
                      </label>
                      <button
                        type="button"
                        className="thx-field"
                        onClick={() => api.template(datasetType)}
                        style={{ cursor: "pointer", justifyContent: "center", alignItems: "center", textAlign: "center" }}
                      >
                        <span style={{ display: "flex", alignItems: "center", gap: 8, color: "var(--thx-cyan)", fontFamily: "var(--thx-font-display)", letterSpacing: "0.16em", fontSize: 11, textTransform: "uppercase" }}>
                          <Download size={13} /> Download Template
                        </span>
                      </button>
                    </div>
                  </div>

                  {csvValidation && (
                    <div className="thx-panel">
                      <div className="thx-panel-h">
                        <h3>Validation</h3>
                        <span className="thx-tag">{csvValidation.valid ? "PASS" : "ISSUES"}</span>
                      </div>
                      <div className="thx-checklist">
                        <div className={"thx-check " + (csvValidation.valid ? "is-ok" : "is-warn")}>
                          <span className="marker">▣</span>
                          <span>{csvValidation.accepted_count} canonical rows accepted</span>
                        </div>
                        {csvValidation.errors.slice(0, 6).map((e) => (
                          <div className="thx-check is-bad" key={`${e.row_number}:${e.code}`}>
                            <span className="marker">×</span>
                            <span>row {e.row_number} · {e.field} — {e.message}</span>
                          </div>
                        ))}
                        {csvValidation.warnings.slice(0, 4).map((w) => (
                          <div className="thx-check is-warn" key={`w:${w.row_number}:${w.code}`}>
                            <span className="marker">!</span>
                            <span>row {w.row_number} · {w.field} — {w.message}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              )}

              {source === "url" && (
                <div className="thx-panel">
                  <div className="thx-panel-h">
                    <h3>Direct URL Fetch</h3>
                    <span className="thx-tag">POST /api/datasets/import-url</span>
                  </div>
                  <form className="thx-params" onSubmit={importUrl}>
                    <label className="thx-field" style={{ gridColumn: "span 2" }}>
                      <div className="thx-field-label"><span>url</span></div>
                      <input
                        value={urlValue}
                        onChange={(e) => setUrlValue(e.target.value)}
                        placeholder="https://… or file:///…"
                      />
                      {urlErrors.url && <div className="thx-field-err">{urlErrors.url}</div>}
                    </label>
                    <label className="thx-field">
                      <div className="thx-field-label"><span>title</span></div>
                      <input value={urlTitle} onChange={(e) => setUrlTitle(e.target.value)} />
                      {urlErrors.title && <div className="thx-field-err">{urlErrors.title}</div>}
                    </label>
                    <label className="thx-field">
                      <div className="thx-field-label"><span>slug</span></div>
                      <input value={urlSlug} onChange={(e) => setUrlSlug(slugify(e.target.value))} />
                      {urlErrors.slug && <div className="thx-field-err">{urlErrors.slug}</div>}
                    </label>
                    <label className="thx-field">
                      <div className="thx-field-label"><span>max_rows</span><span className="v">{urlMaxRows}</span></div>
                      <input
                        type="number"
                        min={1}
                        max={100000}
                        value={urlMaxRows}
                        onChange={(e) => setUrlMaxRows(e.target.value)}
                      />
                      {urlErrors.maxRows && <div className="thx-field-err">{urlErrors.maxRows}</div>}
                    </label>
                  </form>
                </div>
              )}
            </section>
          )}

          {/* === PROCESS STEP === */}
          {step === "process" && (
            <section className="thx-section is-active">
              <div className="thx-pipe">
                <PipelineNode
                  label="FETCH"
                  hint="download from origin"
                  state={pipelineState(activeJob, 0)}
                />
                <div className={"thx-pipe-edge " + edgeState(activeJob, 1)} />
                <PipelineNode
                  label="NORMALIZE"
                  hint="canonical schema"
                  state={pipelineState(activeJob, 1)}
                />
                <div className={"thx-pipe-edge " + edgeState(activeJob, 2)} />
                <PipelineNode
                  label="CLEAN"
                  hint="local inference"
                  state={pipelineState(activeJob, 2)}
                />
                <div className={"thx-pipe-edge " + edgeState(activeJob, 3)} />
                <PipelineNode
                  label="WRITE"
                  hint="register version"
                  state={pipelineState(activeJob, 3)}
                />
                <div className={"thx-pipe-edge " + edgeState(activeJob, 4)} />
                <PipelineNode
                  label="CLEANUP"
                  hint="delete raw"
                  state={pipelineState(activeJob, 4)}
                />
              </div>

              <div className="thx-panel">
                <div className="thx-panel-h">
                  <h3>Active Job</h3>
                  <span className="thx-tag">
                    {activeJob ? activeJob.status.toUpperCase() : "NO ACTIVE JOB"}
                  </span>
                </div>
                {!activeJob && (
                  <div className="thx-empty">
                    [ NO ACTIVE INGESTION · GO TO INGEST OR PICK A RECENT JOB BELOW ]
                  </div>
                )}
                {activeJob && (
                  <div className="thx-summary">
                    <div className="thx-summary-item">
                      <span className="k">Job ID</span>
                      <span className="v">{activeJob.job_id.slice(0, 16)}</span>
                    </div>
                    <div className="thx-summary-item">
                      <span className="k">Type</span>
                      <span className="v">{activeJob.job_type}</span>
                    </div>
                    <div className="thx-summary-item">
                      <span className="k">Slug</span>
                      <span className="v">{activeJob.slug}</span>
                    </div>
                    <div className="thx-summary-item">
                      <span className="k">Worker</span>
                      <span className="v">{activeJob.worker_pid ?? "—"}</span>
                    </div>
                    <div className="thx-summary-item">
                      <span className="k">Created</span>
                      <span className="v">{new Date(activeJob.created_at * 1000).toLocaleTimeString()}</span>
                    </div>
                    <div className="thx-summary-item">
                      <span className="k">Finished</span>
                      <span className="v">
                        {activeJob.finished_at
                          ? new Date(activeJob.finished_at * 1000).toLocaleTimeString()
                          : "—"}
                      </span>
                    </div>
                  </div>
                )}
                {activeJob?.terminal_message && (
                  <div className="thx-instructions" style={{ marginTop: 12 }}>
                    <strong>MSG</strong>{activeJob.terminal_message}
                  </div>
                )}
              </div>

              <div className="thx-panel">
                <div className="thx-panel-h">
                  <h3>Recent Imports</h3>
                  <span className="thx-tag">{recentImports.length} JOBS</span>
                </div>
                {recentImports.length === 0 ? (
                  <div className="thx-empty">[ EMPTY · NO DATASET JOBS RECENT ]</div>
                ) : (
                  <div className="thx-runs">
                    {recentImports.map((j) => {
                      const cls =
                        j.status === "running"
                          ? "is-running"
                          : j.status === "queued"
                          ? "is-queued"
                          : j.status === "failed"
                          ? "is-failed"
                          : "is-done";
                      return (
                        <button
                          key={j.job_id}
                          type="button"
                          className={"thx-run " + cls}
                          onClick={() => {
                            setActiveJobId(j.job_id);
                            setSelectedJob(j);
                          }}
                        >
                          <span className="thx-run-dot" />
                          <span className="thx-run-id">{j.job_id.slice(0, 18)}</span>
                          <span className="thx-run-meta">{j.slug}</span>
                          <span className="thx-run-status">{j.status}</span>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            </section>
          )}

          {/* === INSPECT STEP === */}
          {step === "inspect" && (
            <section className="thx-section is-active">
              <div className="thx-panel">
                <div className="thx-panel-h">
                  <h3>Choose Version</h3>
                  <span className="thx-tag">{datasets.length} LOCAL</span>
                </div>
                {datasets.length === 0 ? (
                  <div className="thx-empty">[ NO LOCAL VERSIONS · INGEST FIRST ]</div>
                ) : (
                  <div className="thx-cards">
                    {datasets.map((d) => (
                      <button
                        key={d.dataset_id}
                        type="button"
                        className={"thx-card" + (d.dataset_id === inspectId ? " is-selected" : "")}
                        onClick={() => setInspectId(d.dataset_id)}
                      >
                        <div className="thx-card-row">
                          <span className="thx-card-sub">{d.version_id.slice(0, 12)}</span>
                          <span className={"thx-cap " + (d.approved ? "thx-cap--ok" : d.validation.valid ? "thx-cap--w" : "thx-cap--no")}>
                            {d.approved ? "APPROVED" : d.validation.valid ? "PENDING" : "ISSUES"}
                          </span>
                        </div>
                        <div className="thx-card-title">{d.title}</div>
                        <div className="thx-card-stats">
                          <div className="thx-card-stat">
                            <span className="k">ROWS</span>
                            <span className="v">{fmtCount(d.row_count)}</span>
                          </div>
                          <div className="thx-card-stat">
                            <span className="k">TYPE</span>
                            <span className="v">{d.dataset_type}</span>
                          </div>
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {inspectDataset && (
                <div className="thx-panel">
                  <div className="thx-panel-h">
                    <h3>Random Sample</h3>
                    <span className="thx-tag">N · {sampleSize} OF 100</span>
                  </div>
                  <div className="thx-sample-bar">
                    <div className="thx-sample-meta">
                      <span className="k">SAMPLE SIZE</span>
                      <span className="v">{sampleSize}</span>
                    </div>
                    <input
                      type="range"
                      min={1}
                      max={100}
                      value={sampleSize}
                      onChange={(e) => setSampleSize(Number(e.target.value))}
                      className="thx-sample-slider"
                    />
                    <div className="thx-sample-marks">
                      <span>1</span>
                      <span>25</span>
                      <span>50</span>
                      <span>75</span>
                      <span>100</span>
                    </div>
                    <button
                      type="button"
                      className="thx-btn"
                      onClick={() => setShuffleSeed((s) => s + 1)}
                      disabled={sampleLoading}
                    >
                      <Shuffle size={14} /> Reshuffle
                    </button>
                  </div>

                  {sampleLoading && (
                    <div className="thx-empty" style={{ marginTop: 10 }}>
                      [ DRAWING {sampleSize} ROWS … ]
                    </div>
                  )}

                  {!sampleLoading && sample && (
                    <div className="thx-rows">
                      {sample.records.map((rec, i) => (
                        <SampleRow key={`${rec.index}-${i}`} record={rec} index={i + 1} />
                      ))}
                      {sample.records.length === 0 && (
                        <div className="thx-empty">[ NO ROWS RETURNED ]</div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </section>
          )}

          {/* === APPROVE STEP === */}
          {step === "approve" && (
            <section className="thx-section is-active">
              {!inspectDataset && (
                <div className="thx-empty">[ SELECT A DATASET ON INSPECT FIRST ]</div>
              )}
              {inspectDataset && (
                <>
                  <div className="thx-panel thx-panel--accent">
                    <div className="thx-panel-h">
                      <h3>{inspectDataset.title}</h3>
                      <span className="thx-tag">{inspectDataset.version_id.slice(0, 14)}</span>
                    </div>
                    <div className="thx-summary">
                      <div className="thx-summary-item">
                        <span className="k">Rows</span>
                        <span className="v">{fmtCount(inspectDataset.row_count)}</span>
                      </div>
                      <div className="thx-summary-item">
                        <span className="k">Type</span>
                        <span className="v">{inspectDataset.dataset_type}</span>
                      </div>
                      <div className="thx-summary-item">
                        <span className="k">Splits</span>
                        <span className="v">
                          {Object.entries(inspectDataset.split_counts)
                            .map(([k, v]) => `${k}:${fmtCount(v)}`)
                            .join(" · ") || "—"}
                        </span>
                      </div>
                      <div className="thx-summary-item">
                        <span className="k">Validation</span>
                        <span className="v">
                          {inspectDataset.validation.valid ? "PASS" : "ISSUES"}
                        </span>
                      </div>
                      <div className="thx-summary-item">
                        <span className="k">Sampled</span>
                        <span className="v">{sample?.records.length ?? 0} rows</span>
                      </div>
                      <div className="thx-summary-item">
                        <span className="k">Status</span>
                        <span className="v" style={{ color: inspectDataset.approved ? "var(--thx-green)" : "var(--thx-yellow)" }}>
                          {inspectDataset.approved ? "APPROVED" : "PENDING"}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="thx-panel">
                    <div className="thx-panel-h">
                      <h3>Pre-Flight Checks</h3>
                      <span className="thx-tag">CONSISTENCY</span>
                    </div>
                    <div className="thx-checklist">
                      <div className={"thx-check " + (inspectDataset.validation.valid ? "is-ok" : "is-bad")}>
                        <span className="marker">▣</span>
                        <span>schema validates ({inspectDataset.validation.accepted_count} accepted)</span>
                      </div>
                      <div className={"thx-check " + ((sample?.records.length ?? 0) > 0 ? "is-ok" : "is-warn")}>
                        <span className="marker">▣</span>
                        <span>random sample reviewed ({sample?.records.length ?? 0} rows)</span>
                      </div>
                      <div className={"thx-check " + (inspectDataset.row_count > 0 ? "is-ok" : "is-bad")}>
                        <span className="marker">▣</span>
                        <span>row count &gt; 0</span>
                      </div>
                      <div className={"thx-check " + (Object.keys(inspectDataset.split_counts).length > 0 ? "is-ok" : "is-warn")}>
                        <span className="marker">▣</span>
                        <span>at least one split present</span>
                      </div>
                    </div>
                  </div>

                  <div className="thx-action" style={{ marginTop: 4 }}>
                    <button
                      type="button"
                      className="thx-btn thx-btn--danger"
                      onClick={() => setPendingDelete(inspectDataset)}
                      disabled={busy}
                    >
                      <Trash2 size={14} /> Reject &amp; Delete
                    </button>
                    <div className="thx-progress">
                      <div className="thx-progress-meta">
                        <span>WIZARD · {progressPct}%</span>
                        <span className="v">{step.toUpperCase()}</span>
                      </div>
                      <div className="thx-progress-bar" style={{ ["--p" as string]: `${progressPct}%` }} />
                    </div>
                    <button
                      type="button"
                      className="thx-btn thx-btn--primary"
                      onClick={() => approveDataset(inspectDataset.dataset_id)}
                      disabled={busy || inspectDataset.approved}
                    >
                      <CheckCircle size={14} />
                      {inspectDataset.approved ? "Approved" : "Approve for Training"}
                    </button>
                  </div>
                </>
              )}
            </section>
          )}

          {/* === LIBRARY STEP === */}
          {step === "library" && (
            <section className="thx-section is-active">
              <div className="thx-mons">
                <div className="thx-mon">
                  <div className="k">VERSIONS</div>
                  <div className="v">{datasets.length}</div>
                </div>
                <div className="thx-mon">
                  <div className="k">APPROVED</div>
                  <div className="v gr">{approvedCount}</div>
                </div>
                <div className="thx-mon">
                  <div className="k">PENDING</div>
                  <div className="v yl">{datasets.length - approvedCount}</div>
                </div>
                <div className="thx-mon">
                  <div className="k">TOTAL ROWS</div>
                  <div className="v cy">{fmtCount(totalRows)}</div>
                </div>
              </div>

              <div className="thx-panel">
                <div className="thx-panel-h">
                  <h3>Local Versions</h3>
                  <span className="thx-tag">REGISTRY</span>
                </div>
                {datasets.length === 0 ? (
                  <div className="thx-empty">[ NO VERSIONS · ACQUIRE A DATASET ]</div>
                ) : (
                  <div className="thx-lib">
                    {datasets.map((d) => (
                      <div className="thx-lib-row" key={d.dataset_id}>
                        <div className="thx-lib-main">
                          <div className="thx-lib-title">{d.title}</div>
                          <div className="thx-lib-sub">
                            {d.slug} · {d.version_id.slice(0, 12)} · {d.dataset_type}
                          </div>
                        </div>
                        <div className="thx-lib-stats">
                          <div className="thx-lib-stat">
                            <span className="k">ROWS</span>
                            <span className="v">{fmtCount(d.row_count)}</span>
                          </div>
                          <div className="thx-lib-stat">
                            <span className="k">VAL</span>
                            <span className="v" style={{ color: d.validation.valid ? "var(--thx-green)" : "var(--thx-yellow)" }}>
                              {d.validation.valid ? "PASS" : `${d.validation.errors.length}E/${d.validation.warnings.length}W`}
                            </span>
                          </div>
                          <div className="thx-lib-stat">
                            <span className="k">STATUS</span>
                            <span className="v" style={{ color: d.approved ? "var(--thx-green)" : "var(--thx-yellow)" }}>
                              {d.approved ? "APPROVED" : "PENDING"}
                            </span>
                          </div>
                        </div>
                        <div className="thx-lib-actions">
                          <button
                            type="button"
                            className="thx-btn"
                            onClick={() => {
                              setInspectId(d.dataset_id);
                              go("inspect");
                            }}
                          >
                            <Eye size={13} /> Inspect
                          </button>
                          {!d.approved && (
                            <button
                              type="button"
                              className="thx-btn"
                              onClick={() => approveDataset(d.dataset_id)}
                              disabled={busy}
                            >
                              <CheckCircle size={13} /> Approve
                            </button>
                          )}
                          <button
                            type="button"
                            className="thx-btn thx-btn--danger"
                            onClick={() => setPendingDelete(d)}
                            disabled={busy}
                          >
                            <Trash2 size={13} /> Delete
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </section>
          )}
        </main>

        {/* ASIDE */}
        <aside className="thx-panel thx-aside">
          <div className="thx-aside-h">
            <span>[ CONTEXT ]</span>
            <span className="ping">live</span>
          </div>
          <div className="thx-aside-meta">{activeAside.meta}</div>
          <h4 className="thx-aside-title">{activeAside.title}</h4>
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
          {activeAside.note && <div className="thx-aside-note">{activeAside.note}</div>}
        </aside>

        {/* ACTION BAR */}
        <div className="thx-action">
          <button type="button" className="thx-btn" onClick={goPrev} disabled={stepIndex === 0}>
            <ChevronLeft size={14} /> Back
          </button>
          <div className="thx-progress">
            <div className="thx-progress-meta">
              <span>WIZARD · {progressPct}%</span>
              <span className="v">{step.toUpperCase()}</span>
            </div>
            <div className="thx-progress-bar" style={{ ["--p" as string]: `${progressPct}%` }} />
          </div>
          {step === "ingest" && source === "huggingface" ? (
            <button
              type="button"
              className="thx-btn thx-btn--primary"
              onClick={startHubImport}
              disabled={!hfResolved || busy}
            >
              <Zap size={14} /> {busy ? "Queueing" : "Queue Import"}
            </button>
          ) : step === "ingest" && source === "csv" ? (
            <button
              type="button"
              className="thx-btn thx-btn--primary"
              onClick={() => uploadCsv()}
              disabled={!csvFile || busy}
            >
              <Upload size={14} /> {busy ? "Uploading" : "Validate &amp; Upload"}
            </button>
          ) : step === "ingest" && source === "url" ? (
            <button
              type="button"
              className="thx-btn thx-btn--primary"
              onClick={() => importUrl()}
              disabled={!urlValue || busy}
            >
              <Download size={14} /> {busy ? "Queueing" : "Queue Import"}
            </button>
          ) : (
            <button
              type="button"
              className="thx-btn thx-btn--primary"
              onClick={goNext}
              disabled={stepIndex === STEPS.length - 1}
            >
              Next <ChevronRight size={14} />
            </button>
          )}
        </div>
      </div>

      {/* DELETE CONFIRM OVERLAY */}
      {pendingDelete && (
        <div className="thx-confirm-back" onClick={() => setPendingDelete(null)}>
          <div className="thx-confirm" onClick={(e) => e.stopPropagation()}>
            <div className="thx-confirm-icon">
              <AlertTriangle size={26} />
            </div>
            <div className="thx-confirm-meta">DESTRUCTIVE · IRREVERSIBLE</div>
            <h3>Delete {pendingDelete.title}?</h3>
            <p>
              This removes the JSONL from disk and drops the registry entry. Any training job referencing
              this version will fail to start.
            </p>
            <div className="thx-confirm-stats">
              <div><span>VERSION</span><strong>{pendingDelete.version_id.slice(0, 12)}</strong></div>
              <div><span>ROWS</span><strong>{fmtCount(pendingDelete.row_count)}</strong></div>
              <div><span>STATUS</span><strong>{pendingDelete.approved ? "APPROVED" : "PENDING"}</strong></div>
            </div>
            <div className="thx-confirm-actions">
              <button className="thx-btn" onClick={() => setPendingDelete(null)} disabled={busy}>
                Cancel
              </button>
              <button
                className="thx-btn thx-btn--primary thx-btn--danger-fill"
                onClick={() => deleteDataset(pendingDelete.dataset_id)}
                disabled={busy}
              >
                <Trash2 size={14} /> {busy ? "Deleting" : "Delete Forever"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ---- pipeline helpers
function pipelineState(job: JobRecord | null, idx: number): "idle" | "active" | "done" | "fail" {
  if (!job) return "idle";
  const status = job.status;
  if (status === "failed") return idx === 0 ? "fail" : "idle";
  if (status === "succeeded" || status === "completed") return "done";
  if (status === "queued") return idx === 0 ? "active" : "idle";
  // running — animate the first three nodes; later nodes idle
  if (idx <= 2) return "active";
  return "idle";
}
function edgeState(job: JobRecord | null, idx: number): string {
  const ps = pipelineState(job, idx - 1);
  return ps === "done" ? "is-flow" : ps === "active" ? "is-active" : "";
}

function PipelineNode({
  label,
  hint,
  state,
}: {
  label: string;
  hint: string;
  state: "idle" | "active" | "done" | "fail";
}) {
  return (
    <div className={"thx-pipe-node is-" + state}>
      <div className="thx-pipe-node-ring" />
      <div className="thx-pipe-node-label">{label}</div>
      <div className="thx-pipe-node-hint">{hint}</div>
    </div>
  );
}

// ---- sample row (collapsed/expanded record card)
function SampleRow({ record, index }: { record: DatasetRecordView; index: number }) {
  const [open, setOpen] = useState(false);
  const splitTag = String((record.metadata?.split as string) || "—");
  return (
    <button
      type="button"
      className={"thx-row" + (open ? " is-open" : "")}
      onClick={() => setOpen((v) => !v)}
    >
      <div className="thx-row-h">
        <span className="thx-row-num">#{String(index).padStart(3, "0")}</span>
        <span className="thx-row-tag">SPLIT · {splitTag.toUpperCase()}</span>
        <span className="thx-row-idx">idx {record.index}</span>
      </div>
      <div className="thx-row-prompt">
        <span className="thx-row-k">PROMPT</span>
        <span>{truncate(record.prompt, open ? 4000 : 220)}</span>
      </div>
      <div className="thx-row-resp">
        <span className="thx-row-k">RESPONSE</span>
        <span>{truncate(record.response, open ? 4000 : 220)}</span>
      </div>
      {open && record.system && (
        <div className="thx-row-sys">
          <span className="thx-row-k">SYSTEM</span>
          <span>{record.system}</span>
        </div>
      )}
      {open && Object.keys(record.metadata || {}).length > 0 && (
        <div className="thx-row-meta">
          <span className="thx-row-k">META</span>
          <span>{JSON.stringify(record.metadata)}</span>
        </div>
      )}
      <div className="thx-row-toggle">{open ? "▾ COLLAPSE" : "▸ EXPAND"}</div>
    </button>
  );
}

function truncate(s: string, n: number): string {
  if (!s) return "—";
  return s.length > n ? s.slice(0, n) + "…" : s;
}
