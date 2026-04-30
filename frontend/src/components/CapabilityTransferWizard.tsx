import { ReactNode, useEffect, useMemo, useState } from "react";
import { AlertTriangle, ArrowRight, CheckCircle, Database, GitCompare, Layers, Play, Rocket, Sliders, Wand2 } from "lucide-react";
import {
  api,
  ArtifactRecord,
  CapabilityRuntime,
  CapabilityTransferRecord,
  DatasetRecord,
  InferenceTarget,
  JobRecord,
  ModelRecord
} from "../api/client";
import { FieldNote } from "./FieldNote";
import { JobLogPanel } from "./JobLogPanel";
import { ActiveTransferPill } from "./ActiveTransferPill";
import "../styles/cyberpunk.css";

type StepKey = "source" | "target" | "calibration" | "extract" | "align" | "deploy";
type ToastHandler = (message: string, tone?: "info" | "success" | "error", title?: string) => void;

type Candidate = {
  id: string;
  runtime: CapabilityRuntime;
  modelSlug: string;
  artifactId: string;
  displayName: string;
  detail: string;
  params: number;
  mode: "hf" | "gguf";
  enabled: boolean;
};

type Props = {
  models: ModelRecord[];
  datasets: DatasetRecord[];
  jobs: JobRecord[];
  artifacts: ArtifactRecord[];
  transfers: CapabilityTransferRecord[];
  activeInferenceTarget: InferenceTarget | null;
  refresh: () => void;
  setSelectedJob: (job: JobRecord) => void;
  onToast: ToastHandler;
};

const STEPS: { key: StepKey; label: string; meta: string }[] = [
  { key: "source", label: "Source", meta: "CAPABILITY" },
  { key: "target", label: "Target", meta: "RECEIVER" },
  { key: "calibration", label: "Calibration", meta: "PAIRS" },
  { key: "extract", label: "Extract", meta: "VECTOR" },
  { key: "align", label: "Align", meta: "MAP" },
  { key: "deploy", label: "Deploy", meta: "INFERENCE" }
];

export function CapabilityTransferWizard({
  models,
  datasets,
  jobs,
  artifacts,
  transfers,
  activeInferenceTarget,
  refresh,
  setSelectedJob,
  onToast
}: Props) {
  const [step, setStep] = useState<StepKey>("source");
  const [sourceId, setSourceId] = useState("");
  const [targetId, setTargetId] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [contrastMode, setContrastMode] = useState<"prompt_pair" | "system_pair">("prompt_pair");
  const [layerTargets, setLayerTargets] = useState<string>("all");
  const [rank, setRank] = useState(16);
  const [layerPairsText, setLayerPairsText] = useState("");
  const [alpha, setAlpha] = useState(1.0);
  const [dryRun, setDryRun] = useState(true);
  const [selectedTransferId, setSelectedTransferId] = useState("");
  const [statusMessage, setStatusMessage] = useState("");

  const candidates = useMemo(() => buildCandidates(models, artifacts), [models, artifacts]);
  const source = candidates.find((item) => item.id === sourceId);
  const targets = useMemo(() => candidates.filter((item) => item.id !== sourceId && targetAllowed(source, item)), [candidates, sourceId]);
  const target = targets.find((item) => item.id === targetId);
  const approvedDatasets = useMemo(() => datasets.filter((dataset) => dataset.approved), [datasets]);
  const selectedTransfer = transfers.find((item) => item.transfer_id === selectedTransferId) || transfers[0];
  const extractJob = selectedTransfer?.extract_job_id ? jobs.find((job) => job.job_id === selectedTransfer.extract_job_id) : undefined;
  const alignJob = selectedTransfer?.align_job_id ? jobs.find((job) => job.job_id === selectedTransfer.align_job_id) : undefined;
  const activeTransferId = activeInferenceTarget?.capability_transfer_id || "";

  useEffect(() => {
    if (!sourceId && candidates.length > 0) {
      setSourceId(candidates[0].id);
    }
  }, [candidates, sourceId]);

  useEffect(() => {
    if (targetId && !targets.some((item) => item.id === targetId)) {
      setTargetId("");
    }
    if (!targetId && targets.length > 0) {
      setTargetId(targets[0].id);
    }
  }, [targetId, targets]);

  useEffect(() => {
    if (!datasetId && approvedDatasets.length > 0) {
      setDatasetId(approvedDatasets[0].dataset_id);
    }
  }, [approvedDatasets, datasetId]);

  useEffect(() => {
    if (!selectedTransferId && transfers.length > 0) {
      setSelectedTransferId(transfers[0].transfer_id);
    }
  }, [selectedTransferId, transfers]);

  async function startExtraction() {
    if (!source || !target || !datasetId) {
      setStatusMessage("Select a source, target, and approved calibration set.");
      return;
    }
    try {
      const transfer = await api.capabilityTransfers.create({
        display_name: displayName.trim() || `${source.displayName} to ${target.displayName}`,
        source_model_slug: source.modelSlug,
        source_runtime: source.runtime,
        source_artifact_id: source.artifactId,
        target_model_slug: target.modelSlug,
        target_runtime: target.runtime,
        target_artifact_id: target.artifactId,
        calibration_dataset_id: datasetId,
        layer_targets: parseLayerTargets(layerTargets),
        contrast_mode: contrastMode,
        rank,
        dry_run: dryRun
      });
      setSelectedTransferId(transfer.transfer_id);
      setStatusMessage(`${transfer.transfer_id} queued for extraction.`);
      onToast(`${transfer.display_name} queued for extraction.`, "success", "Extraction queued");
      await refresh();
      setStep("extract");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Capability extraction could not be queued.";
      setStatusMessage(message);
      onToast(message, "error", "Extraction failed");
    }
  }

  async function startAlignment() {
    if (!selectedTransfer) {
      return;
    }
    try {
      const transfer = await api.capabilityTransfers.align(selectedTransfer.transfer_id, {
        rank,
        layer_pairs: parseLayerPairs(layerPairsText)
      });
      setSelectedTransferId(transfer.transfer_id);
      setStatusMessage(`${transfer.transfer_id} queued for alignment.`);
      onToast(`${transfer.display_name} queued for alignment.`, "success", "Alignment queued");
      await refresh();
      setStep("align");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Capability alignment could not be queued.";
      setStatusMessage(message);
      onToast(message, "error", "Alignment failed");
    }
  }

  async function activate() {
    if (!selectedTransfer) {
      return;
    }
    try {
      const response = await api.capabilityTransfers.activate(selectedTransfer.transfer_id, {
        alpha,
        layer_targets: parseLayerTargets(layerTargets)
      });
      const warning = response.warning || "";
      onToast(warning || "Capability transfer activated on the inference target.", warning ? "info" : "success", "Transfer activated");
      await refresh();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Capability transfer could not be activated.";
      onToast(message, "error", "Activation failed");
    }
  }

  async function deleteSelected() {
    if (!selectedTransfer) {
      return;
    }
    try {
      await api.capabilityTransfers.delete(selectedTransfer.transfer_id);
      onToast("Capability transfer deleted and artifacts removed.", "success", "Transfer deleted");
      setSelectedTransferId("");
      await refresh();
    } catch (err) {
      onToast(err instanceof Error ? err.message : "Capability transfer could not be deleted.", "error", "Delete failed");
    }
  }

  return (
    <div className="thx thx-page thx-xfer">
      <div className="thx-stage-h">
        <div>
          <div className="crumb">MODEL · UNLOCK · TRANSFER</div>
          <h2>
            <span className="thx-glitch" data-text="CAPABILITY TRANSFER">
              CAPABILITY TRANSFER
            </span>
          </h2>
          <p className="lede">Extract a source capability direction, align it into a target model, then steer inference without changing weights.</p>
        </div>
        <div className="stamp">
          STATUS
          <span>{selectedTransfer ? selectedTransfer.status.toUpperCase() : "NEW"}</span>
        </div>
      </div>

      <section className="thx-panel thx-panel--accent thx-xfer-overview">
        <div className="thx-xfer-flow" aria-label="Capability transfer workflow">
          {STEPS.map((item, index) => (
            <button type="button" className={`thx-xfer-step ${step === item.key ? "is-active" : ""}`} onClick={() => setStep(item.key)} key={item.key}>
              <span>{String(index + 1).padStart(2, "0")}</span>
              <strong>{item.label}</strong>
              <em>{item.meta}</em>
            </button>
          ))}
        </div>
        <ActiveTransferPill activeInferenceTarget={activeInferenceTarget} transfers={transfers} refresh={refresh} onToast={onToast} />
      </section>

      <div className="thx-page-grid thx-page-grid--wide">
        <section className="thx-panel thx-xfer-main">
          {step === "source" && (
            <WizardSection title="Source Model" tag={`SOURCE · ${candidates.length}`}>
              <CandidateGrid candidates={candidates} selectedId={sourceId} onSelect={setSourceId} />
              <FieldNote note="Use an HF source for full per-layer extraction. GGUF sources are last-layer degraded mode." link="#gguf-degraded-mode" />
            </WizardSection>
          )}

          {step === "target" && (
            <WizardSection title="Target Model" tag={`TARGET · ${targets.length}`}>
              <div className="thx-xfer-bridge">
                <span>{source?.displayName || "Select source"}</span>
                <ArrowRight size={18} />
                <span>{target?.displayName || "Select target"}</span>
              </div>
              <CandidateGrid candidates={targets} selectedId={targetId} onSelect={setTargetId} />
              <FieldNote note="The picker favors smaller targets when parameter counts are known, matching the transfer-across-scales workflow." link="#capability-vectors" />
            </WizardSection>
          )}

          {step === "calibration" && (
            <WizardSection title="Calibration Set" tag={`DATASETS · ${approvedDatasets.length}`}>
              <div className="thx-cards thx-card-grid--compact">
                {approvedDatasets.map((dataset) => (
                  <button
                    type="button"
                    className={`thx-card ${dataset.dataset_id === datasetId ? "is-selected" : ""}`}
                    onClick={() => setDatasetId(dataset.dataset_id)}
                    key={dataset.dataset_id}
                  >
                    <span className="thx-card-row">
                      <span className="thx-card-title">{dataset.title}</span>
                      <span className="thx-card-status">{dataset.dataset_type}</span>
                    </span>
                    <span className="thx-card-sub">{dataset.slug}</span>
                    <span className="thx-card-stats">
                      <span className="thx-card-stat"><span className="k">Rows</span><span className="v">{dataset.row_count}</span></span>
                      <span className="thx-card-stat"><span className="k">Approved</span><span className="v">{dataset.approved ? "yes" : "no"}</span></span>
                    </span>
                  </button>
                ))}
                {approvedDatasets.length === 0 && <div className="thx-empty">NO APPROVED CALIBRATION DATASETS</div>}
              </div>
              <FieldNote note="Calibration JSONL needs prompt_present and prompt_absent fields, or system_present and system_absent for system-pair mode." link="#calibration-pairs" />
            </WizardSection>
          )}

          {step === "extract" && (
            <WizardSection title="Extract Capability Vector" tag="VECTOR · JOB">
              <div className="thx-params">
                <label className="thx-field thx-field--wide">
                  <span className="thx-field-label"><span>Display name</span><span className="v">{displayName ? "custom" : "auto"}</span></span>
                  <input value={displayName} onChange={(event) => setDisplayName(event.target.value)} placeholder="CoT transfer to compact instruct model" />
                </label>
                <label className="thx-field">
                  <span className="thx-field-label"><span>Contrast mode</span><span className="v">{contrastMode}</span></span>
                  <div className="thx-seg">
                    <button type="button" className={`thx-seg-item ${contrastMode === "prompt_pair" ? "is-active" : ""}`} onClick={() => setContrastMode("prompt_pair")}>Prompt pair</button>
                    <button type="button" className={`thx-seg-item ${contrastMode === "system_pair" ? "is-active" : ""}`} onClick={() => setContrastMode("system_pair")}>System pair</button>
                  </div>
                </label>
                <label className="thx-field">
                  <span className="thx-field-label"><span>Layer targets</span><span className="v">{layerTargets}</span></span>
                  <select value={layerTargets} onChange={(event) => setLayerTargets(event.target.value)}>
                    <option value="all">All layers</option>
                    <option value="every-4">Every 4th layer</option>
                    <option value="last">Last layer</option>
                  </select>
                </label>
                <label className="thx-field thx-field--toggle">
                  <span className="thx-field-label"><span>Dry run</span><span className="v">{dryRun ? "on" : "off"}</span></span>
                  <span className="thx-toggle">
                    <input type="checkbox" checked={dryRun} onChange={(event) => setDryRun(event.target.checked)} />
                    <span className="thx-toggle-track" />
                    <span className="thx-toggle-copy">Use deterministic smoke vectors</span>
                  </span>
                </label>
              </div>
              <div className="thx-form-actions">
                <button type="button" className="thx-btn thx-btn--primary" onClick={startExtraction} disabled={!source || !target || !datasetId}>
                  <Play size={16} /> Start Extraction
                </button>
              </div>
              <JobLogPanel job={extractJob} />
            </WizardSection>
          )}

          {step === "align" && (
            <WizardSection title="Align Into Target Space" tag="ALIGN · SVD">
              <div className="thx-params">
                <label className="thx-field">
                  <span className="thx-field-label"><span>Rank</span><span className="v">{rank}</span></span>
                  <input type="range" min={4} max={64} step={4} value={rank} onChange={(event) => setRank(Number(event.target.value))} />
                  <FieldNote note="Lower rank is smoother and lossier. Higher rank preserves more source detail." link="#linear-alignment" />
                </label>
                <label className="thx-field thx-field--wide">
                  <span className="thx-field-label"><span>Layer pairs</span><span className="v">{layerPairsText ? "custom" : "auto"}</span></span>
                  <input value={layerPairsText} onChange={(event) => setLayerPairsText(event.target.value)} placeholder="auto, or 12:8,16:12,20:16" />
                </label>
              </div>
              <div className="thx-form-actions">
                <button type="button" className="thx-btn thx-btn--primary" onClick={startAlignment} disabled={!selectedTransfer || selectedTransfer.status !== "extracted"}>
                  <GitCompare size={16} /> Start Alignment
                </button>
              </div>
              <JobLogPanel job={alignJob} />
            </WizardSection>
          )}

          {step === "deploy" && (
            <WizardSection title="Deploy Steering" tag="ACTIVE · CHAT">
              <div className="thx-params">
                <label className="thx-field">
                  <span className="thx-field-label"><span>Alpha</span><span className="v">{alpha.toFixed(2)}</span></span>
                  <input type="range" min={0} max={4} step={0.05} value={alpha} onChange={(event) => setAlpha(Number(event.target.value))} />
                  <FieldNote note="Alpha is the strength knob. Start around 0.5 to 1.5 and raise carefully." link="#alpha-tuning" />
                </label>
                <label className="thx-field">
                  <span className="thx-field-label"><span>Active layers</span><span className="v">{layerTargets}</span></span>
                  <select value={layerTargets} onChange={(event) => setLayerTargets(event.target.value)}>
                    <option value="all">All aligned layers</option>
                    <option value="every-4">Every 4th layer</option>
                    <option value="last">Last layer only</option>
                  </select>
                </label>
              </div>
              {selectedTransfer?.degraded_mode && (
                <div className="thx-xfer-warning">
                  <AlertTriangle size={16} />
                  <span>Last-layer mode. Fidelity is degraded versus the paper's per-layer HF path.</span>
                </div>
              )}
              <div className="thx-form-actions">
                <button type="button" className="thx-btn thx-btn--primary" onClick={activate} disabled={!selectedTransfer || selectedTransfer.status !== "ready"}>
                  <Rocket size={16} /> Activate Transfer On Inference
                </button>
                <a className="thx-btn" href="#/chat">Open Chat</a>
              </div>
            </WizardSection>
          )}
        </section>

        <aside className="thx-panel thx-xfer-side">
          <div className="thx-panel-h">
            <h3>Transfers</h3>
            <span className="thx-tag">[ RECORDS · {String(transfers.length).padStart(2, "0")} ]</span>
          </div>
          {statusMessage && <div className="thx-status-line">{statusMessage}</div>}
          {transfers.length === 0 ? (
            <div className="thx-empty">NO CAPABILITY TRANSFERS</div>
          ) : (
            <div className="thx-runs thx-xfer-runs">
              {transfers.map((transfer) => (
                <button
                  type="button"
                  className={`thx-run thx-xfer-run ${transfer.transfer_id === selectedTransfer?.transfer_id ? "is-selected" : ""}`}
                  onClick={() => setSelectedTransferId(transfer.transfer_id)}
                  key={transfer.transfer_id}
                >
                  <span className={`thx-run-dot ${transfer.status === "ready" ? "ok" : transfer.status === "failed" ? "bad" : "warn"}`} />
                  <span className="thx-run-id">
                    {transfer.display_name}
                    <span className="thx-bench-result-sub">{transfer.source_display_name || transfer.source_model_slug} → {transfer.target_display_name || transfer.target_model_slug}</span>
                  </span>
                  <span className={`thx-cap ${transfer.status === "ready" ? "thx-cap--ok" : transfer.status === "failed" ? "thx-cap--no" : "thx-cap--w"}`}>{transfer.status}</span>
                  {transfer.transfer_id === activeTransferId && <CheckCircle size={15} />}
                </button>
              ))}
            </div>
          )}
          {selectedTransfer && (
            <div className="thx-xfer-detail">
              <div className="thx-summary">
                <div className="thx-summary-item"><span className="k">Vector</span><span className="v">{selectedTransfer.vector_artifact_id ? "ready" : "pending"}</span></div>
                <div className="thx-summary-item"><span className="k">Alignment</span><span className="v">{selectedTransfer.alignment_artifact_id ? "ready" : "pending"}</span></div>
                <div className="thx-summary-item"><span className="k">Runtime</span><span className="v">{selectedTransfer.target_runtime}</span></div>
              </div>
              <button type="button" className="thx-btn thx-btn--danger" onClick={deleteSelected}>Delete Transfer</button>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}

function WizardSection({ title, tag, children }: { title: string; tag: string; children: ReactNode }) {
  return (
    <div className="thx-section-stack">
      <div className="thx-panel-h">
        <h3>{title}</h3>
        <span className="thx-tag">[ {tag} ]</span>
      </div>
      {children}
    </div>
  );
}

function CandidateGrid({ candidates, selectedId, onSelect }: { candidates: Candidate[]; selectedId: string; onSelect: (id: string) => void }) {
  return (
    <div className="thx-cards thx-card-grid--compact">
      {candidates.map((candidate) => (
        <button
          type="button"
          className={`thx-card ${candidate.id === selectedId ? "is-selected" : ""} ${candidate.enabled ? "" : "is-disabled"}`}
          disabled={!candidate.enabled}
          onClick={() => onSelect(candidate.id)}
          key={candidate.id}
        >
          <span className="thx-card-row">
            <span className="thx-card-title">{candidate.displayName}</span>
            <span className="thx-card-status">{candidate.mode === "gguf" ? "GGUF" : "HF"}</span>
          </span>
          <span className="thx-card-sub">{candidate.detail}</span>
          <span className="thx-card-stats">
            <span className="thx-card-stat"><span className="k">Params</span><span className="v">{candidate.params ? formatParams(candidate.params) : "unknown"}</span></span>
            <span className="thx-card-stat"><span className="k">Runtime</span><span className="v">{candidate.runtime}</span></span>
          </span>
        </button>
      ))}
      {candidates.length === 0 && <div className="thx-empty">NO COMPATIBLE MODEL CANDIDATES</div>}
    </div>
  );
}

function buildCandidates(models: ModelRecord[], artifacts: ArtifactRecord[]): Candidate[] {
  const hf = models
    .filter((model) => model.supports_bf16_inference)
    .map((model) => ({
      id: `hf:${model.slug}`,
      runtime: "transformers" as CapabilityRuntime,
      modelSlug: model.slug,
      artifactId: "",
      displayName: model.display_name,
      detail: model.provider_id,
      params: parseParams(model.parameter_count),
      mode: "hf" as const,
      enabled: true
    }));
  const gguf = artifacts
    .filter((artifact) => artifact.artifact_type === "gguf_fp16" || artifact.artifact_type === "gguf_quantized")
    .map((artifact) => ({
      id: `gguf:${artifact.artifact_id}`,
      runtime: "llama_cpp" as CapabilityRuntime,
      modelSlug: String(artifact.metadata?.model_slug || artifact.artifact_id),
      artifactId: artifact.artifact_id,
      displayName: artifact.display_name,
      detail: artifact.path,
      params: 0,
      mode: "gguf" as const,
      enabled: true
    }));
  return [...hf, ...gguf];
}

function targetAllowed(source: Candidate | undefined, target: Candidate) {
  if (!source) {
    return true;
  }
  if (source.params > 0 && target.params > 0) {
    return target.params <= source.params;
  }
  return true;
}

function parseParams(raw: string) {
  const value = Number(raw);
  return Number.isFinite(value) ? value : 0;
}

function formatParams(value: number) {
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  return String(value);
}

function parseLayerTargets(value: string): string | number[] {
  if (["all", "every-4", "last"].includes(value)) {
    return value;
  }
  return value.split(",").map((item) => Number(item.trim())).filter((item) => Number.isFinite(item));
}

function parseLayerPairs(value: string): number[][] {
  if (!value.trim()) {
    return [];
  }
  return value.split(",").map((pair) => {
    const [source, target] = pair.split(":").map((item) => Number(item.trim()));
    return [source, target];
  }).filter((pair) => pair.every((item) => Number.isFinite(item)));
}
