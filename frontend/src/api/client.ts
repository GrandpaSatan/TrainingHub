export type ApiError = {
  detail?: string;
};

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    credentials: "include",
    headers: init.body instanceof FormData ? init.headers : { "Content-Type": "application/json", ...init.headers },
    ...init
  });
  if (!response.ok) {
    let message = response.statusText;
    try {
      const body = (await response.json()) as ApiError;
      message = body.detail || message;
    } catch {
      // Keep HTTP status text.
    }
    throw new Error(message);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body: unknown) => request<T>(path, { method: "POST", body: JSON.stringify(body) }),
  delete: <T>(path: string) => request<T>(path, { method: "DELETE" }),
  postForm: <T>(path: string, body: FormData) => request<T>(path, { method: "POST", body }),
  capabilityTransfers: {
    list: () => request<CapabilityTransferRecord[]>("/api/capability-transfers"),
    create: (body: CapabilityTransferCreateRequest) =>
      request<CapabilityTransferRecord>("/api/capability-transfers", { method: "POST", body: JSON.stringify(body) }),
    align: (id: string, body: CapabilityTransferAlignRequest) =>
      request<CapabilityTransferRecord>(`/api/capability-transfers/${id}/align`, { method: "POST", body: JSON.stringify(body) }),
    activate: (id: string, body: CapabilityTransferActivateRequest) =>
      request<CapabilityTransferActivationResponse>(`/api/capability-transfers/${id}/activate`, { method: "POST", body: JSON.stringify(body) }),
    deactivate: (id: string) =>
      request<CapabilityTransferActivationResponse>(`/api/capability-transfers/${id}/deactivate`, { method: "POST", body: JSON.stringify({}) }),
    delete: (id: string) => request<{ deleted: boolean; transfer_id: string; removed_paths: string[]; removed_artifacts: string[] }>(
      `/api/capability-transfers/${id}`,
      { method: "DELETE" }
    )
  },
  template: async (datasetType: string) => {
    const response = await fetch("/api/datasets/template", {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_type: datasetType })
    });
    if (!response.ok) {
      throw new Error(response.statusText);
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${datasetType}_template.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
  }
};

export type ModelRecord = {
  slug: string;
  provider_id: string;
  display_name: string;
  family: string;
  parameter_count: string;
  supports_lora: boolean;
  supports_qlora: boolean;
  supports_full_finetune: boolean;
  supports_bf16_inference: boolean;
  supports_benchmark: boolean;
  supports_quantization: boolean;
  supports_gguf_path: boolean;
  hardware_note: string;
  default_dtype: string;
  max_sequence_length: number;
  metadata: Record<string, unknown>;
  deletable: boolean;
  seeded: boolean;
  local_path: string;
  local_size_bytes: number;
};

export type DatasetRecord = {
  dataset_id: string;
  slug: string;
  dataset_type: string;
  title: string;
  version_id: string;
  approved: boolean;
  reviewed_at?: number;
  review_sample_size?: number;
  row_count: number;
  split_counts: Record<string, number>;
  validation: ValidationResult;
  jsonl_path: string;
};

export type DatasetRecordView = {
  index: number;
  system: string;
  prompt: string;
  response: string;
  metadata: Record<string, unknown>;
};

export type DatasetRecordsResponse = {
  dataset_id: string;
  version_id: string;
  approved: boolean;
  row_count: number;
  total_matching: number;
  offset: number;
  limit: number;
  sample_size?: number;
  required_review_sample_size?: number;
  records: DatasetRecordView[];
};

export type HubResolvedResource = {
  found: boolean;
  resource_type: "model" | "dataset";
  repo_id: string;
  sha: string;
  last_modified?: string;
  private: boolean;
  gated: false | "auto" | "manual";
  downloads?: number;
  likes?: number;
  tags: string[];
  siblings: string[];
  summary: Record<string, unknown>;
};

export type ValidationIssue = {
  row_number: number;
  field: string;
  code: string;
  message: string;
};

export type ValidationResult = {
  valid: boolean;
  errors: ValidationIssue[];
  warnings: ValidationIssue[];
  accepted_count: number;
  rows?: Record<string, string>[];
};

export type JobRecord = {
  job_id: string;
  job_type: string;
  status: string;
  slug: string;
  work_dir: string;
  worker_module: string;
  worker_pid?: number;
  gpu_ids: number[];
  created_at: number;
  started_at?: number;
  finished_at?: number;
  terminal_message: string;
  payload: Record<string, unknown>;
};

export type ArtifactRecord = {
  artifact_id: string;
  job_id?: string;
  artifact_type: string;
  display_name: string;
  path: string;
  size_bytes: number;
  checksum_sha256: string;
  metadata: Record<string, unknown>;
  created_at: number;
};

export type BenchmarkCatalogItem = {
  id: string;
  family: string;
  label: string;
  description: string;
  smoke_default: number;
  full_default: number;
};

export type BenchmarkResultRecord = {
  result_id: string;
  job_id: string;
  model_slug: string;
  benchmark_name: string;
  metrics: Record<string, unknown>;
  result_path: string;
  artifact_id?: string;
  created_at: number;
};

export type InferenceTarget = {
  target_type: "base_model" | "gguf_artifact";
  model_slug: string;
  artifact_id: string;
  display_name: string;
  provider_id: string;
  path: string;
  updated_at: number;
  capability_transfer_id?: string;
};

export type InferenceOption = {
  target_type: "base_model" | "gguf_artifact";
  model_slug: string;
  artifact_id: string;
  display_name: string;
  description: string;
  enabled: boolean;
  disabled_reason: string;
  provider_id: string;
  path: string;
};

export type CapabilityRuntime = "transformers" | "llama_cpp";
export type CapabilityTransferStatus = "extracting" | "extracted" | "aligning" | "ready" | "failed" | "deleted";

export type CapabilityTransferRecord = {
  transfer_id: string;
  display_name: string;
  source_model_slug: string;
  source_runtime: CapabilityRuntime;
  target_model_slug: string;
  target_runtime: CapabilityRuntime;
  vector_artifact_id?: string;
  alignment_artifact_id?: string;
  alpha: number;
  layer_targets: string | number[];
  status: CapabilityTransferStatus;
  config: Record<string, unknown>;
  created_at: number;
  updated_at: number;
  deleted_at?: number;
  extract_job_id?: string;
  align_job_id?: string;
  source_artifact_id?: string;
  target_artifact_id?: string;
  source_display_name?: string;
  target_display_name?: string;
  degraded_mode?: boolean;
};

export type CapabilityTransferCreateRequest = {
  display_name: string;
  source_model_slug: string;
  source_runtime: CapabilityRuntime;
  source_artifact_id?: string;
  target_model_slug: string;
  target_runtime: CapabilityRuntime;
  target_artifact_id?: string;
  calibration_dataset_id: string;
  layer_targets: string | number[];
  contrast_mode: "prompt_pair" | "system_pair";
  rank: number;
  dry_run?: boolean;
};

export type CapabilityTransferAlignRequest = {
  rank: number;
  layer_pairs: number[][];
};

export type CapabilityTransferActivateRequest = {
  alpha: number;
  layer_targets: string | number[];
};

export type CapabilityTransferActivationResponse = {
  transfer?: CapabilityTransferRecord;
  active_target?: InferenceTarget;
  warning?: string;
};
