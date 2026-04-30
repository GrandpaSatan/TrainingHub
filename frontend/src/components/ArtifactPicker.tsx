import { ArtifactRecord } from "../api/client";

type ArtifactPickerValueMode = "artifact_id" | "path";

type ArtifactPickerDefaultOption = {
  label: string;
  value: string;
  detail: string;
};

type ArtifactPickerProps = {
  artifacts: ArtifactRecord[];
  selectedValue: string;
  valueMode?: ArtifactPickerValueMode;
  onSelect: (artifact: ArtifactRecord | null) => void;
  emptyMessage: string;
  defaultOption?: ArtifactPickerDefaultOption;
  className?: string;
};

export function ArtifactPicker({
  artifacts,
  selectedValue,
  valueMode = "artifact_id",
  onSelect,
  emptyMessage,
  defaultOption,
  className = "",
}: ArtifactPickerProps) {
  return (
    <div className={`thx-cards thx-artifact-picker ${className}`}>
      {defaultOption && (
        <button
          type="button"
          className={`thx-card thx-artifact-picker-default ${selectedValue === defaultOption.value ? "is-selected" : ""}`}
          onClick={() => onSelect(null)}
        >
          <span className="thx-card-row">
            <span className="thx-card-title">{defaultOption.label}</span>
            <span className="thx-card-status">default</span>
          </span>
          <span className="thx-card-sub">{defaultOption.detail}</span>
        </button>
      )}
      {artifacts.map((artifact) => {
        const value = artifactValue(artifact, valueMode);
        return (
          <button
            type="button"
            className={`thx-card ${selectedValue === value ? "is-selected" : ""}`}
            onClick={() => onSelect(artifact)}
            key={artifact.artifact_id}
          >
            <span className="thx-card-row">
              <span className="thx-card-title">{artifact.display_name}</span>
              <span className="thx-card-status">{artifact.artifact_type}</span>
            </span>
            <span className="thx-card-sub">{artifact.path}</span>
            <span className="thx-card-stats">
              <span className="thx-card-stat">
                <span className="k">Size</span>
                <span className="v">{formatArtifactBytes(artifact.size_bytes)}</span>
              </span>
              <span className="thx-card-stat">
                <span className="k">Created</span>
                <span className="v">{formatArtifactTime(artifact.created_at)}</span>
              </span>
            </span>
          </button>
        );
      })}
      {artifacts.length === 0 && !defaultOption && <div className="thx-empty">{emptyMessage}</div>}
      {artifacts.length === 0 && defaultOption && <div className="thx-empty thx-artifact-picker-empty">{emptyMessage}</div>}
    </div>
  );
}

function artifactValue(artifact: ArtifactRecord, mode: ArtifactPickerValueMode) {
  return mode === "path" ? artifact.path : artifact.artifact_id;
}

function formatArtifactBytes(value: number) {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let next = value;
  let index = 0;
  while (next >= 1024 && index < units.length - 1) {
    next /= 1024;
    index += 1;
  }
  return `${next.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function formatArtifactTime(value?: number) {
  if (!value) {
    return "unknown";
  }
  return new Date(value * 1000).toLocaleString();
}
