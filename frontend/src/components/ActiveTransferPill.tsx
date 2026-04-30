import { Wand2, X } from "lucide-react";
import { api, CapabilityTransferRecord, InferenceTarget } from "../api/client";

type Props = {
  activeInferenceTarget: InferenceTarget | null;
  transfers: CapabilityTransferRecord[];
  refresh: () => void;
  onToast?: (message: string, tone?: "info" | "success" | "error", title?: string) => void;
};

export function ActiveTransferPill({ activeInferenceTarget, transfers, refresh, onToast }: Props) {
  const transferId = activeInferenceTarget?.capability_transfer_id || "";
  const transfer = transfers.find((item) => item.transfer_id === transferId);
  if (!transferId || !transfer) {
    return (
      <div className="thx-active-transfer is-empty">
        <Wand2 size={14} />
        <span>No transfer active</span>
      </div>
    );
  }

  async function deactivate() {
    try {
      await api.capabilityTransfers.deactivate(transferId);
      onToast?.("Capability transfer deactivated.", "success", "Transfer off");
      await refresh();
    } catch (err) {
      onToast?.(err instanceof Error ? err.message : "Unable to deactivate capability transfer.", "error", "Transfer update failed");
    }
  }

  return (
    <div className={`thx-active-transfer ${transfer.degraded_mode ? "is-degraded" : "is-active"}`}>
      <Wand2 size={14} />
      <span className="thx-active-transfer-main">
        {transfer.source_display_name || transfer.source_model_slug} → {transfer.target_display_name || transfer.target_model_slug}
      </span>
      <span className="thx-active-transfer-meta">α={Number(transfer.alpha || 0).toFixed(2)} · L:{formatLayers(transfer.layer_targets)}</span>
      {transfer.degraded_mode && <span className="thx-cap thx-cap--w">LAST-LAYER</span>}
      <button type="button" className="thx-active-transfer-off" onClick={deactivate} aria-label="Deactivate capability transfer">
        <X size={13} />
      </button>
    </div>
  );
}

function formatLayers(layerTargets: string | number[]) {
  if (Array.isArray(layerTargets)) {
    return layerTargets.length > 3 ? `${layerTargets.length} layers` : layerTargets.join(",");
  }
  return layerTargets;
}
