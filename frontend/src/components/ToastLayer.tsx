import { AlertTriangle, CheckCircle, Info, X } from "lucide-react";

export type ToastTone = "success" | "error" | "info";

export type ToastMessage = {
  id: number;
  tone: ToastTone;
  title: string;
  message: string;
};

type ToastLayerProps = {
  toasts: ToastMessage[];
  onDismiss: (id: number) => void;
};

export function ToastLayer({ toasts, onDismiss }: ToastLayerProps) {
  if (toasts.length === 0) {
    return null;
  }

  return (
    <div className="thx-toast-layer" aria-live="polite" aria-label="Notifications">
      {toasts.map((toast) => (
        <article className={`thx-toast thx-toast--${toast.tone}`} key={toast.id}>
          <span className="thx-toast-icon">{toastIcon(toast.tone)}</span>
          <span className="thx-toast-copy">
            <strong>{toast.title}</strong>
            <span>{toast.message}</span>
          </span>
          <button type="button" className="thx-toast-close" onClick={() => onDismiss(toast.id)} aria-label="Dismiss notification">
            <X size={14} />
          </button>
        </article>
      ))}
    </div>
  );
}

function toastIcon(tone: ToastTone) {
  if (tone === "success") {
    return <CheckCircle size={17} />;
  }
  if (tone === "error") {
    return <AlertTriangle size={17} />;
  }
  return <Info size={17} />;
}
