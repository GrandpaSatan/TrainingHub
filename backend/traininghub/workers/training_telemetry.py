from __future__ import annotations

import math
import time
from typing import Any

try:
    from transformers import TrainerCallback as _TrainerCallbackBase
except ImportError:

    class _TrainerCallbackBase:
        def on_init_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            pass


class TrainingTelemetryCallback(_TrainerCallbackBase):
    def __init__(self, context: Any) -> None:
        self.context = context
        self.started_at = time.monotonic()
        self.latest_loss: float | None = None
        self.latest_metric: dict[str, Any] = {}

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        metric = normalize_trainer_log(logs or {}, getattr(state, "global_step", None), self.started_at)
        if not metric:
            return
        loss = metric.get("loss")
        if isinstance(loss, float):
            self.latest_loss = loss
        self.latest_metric = metric
        self.context.metric(metric)


def normalize_trainer_log(logs: dict[str, Any], global_step: Any, started_at: float | None = None) -> dict[str, Any]:
    metric: dict[str, Any] = {}
    step = _coerce_int(logs.get("step")) or _coerce_int(global_step)
    if step is not None:
        metric["step"] = step

    for key in [
        "loss",
        "train_loss",
        "eval_loss",
        "learning_rate",
        "grad_norm",
        "epoch",
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
    ]:
        value = _coerce_float(logs.get(key))
        if value is not None:
            metric[key] = value

    if started_at is not None:
        metric["runtime_seconds"] = round(time.monotonic() - started_at, 3)
    return metric


def normalize_training_summary(train_output: Any, telemetry: TrainingTelemetryCallback, started_at: float) -> dict[str, Any]:
    metrics = dict(getattr(train_output, "metrics", {}) or {})
    steps_ran = _coerce_int(getattr(train_output, "global_step", None)) or _coerce_int(metrics.get("global_step")) or 0
    final_train_loss = (
        _coerce_float(metrics.get("train_loss"))
        or _coerce_float(getattr(train_output, "training_loss", None))
        or _coerce_float(telemetry.latest_metric.get("train_loss"))
    )
    runtime_seconds = _coerce_float(metrics.get("train_runtime"))
    if runtime_seconds is None:
        runtime_seconds = round(time.monotonic() - started_at, 3)

    latest_loss = telemetry.latest_loss
    summary: dict[str, Any] = {
        "loss": latest_loss,
        "latest_loss": latest_loss,
        "steps_ran": steps_ran,
        "runtime_seconds": round(runtime_seconds, 3),
        "final_train_loss": final_train_loss,
        "dry_run": False,
    }
    for key in ["train_samples_per_second", "train_steps_per_second", "epoch"]:
        value = _coerce_float(metrics.get(key))
        if value is not None:
            summary[key] = value
    return summary


def synthetic_training_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "loss": None,
        "latest_loss": None,
        "steps_ran": int(payload.get("max_steps") or 0),
        "runtime_seconds": 0.0,
        "final_train_loss": None,
        "dry_run": True,
    }


def training_summary_message(summary: dict[str, Any]) -> str:
    return (
        "Training completed. "
        f"loss={_display_metric(summary.get('loss'))}, "
        f"steps={summary.get('steps_ran', 0)}, "
        f"runtime={_display_runtime(summary.get('runtime_seconds'))}, "
        f"final_train_loss={_display_metric(summary.get('final_train_loss'))}."
    )


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _coerce_int(value: Any) -> int | None:
    number = _coerce_float(value)
    if number is None:
        return None
    return int(number)


def _display_metric(value: Any) -> str:
    number = _coerce_float(value)
    if number is None:
        return "n/a"
    if abs(number) < 0.001 and number != 0:
        return f"{number:.3e}"
    return f"{number:.5g}"


def _display_runtime(value: Any) -> str:
    seconds = _coerce_float(value)
    if seconds is None:
        return "n/a"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remainder = int(round(seconds % 60))
    if minutes < 60:
        return f"{minutes}m {remainder}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h {minutes}m"
