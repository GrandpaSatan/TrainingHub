from __future__ import annotations

from types import SimpleNamespace

from traininghub.workers.training_telemetry import (
    TrainingTelemetryCallback,
    normalize_training_summary,
    normalize_trainer_log,
    synthetic_training_summary,
    training_summary_message,
)


class FakeContext:
    def __init__(self) -> None:
        self.metrics: list[dict[str, object]] = []

    def metric(self, values: dict[str, object]) -> None:
        self.metrics.append(values)


def test_normalize_trainer_log_keeps_numeric_loss_and_step() -> None:
    metric = normalize_trainer_log(
        {"loss": "0.1234", "grad_norm": "1.5", "learning_rate": "2e-5", "ignored": "x"},
        global_step=7,
        started_at=None,
    )

    assert metric == {"step": 7, "loss": 0.1234, "learning_rate": 2e-05, "grad_norm": 1.5}


def test_training_telemetry_callback_emits_metric_events() -> None:
    context = FakeContext()
    callback = TrainingTelemetryCallback(context)

    callback.on_log(None, SimpleNamespace(global_step=3), None, {"loss": 0.42, "epoch": 0.1})

    assert len(context.metrics) == 1
    assert context.metrics[0]["step"] == 3
    assert context.metrics[0]["loss"] == 0.42
    assert context.metrics[0]["epoch"] == 0.1
    assert callback.latest_loss == 0.42


def test_training_telemetry_callback_accepts_trainer_init_hook() -> None:
    context = FakeContext()
    callback = TrainingTelemetryCallback(context)

    callback.on_init_end(None, SimpleNamespace(global_step=0), None)

    assert context.metrics == []


def test_normalize_training_summary_includes_completion_fields() -> None:
    context = FakeContext()
    callback = TrainingTelemetryCallback(context)
    callback.latest_loss = 0.02
    output = SimpleNamespace(
        global_step=1100,
        training_loss=0.02625,
        metrics={"train_runtime": 343.1, "train_steps_per_second": 3.206, "epoch": 1.1},
    )

    summary = normalize_training_summary(output, callback, callback.started_at)

    assert summary["loss"] == 0.02
    assert summary["latest_loss"] == 0.02
    assert summary["steps_ran"] == 1100
    assert summary["runtime_seconds"] == 343.1
    assert summary["final_train_loss"] == 0.02625
    assert summary["dry_run"] is False


def test_synthetic_training_summary_is_marked_dry_run_without_fake_loss() -> None:
    summary = synthetic_training_summary({"max_steps": 5})

    assert summary["dry_run"] is True
    assert summary["loss"] is None
    assert summary["final_train_loss"] is None
    assert summary["steps_ran"] == 5
    assert "loss=n/a" in training_summary_message(summary)
