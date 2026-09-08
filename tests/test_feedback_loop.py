# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for hybrid feedback loop
"""Tests for hardware.feedback_loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pytest

from scpn_quantum_control.control.realtime_feedback import (
    RealtimeFeedbackConfig,
    RealtimeSyncFeedbackController,
)
from scpn_quantum_control.hardware.feedback_loop import (
    FeedbackCommand,
    FeedbackLoopConfig,
    FeedbackLoopLatencySLA,
    FeedbackResult,
    FeedbackRunner,
    FeedbackStepRecord,
    ProportionalMetricObserver,
    RealtimeControllerScheduler,
)


@dataclass
class DummyScheduler:
    """Deterministic local scheduler used by feedback-loop contract tests."""

    metrics: list[float]
    is_hardware: bool = False
    qpu_seconds: float = 0.0

    def __post_init__(self) -> None:
        """Initialize command-custody state."""
        self.submitted: list[FeedbackCommand] = []

    def submit(self, command: FeedbackCommand) -> FeedbackResult:
        """Record a command and return the next deterministic metric."""
        self.submitted.append(command)
        metric = self.metrics[min(len(self.submitted) - 1, len(self.metrics) - 1)]
        return FeedbackResult(metrics={"r": metric}, qpu_seconds=self.qpu_seconds)


def test_feedback_runner_rejects_unapproved_hardware_scheduler() -> None:
    """Refuse hardware-marked schedulers without explicit approval."""
    scheduler = DummyScheduler(metrics=[0.0], is_hardware=True)
    observer = ProportionalMetricObserver(initial_value=0.1, metric_name="r", target=0.5, gain=1.0)

    with pytest.raises(PermissionError, match="explicit approval"):
        FeedbackRunner(scheduler, observer, FeedbackLoopConfig(max_steps=1))


@pytest.mark.parametrize("encoded", ['"false"', '"true"', "1", "0", "null"])
def test_runner_rejects_nonboolean_approval(encoded: str) -> None:
    """Decoded truthy values are not explicit hardware approval."""
    import json

    scheduler = DummyScheduler(metrics=[0.0], is_hardware=True)
    observer = ProportionalMetricObserver(initial_value=0.1, metric_name="r", target=0.5, gain=1.0)
    with pytest.raises(ValueError, match="hardware_approved must be a boolean"):
        FeedbackRunner(
            scheduler,
            observer,
            FeedbackLoopConfig(max_steps=1),
            hardware_approved=json.loads(encoded),
        )
    assert scheduler.submitted == []


@pytest.mark.parametrize("encoded", ['"false"', "0", "null"])
def test_config_rejects_nonboolean_approval_policy(encoded: str) -> None:
    """A decoded falsy value cannot disable the approval gate implicitly."""
    import json

    with pytest.raises(ValueError, match="require_hardware_approval must be a boolean"):
        FeedbackLoopConfig(max_steps=1, require_hardware_approval=json.loads(encoded))


def test_runner_rechecks_hardware_identity_before_dispatch() -> None:
    """A scheduler switched from simulation to hardware requires fresh approval."""
    scheduler = DummyScheduler(metrics=[0.0])
    observer = ProportionalMetricObserver(initial_value=0.1, metric_name="r", target=0.5, gain=1.0)
    runner = FeedbackRunner(scheduler, observer, FeedbackLoopConfig(max_steps=1))
    scheduler.is_hardware = True
    with pytest.raises(PermissionError, match="explicit approval"):
        runner.run()
    assert scheduler.submitted == []


@pytest.mark.parametrize("encoded", ['"false"', "0", "null"])
def test_runner_rejects_ambiguous_scheduler_identity(encoded: str) -> None:
    """Missing or malformed hardware identity is not evidence of simulation."""
    import json

    scheduler = DummyScheduler(metrics=[0.0])
    scheduler.is_hardware = json.loads(encoded)
    observer = ProportionalMetricObserver(initial_value=0.1, metric_name="r", target=0.5, gain=1.0)
    with pytest.raises(ValueError, match="is_hardware must be a boolean"):
        FeedbackRunner(scheduler, observer, FeedbackLoopConfig(max_steps=1))
    assert scheduler.submitted == []


@pytest.mark.parametrize("change_at", ["initial", "update"])
def test_observer_cannot_revoke_approval_then_dispatch(change_at: str) -> None:
    """A callback's approval revocation takes effect before the next command."""
    from collections.abc import Mapping, Sequence

    scheduler = DummyScheduler(metrics=[0.0], is_hardware=True)

    class RevokingObserver(ProportionalMetricObserver):
        def initial_command(self) -> FeedbackCommand:
            if change_at == "initial":
                runner.hardware_approved = False
            return super().initial_command()

        def update(
            self, result: FeedbackResult, history: Sequence[FeedbackStepRecord]
        ) -> tuple[FeedbackCommand | None, Mapping[str, object]]:
            runner.hardware_approved = False
            return super().update(result, history)

    runner = FeedbackRunner(
        scheduler,
        RevokingObserver(initial_value=0.1, metric_name="r", target=0.5, gain=1.0),
        FeedbackLoopConfig(max_steps=2),
        hardware_approved=True,
    )
    with pytest.raises(PermissionError, match="explicit approval"):
        runner.run()
    assert len(scheduler.submitted) == (0 if change_at == "initial" else 1)


def test_explicit_approval_policy_opt_out_remains_supported() -> None:
    """An exact false policy preserves the existing caller-controlled opt-out."""
    scheduler = DummyScheduler(metrics=[0.5], is_hardware=True)
    observer = ProportionalMetricObserver(initial_value=0.1, metric_name="r", target=0.5, gain=1.0)
    runner = FeedbackRunner(
        scheduler,
        observer,
        FeedbackLoopConfig(max_steps=1, require_hardware_approval=False),
    )
    assert len(runner.run()) == 1
    assert len(scheduler.submitted) == 1


def test_feedback_runner_records_steps_until_observer_converges() -> None:
    """Record each local step until the observer requests termination."""
    scheduler = DummyScheduler(metrics=[0.2, 0.5])
    observer = ProportionalMetricObserver(
        initial_value=0.1,
        metric_name="r",
        target=0.5,
        gain=0.5,
        max_value=1.0,
        tolerance=0.01,
    )
    runner = FeedbackRunner(
        scheduler,
        observer,
        FeedbackLoopConfig(max_steps=5, max_qpu_seconds=0.0),
    )

    history = runner.run()

    assert len(history) == 2
    assert isinstance(history[0], FeedbackStepRecord)
    assert history[0].command.payload == {"value": 0.1}
    assert history[0].observer_state["value_out"] == pytest.approx(0.25)
    assert history[-1].stop_requested is True
    assert len(scheduler.submitted) == 2


def test_feedback_runner_enforces_qpu_budget_before_submission() -> None:
    """Refuse estimated QPU spend before scheduler submission."""
    scheduler = DummyScheduler(metrics=[0.0])

    class CostlyObserver(ProportionalMetricObserver):
        def initial_command(self) -> FeedbackCommand:
            return FeedbackCommand(payload={"value": 0.1}, estimated_qpu_seconds=2.0)

    observer = CostlyObserver(initial_value=0.1, metric_name="r", target=0.5, gain=1.0)
    runner = FeedbackRunner(
        scheduler,
        observer,
        FeedbackLoopConfig(max_steps=1, max_qpu_seconds=1.0),
    )

    with pytest.raises(RuntimeError, match="configured QPU budget"):
        runner.run()
    assert scheduler.submitted == []


def test_feedback_runner_allows_hardware_only_when_approval_flag_is_explicit() -> None:
    """Admit a hardware-marked double only with explicit approval."""
    scheduler = DummyScheduler(metrics=[0.5], is_hardware=True)
    observer = ProportionalMetricObserver(
        initial_value=0.1,
        metric_name="r",
        target=0.5,
        gain=1.0,
        tolerance=0.0,
    )

    runner = FeedbackRunner(
        scheduler,
        observer,
        FeedbackLoopConfig(max_steps=1, max_qpu_seconds=1.0),
        hardware_approved=True,
    )

    history = runner.run()

    assert len(history) == 1
    assert history[0].stop_requested is True
    assert scheduler.submitted[0].label == "proportional"


def test_feedback_runner_enforces_qpu_budget_after_result() -> None:
    """Refuse observed QPU spend that exceeds the configured budget."""
    scheduler = DummyScheduler(metrics=[0.0], qpu_seconds=2.0)
    observer = ProportionalMetricObserver(initial_value=0.1, metric_name="r", target=0.5, gain=1.0)
    runner = FeedbackRunner(
        scheduler,
        observer,
        FeedbackLoopConfig(max_steps=1, max_qpu_seconds=1.0),
    )

    with pytest.raises(RuntimeError, match="exceeded max_qpu_seconds"):
        runner.run()


def test_proportional_observer_clips_and_requires_metric() -> None:
    """Clip proportional updates and require the configured metric."""
    observer = ProportionalMetricObserver(
        initial_value=0.9,
        metric_name="r",
        target=1.0,
        gain=10.0,
        max_value=1.0,
    )

    command, state = observer.update(FeedbackResult(metrics={"r": 0.5}), ())

    assert command is not None
    assert command.payload == {"value": 1.0}
    assert state["converged"] is False
    with pytest.raises(KeyError, match="missing feedback metric"):
        observer.update(FeedbackResult(metrics={"other": 0.5}), ())


@pytest.mark.parametrize("encoded", ["1.5", "true", "NaN", "Infinity"])
def test_feedback_result_rejects_invalid_counts(encoded: str) -> None:
    """Shot counts cannot be fractional, boolean or non-finite values."""
    import json

    with pytest.raises(ValueError, match="count"):
        FeedbackResult(counts={"0": json.loads(encoded)})


@pytest.mark.parametrize("encoded", ["NaN", "Infinity", "-Infinity", "true"])
def test_feedback_result_rejects_invalid_control_metric(encoded: str) -> None:
    """Invalid metrics must fail before they can generate a control command."""
    import json

    with pytest.raises(ValueError, match="metric"):
        FeedbackResult(metrics={"r": json.loads(encoded)})


def test_result_snapshots_caller_owned_counts_and_metrics() -> None:
    """Later edits to provider dictionaries cannot rewrite admitted results."""
    counts = {"0": 4}
    metrics = {"r": 0.5}
    result = FeedbackResult(counts=counts, metrics=metrics)
    counts["0"] = 9
    metrics["r"] = float("nan")
    assert dict(result.counts) == {"0": 4}
    assert dict(result.metrics) == {"r": 0.5}


def test_observer_rejects_corrupted_metric_without_changing_control_state() -> None:
    """A mutated result must not poison the observer's next control command."""
    result = FeedbackResult(metrics={"r": 0.5})
    assert isinstance(result.metrics, dict)
    result.metrics["r"] = float("nan")
    observer = ProportionalMetricObserver(initial_value=0.1, metric_name="r", target=0.5, gain=1.0)
    with pytest.raises(ValueError, match="metric"):
        observer.update(result, ())
    assert observer.initial_command().payload == {"value": 0.1}


def test_realtime_controller_scheduler_runs_deterministic_simulator_steps() -> None:
    """Run seeded simulator steps with complete provenance metadata."""
    controller = RealtimeSyncFeedbackController(
        K_coupling=np.array([[0.0, 0.2], [0.2, 0.0]], dtype=np.float64),
        omega_natural=np.array([0.1, 0.3], dtype=np.float64),
        config=RealtimeFeedbackConfig(measurement_shots=32, base_dt=0.02, trotter_steps=1),
    )
    scheduler = RealtimeControllerScheduler(controller, base_seed=10)

    result = scheduler.submit(FeedbackCommand(payload={"coupling_scale": 1.2}, label="sim"))

    assert scheduler.is_hardware is False
    assert scheduler.submitted == 1
    assert result.qpu_seconds == 0.0
    assert result.metadata["source"] == "realtime_sync_feedback_controller"
    assert result.metadata["seed"] == 10
    assert result.metadata["action"] in {"release", "hold", "synchronise"}
    assert result.metrics["applied_coupling_scale"] == pytest.approx(1.2)
    assert result.metrics["next_coupling_scale"] == pytest.approx(controller.coupling_scale)
    assert sum(result.counts.values()) == 32


def test_realtime_controller_scheduler_rejects_invalid_payloads() -> None:
    """Reject malformed simulator payload, seed, and coupling values."""
    controller = RealtimeSyncFeedbackController(
        K_coupling=np.array([[0.0, 0.2], [0.2, 0.0]], dtype=np.float64),
        omega_natural=np.array([0.1, 0.3], dtype=np.float64),
    )
    scheduler = RealtimeControllerScheduler(controller)

    with pytest.raises(TypeError, match="payload must be a mapping"):
        scheduler.submit(FeedbackCommand(payload=0.5))
    with pytest.raises(ValueError, match="seed must be"):
        scheduler.submit(FeedbackCommand(payload={"seed": -1}))
    with pytest.raises(ValueError, match="scale must be"):
        scheduler.submit(FeedbackCommand(payload={"coupling_scale": 3.0}))


def test_realtime_controller_scheduler_payload_seed_overrides_base_seed_provenance() -> None:
    """Prefer an explicit payload seed over the scheduler seed stream."""
    controller = RealtimeSyncFeedbackController(
        K_coupling=np.array([[0.0, 0.2], [0.2, 0.0]], dtype=np.float64),
        omega_natural=np.array([0.1, 0.3], dtype=np.float64),
        config=RealtimeFeedbackConfig(measurement_shots=16, base_dt=0.02, trotter_steps=1),
    )
    scheduler = RealtimeControllerScheduler(controller, base_seed=10)

    result = scheduler.submit(FeedbackCommand(payload={"seed": 42}, label="override"))

    assert result.metadata["seed"] == 42
    assert result.metadata["command_label"] == "override"
    assert result.metadata["step_index"] == 0
    assert sum(result.counts.values()) == 16


def test_feedback_loop_value_objects_reject_invalid_runtime_boundaries() -> None:
    """Reject invalid loop, command, count, and metric values."""
    with pytest.raises(ValueError, match="max_steps"):
        FeedbackLoopConfig(max_steps=0)
    with pytest.raises(ValueError, match="estimated_qpu_seconds"):
        FeedbackCommand(payload={}, estimated_qpu_seconds=-0.1)
    with pytest.raises(ValueError, match="count"):
        FeedbackResult(counts={"0": -1})
    with pytest.raises(ValueError, match="metric"):
        FeedbackResult(metrics=cast(dict[str, float], {"r": "not numeric"}))


def test_realtime_controller_scheduler_accepts_empty_payload_without_seed_provenance() -> None:
    """Run an unseeded simulator step from an empty payload."""
    controller = RealtimeSyncFeedbackController(
        K_coupling=np.array([[0.0, 0.2], [0.2, 0.0]], dtype=np.float64),
        omega_natural=np.array([0.1, 0.3], dtype=np.float64),
        config=RealtimeFeedbackConfig(measurement_shots=8, base_dt=0.02, trotter_steps=1),
    )
    scheduler = RealtimeControllerScheduler(controller)

    result = scheduler.submit(FeedbackCommand(payload=None, label="empty"))

    assert result.metadata["seed"] is None
    assert result.metadata["command_label"] == "empty"
    assert sum(result.counts.values()) == 8


def test_proportional_observer_rejects_invalid_bounds() -> None:
    """Reject an observer whose maximum is below its minimum."""
    with pytest.raises(ValueError, match="max_value"):
        ProportionalMetricObserver(
            initial_value=0.5,
            metric_name="r",
            target=0.5,
            gain=1.0,
            min_value=1.0,
            max_value=0.0,
        )


def test_feedback_runner_latency_sla_accepts_sub_millisecond_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept a deterministic local profile within every SLA bound."""
    scheduler = DummyScheduler(metrics=[0.1, 0.2, 0.3])
    observer = ProportionalMetricObserver(
        initial_value=0.1,
        metric_name="r",
        target=0.3,
        gain=0.5,
        tolerance=0.01,
    )
    config = FeedbackLoopConfig(
        max_steps=3,
        max_qpu_seconds=0.0,
        max_step_latency_s=0.01,
        latency_sla=FeedbackLoopLatencySLA(
            max_latency_s=0.001,
            p95_latency_s=0.001,
            p99_latency_s=0.001,
        ),
    )
    runner = FeedbackRunner(scheduler, observer, config)

    timeline = iter(
        [
            0.000000,
            0.000450,
            0.001000,
            0.001700,
            0.002500,
            0.003250,
        ]
    )
    monkeypatch.setattr(
        "scpn_quantum_control.hardware.feedback_loop.time.monotonic",
        lambda: next(timeline),
    )

    history = runner.run()

    assert len(history) == 3
    assert all(record.latency_s <= 0.001 for record in history)


def test_feedback_runner_latency_sla_rejects_p99_breach(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject a deterministic local profile that breaches its p99 bound."""
    scheduler = DummyScheduler(metrics=[0.1, 0.2, 0.3, 0.4])
    observer = ProportionalMetricObserver(
        initial_value=0.1,
        metric_name="r",
        target=0.9,
        gain=0.2,
        tolerance=0.0,
    )
    config = FeedbackLoopConfig(
        max_steps=4,
        max_qpu_seconds=0.0,
        max_step_latency_s=0.01,
        latency_sla=FeedbackLoopLatencySLA(
            max_latency_s=0.005,
            p95_latency_s=0.005,
            p99_latency_s=0.001,
        ),
    )
    runner = FeedbackRunner(scheduler, observer, config)

    timeline = iter(
        [
            0.000000,
            0.000200,
            0.001000,
            0.001250,
            0.002000,
            0.002300,
            0.003000,
            0.004500,
        ]
    )
    monkeypatch.setattr(
        "scpn_quantum_control.hardware.feedback_loop.time.monotonic",
        lambda: next(timeline),
    )

    with pytest.raises(RuntimeError, match="p99 latency"):
        runner.run()
