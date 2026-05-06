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

import numpy as np
import pytest

from scpn_quantum_control.control.realtime_feedback import (
    RealtimeFeedbackConfig,
    RealtimeSyncFeedbackController,
)
from scpn_quantum_control.hardware.feedback_loop import (
    FeedbackCommand,
    FeedbackLoopConfig,
    FeedbackResult,
    FeedbackRunner,
    FeedbackStepRecord,
    ProportionalMetricObserver,
    RealtimeControllerScheduler,
)


@dataclass
class DummyScheduler:
    metrics: list[float]
    is_hardware: bool = False
    qpu_seconds: float = 0.0

    def __post_init__(self) -> None:
        self.submitted: list[FeedbackCommand] = []

    def submit(self, command: FeedbackCommand) -> FeedbackResult:
        self.submitted.append(command)
        metric = self.metrics[min(len(self.submitted) - 1, len(self.metrics) - 1)]
        return FeedbackResult(metrics={"r": metric}, qpu_seconds=self.qpu_seconds)


def test_feedback_runner_rejects_unapproved_hardware_scheduler():
    scheduler = DummyScheduler(metrics=[0.0], is_hardware=True)
    observer = ProportionalMetricObserver(initial_value=0.1, metric_name="r", target=0.5, gain=1.0)

    with pytest.raises(PermissionError, match="explicit approval"):
        FeedbackRunner(scheduler, observer, FeedbackLoopConfig(max_steps=1))


def test_feedback_runner_records_steps_until_observer_converges():
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


def test_feedback_runner_enforces_qpu_budget_before_submission():
    scheduler = DummyScheduler(metrics=[0.0])
    observer = ProportionalMetricObserver(initial_value=0.1, metric_name="r", target=0.5, gain=1.0)

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


def test_feedback_runner_enforces_qpu_budget_after_result():
    scheduler = DummyScheduler(metrics=[0.0], qpu_seconds=2.0)
    observer = ProportionalMetricObserver(initial_value=0.1, metric_name="r", target=0.5, gain=1.0)
    runner = FeedbackRunner(
        scheduler,
        observer,
        FeedbackLoopConfig(max_steps=1, max_qpu_seconds=1.0),
    )

    with pytest.raises(RuntimeError, match="exceeded max_qpu_seconds"):
        runner.run()


def test_proportional_observer_clips_and_requires_metric():
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


def test_realtime_controller_scheduler_runs_deterministic_simulator_steps():
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


def test_realtime_controller_scheduler_rejects_invalid_payloads():
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
