# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the cross-shot feedback loop
"""Guard and branch tests for the cross-shot feedback-loop orchestration.

Covers the config and scheduler type guards, the step/total latency breaches,
the finiteness helpers, the latency SLA enforcement and the linear-quantile
edge cases.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from scpn_quantum_control.control.realtime_feedback import RealtimeSyncFeedbackController
from scpn_quantum_control.hardware.feedback_loop import (
    FeedbackCommand,
    FeedbackLoopConfig,
    FeedbackLoopLatencySLA,
    FeedbackResult,
    FeedbackRunner,
    FeedbackStepRecord,
    ProportionalMetricObserver,
    RealtimeControllerScheduler,
    _enforce_latency_sla,
    _finite_float,
    _linear_quantile,
    _require_finite,
)


class _SleepScheduler:
    """Scheduler whose submit takes a small, measurable wall-clock latency."""

    is_hardware = False

    def __init__(self, sleep_s: float = 0.002) -> None:
        self._sleep_s = sleep_s

    def submit(self, command: FeedbackCommand) -> FeedbackResult:
        time.sleep(self._sleep_s)
        return FeedbackResult(metrics={"r": 0.0}, qpu_seconds=0.0)


def _observer() -> ProportionalMetricObserver:
    return ProportionalMetricObserver(initial_value=0.1, metric_name="r", target=0.5, gain=1.0)


def test_config_rejects_non_sla_latency_object() -> None:
    """A latency_sla that is not a FeedbackLoopLatencySLA is rejected."""
    with pytest.raises(TypeError, match="latency_sla must be a FeedbackLoopLatencySLA"):
        FeedbackLoopConfig(max_steps=1, latency_sla="fast")  # type: ignore[arg-type]


def test_runner_raises_on_step_latency_breach() -> None:
    """A submit exceeding the per-step latency budget aborts the loop."""
    runner = FeedbackRunner(
        _SleepScheduler(),
        _observer(),
        FeedbackLoopConfig(max_steps=2, max_step_latency_s=0.0, max_qpu_seconds=0.0),
    )
    with pytest.raises(RuntimeError, match="exceeded max_step_latency_s"):
        runner.run()


def test_runner_raises_on_total_latency_breach() -> None:
    """A cumulative latency exceeding the total budget aborts the loop."""
    runner = FeedbackRunner(
        _SleepScheduler(),
        _observer(),
        FeedbackLoopConfig(
            max_steps=2,
            max_step_latency_s=10.0,
            max_total_latency_s=0.0,
            max_qpu_seconds=0.0,
        ),
    )
    with pytest.raises(RuntimeError, match="exceeded max_total_latency_s"):
        runner.run()


def test_realtime_scheduler_rejects_non_controller() -> None:
    """A scheduler controller must be a RealtimeSyncFeedbackController."""
    with pytest.raises(TypeError, match="controller must be a RealtimeSyncFeedbackController"):
        RealtimeControllerScheduler("not a controller")  # type: ignore[arg-type]


def test_realtime_scheduler_rejects_negative_base_seed() -> None:
    """A negative base seed is rejected."""
    controller = RealtimeSyncFeedbackController(
        np.array([[0.0, 0.3], [0.3, 0.0]], dtype=np.float64),
        np.array([0.2, 0.5], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="base_seed must be a non-negative integer or None"):
        RealtimeControllerScheduler(controller, base_seed=-1)


def test_require_finite_rejects_non_finite() -> None:
    """The finiteness guard rejects non-finite values."""
    with pytest.raises(ValueError, match="budget must be finite"):
        _require_finite(float("inf"), "budget")


def test_finite_float_rejects_non_finite() -> None:
    """The finite-float coercion rejects non-finite values."""
    with pytest.raises(ValueError, match="weight must be finite"):
        _finite_float(float("nan"), "weight")


def _record(latency: float) -> FeedbackStepRecord:
    return FeedbackStepRecord(
        index=0,
        command=FeedbackCommand(payload={"value": 0.1}),
        result=FeedbackResult(metrics={"r": 0.0}),
        latency_s=latency,
        cumulative_qpu_seconds=0.0,
        stop_requested=False,
    )


def test_enforce_latency_sla_flags_max_latency_breach() -> None:
    """A maximum-latency breach raises an SLA error."""
    history = [_record(1.0)]
    with pytest.raises(RuntimeError, match="max latency"):
        _enforce_latency_sla(history, FeedbackLoopLatencySLA(max_latency_s=0.5))


def test_enforce_latency_sla_flags_p95_breach() -> None:
    """A p95-latency breach raises an SLA error."""
    history = [_record(1.0)]
    with pytest.raises(RuntimeError, match="p95 latency"):
        _enforce_latency_sla(
            history, FeedbackLoopLatencySLA(max_latency_s=10.0, p95_latency_s=0.0)
        )


def test_linear_quantile_rejects_empty() -> None:
    """An empty value sequence has no quantile."""
    with pytest.raises(ValueError, match="quantile values must not be empty"):
        _linear_quantile([], 0.5)


def test_linear_quantile_clamps_low_and_high() -> None:
    """Quantiles at or beyond the bounds return the extreme order statistics."""
    assert _linear_quantile([1.0, 2.0, 3.0], 0.0) == 1.0
    assert _linear_quantile([1.0, 2.0, 3.0], 1.0) == 3.0


def test_linear_quantile_returns_exact_index_value() -> None:
    """An interior quantile landing on an integer index returns that value."""
    assert _linear_quantile([1.0, 2.0, 3.0], 0.5) == 2.0
