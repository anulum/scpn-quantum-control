# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Hybrid feedback loop
"""Cross-shot hybrid classical-quantum feedback orchestration.

This module implements the safe S1 foundation: a scheduler/observer loop that
can run against simulators, mocks, or an explicitly approved hardware scheduler.
It does not create IBM sessions, fetch credentials, or submit cloud jobs by
itself. Python-level feedback is treated as cross-shot or cross-circuit control;
sub-microsecond intra-shot feedback must use provider-side dynamic-circuit
primitives rather than this runner.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class FeedbackLoopConfig:
    """Safety and budget configuration for a cross-shot feedback loop."""

    max_steps: int
    max_total_latency_s: float = 300.0
    max_step_latency_s: float = 1.0
    max_qpu_seconds: float = 0.0
    require_hardware_approval: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        _require_non_negative(self.max_total_latency_s, "max_total_latency_s")
        _require_non_negative(self.max_step_latency_s, "max_step_latency_s")
        _require_non_negative(self.max_qpu_seconds, "max_qpu_seconds")


@dataclass(frozen=True)
class FeedbackCommand:
    """One scheduler command proposed by a classical observer."""

    payload: Any
    label: str = "step"
    estimated_qpu_seconds: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_negative(self.estimated_qpu_seconds, "estimated_qpu_seconds")


@dataclass(frozen=True)
class FeedbackResult:
    """Observed result returned by a feedback scheduler."""

    counts: Mapping[str, int] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    job_id: str | None = None
    qpu_seconds: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_negative(self.qpu_seconds, "qpu_seconds")
        for bitstring, count in self.counts.items():
            if count < 0:
                raise ValueError(f"count for {bitstring!r} must be non-negative")
        for name, value in self.metrics.items():
            if not isinstance(value, int | float):
                raise ValueError(f"metric {name!r} must be numeric")


@dataclass(frozen=True)
class FeedbackStepRecord:
    """Auditable record for one feedback-loop iteration."""

    index: int
    command: FeedbackCommand
    result: FeedbackResult
    latency_s: float
    cumulative_qpu_seconds: float
    stop_requested: bool
    observer_state: Mapping[str, Any] = field(default_factory=dict)


class FeedbackScheduler(Protocol):
    """Scheduler boundary for simulator, mock, or approved hardware backends."""

    is_hardware: bool

    def submit(self, command: FeedbackCommand) -> FeedbackResult:
        """Submit one command and return observed counts/metrics."""
        ...


class FeedbackObserver(Protocol):
    """Classical observer that proposes the next command from prior results."""

    def initial_command(self) -> FeedbackCommand:
        """Return the first command to submit."""
        ...

    def update(
        self,
        result: FeedbackResult,
        history: Sequence[FeedbackStepRecord],
    ) -> tuple[FeedbackCommand | None, Mapping[str, Any]]:
        """Return the next command and observer diagnostics.

        Returning ``None`` requests clean loop termination.
        """
        ...


class FeedbackRunner:
    """Run a bounded cross-shot feedback loop.

    The runner is intentionally backend-agnostic. It records latency and QPU
    spend, but it does not know how to create provider sessions. A real hardware
    scheduler must be supplied by the caller after separate approval.
    """

    def __init__(
        self,
        scheduler: FeedbackScheduler,
        observer: FeedbackObserver,
        config: FeedbackLoopConfig,
        *,
        hardware_approved: bool = False,
    ) -> None:
        self.scheduler = scheduler
        self.observer = observer
        self.config = config
        self.hardware_approved = hardware_approved
        if (
            config.require_hardware_approval
            and getattr(scheduler, "is_hardware", False)
            and not hardware_approved
        ):
            raise PermissionError(
                "hardware feedback scheduler requires explicit approval before execution"
            )

    def run(self) -> list[FeedbackStepRecord]:
        """Execute the feedback loop until stop, step, latency, or QPU budget limit."""
        history: list[FeedbackStepRecord] = []
        command = self.observer.initial_command()
        total_qpu = 0.0
        total_latency = 0.0
        for index in range(self.config.max_steps):
            if total_qpu + command.estimated_qpu_seconds > self.config.max_qpu_seconds:
                raise RuntimeError("feedback loop would exceed configured QPU budget")
            started = time.monotonic()
            result = self.scheduler.submit(command)
            latency = time.monotonic() - started
            total_latency += latency
            total_qpu += result.qpu_seconds
            if latency > self.config.max_step_latency_s:
                raise RuntimeError("feedback step exceeded max_step_latency_s")
            if total_latency > self.config.max_total_latency_s:
                raise RuntimeError("feedback loop exceeded max_total_latency_s")
            if total_qpu > self.config.max_qpu_seconds:
                raise RuntimeError("feedback loop exceeded max_qpu_seconds")
            next_command, observer_state = self.observer.update(result, tuple(history))
            record = FeedbackStepRecord(
                index=index,
                command=command,
                result=result,
                latency_s=latency,
                cumulative_qpu_seconds=total_qpu,
                stop_requested=next_command is None,
                observer_state=dict(observer_state),
            )
            history.append(record)
            if next_command is None:
                break
            command = next_command
        return history


class ProportionalMetricObserver:
    """Reference observer that tunes one numeric command parameter from a metric."""

    def __init__(
        self,
        *,
        initial_value: float,
        metric_name: str,
        target: float,
        gain: float,
        min_value: float = 0.0,
        max_value: float = 1.0,
        tolerance: float = 0.0,
        label: str = "proportional",
    ) -> None:
        _require_finite(initial_value, "initial_value")
        _require_finite(target, "target")
        _require_finite(gain, "gain")
        _require_finite(min_value, "min_value")
        _require_finite(max_value, "max_value")
        _require_non_negative(tolerance, "tolerance")
        if max_value < min_value:
            raise ValueError("max_value must be >= min_value")
        self.current = float(initial_value)
        self.metric_name = metric_name
        self.target = float(target)
        self.gain = float(gain)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.tolerance = float(tolerance)
        self.label = label

    def initial_command(self) -> FeedbackCommand:
        """Return the initial parameter command."""
        return FeedbackCommand(payload={"value": self.current}, label=self.label)

    def update(
        self,
        result: FeedbackResult,
        history: Sequence[FeedbackStepRecord],
    ) -> tuple[FeedbackCommand | None, Mapping[str, Any]]:
        """Update the parameter toward the configured target metric."""
        if self.metric_name not in result.metrics:
            raise KeyError(f"missing feedback metric {self.metric_name!r}")
        observed = float(result.metrics[self.metric_name])
        error = self.target - observed
        state = {"observed": observed, "error": error, "value_in": self.current}
        if abs(error) <= self.tolerance:
            return None, state | {"value_out": self.current, "converged": True}
        self.current = min(max(self.current + self.gain * error, self.min_value), self.max_value)
        return (
            FeedbackCommand(payload={"value": self.current}, label=self.label),
            state | {"value_out": self.current, "converged": False},
        )


def _require_non_negative(value: float, name: str) -> None:
    _require_finite(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _require_finite(value: float, name: str) -> None:
    if (
        not isinstance(value, int | float)
        or value != value
        or value in {float("inf"), float("-inf")}
    ):
        raise ValueError(f"{name} must be finite")
