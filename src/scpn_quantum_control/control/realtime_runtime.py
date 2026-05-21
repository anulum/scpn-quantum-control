# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- deadline-aware realtime runtime
"""Deadline-aware realtime control runtime.

This is a deterministic software-control runtime for bounded feedback loops.
It accounts for latency, jitter, and deadline misses around injected control
steps. It is not an intra-shot hardware-latency claim.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

import numpy as np


class RealtimeClock(Protocol):
    """Clock boundary used by the realtime runtime."""

    def now(self) -> float:
        """Return monotonic seconds."""
        ...

    def sleep_until(self, target_s: float) -> None:
        """Block or advance until ``target_s``."""
        ...


class MonotonicRealtimeClock:
    """Wall-clock realtime clock using ``time.monotonic``."""

    def now(self) -> float:
        """Return monotonic wall-clock seconds."""
        return time.monotonic()

    def sleep_until(self, target_s: float) -> None:
        """Sleep until the target monotonic time."""
        delay = target_s - self.now()
        if delay > 0.0:
            time.sleep(delay)


class VirtualRealtimeClock:
    """Deterministic test clock for realtime control loops."""

    def __init__(self) -> None:
        self._now = 0.0

    def now(self) -> float:
        """Return virtual monotonic seconds."""
        return self._now

    def sleep_until(self, target_s: float) -> None:
        """Advance to the target time without wall-clock sleep."""
        if target_s > self._now:
            self._now = float(target_s)

    def advance(self, seconds: float) -> None:
        """Advance virtual time by a finite non-negative duration."""
        _require_non_negative(seconds, "seconds")
        self._now += float(seconds)


@dataclass(frozen=True)
class RealtimeRuntimeConfig:
    """Timing contract for a realtime software control loop."""

    sample_period_s: float
    deadline_s: float
    jitter_budget_s: float = 0.0
    max_missed_deadlines: int = 0
    align_to_period: bool = True

    def __post_init__(self) -> None:
        _require_positive(self.sample_period_s, "sample_period_s")
        _require_positive(self.deadline_s, "deadline_s")
        _require_non_negative(self.jitter_budget_s, "jitter_budget_s")
        if self.deadline_s > self.sample_period_s:
            raise ValueError("deadline_s must be <= sample_period_s")
        if not isinstance(self.max_missed_deadlines, int) or self.max_missed_deadlines < 0:
            raise ValueError("max_missed_deadlines must be a non-negative integer")


@dataclass(frozen=True)
class RealtimeTickRecord:
    """Auditable timing record for one realtime tick."""

    index: int
    scheduled_start_s: float
    actual_start_s: float
    finish_s: float
    latency_s: float
    jitter_s: float
    deadline_missed: bool
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


@dataclass(frozen=True)
class RealtimeRunResult:
    """Aggregate result for a realtime control-loop run."""

    records: tuple[RealtimeTickRecord, ...]
    completed: bool
    missed_deadlines: int
    max_latency_s: float
    max_jitter_s: float


@dataclass(frozen=True)
class RealtimeSLAConfig:
    """Service-level contract for realtime-loop latency and jitter."""

    max_latency_s: float
    max_jitter_s: float = 0.0
    p95_latency_s: float | None = None
    p99_latency_s: float | None = None
    max_deadline_miss_rate: float = 0.0

    def __post_init__(self) -> None:
        _require_positive(self.max_latency_s, "max_latency_s")
        _require_non_negative(self.max_jitter_s, "max_jitter_s")
        if self.p95_latency_s is not None:
            _require_positive(self.p95_latency_s, "p95_latency_s")
        if self.p99_latency_s is not None:
            _require_positive(self.p99_latency_s, "p99_latency_s")
        _require_non_negative(self.max_deadline_miss_rate, "max_deadline_miss_rate")
        if self.max_deadline_miss_rate > 1.0:
            raise ValueError("max_deadline_miss_rate must be <= 1.0")


@dataclass(frozen=True)
class RealtimeSLAReport:
    """Measured SLA verdict for one realtime run."""

    compliant: bool
    breach_reasons: tuple[str, ...]
    observed_max_latency_s: float
    observed_max_jitter_s: float
    observed_p95_latency_s: float
    observed_p99_latency_s: float
    observed_deadline_miss_rate: float
    n_ticks: int


RealtimeStep = Callable[[int], Mapping[str, float]]


def run_realtime_control_loop(
    n_ticks: int,
    step: RealtimeStep,
    *,
    config: RealtimeRuntimeConfig,
    clock: RealtimeClock | None = None,
) -> RealtimeRunResult:
    """Run ``step`` on a fixed-period realtime schedule with deadline accounting."""

    if not isinstance(n_ticks, int) or n_ticks < 1:
        raise ValueError("n_ticks must be a positive integer")
    runtime_clock = clock or MonotonicRealtimeClock()
    start_s = runtime_clock.now()
    records: list[RealtimeTickRecord] = []
    missed = 0
    previous_start = start_s

    for index in range(n_ticks):
        scheduled = start_s + index * config.sample_period_s
        if config.align_to_period:
            runtime_clock.sleep_until(scheduled)
        actual_start = runtime_clock.now()
        raw_metrics = step(index)
        finish = runtime_clock.now()
        latency = finish - actual_start
        jitter = (
            0.0 if index == 0 else abs((actual_start - previous_start) - config.sample_period_s)
        )
        if jitter <= config.jitter_budget_s:
            jitter = 0.0
        deadline_missed = latency > config.deadline_s or jitter > config.jitter_budget_s
        if deadline_missed:
            missed += 1
        if missed > config.max_missed_deadlines:
            raise RuntimeError("realtime control loop exceeded deadline miss budget")
        record = RealtimeTickRecord(
            index=index,
            scheduled_start_s=scheduled,
            actual_start_s=actual_start,
            finish_s=finish,
            latency_s=latency,
            jitter_s=jitter,
            deadline_missed=deadline_missed,
            metrics=_normalise_metrics(raw_metrics),
        )
        records.append(record)
        previous_start = actual_start

    max_latency = max(record.latency_s for record in records)
    max_jitter = max(record.jitter_s for record in records)
    return RealtimeRunResult(
        records=tuple(records),
        completed=True,
        missed_deadlines=missed,
        max_latency_s=max_latency,
        max_jitter_s=max_jitter,
    )


def evaluate_realtime_sla(
    result: RealtimeRunResult,
    *,
    sla: RealtimeSLAConfig,
) -> RealtimeSLAReport:
    """Evaluate a realtime run against a strict SLA contract."""

    if not result.records:
        raise ValueError("realtime result must contain at least one tick record")
    latencies = np.asarray([record.latency_s for record in result.records], dtype=np.float64)
    miss_rate = float(result.missed_deadlines) / float(len(result.records))
    p95 = float(np.quantile(latencies, 0.95, method="linear"))
    p99 = float(np.quantile(latencies, 0.99, method="linear"))
    reasons: list[str] = []
    if float(result.max_latency_s) > sla.max_latency_s:
        reasons.append(
            f"max latency {float(result.max_latency_s):.9f}s exceeds SLA {sla.max_latency_s:.9f}s"
        )
    if float(result.max_jitter_s) > sla.max_jitter_s:
        reasons.append(
            f"max jitter {float(result.max_jitter_s):.9f}s exceeds SLA {sla.max_jitter_s:.9f}s"
        )
    if sla.p95_latency_s is not None and p95 > sla.p95_latency_s:
        reasons.append(f"p95 latency {p95:.9f}s exceeds SLA {sla.p95_latency_s:.9f}s")
    if sla.p99_latency_s is not None and p99 > sla.p99_latency_s:
        reasons.append(f"p99 latency {p99:.9f}s exceeds SLA {sla.p99_latency_s:.9f}s")
    if miss_rate > sla.max_deadline_miss_rate:
        reasons.append(
            f"deadline miss rate {miss_rate:.6f} exceeds SLA {sla.max_deadline_miss_rate:.6f}"
        )
    return RealtimeSLAReport(
        compliant=not reasons,
        breach_reasons=tuple(reasons),
        observed_max_latency_s=float(result.max_latency_s),
        observed_max_jitter_s=float(result.max_jitter_s),
        observed_p95_latency_s=p95,
        observed_p99_latency_s=p99,
        observed_deadline_miss_rate=miss_rate,
        n_ticks=len(result.records),
    )


def enforce_realtime_sla(
    result: RealtimeRunResult,
    *,
    sla: RealtimeSLAConfig,
) -> RealtimeSLAReport:
    """Fail closed when a realtime run violates the configured SLA."""

    report = evaluate_realtime_sla(result, sla=sla)
    if not report.compliant:
        raise RuntimeError("; ".join(report.breach_reasons))
    return report


def _normalise_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    normalised: dict[str, float] = {}
    for key, value in metrics.items():
        if not key:
            raise ValueError("metric names must be non-empty")
        if not isinstance(value, int | float) or not math.isfinite(value):
            raise ValueError(f"metric {key!r} must be finite")
        normalised[str(key)] = float(value)
    return normalised


def _require_positive(value: float, name: str) -> None:
    if not isinstance(value, int | float) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _require_non_negative(value: float, name: str) -> None:
    if not isinstance(value, int | float) or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


__all__ = [
    "RealtimeSLAConfig",
    "RealtimeSLAReport",
    "MonotonicRealtimeClock",
    "RealtimeClock",
    "RealtimeRunResult",
    "RealtimeRuntimeConfig",
    "RealtimeTickRecord",
    "VirtualRealtimeClock",
    "enforce_realtime_sla",
    "evaluate_realtime_sla",
    "run_realtime_control_loop",
]
