# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — realtime runtime module
# scpn-quantum-control -- deadline-aware realtime runtime
"""Deadline-aware realtime control runtime.

This is a deterministic software-control runtime for bounded feedback loops.
It accounts for latency, jitter, and deadline misses around injected control
steps. It is not an intra-shot hardware-latency claim.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

_NS_PER_SECOND = 1_000_000_000


class RealtimeClock(Protocol):
    """Monotonic clock boundary used by the realtime runtime.

    Implementations expose seconds from an arbitrary monotonic epoch. Runtime
    calculations depend only on differences between readings, never on the
    epoch itself.
    """

    def now(self) -> float:
        """Return the current monotonic time.

        Returns
        -------
        float
            Monotonic time in seconds.

        """
        ...

    def sleep_until(self, target_s: float) -> None:
        """Block or advance until a monotonic target.

        Parameters
        ----------
        target_s : float
            Target time in monotonic seconds. Targets at or before the current
            time return without sleeping.

        """
        ...


class MonotonicRealtimeClock:
    """Wall-clock implementation backed by :func:`time.monotonic`."""

    def now(self) -> float:
        """Return the current wall-clock monotonic time.

        Returns
        -------
        float
            Monotonic time in seconds.

        """
        return time.monotonic()

    def sleep_until(self, target_s: float) -> None:
        """Sleep until a future monotonic target.

        Parameters
        ----------
        target_s : float
            Target time in monotonic seconds. Past targets return immediately.

        """
        delay = target_s - self.now()
        if delay > 0.0:
            time.sleep(delay)


class VirtualRealtimeClock:
    """Deterministic clock for simulation and control-loop verification."""

    def __init__(self) -> None:
        """Initialise the clock at zero monotonic seconds."""
        self._now = 0.0

    def now(self) -> float:
        """Return the current virtual time.

        Returns
        -------
        float
            Virtual monotonic time in seconds.

        """
        return self._now

    def sleep_until(self, target_s: float) -> None:
        """Advance to a future target without wall-clock sleep.

        Parameters
        ----------
        target_s : float
            Target time in virtual monotonic seconds. Past targets leave the
            clock unchanged.

        """
        if target_s > self._now:
            self._now = float(target_s)

    def advance(self, seconds: float) -> None:
        """Advance virtual time by a finite non-negative duration.

        Parameters
        ----------
        seconds : float
            Duration to add, in seconds.

        Raises
        ------
        ValueError
            If ``seconds`` is negative or non-finite.

        """
        _require_non_negative(seconds, "seconds")
        self._now += float(seconds)


@dataclass(frozen=True)
class RealtimeRuntimeConfig:
    """Timing contract for a realtime software control loop.

    Attributes
    ----------
    sample_period_s : float
        Scheduled start-to-start period in seconds.
    deadline_s : float
        Maximum permitted step execution latency in seconds. It cannot exceed
        ``sample_period_s``.
    jitter_budget_s : float
        Maximum start-to-start scheduling deviation in seconds. Deviations at
        or below this value are recorded as zero.
    max_missed_deadlines : int
        Number of missed ticks tolerated before the runtime fails closed.
    align_to_period : bool
        Whether each tick waits for its scheduled start. When false, ticks run
        immediately while retaining the same schedule for telemetry.

    """

    sample_period_s: float
    deadline_s: float
    jitter_budget_s: float = 0.0
    max_missed_deadlines: int = 0
    align_to_period: bool = True

    def __post_init__(self) -> None:
        """Validate the timing contract.

        Raises
        ------
        ValueError
            If a duration is invalid, the deadline exceeds the period, or the
            missed-deadline budget is not a non-negative integer.

        """
        _require_positive(self.sample_period_s, "sample_period_s")
        _require_positive(self.deadline_s, "deadline_s")
        _require_non_negative(self.jitter_budget_s, "jitter_budget_s")
        if self.deadline_s > self.sample_period_s:
            raise ValueError("deadline_s must be <= sample_period_s")
        if not isinstance(self.max_missed_deadlines, int) or self.max_missed_deadlines < 0:
            raise ValueError("max_missed_deadlines must be a non-negative integer")


@dataclass(frozen=True)
class RealtimeTickRecord:
    """Auditable timing record for one realtime tick.

    Attributes
    ----------
    index : int
        Zero-based tick index.
    scheduled_start_s : float
        Intended start time in monotonic seconds.
    actual_start_s : float
        Observed start time in monotonic seconds.
    finish_s : float
        Observed finish time in monotonic seconds.
    latency_s : float
        Step execution duration in seconds.
    jitter_s : float
        Budget-filtered start-to-start deviation in seconds.
    deadline_missed : bool
        Whether latency or jitter exceeded its configured bound.
    metrics : Mapping[str, float]
        Finite numeric metrics returned by the control step. The stored mapping
        is an immutable copy.

    """

    index: int
    scheduled_start_s: float
    actual_start_s: float
    finish_s: float
    latency_s: float
    jitter_s: float
    deadline_missed: bool
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Copy metrics into an immutable mapping."""
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


@dataclass(frozen=True)
class RealtimeRunResult:
    """Aggregate result for a realtime control-loop run.

    Attributes
    ----------
    records : tuple[RealtimeTickRecord, ...]
        Ordered per-tick telemetry.
    completed : bool
        Whether every requested tick completed.
    missed_deadlines : int
        Number of ticks whose latency or jitter exceeded its bound.
    max_latency_s : float
        Maximum observed execution latency in seconds.
    max_jitter_s : float
        Maximum observed budget-filtered jitter in seconds.

    """

    records: tuple[RealtimeTickRecord, ...]
    completed: bool
    missed_deadlines: int
    max_latency_s: float
    max_jitter_s: float


@dataclass(frozen=True)
class RealtimeSLAConfig:
    """Service-level contract for realtime-loop latency and jitter.

    Attributes
    ----------
    max_latency_s : float
        Maximum permitted observed latency in seconds.
    max_jitter_s : float
        Maximum permitted observed jitter in seconds.
    p95_latency_s : float or None
        Optional 95th-percentile latency ceiling in seconds.
    p99_latency_s : float or None
        Optional 99th-percentile latency ceiling in seconds.
    max_deadline_miss_rate : float
        Maximum permitted fraction of missed ticks in the closed interval
        ``[0, 1]``.

    """

    max_latency_s: float
    max_jitter_s: float = 0.0
    p95_latency_s: float | None = None
    p99_latency_s: float | None = None
    max_deadline_miss_rate: float = 0.0

    def __post_init__(self) -> None:
        """Validate the service-level contract.

        Raises
        ------
        ValueError
            If a limit is invalid or ``max_deadline_miss_rate`` exceeds one.

        """
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
    """Measured SLA verdict for one realtime run.

    Attributes
    ----------
    compliant : bool
        Whether every configured SLA bound passed.
    breach_reasons : tuple[str, ...]
        Human-readable descriptions of every breached bound.
    observed_max_latency_s : float
        Maximum observed latency in seconds.
    observed_max_jitter_s : float
        Maximum observed jitter in seconds.
    observed_p95_latency_s : float
        Linear-interpolated 95th-percentile latency in seconds.
    observed_p99_latency_s : float
        Linear-interpolated 99th-percentile latency in seconds.
    observed_deadline_miss_rate : float
        Fraction of ticks recorded as deadline misses.
    n_ticks : int
        Number of tick records evaluated.

    """

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
    """Run a control step on a fixed-period schedule.

    Parameters
    ----------
    n_ticks : int
        Number of ticks to execute.
    step : RealtimeStep
        Callable receiving the zero-based tick index and returning finite
        numeric metrics.
    config : RealtimeRuntimeConfig
        Period, deadline, jitter, miss-budget, and alignment contract.
    clock : RealtimeClock or None
        Monotonic clock implementation. The wall-clock implementation is used
        when omitted.

    Returns
    -------
    RealtimeRunResult
        Ordered tick records and aggregate latency, jitter, and miss telemetry.

    Raises
    ------
    ValueError
        If ``n_ticks`` is not a positive integer or a metric is invalid.
    RuntimeError
        If the number of missed ticks exceeds ``max_missed_deadlines``.

    Notes
    -----
    A tick misses when its execution latency exceeds ``deadline_s`` or its
    unsuppressed start-to-start jitter exceeds ``jitter_budget_s``.

    """
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
    """Evaluate a realtime run against an SLA contract.

    Parameters
    ----------
    result : RealtimeRunResult
        Completed run telemetry to evaluate.
    sla : RealtimeSLAConfig
        Maximum, percentile, and miss-rate bounds.

    Returns
    -------
    RealtimeSLAReport
        Observed statistics and all breach reasons.

    Raises
    ------
    ValueError
        If ``result`` contains no tick records.

    Notes
    -----
    Percentiles use NumPy's linear interpolation method.

    """
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
    """Require a realtime run to satisfy an SLA contract.

    Parameters
    ----------
    result : RealtimeRunResult
        Completed run telemetry to evaluate.
    sla : RealtimeSLAConfig
        Maximum, percentile, and miss-rate bounds.

    Returns
    -------
    RealtimeSLAReport
        Compliant report for the run.

    Raises
    ------
    ValueError
        If ``result`` contains no tick records.
    RuntimeError
        If any configured SLA bound is breached.

    """
    report = evaluate_realtime_sla(result, sla=sla)
    if not report.compliant:
        raise RuntimeError("; ".join(report.breach_reasons))
    return report


@dataclass(frozen=True)
class CycleSample:
    """Sub-microsecond timing record for one outer-loop cycle.

    All timestamps are integer nanoseconds on a monotonic clock. A cycle misses
    its deadline when ``end_ns`` exceeds ``deadline_ns``.

    Attributes
    ----------
    cycle_id : int
        Caller-defined cycle identifier.
    start_ns : int
        Monotonic cycle start in nanoseconds.
    end_ns : int
        Monotonic cycle finish in nanoseconds.
    deadline_ns : int
        Absolute monotonic deadline in nanoseconds.

    """

    cycle_id: int
    start_ns: int
    end_ns: int
    deadline_ns: int

    def __post_init__(self) -> None:
        """Validate timestamp types and ordering.

        Raises
        ------
        TypeError
            If any field is not a plain integer.
        ValueError
            If the finish or deadline precedes the start.

        """
        for name in ("cycle_id", "start_ns", "end_ns", "deadline_ns"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be a plain int, got {type(value).__name__}")
        if self.end_ns < self.start_ns:
            raise ValueError("end_ns must be >= start_ns")
        if self.deadline_ns < self.start_ns:
            raise ValueError("deadline_ns must be >= start_ns")

    @property
    def duration_ns(self) -> int:
        """Return the executed cycle duration.

        Returns
        -------
        int
            Duration in nanoseconds.

        """
        return self.end_ns - self.start_ns

    @property
    def deadline_missed(self) -> bool:
        """Return whether the cycle finished after its deadline.

        Returns
        -------
        bool
            True when ``end_ns`` is greater than ``deadline_ns``.

        """
        return self.end_ns > self.deadline_ns


@dataclass(frozen=True)
class SubMicrosecondReport:
    """Aggregate inter-cycle jitter and deadline-miss telemetry.

    Attributes
    ----------
    jitter_p50_ns : float
        Median retained jitter in nanoseconds.
    jitter_p95_ns : float
        Linear-interpolated 95th-percentile jitter in nanoseconds.
    jitter_p99_ns : float
        Linear-interpolated 99th-percentile jitter in nanoseconds.
    jitter_max_ns : float
        Maximum retained jitter in nanoseconds.
    deadline_misses : int
        Total missed cycles, including samples outside the retained window.
    cycles_observed : int
        Total cycles recorded or summarised.
    target_period_ns : float
        Expected start-to-start interval in nanoseconds.
    window_size : int
        Number of jitter samples used for percentile estimation.

    """

    jitter_p50_ns: float
    jitter_p95_ns: float
    jitter_p99_ns: float
    jitter_max_ns: float
    deadline_misses: int
    cycles_observed: int
    target_period_ns: float
    window_size: int


class SubMicrosecondTracker:
    """Sub-microsecond outer-loop jitter and deadline tracker.

    Records integer-nanosecond cycle samples and reports inter-cycle jitter
    percentiles against the target period plus a deadline-miss count. The jitter
    of a cycle is the absolute deviation of its start-to-start interval from the
    target period ``1e9 / target_rate_hz`` nanoseconds; the first observed cycle
    has zero jitter. Recent jitter samples are kept in a bounded ring of
    ``ring_buffer_capacity`` entries for percentile estimation, while total
    cycles and total deadline misses are running counters and stay exact across
    ring overwrites.

    This is software telemetry for the microsecond-scale outer loop, not an
    intra-shot hardware-latency claim; the downstream sub-50 ns FPGA path is
    covered by RTL assertions in the consumer.

    Parameters
    ----------
    target_rate_hz : int
        Positive target cycle rate in hertz.
    ring_buffer_capacity : int
        Positive maximum number of jitter samples retained for percentiles.

    Raises
    ------
    TypeError
        If either parameter is not a plain integer.
    ValueError
        If either parameter is less than one.

    """

    def __init__(
        self,
        target_rate_hz: int = 100_000,
        ring_buffer_capacity: int = 1 << 16,
    ) -> None:
        """Initialise an empty tracker for a target rate and window size."""
        if not isinstance(target_rate_hz, int) or isinstance(target_rate_hz, bool):
            raise TypeError("target_rate_hz must be an int")
        if target_rate_hz < 1:
            raise ValueError("target_rate_hz must be a positive integer")
        if not isinstance(ring_buffer_capacity, int) or isinstance(ring_buffer_capacity, bool):
            raise TypeError("ring_buffer_capacity must be an int")
        if ring_buffer_capacity < 1:
            raise ValueError("ring_buffer_capacity must be a positive integer")
        self._target_rate_hz = target_rate_hz
        self._target_period_ns = _NS_PER_SECOND / float(target_rate_hz)
        self._capacity = ring_buffer_capacity
        self._jitter_ring: deque[float] = deque(maxlen=ring_buffer_capacity)
        self._total_cycles = 0
        self._total_misses = 0
        self._prev_start_ns: int | None = None

    @property
    def target_period_ns(self) -> float:
        """Return the target start-to-start period.

        Returns
        -------
        float
            Period in nanoseconds.

        """
        return self._target_period_ns

    @property
    def cycles_observed(self) -> int:
        """Return the total number of recorded cycles.

        Returns
        -------
        int
            Count since construction or the most recent reset.

        """
        return self._total_cycles

    def record(self, sample: CycleSample) -> None:
        """Record one cycle and update jitter and miss telemetry.

        Parameters
        ----------
        sample : CycleSample
            Validated monotonic timestamps for one cycle.

        Raises
        ------
        TypeError
            If ``sample`` is not a :class:`CycleSample`.

        """
        if not isinstance(sample, CycleSample):
            raise TypeError("sample must be a CycleSample")
        if self._prev_start_ns is None:
            jitter = 0.0
        else:
            interval = float(sample.start_ns - self._prev_start_ns)
            jitter = abs(interval - self._target_period_ns)
        self._jitter_ring.append(jitter)
        self._total_cycles += 1
        if sample.deadline_missed:
            self._total_misses += 1
        self._prev_start_ns = sample.start_ns

    def report(self) -> SubMicrosecondReport:
        """Return current jitter and deadline-miss telemetry.

        Returns
        -------
        SubMicrosecondReport
            Retained-window percentiles and exact running counts.

        Raises
        ------
        ValueError
            If no cycles have been recorded.

        """
        if self._total_cycles == 0:
            raise ValueError("no cycles recorded")
        jitters = np.asarray(self._jitter_ring, dtype=np.float64)
        p50, p95, p99, jmax = _jitter_percentiles(jitters)
        return SubMicrosecondReport(
            jitter_p50_ns=p50,
            jitter_p95_ns=p95,
            jitter_p99_ns=p99,
            jitter_max_ns=jmax,
            deadline_misses=self._total_misses,
            cycles_observed=self._total_cycles,
            target_period_ns=self._target_period_ns,
            window_size=len(self._jitter_ring),
        )

    def reset(self) -> None:
        """Clear all retained samples, counters, and interval history."""
        self._jitter_ring.clear()
        self._total_cycles = 0
        self._total_misses = 0
        self._prev_start_ns = None


def summarise_cycle_samples(
    start_ns: NDArray[np.int64] | list[int],
    end_ns: NDArray[np.int64] | list[int],
    deadline_ns: NDArray[np.int64] | list[int],
    *,
    target_rate_hz: int = 100_000,
) -> SubMicrosecondReport:
    """Summarise arrays of cycle timestamps in a single pass.

    This is the batch path consumed by the throughput benchmark and by callers
    that buffer cycle timestamps and summarise them periodically. It computes the
    same jitter percentiles and deadline-miss count as
    :class:`SubMicrosecondTracker` over the full input.

    Parameters
    ----------
    start_ns : numpy.ndarray or list[int]
        One-dimensional cycle-start timestamps in monotonic nanoseconds.
    end_ns : numpy.ndarray or list[int]
        One-dimensional cycle-finish timestamps in monotonic nanoseconds.
    deadline_ns : numpy.ndarray or list[int]
        One-dimensional absolute deadlines in monotonic nanoseconds.
    target_rate_hz : int
        Positive target cycle rate in hertz.

    Returns
    -------
    SubMicrosecondReport
        Full-window jitter percentiles and deadline-miss count.

    Raises
    ------
    TypeError
        If ``target_rate_hz`` is not a plain integer.
    ValueError
        If the rate is non-positive, arrays are not one-dimensional and equal
        length, no cycle is supplied, or a finish/deadline precedes its start.

    """
    if not isinstance(target_rate_hz, int) or isinstance(target_rate_hz, bool):
        raise TypeError("target_rate_hz must be an int")
    if target_rate_hz < 1:
        raise ValueError("target_rate_hz must be a positive integer")
    start = np.ascontiguousarray(start_ns, dtype=np.int64)
    end = np.ascontiguousarray(end_ns, dtype=np.int64)
    deadline = np.ascontiguousarray(deadline_ns, dtype=np.int64)
    if not (start.ndim == end.ndim == deadline.ndim == 1):
        raise ValueError("start_ns, end_ns, deadline_ns must be one-dimensional")
    if not (start.shape == end.shape == deadline.shape):
        raise ValueError("start_ns, end_ns, deadline_ns must have equal length")
    if start.size == 0:
        raise ValueError("no cycles to summarise")
    if np.any(end < start):
        raise ValueError("every end_ns must be >= its start_ns")
    if np.any(deadline < start):
        raise ValueError("every deadline_ns must be >= its start_ns")

    target_period_ns = _NS_PER_SECOND / float(target_rate_hz)
    summary = _sub_us_summary(start, end, deadline, target_period_ns)
    p50, p95, p99, jmax, misses, count = summary
    return SubMicrosecondReport(
        jitter_p50_ns=p50,
        jitter_p95_ns=p95,
        jitter_p99_ns=p99,
        jitter_max_ns=jmax,
        deadline_misses=misses,
        cycles_observed=count,
        target_period_ns=target_period_ns,
        window_size=int(start.size),
    )


def _jitter_percentiles(jitters: NDArray[np.float64]) -> tuple[float, float, float, float]:
    """Return ``(p50, p95, p99, max)`` jitter, via the Rust kernel when present."""
    try:
        import scpn_quantum_engine as _engine

        if hasattr(_engine, "sub_us_jitter_percentiles"):
            p50, p95, p99, jmax = _engine.sub_us_jitter_percentiles(
                np.ascontiguousarray(jitters, dtype=np.float64)
            )
            return float(p50), float(p95), float(p99), float(jmax)
    except (ImportError, AttributeError, ValueError):
        pass
    return _jitter_percentiles_numpy(jitters)


def _jitter_percentiles_numpy(
    jitters: NDArray[np.float64],
) -> tuple[float, float, float, float]:
    if jitters.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    p50 = float(np.quantile(jitters, 0.50, method="linear"))
    p95 = float(np.quantile(jitters, 0.95, method="linear"))
    p99 = float(np.quantile(jitters, 0.99, method="linear"))
    return p50, p95, p99, float(jitters.max())


def _sub_us_summary(
    start: NDArray[np.int64],
    end: NDArray[np.int64],
    deadline: NDArray[np.int64],
    target_period_ns: float,
) -> tuple[float, float, float, float, int, int]:
    """Return ``(p50, p95, p99, max, misses, count)``, via Rust when present."""
    try:
        import scpn_quantum_engine as _engine

        if hasattr(_engine, "sub_us_tracker_summary"):
            p50, p95, p99, jmax, misses, count = _engine.sub_us_tracker_summary(
                start, end, deadline, float(target_period_ns)
            )
            return float(p50), float(p95), float(p99), float(jmax), int(misses), int(count)
    except (ImportError, AttributeError, ValueError):
        pass
    return _sub_us_summary_numpy(start, end, deadline, target_period_ns)


def _sub_us_summary_numpy(
    start: NDArray[np.int64],
    end: NDArray[np.int64],
    deadline: NDArray[np.int64],
    target_period_ns: float,
) -> tuple[float, float, float, float, int, int]:
    intervals = np.diff(start.astype(np.float64))
    jitters = np.empty(start.size, dtype=np.float64)
    jitters[0] = 0.0
    jitters[1:] = np.abs(intervals - target_period_ns)
    p50, p95, p99, jmax = _jitter_percentiles_numpy(jitters)
    misses = int(np.count_nonzero(end > deadline))
    return p50, p95, p99, jmax, misses, int(start.size)


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
    "CycleSample",
    "RealtimeSLAConfig",
    "RealtimeSLAReport",
    "MonotonicRealtimeClock",
    "RealtimeClock",
    "RealtimeRunResult",
    "RealtimeRuntimeConfig",
    "RealtimeTickRecord",
    "SubMicrosecondReport",
    "SubMicrosecondTracker",
    "VirtualRealtimeClock",
    "enforce_realtime_sla",
    "evaluate_realtime_sla",
    "run_realtime_control_loop",
    "summarise_cycle_samples",
]
