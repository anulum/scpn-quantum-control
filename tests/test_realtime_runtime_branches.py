# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the realtime control runtime
"""Clock, config, SLA, and summary branch tests for the realtime runtime.

Covers the monotonic wall-clock sleep path, the runtime and SLA config guards,
the control-loop and SLA-evaluation guards, the batch cycle summariser guard,
the metric normalisation guards, the numeric guard helpers, and the numpy
fallbacks for the jitter percentiles and sub-microsecond summary.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.control.realtime_runtime import (
    CycleSample,
    MonotonicRealtimeClock,
    RealtimeRunResult,
    RealtimeRuntimeConfig,
    RealtimeSLAConfig,
    RealtimeTickRecord,
    SubMicrosecondTracker,
    VirtualRealtimeClock,
    _jitter_percentiles_numpy,
    _normalise_metrics,
    evaluate_realtime_sla,
    run_realtime_control_loop,
    summarise_cycle_samples,
)


def test_monotonic_clock_now_and_sleep() -> None:
    """The monotonic clock advances and sleeps for a positive delay."""
    clock = MonotonicRealtimeClock()
    start = clock.now()
    clock.sleep_until(start + 5.0e-4)
    assert clock.now() >= start


def test_monotonic_clock_returns_immediately_for_past_target() -> None:
    """A missed wall-clock target must not request an invalid negative sleep."""
    clock = MonotonicRealtimeClock()
    past_target = clock.now() - 1.0

    clock.sleep_until(past_target)

    assert clock.now() >= past_target


def test_virtual_clock_advance_rejects_negative() -> None:
    """Virtual time cannot advance by a negative duration."""
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        VirtualRealtimeClock().advance(-1.0)


def test_runtime_config_rejects_non_positive_period() -> None:
    """The sample period must be finite and positive."""
    with pytest.raises(ValueError, match="sample_period_s must be finite and positive"):
        RealtimeRuntimeConfig(sample_period_s=0.0, deadline_s=0.001)


def test_runtime_config_rejects_deadline_above_period() -> None:
    """The deadline must not exceed the sample period."""
    with pytest.raises(ValueError, match="deadline_s must be <="):
        RealtimeRuntimeConfig(sample_period_s=0.001, deadline_s=0.002)


def test_runtime_config_rejects_negative_miss_budget() -> None:
    """The missed-deadline budget must be a non-negative integer."""
    with pytest.raises(ValueError, match="max_missed_deadlines must be a non-negative integer"):
        RealtimeRuntimeConfig(sample_period_s=0.001, deadline_s=0.001, max_missed_deadlines=-1)


def test_sla_config_rejects_miss_rate_above_one() -> None:
    """The deadline-miss rate ceiling cannot exceed one."""
    with pytest.raises(ValueError, match="max_deadline_miss_rate must be <= 1.0"):
        RealtimeSLAConfig(max_latency_s=0.01, max_deadline_miss_rate=1.5)


def test_control_loop_requires_positive_tick_count() -> None:
    """The control loop needs at least one tick."""
    config = RealtimeRuntimeConfig(sample_period_s=0.001, deadline_s=0.001)
    with pytest.raises(ValueError, match="n_ticks must be a positive integer"):
        run_realtime_control_loop(0, lambda _index: {}, config=config)


def test_unaligned_loop_reports_start_jitter_without_waiting_for_schedule() -> None:
    """Caller-driven ticks retain their schedule and expose over-budget jitter."""
    clock = VirtualRealtimeClock()
    config = RealtimeRuntimeConfig(
        sample_period_s=0.010,
        deadline_s=0.009,
        jitter_budget_s=0.001,
        max_missed_deadlines=2,
        align_to_period=False,
    )
    durations = (0.002, 0.004)

    def step(index: int) -> dict[str, float]:
        clock.advance(durations[index])
        return {"duration_s": durations[index]}

    result = run_realtime_control_loop(2, step, config=config, clock=clock)

    assert result.completed
    assert result.missed_deadlines == 1
    assert result.records[1].scheduled_start_s == pytest.approx(0.010)
    assert result.records[1].actual_start_s == pytest.approx(0.002)
    assert result.records[1].latency_s == pytest.approx(0.004)
    assert result.records[1].jitter_s == pytest.approx(0.008)
    assert result.records[1].deadline_missed


def test_evaluate_sla_requires_records() -> None:
    """SLA evaluation needs at least one tick record."""
    empty = RealtimeRunResult(
        records=(), completed=False, missed_deadlines=0, max_latency_s=0.0, max_jitter_s=0.0
    )
    with pytest.raises(ValueError, match="at least one tick record"):
        evaluate_realtime_sla(empty, sla=RealtimeSLAConfig(max_latency_s=0.01))


def test_evaluate_sla_reports_every_breach() -> None:
    """A result breaching every axis lists all breach reasons."""
    record = RealtimeTickRecord(
        index=0,
        scheduled_start_s=0.0,
        actual_start_s=0.0,
        finish_s=1.0,
        latency_s=1.0,
        jitter_s=1.0,
        deadline_missed=True,
    )
    result = RealtimeRunResult(
        records=(record,),
        completed=True,
        missed_deadlines=1,
        max_latency_s=1.0,
        max_jitter_s=1.0,
    )
    sla = RealtimeSLAConfig(
        max_latency_s=1e-9,
        max_jitter_s=0.0,
        p95_latency_s=1e-9,
        p99_latency_s=1e-9,
        max_deadline_miss_rate=0.0,
    )
    report = evaluate_realtime_sla(result, sla=sla)
    assert not report.compliant
    reasons = " ".join(report.breach_reasons)
    assert "max latency" in reasons
    assert "max jitter" in reasons
    assert "p95 latency" in reasons
    assert "p99 latency" in reasons
    assert "deadline miss rate" in reasons


def test_summarise_cycle_samples_rejects_non_integer_rate() -> None:
    """The target rate of the batch summariser must be an int."""
    with pytest.raises(TypeError, match="target_rate_hz must be an int"):
        summarise_cycle_samples([0], [1], [1], target_rate_hz=1.5)  # type: ignore[arg-type]


def test_normalise_metrics_rejects_empty_key() -> None:
    """Metric names must be non-empty."""
    with pytest.raises(ValueError, match="metric names must be non-empty"):
        _normalise_metrics({"": 1.0})


def test_normalise_metrics_rejects_non_finite_value() -> None:
    """Metric values must be finite."""
    with pytest.raises(ValueError, match="must be finite"):
        _normalise_metrics({"latency": float("inf")})


def test_jitter_percentiles_numpy_empty_is_zero() -> None:
    """An empty jitter array yields zero percentiles."""
    assert _jitter_percentiles_numpy(np.array([], dtype=np.float64)) == (0.0, 0.0, 0.0, 0.0)


def test_public_reports_recover_when_optional_native_kernels_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public streaming and batch reports survive a rejected native dispatch."""
    native_present = False
    calls = {"percentiles": 0, "summary": 0}
    try:
        import scpn_quantum_engine as engine

        if hasattr(engine, "sub_us_jitter_percentiles") and hasattr(
            engine, "sub_us_tracker_summary"
        ):
            native_present = True

            def reject_percentiles(*_args: object, **_kwargs: object) -> object:
                calls["percentiles"] += 1
                raise ValueError("rejected percentile input")

            def reject_summary(*_args: object, **_kwargs: object) -> object:
                calls["summary"] += 1
                raise ValueError("rejected summary input")

            monkeypatch.setattr(engine, "sub_us_jitter_percentiles", reject_percentiles)
            monkeypatch.setattr(engine, "sub_us_tracker_summary", reject_summary)
    except ImportError:
        pass

    tracker = SubMicrosecondTracker(target_rate_hz=1_000_000)
    samples = (
        CycleSample(cycle_id=0, start_ns=0, end_ns=400, deadline_ns=500),
        CycleSample(cycle_id=1, start_ns=1_100, end_ns=1_600, deadline_ns=1_500),
        CycleSample(cycle_id=2, start_ns=1_900, end_ns=2_500, deadline_ns=2_600),
    )
    for sample in samples:
        tracker.record(sample)
    streamed = tracker.report()
    batched = summarise_cycle_samples(
        [sample.start_ns for sample in samples],
        [sample.end_ns for sample in samples],
        [sample.deadline_ns for sample in samples],
        target_rate_hz=1_000_000,
    )

    assert streamed == batched
    assert streamed.jitter_p50_ns == 100.0
    assert streamed.jitter_p95_ns == 190.0
    assert streamed.jitter_p99_ns == 198.0
    assert streamed.jitter_max_ns == 200.0
    assert streamed.deadline_misses == 1
    if native_present:
        assert calls == {"percentiles": 1, "summary": 1}


def test_public_reports_use_numpy_when_native_exports_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial optional engine install must retain public report semantics."""
    try:
        import scpn_quantum_engine as engine

        monkeypatch.delattr(engine, "sub_us_jitter_percentiles", raising=False)
        monkeypatch.delattr(engine, "sub_us_tracker_summary", raising=False)
    except ImportError:
        pass

    tracker = SubMicrosecondTracker(target_rate_hz=1_000_000)
    samples = (
        CycleSample(cycle_id=0, start_ns=0, end_ns=400, deadline_ns=500),
        CycleSample(cycle_id=1, start_ns=1_100, end_ns=1_600, deadline_ns=1_500),
        CycleSample(cycle_id=2, start_ns=1_900, end_ns=2_500, deadline_ns=2_600),
    )
    for sample in samples:
        tracker.record(sample)
    streamed = tracker.report()
    batched = summarise_cycle_samples(
        [sample.start_ns for sample in samples],
        [sample.end_ns for sample in samples],
        [sample.deadline_ns for sample in samples],
        target_rate_hz=1_000_000,
    )

    assert streamed == batched
    assert streamed.cycles_observed == 3
    assert streamed.jitter_p99_ns == 198.0
    assert streamed.deadline_misses == 1
