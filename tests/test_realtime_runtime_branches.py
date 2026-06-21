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
    MonotonicRealtimeClock,
    RealtimeRunResult,
    RealtimeRuntimeConfig,
    RealtimeSLAConfig,
    RealtimeTickRecord,
    VirtualRealtimeClock,
    _jitter_percentiles,
    _jitter_percentiles_numpy,
    _normalise_metrics,
    _sub_us_summary,
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


def test_jitter_percentiles_falls_back_to_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the native percentile kernel errors, the numpy path is used."""
    jitters = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    try:
        import scpn_quantum_engine as engine

        if hasattr(engine, "sub_us_jitter_percentiles"):

            def _raise(*_args: object, **_kwargs: object) -> object:
                raise ValueError("forced numpy fallback")

            monkeypatch.setattr(engine, "sub_us_jitter_percentiles", _raise)
    except ImportError:
        pass
    assert _jitter_percentiles(jitters) == _jitter_percentiles_numpy(jitters)


def test_sub_us_summary_falls_back_to_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the native summary kernel errors, the numpy path is used."""
    start = np.array([0, 1000, 2000], dtype=np.int64)
    end = np.array([500, 1500, 2500], dtype=np.int64)
    deadline = np.array([1000, 2000, 3000], dtype=np.int64)
    try:
        import scpn_quantum_engine as engine

        if hasattr(engine, "sub_us_tracker_summary"):

            def _raise(*_args: object, **_kwargs: object) -> object:
                raise ValueError("forced numpy fallback")

            monkeypatch.setattr(engine, "sub_us_tracker_summary", _raise)
    except ImportError:
        pass
    report = summarise_cycle_samples(start, end, deadline, target_rate_hz=1000)
    assert report.cycles_observed == 3
    assert _sub_us_summary(start, end, deadline, 1.0e6)[5] == 3
