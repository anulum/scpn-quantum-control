# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the sub-microsecond realtime tracker
"""Tests for control/realtime_runtime.py sub-microsecond tracker (QUA-C.6)."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scpn_quantum_control.control.realtime_runtime import (
    CycleSample,
    SubMicrosecondTracker,
    _jitter_percentiles_numpy,
    _sub_us_summary,
    _sub_us_summary_numpy,
    summarise_cycle_samples,
)

try:
    import scpn_quantum_engine as _engine

    _HAS_RUST = hasattr(_engine, "sub_us_jitter_percentiles") and hasattr(
        _engine, "sub_us_tracker_summary"
    )
except ImportError:  # pragma: no cover - engine optional
    _engine = None
    _HAS_RUST = False

_RATE = 100_000  # 10_000 ns target period
_PERIOD = 10_000.0


def _cadence(jitters_ns, *, durations_ns=3_000, deadline_offset_ns=5_000):
    """Build a CycleSample list whose inter-start deviations equal ``jitters_ns``."""
    samples = []
    start = 0
    for idx, jitter in enumerate([0, *jitters_ns]):
        start += int(_PERIOD) + int(jitter) if idx else 0
        dur = durations_ns[idx] if isinstance(durations_ns, list) else durations_ns
        dl = (
            deadline_offset_ns[idx] if isinstance(deadline_offset_ns, list) else deadline_offset_ns
        )
        samples.append(
            CycleSample(cycle_id=idx, start_ns=start, end_ns=start + dur, deadline_ns=start + dl)
        )
    return samples


# --------------------------------------------------------------------------- #
# CycleSample contract
# --------------------------------------------------------------------------- #
def test_cycle_sample_valid_properties():
    s = CycleSample(cycle_id=2, start_ns=1_000, end_ns=4_000, deadline_ns=6_000)
    assert s.duration_ns == 3_000
    assert s.deadline_missed is False
    late = CycleSample(cycle_id=3, start_ns=1_000, end_ns=7_000, deadline_ns=6_000)
    assert late.deadline_missed is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cycle_id": 1.0, "start_ns": 0, "end_ns": 1, "deadline_ns": 1},
        {"cycle_id": True, "start_ns": 0, "end_ns": 1, "deadline_ns": 1},
        {"cycle_id": 0, "start_ns": 0.0, "end_ns": 1, "deadline_ns": 1},
    ],
)
def test_cycle_sample_rejects_non_int(kwargs):
    with pytest.raises(TypeError):
        CycleSample(**kwargs)


def test_cycle_sample_rejects_inverted_intervals():
    with pytest.raises(ValueError, match="end_ns"):
        CycleSample(cycle_id=0, start_ns=10, end_ns=5, deadline_ns=20)
    with pytest.raises(ValueError, match="deadline_ns"):
        CycleSample(cycle_id=0, start_ns=10, end_ns=12, deadline_ns=5)


# --------------------------------------------------------------------------- #
# Tracker construction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_rate_hz": 0},
        {"target_rate_hz": -1},
        {"ring_buffer_capacity": 0},
    ],
)
def test_tracker_rejects_non_positive(kwargs):
    with pytest.raises(ValueError):
        SubMicrosecondTracker(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"target_rate_hz": 1.0}, {"target_rate_hz": True}, {"ring_buffer_capacity": 4.0}],
)
def test_tracker_rejects_non_int(kwargs):
    with pytest.raises(TypeError):
        SubMicrosecondTracker(**kwargs)


def test_tracker_target_period():
    assert SubMicrosecondTracker(target_rate_hz=_RATE).target_period_ns == _PERIOD


# --------------------------------------------------------------------------- #
# Recording and reporting
# --------------------------------------------------------------------------- #
def test_perfect_cadence_has_zero_jitter():
    tracker = SubMicrosecondTracker(target_rate_hz=_RATE)
    for sample in _cadence([0, 0, 0, 0]):
        tracker.record(sample)
    report = tracker.report()
    assert report.cycles_observed == 5
    assert report.deadline_misses == 0
    assert report.jitter_p50_ns == 0.0
    assert report.jitter_p95_ns == 0.0
    assert report.jitter_max_ns == 0.0


def test_known_jitter_matches_numpy_quantile():
    jitters = [200, 0, 1_500, 50, 900, 0, 3_000]
    tracker = SubMicrosecondTracker(target_rate_hz=_RATE)
    for sample in _cadence(jitters):
        tracker.record(sample)
    report = tracker.report()
    observed = np.asarray([0.0, *jitters], dtype=np.float64)
    assert report.jitter_p50_ns == float(np.quantile(observed, 0.50, method="linear"))
    assert report.jitter_p95_ns == float(np.quantile(observed, 0.95, method="linear"))
    assert report.jitter_p99_ns == float(np.quantile(observed, 0.99, method="linear"))
    assert report.jitter_max_ns == float(observed.max())


def test_deadline_miss_count():
    tracker = SubMicrosecondTracker(target_rate_hz=_RATE)
    # durations: cycles 1 and 3 finish after their 5_000 ns deadline offset
    durations = [3_000, 6_000, 3_000, 7_000, 3_000]
    for sample in _cadence([0, 0, 0, 0], durations_ns=durations):
        tracker.record(sample)
    assert tracker.report().deadline_misses == 2


def test_report_before_record_raises():
    with pytest.raises(ValueError, match="no cycles recorded"):
        SubMicrosecondTracker().report()


def test_record_rejects_non_sample():
    with pytest.raises(TypeError):
        SubMicrosecondTracker().record((0, 1, 2, 3))


def test_reset_clears_state():
    tracker = SubMicrosecondTracker(target_rate_hz=_RATE)
    for sample in _cadence([500, 500]):
        tracker.record(sample)
    tracker.reset()
    assert tracker.cycles_observed == 0
    with pytest.raises(ValueError):
        tracker.report()
    tracker.record(CycleSample(cycle_id=0, start_ns=0, end_ns=1_000, deadline_ns=2_000))
    assert tracker.report().cycles_observed == 1


def test_ring_buffer_overwrite_keeps_exact_counters():
    tracker = SubMicrosecondTracker(target_rate_hz=_RATE, ring_buffer_capacity=4)
    durations = [3_000] * 5 + [9_000] * 5  # last five all miss the 5_000 ns deadline
    for sample in _cadence([0] * 9, durations_ns=durations):
        tracker.record(sample)
    report = tracker.report()
    assert report.cycles_observed == 10  # exact total across overwrites
    assert report.deadline_misses == 5  # exact total across overwrites
    assert report.window_size == 4  # bounded jitter window


# --------------------------------------------------------------------------- #
# Batch summary path
# --------------------------------------------------------------------------- #
def test_summarise_matches_tracker():
    jitters = [120, 800, 0, 4_000, 60, 900]
    # _cadence prepends a zero-jitter first cycle, so durations spans len+1 cycles
    durations = [3_000, 6_000, 3_000, 3_000, 8_000, 3_000, 3_000]
    samples = _cadence(jitters, durations_ns=durations)
    tracker = SubMicrosecondTracker(target_rate_hz=_RATE)
    for sample in samples:
        tracker.record(sample)
    expected = tracker.report()
    start = np.asarray([s.start_ns for s in samples], dtype=np.int64)
    end = np.asarray([s.end_ns for s in samples], dtype=np.int64)
    deadline = np.asarray([s.deadline_ns for s in samples], dtype=np.int64)
    got = summarise_cycle_samples(start, end, deadline, target_rate_hz=_RATE)
    assert got.jitter_p50_ns == expected.jitter_p50_ns
    assert got.jitter_p95_ns == expected.jitter_p95_ns
    assert got.jitter_p99_ns == expected.jitter_p99_ns
    assert got.jitter_max_ns == expected.jitter_max_ns
    assert got.deadline_misses == expected.deadline_misses
    assert got.cycles_observed == expected.cycles_observed


def test_summarise_single_cycle():
    report = summarise_cycle_samples([0], [3_000], [5_000], target_rate_hz=_RATE)
    assert report.cycles_observed == 1
    assert report.jitter_max_ns == 0.0
    assert report.deadline_misses == 0


@pytest.mark.parametrize(
    "start,end,deadline,err",
    [
        ([0, 1], [1], [1, 2], "equal length"),
        ([], [], [], "no cycles"),
        ([0, 5], [1, 4], [9, 9], "end_ns"),
        ([0, 5], [1, 6], [9, 4], "deadline_ns"),
    ],
)
def test_summarise_validation(start, end, deadline, err):
    with pytest.raises(ValueError, match=err):
        summarise_cycle_samples(start, end, deadline)


def test_summarise_rejects_2d():
    with pytest.raises(ValueError, match="one-dimensional"):
        summarise_cycle_samples(
            np.zeros((2, 2), dtype=np.int64),
            np.ones((2, 2), dtype=np.int64),
            np.ones((2, 2), dtype=np.int64),
        )


def test_summarise_rejects_invalid_rate():
    with pytest.raises((TypeError, ValueError)):
        summarise_cycle_samples([0], [1], [1], target_rate_hz=0)


# --------------------------------------------------------------------------- #
# Property-based determinism
# --------------------------------------------------------------------------- #
@settings(max_examples=60, deadline=None)
@given(
    intervals=st.lists(st.integers(min_value=1, max_value=40_000), min_size=1, max_size=64),
)
def test_summary_is_deterministic_and_matches_numpy(intervals):
    start = np.cumsum(np.asarray(intervals, dtype=np.int64))
    end = start + 2_000
    deadline = start + 5_000
    first = summarise_cycle_samples(start, end, deadline, target_rate_hz=_RATE)
    second = summarise_cycle_samples(start, end, deadline, target_rate_hz=_RATE)
    assert first == second
    ref = _sub_us_summary_numpy(start, end, deadline, _PERIOD)
    assert first.jitter_p50_ns == ref[0]
    assert first.jitter_p99_ns == ref[2]
    assert first.jitter_max_ns == ref[3]
    assert first.deadline_misses == ref[4]


# --------------------------------------------------------------------------- #
# Python ↔ Rust parity
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_RUST, reason="scpn_quantum_engine sub-µs kernels not built")
def test_rust_python_parity_percentiles():
    rng = np.random.default_rng(11)
    for _ in range(50):
        size = int(rng.integers(1, 256))
        jitters = np.abs(rng.normal(0.0, 750.0, size=size)).astype(np.float64)
        rust = _engine.sub_us_jitter_percentiles(np.ascontiguousarray(jitters))
        numpy_ref = _jitter_percentiles_numpy(jitters)
        assert rust == numpy_ref  # bit-true


@pytest.mark.skipif(not _HAS_RUST, reason="scpn_quantum_engine sub-µs kernels not built")
def test_rust_python_parity_summary():
    rng = np.random.default_rng(13)
    for _ in range(50):
        size = int(rng.integers(1, 256))
        start = np.cumsum(rng.integers(6_000, 14_000, size=size)).astype(np.int64)
        end = start + rng.integers(1_000, 7_000, size=size).astype(np.int64)
        deadline = start + 5_000
        rust = _engine.sub_us_tracker_summary(
            np.ascontiguousarray(start),
            np.ascontiguousarray(end),
            np.ascontiguousarray(deadline),
            _PERIOD,
        )
        numpy_ref = _sub_us_summary_numpy(start, end, deadline, _PERIOD)
        assert rust[:4] == numpy_ref[:4]  # bit-true percentiles + max
        assert rust[4] == numpy_ref[4]
        assert rust[5] == numpy_ref[5]


@pytest.mark.skipif(not _HAS_RUST, reason="scpn_quantum_engine sub-µs kernels not built")
def test_dispatch_prefers_rust():
    # _sub_us_summary routes through the Rust kernel when present
    start = np.array([0, 10_000, 21_000], dtype=np.int64)
    end = start + 2_000
    deadline = start + 5_000
    routed = _sub_us_summary(start, end, deadline, _PERIOD)
    direct = _engine.sub_us_tracker_summary(
        np.ascontiguousarray(start),
        np.ascontiguousarray(end),
        np.ascontiguousarray(deadline),
        _PERIOD,
    )
    assert routed[:4] == direct[:4]
    assert routed[4] == direct[4] and routed[5] == direct[5]
