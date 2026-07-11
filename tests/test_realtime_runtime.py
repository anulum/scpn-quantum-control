# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Realtime Runtime Tests
"""Runtime and package-export tests for deterministic realtime control."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

import scpn_quantum_control as scpn
from scpn_quantum_control.control.realtime_runtime import (
    RealtimeRuntimeConfig,
    RealtimeSLAConfig,
    VirtualRealtimeClock,
    enforce_realtime_sla,
    evaluate_realtime_sla,
    run_realtime_control_loop,
)


def test_realtime_control_loop_records_deadline_jitter_and_misses() -> None:
    """Realtime runtime should account for deterministic deadline misses."""

    clock = VirtualRealtimeClock()
    config = RealtimeRuntimeConfig(
        sample_period_s=0.01,
        deadline_s=0.005,
        jitter_budget_s=0.002,
        max_missed_deadlines=2,
    )

    durations = [0.001, 0.006, 0.004]

    def step(index: int) -> Mapping[str, float]:
        clock.advance(durations[index])
        return {"index": float(index)}

    result = run_realtime_control_loop(3, step, config=config, clock=clock)

    assert result.completed
    assert result.missed_deadlines == 1
    assert result.max_latency_s == pytest.approx(0.006)
    assert result.records[1].deadline_missed
    assert result.records[2].jitter_s == pytest.approx(0.0)


def test_realtime_control_loop_fails_when_deadline_budget_is_exceeded() -> None:
    """Realtime runtime should abort instead of hiding repeated deadline misses."""

    clock = VirtualRealtimeClock()
    config = RealtimeRuntimeConfig(
        sample_period_s=0.01,
        deadline_s=0.002,
        max_missed_deadlines=0,
    )

    def step(_index: int) -> Mapping[str, float]:
        clock.advance(0.003)
        return {}

    with pytest.raises(RuntimeError, match="deadline"):
        run_realtime_control_loop(1, step, config=config, clock=clock)


def test_realtime_sla_accepts_sub_millisecond_loop() -> None:
    """SLA gate should pass when all observed latency stays within <1 ms budget."""

    clock = VirtualRealtimeClock()
    config = RealtimeRuntimeConfig(
        sample_period_s=0.0012,
        deadline_s=0.00095,
        jitter_budget_s=0.00015,
        max_missed_deadlines=0,
    )
    durations = [0.00052, 0.00061, 0.00074, 0.00068, 0.00057]

    def step(index: int) -> Mapping[str, float]:
        clock.advance(durations[index])
        return {"index": float(index)}

    result = run_realtime_control_loop(len(durations), step, config=config, clock=clock)
    sla = RealtimeSLAConfig(
        max_latency_s=0.001,
        max_jitter_s=0.00025,
        p95_latency_s=0.001,
        p99_latency_s=0.001,
        max_deadline_miss_rate=0.0,
    )
    report = evaluate_realtime_sla(result, sla=sla)

    assert report.compliant is True
    assert report.breach_reasons == ()
    assert report.observed_max_latency_s <= 0.001
    assert enforce_realtime_sla(result, sla=sla) == report


def test_realtime_sla_rejects_latency_breach() -> None:
    """SLA gate should fail closed on millisecond-class latency overshoot."""

    clock = VirtualRealtimeClock()
    config = RealtimeRuntimeConfig(
        sample_period_s=0.0014,
        deadline_s=0.0012,
        jitter_budget_s=0.00020,
        max_missed_deadlines=5,
    )
    durations = [0.00055, 0.00145, 0.00066]

    def step(index: int) -> Mapping[str, float]:
        clock.advance(durations[index])
        return {"index": float(index)}

    result = run_realtime_control_loop(len(durations), step, config=config, clock=clock)
    sla = RealtimeSLAConfig(
        max_latency_s=0.001,
        max_jitter_s=0.00030,
        p95_latency_s=0.001,
        p99_latency_s=0.001,
        max_deadline_miss_rate=0.5,
    )
    report = evaluate_realtime_sla(result, sla=sla)

    assert report.compliant is False
    assert any("max latency" in item for item in report.breach_reasons)
    with pytest.raises(RuntimeError, match="max latency"):
        enforce_realtime_sla(result, sla=sla)


def test_realtime_runtime_api_exported_from_package_root() -> None:
    """Realtime runtime surfaces should remain stable package-root imports."""

    assert scpn.RealtimeRuntimeConfig is RealtimeRuntimeConfig
    assert scpn.RealtimeSLAConfig is RealtimeSLAConfig
    assert scpn.run_realtime_control_loop is run_realtime_control_loop
    assert scpn.evaluate_realtime_sla is evaluate_realtime_sla
    assert scpn.enforce_realtime_sla is enforce_realtime_sla
