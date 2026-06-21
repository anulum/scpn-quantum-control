# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Wall-clock latency-path tests for closed-loop analysis
"""Wall-clock measurement and budget-blocker tests for closed-loop latency.

Covers the live ``time.perf_counter_ns`` measurement path (taken when no
observed samples are replayed), the minimum-round guard, the observed-sample
validation guard, and the percentile and total-latency budget blockers.
"""

from __future__ import annotations

import warnings

import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import build_kuramoto_ring
from scpn_quantum_control.control.closed_loop_analysis import (
    ClosedLoopLatencyBudget,
    measure_closed_loop_latency_budget,
)
from scpn_quantum_control.control.realtime_feedback import (
    RealtimeFeedbackConfig,
    RealtimeSyncFeedbackController,
)


def _controller() -> RealtimeSyncFeedbackController:
    """Build a small ring controller for closed-loop latency measurement."""
    coupling, omega = build_kuramoto_ring(4, coupling=0.6, rng_seed=0)
    return RealtimeSyncFeedbackController(
        coupling, omega, config=RealtimeFeedbackConfig(target_r=0.6)
    )


def test_latency_budget_requires_two_rounds() -> None:
    """A single round cannot yield a latency verdict."""
    with pytest.raises(ValueError, match="at least two"):
        measure_closed_loop_latency_budget(_controller(), 1)


def test_latency_budget_rejects_non_finite_observed_samples() -> None:
    """Replayed observed samples must be finite and non-negative."""
    with pytest.raises(ValueError, match="finite non-negative samples"):
        measure_closed_loop_latency_budget(
            _controller(), 2, observed_round_latencies_s=(-1.0, 0.5)
        )


def test_wall_clock_measurement_path_flags_all_budget_blockers() -> None:
    """Live wall-clock measurement breaches a sub-nanosecond budget on every axis."""
    budget = ClosedLoopLatencyBudget(
        max_round_latency_s=1e-12,
        p95_round_latency_s=1e-12,
        p99_round_latency_s=1e-12,
        max_total_latency_s=1e-12,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = measure_closed_loop_latency_budget(_controller(), 4, budget=budget, seed=0)
    assert report.clock_source == "time.perf_counter_ns"
    assert report.samples == 4
    assert not report.passes
    blockers = " ".join(report.blockers)
    assert "p95 round latency" in blockers
    assert "p99 round latency" in blockers
    assert "total latency" in blockers
