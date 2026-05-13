# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Trotter Error
"""Tests for Trotter error analysis — multi-angle coverage."""

import numpy as np
import pytest

import scpn_quantum_control.phase.trotter_error as trotter_module
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.phase.trotter_error import trotter_error_norm, trotter_error_sweep


@pytest.fixture
def small_system():
    n = 3
    return build_knm_paper27(L=n), OMEGA_N_16[:n]


def test_error_at_t_zero(small_system):
    K, omega = small_system
    err = trotter_error_norm(K, omega, t=0.0, reps=1)
    assert err < 1e-10


def test_error_norm_rejects_dense_budget_before_hamiltonian_allocation(monkeypatch):
    K = build_knm_paper27(L=10)
    omega = OMEGA_N_16[:10]

    def fail_if_dense_hamiltonian_is_requested(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("dense Hamiltonian allocation happened before budget gate")

    monkeypatch.setattr(
        trotter_module, "knm_to_dense_matrix", fail_if_dense_hamiltonian_is_requested
    )

    with pytest.raises(DenseAllocationError, match="Trotter dense"):
        trotter_error_norm(K, omega, t=0.1, reps=1, max_dense_gib=1e-12)


def test_error_norm_passes_dense_budget_to_bridge(monkeypatch):
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    seen_budgets: list[float | None] = []

    def fake_dense_matrix(K_arg, omega_arg, **kwargs):  # noqa: ARG001
        seen_budgets.append(kwargs.get("max_dense_gib"))
        return np.zeros((4, 4), dtype=complex)

    monkeypatch.setattr(trotter_module, "knm_to_dense_matrix", fake_dense_matrix)

    trotter_error_norm(K, omega, t=0.1, reps=1, max_dense_gib=0.25)

    assert seen_budgets == [0.25]


def test_error_decreases_with_reps(small_system):
    K, omega = small_system
    err_1 = trotter_error_norm(K, omega, t=1.0, reps=1)
    err_4 = trotter_error_norm(K, omega, t=1.0, reps=4)
    assert err_4 < err_1


def test_error_increases_with_time(small_system):
    K, omega = small_system
    err_short = trotter_error_norm(K, omega, t=0.05, reps=2)
    err_long = trotter_error_norm(K, omega, t=0.5, reps=2)
    assert err_long > err_short


def test_raises_for_large_n():
    K = build_knm_paper27(L=11)
    omega = OMEGA_N_16[:11]
    with pytest.raises(ValueError, match="too large"):
        trotter_error_norm(K, omega, t=0.1, reps=1)


def test_trotter_convergence_rate(small_system):
    """Lie-Trotter error scales as O(t^2) at fixed reps.

    Halving t should reduce error by ~4x. Accept >2x for finite-size effects.
    """
    K, omega = small_system
    err_coarse = trotter_error_norm(K, omega, t=1.0, reps=1)
    err_fine = trotter_error_norm(K, omega, t=0.5, reps=1)
    ratio = err_coarse / max(err_fine, 1e-15)
    assert ratio > 2.0, f"expected ~4x improvement, got {ratio:.1f}x"


def test_sweep_returns_2d(small_system):
    K, omega = small_system
    result = trotter_error_sweep(K, omega, t_values=[0.05, 0.1], reps_values=[1, 2])
    assert len(result["errors"]) == 2
    assert len(result["errors"][0]) == 2
    assert all(e >= 0 for row in result["errors"] for e in row)


def test_sweep_propagates_dense_budget(monkeypatch, small_system):
    K, omega = small_system
    seen: list[float | None] = []

    def fake_error_norm(K_arg, omega_arg, t, reps, *, max_dense_gib=None):  # noqa: ARG001
        seen.append(max_dense_gib)
        return 0.0

    monkeypatch.setattr(trotter_module, "trotter_error_norm", fake_error_norm)

    trotter_error_sweep(K, omega, t_values=[0.05, 0.1], reps_values=[1, 2], max_dense_gib=0.5)

    assert seen == [0.5, 0.5, 0.5, 0.5]


def test_error_positive_at_nonzero_time(small_system):
    K, omega = small_system
    err = trotter_error_norm(K, omega, t=0.5, reps=1)
    assert err > 0


def test_sweep_t_values_in_result(small_system):
    K, omega = small_system
    result = trotter_error_sweep(K, omega, t_values=[0.1, 0.2, 0.3], reps_values=[1, 2])
    assert len(result["errors"]) == 3
    assert len(result["errors"][0]) == 2


@pytest.mark.parametrize("n", [2, 3, 4])
def test_error_norm_various_sizes(n):
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    err = trotter_error_norm(K, omega, t=0.1, reps=2)
    assert np.isfinite(err)
    assert err >= 0


def test_error_norm_high_reps_small(small_system):
    """At very high reps, error should be very small."""
    K, omega = small_system
    err = trotter_error_norm(K, omega, t=0.1, reps=20)
    assert err < 0.01


# ---------------------------------------------------------------------------
# Trotter error physics: scaling laws
# ---------------------------------------------------------------------------


def test_error_scales_quadratically_with_t(small_system):
    """First-order Trotter: ε ~ O(t²/n) at fixed reps."""
    K, omega = small_system
    err_t1 = trotter_error_norm(K, omega, t=0.1, reps=1)
    err_t2 = trotter_error_norm(K, omega, t=0.2, reps=1)
    # Doubling t should roughly quadruple error
    ratio = err_t2 / max(err_t1, 1e-15)
    assert ratio > 2.0


# ---------------------------------------------------------------------------
# Pipeline: Knm → Trotter error → sweep → wired
# ---------------------------------------------------------------------------


def test_pipeline_knm_to_trotter_sweep(small_system):
    """Full pipeline: Knm → error sweep → 2D error map.
    Verifies Trotter error module is wired and produces actionable data.
    """
    import time

    K, omega = small_system

    t0 = time.perf_counter()
    result = trotter_error_sweep(K, omega, t_values=[0.05, 0.1, 0.2], reps_values=[1, 2, 5])
    dt = (time.perf_counter() - t0) * 1000

    assert len(result["errors"]) == 3
    assert len(result["errors"][0]) == 3
    # Error should decrease with more reps
    for row in result["errors"]:
        assert row[0] >= row[-1] - 1e-10  # reps=1 ≥ reps=5

    print(f"\n  PIPELINE Knm→TrotterSweep (3q, 3×3): {dt:.1f} ms")
    print(f"  ε(t=0.05,reps=1)={result['errors'][0][0]:.6f}")
    print(f"  ε(t=0.2,reps=5)={result['errors'][2][2]:.6f}")
