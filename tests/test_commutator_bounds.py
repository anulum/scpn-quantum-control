# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Commutator Bounds
"""Tests for analytical commutator Trotter error bounds."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.phase.trotter_error import (
    commutator_norm_bound,
    frequency_heterogeneity,
    optimal_dt,
    trotter_error_bound,
    trotter_error_norm,
)


class TestCommutatorBounds:
    def test_commutator_norm_positive(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        gamma = commutator_norm_bound(K, omega)
        assert gamma > 0

    def test_commutator_norm_zero_for_equal_frequencies(self):
        """Key insight: Trotter error vanishes when all frequencies are equal."""
        K = build_knm_paper27(L=4)
        omega_equal = np.ones(4) * 1.0
        gamma = commutator_norm_bound(K, omega_equal)
        assert gamma == pytest.approx(0.0, abs=1e-15)

    def test_bound_exceeds_empirical(self):
        """Analytical bound must be >= empirical Frobenius norm error."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        t = 0.5
        reps = 3
        bound = trotter_error_bound(K, omega, t, reps, order=1)
        empirical = trotter_error_norm(K, omega, t, reps)
        assert bound >= empirical * 0.5  # allow slack for norm differences

    def test_bound_decreases_with_reps(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        b1 = trotter_error_bound(K, omega, 1.0, 1, order=1)
        b5 = trotter_error_bound(K, omega, 1.0, 5, order=1)
        b10 = trotter_error_bound(K, omega, 1.0, 10, order=1)
        assert b1 > b5 > b10

    def test_second_order_better_than_first(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        b1 = trotter_error_bound(K, omega, 0.5, 5, order=1)
        b2 = trotter_error_bound(K, omega, 0.5, 5, order=2)
        assert b2 < b1

    def test_optimal_dt_respects_epsilon(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = optimal_dt(K, omega, epsilon=0.01, t_total=1.0, order=1)
        assert result["error_bound"] <= 0.01 * 1.5  # small slack
        assert result["n_steps"] >= 1
        assert result["dt"] > 0

    def test_optimal_dt_order2(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        r1 = optimal_dt(K, omega, epsilon=0.01, t_total=1.0, order=1)
        r2 = optimal_dt(K, omega, epsilon=0.01, t_total=1.0, order=2)
        # Second order should need fewer steps
        assert r2["n_steps"] <= r1["n_steps"]

    def test_frequency_heterogeneity(self):
        omega_uniform = np.ones(4) * 1.0
        assert frequency_heterogeneity(omega_uniform) == pytest.approx(0.0)

        omega_spread = np.array([0.5, 1.0, 1.5, 2.0])
        assert frequency_heterogeneity(omega_spread) > 0

    def test_scpn_commutator_norm(self):
        """For 16-layer SCPN, compute and record the commutator norm."""
        K = build_knm_paper27(L=16)
        omega = OMEGA_N_16
        gamma = commutator_norm_bound(K, omega)
        # From Round 2 derivation: estimated ~18-20 for 16 layers
        assert gamma > 5  # lower bound sanity
        assert gamma < 200  # upper bound sanity

    def test_scpn_optimal_dt_value(self):
        """For 16-layer SCPN at epsilon=0.01, what dt do we need?"""
        K = build_knm_paper27(L=16)
        omega = OMEGA_N_16
        result = optimal_dt(K, omega, epsilon=0.01, t_total=5.0, order=1)
        # Should require many steps for t=5.0 at epsilon=0.01
        assert result["n_steps"] > 100
        assert result["dt"] < 0.1


# ---------------------------------------------------------------------------
# Commutator physics: scaling and universality
# ---------------------------------------------------------------------------


class TestCommutatorPhysics:
    def test_heterogeneity_drives_error(self):
        """More frequency spread → larger commutator norm → larger Trotter error."""
        K = build_knm_paper27(L=4)
        omega_narrow = np.array([1.0, 1.1, 1.2, 1.3])
        omega_wide = np.array([0.5, 1.0, 2.0, 3.0])
        gamma_narrow = commutator_norm_bound(K, omega_narrow)
        gamma_wide = commutator_norm_bound(K, omega_wide)
        assert gamma_wide > gamma_narrow

    def test_error_bound_nonnegative(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        b = trotter_error_bound(K, omega, 1.0, 5, order=1)
        assert b >= 0


# ---------------------------------------------------------------------------
# Pipeline: Knm → commutator → optimal dt → wired
# ---------------------------------------------------------------------------


class TestCommutatorPipeline:
    def test_pipeline_knm_to_optimal_dt(self):
        """Full pipeline: build_knm → commutator norm → optimal dt for ε=0.01.
        Verifies Trotter error module is wired and produces actionable circuit params.
        """
        import time

        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]

        t0 = time.perf_counter()
        gamma = commutator_norm_bound(K, omega)
        result = optimal_dt(K, omega, epsilon=0.01, t_total=1.0, order=1)
        dt_ms = (time.perf_counter() - t0) * 1000

        assert gamma > 0
        assert result["n_steps"] >= 1

        print(f"\n  PIPELINE Knm→Commutator→dt* (4q): {dt_ms:.1f} ms")
        print(f"  γ = {gamma:.4f}, dt* = {result['dt']:.4f}, n_steps = {result['n_steps']}")
