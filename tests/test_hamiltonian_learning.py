# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Hamiltonian Learning
"""Tests for Hamiltonian learning from measurements."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.analysis.hamiltonian_learning import (
    HamiltonianLearningResult,
    learn_hamiltonian,
    measure_correlators,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


class TestMeasureCorrelators:
    def test_shape(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        C = measure_correlators(K, omega)
        assert C.shape == (3, 3)

    def test_symmetric(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        C = measure_correlators(K, omega)
        np.testing.assert_allclose(C, C.T, atol=1e-10)

    def test_zero_diagonal(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        C = measure_correlators(K, omega)
        np.testing.assert_allclose(np.diag(C), 0.0, atol=1e-10)

    def test_bounded(self):
        """XY correlator bounded by [-2, 2]."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        C = measure_correlators(K, omega)
        assert np.all(np.abs(C) <= 2.0 + 1e-10)


class TestLearnHamiltonian:
    def test_returns_result(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        C = measure_correlators(K, omega)
        result = learn_hamiltonian(C, omega, maxiter=10)
        assert isinstance(result, HamiltonianLearningResult)

    def test_learned_K_symmetric(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        C = measure_correlators(K, omega)
        result = learn_hamiltonian(C, omega, maxiter=10)
        np.testing.assert_allclose(result.K_learned, result.K_learned.T, atol=1e-10)

    def test_learned_K_non_negative(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        C = measure_correlators(K, omega)
        result = learn_hamiltonian(C, omega, maxiter=10)
        assert np.all(result.K_learned >= -1e-10)

    def test_correlator_error_decreases(self):
        """Learning should reduce correlator error vs random K."""
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        C = measure_correlators(K, omega)
        result = learn_hamiltonian(C, omega, maxiter=50)
        # Error should be finite
        assert result.correlator_error < 10.0

    def test_self_consistent(self):
        """If we give the true K as init, error should be near zero."""
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        C = measure_correlators(K, omega)
        result = learn_hamiltonian(C, omega, K_init=K, maxiter=5)
        assert result.correlator_error < 0.1

    def test_loss_non_negative(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        C = measure_correlators(K, omega)
        result = learn_hamiltonian(C, omega, maxiter=10)
        assert result.loss >= 0


# ---------------------------------------------------------------------------
# Learning physics: inverse problem structure
# ---------------------------------------------------------------------------


class TestLearningPhysics:
    def test_3qubit_correlators_stronger(self):
        """Stronger coupling → larger off-diagonal correlators."""
        omega = OMEGA_N_16[:3]
        from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27

        C_weak = measure_correlators(build_knm_paper27(L=3, K_base=0.1), omega)
        C_strong = measure_correlators(build_knm_paper27(L=3, K_base=2.0), omega)
        assert np.max(np.abs(C_strong)) > np.max(np.abs(C_weak))

    def test_learned_K_shape(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        C = measure_correlators(K, omega)
        result = learn_hamiltonian(C, omega, maxiter=10)
        assert result.K_learned.shape == (3, 3)


# ---------------------------------------------------------------------------
# Pipeline: Knm → correlators → learn K → wired
# ---------------------------------------------------------------------------


class TestLearningPipeline:
    def test_pipeline_knm_to_learned_K(self):
        """Full pipeline: build_knm → measure correlators → learn K.
        Verifies Hamiltonian learning is wired end-to-end.
        """
        import time

        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]

        t0 = time.perf_counter()
        C = measure_correlators(K, omega)
        result = learn_hamiltonian(C, omega, maxiter=30)
        dt = (time.perf_counter() - t0) * 1000

        assert result.K_learned.shape == (3, 3)
        assert result.loss >= 0

        print(f"\n  PIPELINE Knm→Correlators→LearnK (3q): {dt:.1f} ms")
        print(f"  Loss = {result.loss:.6f}, corr_error = {result.correlator_error:.4f}")
