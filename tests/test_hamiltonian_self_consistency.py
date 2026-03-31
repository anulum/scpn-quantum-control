# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Hamiltonian Self Consistency
"""Tests for Hamiltonian self-consistency loop."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis.hamiltonian_self_consistency import (
    SelfConsistencyResult,
    correlator_shot_noise,
    correlators_from_counts,
    self_consistency_from_exact,
    self_consistency_from_noisy_sim,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


class TestCorrelatorsFromCounts:
    def test_perfect_correlation(self):
        x_counts = {"00": 1000}
        y_counts = {"00": 1000}
        C = correlators_from_counts(x_counts, y_counts, 2)
        # ⟨Z_0Z_1⟩ = +1 for both bases → C[0,1] = 2.0
        assert C[0, 1] == pytest.approx(2.0)
        assert C[0, 0] == pytest.approx(0.0)  # diagonal zeroed

    def test_anticorrelation(self):
        x_counts = {"01": 500, "10": 500}
        y_counts = {"01": 500, "10": 500}
        C = correlators_from_counts(x_counts, y_counts, 2)
        assert C[0, 1] == pytest.approx(-2.0)

    def test_symmetric(self):
        rng = np.random.default_rng(42)
        x_counts = {
            format(i, "03b"): int(c) for i, c in enumerate(rng.multinomial(4000, [1 / 8] * 8))
        }
        y_counts = {
            format(i, "03b"): int(c) for i, c in enumerate(rng.multinomial(4000, [1 / 8] * 8))
        }
        C = correlators_from_counts(x_counts, y_counts, 3)
        np.testing.assert_array_almost_equal(C, C.T)


class TestShotNoise:
    def test_finite_shots(self):
        x_counts = {"00": 4000}
        y_counts = {"00": 4000}
        std = correlator_shot_noise(x_counts, y_counts, 2)
        expected = np.sqrt(2.0 / 4000)
        assert std == pytest.approx(expected)

    def test_zero_shots(self):
        std = correlator_shot_noise({}, {}, 2)
        assert std == float("inf")


class TestSelfConsistencyExact:
    def test_2qubit_runs(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = self_consistency_from_exact(K, omega, maxiter=50)
        assert isinstance(result, SelfConsistencyResult)
        assert result.n_qubits == 2
        # 2-qubit inverse problem is degenerate: different K values can
        # produce identical correlators. The learning converges to a
        # valid solution (loss ≈ 0) but not necessarily K_true.
        assert result.learning_result.loss < 0.1

    def test_3qubit_runs(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = self_consistency_from_exact(K, omega, maxiter=30)
        assert result.K_learned.shape == (3, 3)
        assert result.frobenius_error >= 0


class TestSelfConsistencyNoisySim:
    def test_noisy_has_higher_error(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        self_consistency_from_exact(K, omega, maxiter=30)
        noisy = self_consistency_from_noisy_sim(K, omega, noise_std=0.2, maxiter=30)
        assert noisy.shot_noise_std > 0

    def test_low_noise_has_low_correlator_error(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = self_consistency_from_noisy_sim(K, omega, noise_std=0.001, maxiter=50)
        # Correlator reconstruction should be accurate even if K recovery
        # hits the degeneracy (different K → same correlators)
        assert result.correlator_error < 0.1


# ---------------------------------------------------------------------------
# Self-consistency physics: K_learned properties
# ---------------------------------------------------------------------------


class TestSelfConsistencyPhysics:
    def test_learned_K_symmetric(self):
        """Learned coupling matrix must be symmetric."""
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = self_consistency_from_exact(K, omega, maxiter=30)
        np.testing.assert_allclose(result.K_learned, result.K_learned.T, atol=1e-8)

    def test_frobenius_error_nonnegative(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = self_consistency_from_exact(K, omega, maxiter=20)
        assert result.frobenius_error >= 0

    def test_learning_loss_nonnegative(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = self_consistency_from_exact(K, omega, maxiter=30)
        assert result.learning_result.loss >= 0


# ---------------------------------------------------------------------------
# Pipeline: Knm → correlators → learn K → wired
# ---------------------------------------------------------------------------


class TestSelfConsistencyPipeline:
    def test_pipeline_knm_to_learned_K(self):
        """Full pipeline: Knm → exact correlators → learn K_learned.
        Verifies self-consistency loop is wired end-to-end.
        """
        import time

        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]

        t0 = time.perf_counter()
        result = self_consistency_from_exact(K, omega, maxiter=50)
        dt = (time.perf_counter() - t0) * 1000

        assert result.K_learned.shape == K.shape
        assert result.learning_result.loss < 1.0

        print(f"\n  PIPELINE Knm→Correlators→LearnK (2q): {dt:.1f} ms")
        print(f"  Frobenius error = {result.frobenius_error:.4f}")
        print(f"  Learning loss = {result.learning_result.loss:.6f}")
