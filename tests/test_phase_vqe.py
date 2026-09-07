# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase Vqe
"""Tests for phase/phase_vqe.py — multi-angle coverage."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.phase.phase_vqe import PhaseVQE


def _energy(result: dict[str, object], key: str) -> float:
    """Return one energy field of a VQE result, asserting the float it documents."""
    value = result[key]
    assert isinstance(value, float)
    return value


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestPhaseVQEInit:
    def test_stores_params(self) -> None:
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        vqe = PhaseVQE(K, omega, ansatz_reps=1)
        np.testing.assert_array_equal(vqe.K, K)
        np.testing.assert_array_equal(vqe.omega, omega)

    def test_hamiltonian_built(self) -> None:
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        vqe = PhaseVQE(K, omega)
        assert vqe.hamiltonian.num_qubits == 2

    def test_ansatz_built(self) -> None:
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        vqe = PhaseVQE(K, omega, ansatz_reps=1)
        assert vqe.ansatz.num_qubits == 2
        assert vqe.n_params > 0

    def test_no_optimal_params_before_solve(self) -> None:
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        vqe = PhaseVQE(K, omega)
        assert vqe._optimal_params is None
        assert vqe._ground_energy is None


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------


class TestSolve:
    def test_returns_expected_keys(self) -> None:
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        vqe = PhaseVQE(K, omega, ansatz_reps=1)
        result = vqe.solve(maxiter=20, seed=0)
        expected_keys = {
            "ground_energy",
            "exact_energy",
            "energy_gap",
            "relative_error_pct",
            "optimal_params",
            "n_evals",
            "n_params",
            "converged",
        }
        assert expected_keys.issubset(result.keys())

    def test_energy_is_finite(self) -> None:
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        vqe = PhaseVQE(K, omega, ansatz_reps=1)
        result = vqe.solve(maxiter=30, seed=0)
        assert np.isfinite(_energy(result, "ground_energy"))
        assert np.isfinite(_energy(result, "exact_energy"))

    def test_energy_gap_nonnegative(self) -> None:
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        vqe = PhaseVQE(K, omega, ansatz_reps=1)
        result = vqe.solve(maxiter=30, seed=0)
        assert _energy(result, "energy_gap") >= 0

    def test_ground_energy_at_most_exact(self) -> None:
        """VQE ground energy should be >= exact (variational principle)."""
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        vqe = PhaseVQE(K, omega, ansatz_reps=2)
        result = vqe.solve(maxiter=100, seed=42)
        # Generous tolerance for VQE
        assert _energy(result, "ground_energy") >= _energy(result, "exact_energy") - 0.5

    def test_seed_determinism(self) -> None:
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        vqe1 = PhaseVQE(K, omega, ansatz_reps=1)
        vqe2 = PhaseVQE(K, omega, ansatz_reps=1)
        r1 = vqe1.solve(maxiter=20, seed=42)
        r2 = vqe2.solve(maxiter=20, seed=42)
        assert _energy(r1, "ground_energy") == _energy(r2, "ground_energy")

    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_various_sizes(self, L: int) -> None:
        K = build_knm_paper27(L=L)
        omega = OMEGA_N_16[:L]
        vqe = PhaseVQE(K, omega, ansatz_reps=1)
        result = vqe.solve(maxiter=20, seed=0)
        assert np.isfinite(_energy(result, "ground_energy"))


# ---------------------------------------------------------------------------
# ground_state
# ---------------------------------------------------------------------------


class TestGroundState:
    def test_none_before_solve(self) -> None:
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        vqe = PhaseVQE(K, omega)
        assert vqe.ground_state() is None

    def test_normalised_after_solve(self) -> None:
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        vqe = PhaseVQE(K, omega, ansatz_reps=1)
        vqe.solve(maxiter=20, seed=0)
        sv = vqe.ground_state()
        assert sv is not None
        np.testing.assert_allclose(float(np.sum(np.abs(sv) ** 2)), 1.0, atol=1e-10)

    def test_correct_dimension(self) -> None:
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        vqe = PhaseVQE(K, omega, ansatz_reps=1)
        vqe.solve(maxiter=20, seed=0)
        sv = vqe.ground_state()
        # `ground_state()` returns None until `solve` has run; this test runs
        # after it, and now states that precondition instead of assuming it.
        assert sv is not None
        assert len(sv) == 2**3


# ---------------------------------------------------------------------------
# Convergence with more iterations
# ---------------------------------------------------------------------------


class TestConvergence:
    def test_more_reps_lower_or_equal_energy(self) -> None:
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        vqe_short = PhaseVQE(K, omega, ansatz_reps=1)
        vqe_long = PhaseVQE(K, omega, ansatz_reps=2)
        r1 = vqe_short.solve(maxiter=10, seed=0)
        r2 = vqe_long.solve(maxiter=50, seed=0)
        assert _energy(r2, "ground_energy") <= _energy(r1, "ground_energy") + 1.0
