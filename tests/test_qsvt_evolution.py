# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Qsvt Evolution
"""Tests for QSVT Hamiltonian simulation resource estimation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import scpn_quantum_control.phase.qsvt_evolution as qsvt_mod
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.phase.qsvt_evolution import (
    QSVTResourceEstimate,
    hamiltonian_1norm,
    hamiltonian_spectral_norm,
    qsp_phase_angles,
    qsvt_query_count,
    qsvt_resource_estimate,
    trotter1_step_count,
    trotter2_step_count,
)


class TestHamiltonian1Norm:
    def test_positive(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        alpha = hamiltonian_1norm(K, omega)
        assert alpha > 0

    def test_scales_with_coupling(self):
        omega = OMEGA_N_16[:4]
        a_weak = hamiltonian_1norm(build_knm_paper27(L=4, K_base=0.1), omega)
        a_strong = hamiltonian_1norm(build_knm_paper27(L=4, K_base=1.0), omega)
        assert a_strong > a_weak

    def test_scales_with_n(self):
        a4 = hamiltonian_1norm(build_knm_paper27(L=4), OMEGA_N_16[:4])
        a8 = hamiltonian_1norm(build_knm_paper27(L=8), OMEGA_N_16[:8])
        assert a8 > a4


class TestHamiltonianSpectralNorm:
    def test_positive(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        norm = hamiltonian_spectral_norm(K, omega)
        assert norm > 0

    def test_leq_1norm(self):
        """Spectral norm ≤ 1-norm always."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        spec = hamiltonian_spectral_norm(K, omega)
        alpha = hamiltonian_1norm(K, omega)
        assert spec <= alpha + 1e-10

    def test_large_system_uses_sparse_matrix_without_dense_builder(self, monkeypatch):
        """n >= 14 must not allocate a dense 2^n x 2^n matrix."""
        K = np.zeros((14, 14))
        omega = np.linspace(0.5, 1.5, 14)

        def fail_dense(*args, **kwargs):
            raise AssertionError("dense builder must not be called for sparse QSVT norm path")

        monkeypatch.setattr(qsvt_mod, "knm_to_dense_matrix", fail_dense)
        monkeypatch.setattr(
            qsvt_mod,
            "knm_to_sparse_matrix",
            lambda *_args, **_kwargs: sparse.diags([-2.0, 3.0], format="csc"),
        )

        assert hamiltonian_spectral_norm(K, omega) == pytest.approx(3.0)

    def test_rejects_dense_budget_before_small_dense_allocation(self, monkeypatch):
        """Dense n < 14 branch must respect explicit memory budgets."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]

        def fail_dense(*args, **kwargs):
            raise AssertionError("dense builder must not run after budget rejection")

        monkeypatch.setattr(qsvt_mod, "knm_to_dense_matrix", fail_dense)

        with pytest.raises(DenseAllocationError, match="QSVT dense spectral norm"):
            hamiltonian_spectral_norm(K, omega, max_dense_gib=1e-12)


class TestQueryCounts:
    def test_qsvt_positive(self):
        q = qsvt_query_count(1.0, 1.0, 0.01)
        assert q >= 1

    def test_trotter1_positive(self):
        r = trotter1_step_count(1.0, 1.0, 0.01)
        assert r >= 1

    def test_trotter2_positive(self):
        r = trotter2_step_count(1.0, 1.0, 0.01)
        assert r >= 1

    def test_qsvt_fewer_than_trotter1(self):
        """QSVT should need fewer queries than first-order Trotter."""
        alpha, t, eps = 5.0, 2.0, 0.001
        q = qsvt_query_count(alpha, t, eps)
        r = trotter1_step_count(alpha, t, eps)
        assert q < r

    def test_scales_sublinearly_or_linearly(self):
        """QSVT: O(αt + log(1/ε)). At large t, nearly linear."""
        q10 = qsvt_query_count(1.0, 10.0, 0.01)
        q100 = qsvt_query_count(1.0, 100.0, 0.01)
        ratio = q100 / q10
        assert 5 < ratio < 15  # approximately 10x at large t

    @pytest.mark.parametrize("alpha", [0.0, -1.0, float("inf"), float("nan")])
    def test_query_counts_reject_invalid_alpha(self, alpha):
        with pytest.raises(ValueError, match="alpha"):
            qsvt_query_count(alpha, 1.0, 0.01)
        with pytest.raises(ValueError, match="alpha"):
            trotter1_step_count(alpha, 1.0, 0.01)
        with pytest.raises(ValueError, match="alpha"):
            trotter2_step_count(alpha, 1.0, 0.01)

    @pytest.mark.parametrize("epsilon", [0.0, -0.1, 1.0, 1.5, float("nan")])
    def test_query_counts_reject_invalid_error_budget(self, epsilon):
        with pytest.raises(ValueError, match="epsilon"):
            qsvt_query_count(1.0, 1.0, epsilon)
        with pytest.raises(ValueError, match="epsilon"):
            trotter1_step_count(1.0, 1.0, epsilon)
        with pytest.raises(ValueError, match="epsilon"):
            trotter2_step_count(1.0, 1.0, epsilon)

    @pytest.mark.parametrize("time", [-1.0, float("inf"), float("nan")])
    def test_query_counts_reject_invalid_time(self, time):
        with pytest.raises(ValueError, match="simulation time"):
            qsvt_query_count(1.0, time, 0.01)
        with pytest.raises(ValueError, match="simulation time"):
            trotter1_step_count(1.0, time, 0.01)
        with pytest.raises(ValueError, match="simulation time"):
            trotter2_step_count(1.0, time, 0.01)

    def test_query_counts_reject_string_alpha_coercion(self):
        with pytest.raises(ValueError, match="alpha must be a real numeric scalar"):
            qsvt_query_count("1.0", 1.0, 0.01)

    def test_query_counts_reject_boolean_time_coercion(self):
        with pytest.raises(ValueError, match="simulation time must be a real numeric scalar"):
            trotter1_step_count(1.0, True, 0.01)


class TestQSVTResourceEstimate:
    def test_returns_result(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = qsvt_resource_estimate(K, omega)
        assert isinstance(result, QSVTResourceEstimate)

    def test_speedup_positive(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = qsvt_resource_estimate(K, omega, t=2.0, epsilon=0.001)
        assert result.speedup_vs_trotter1 > 1.0

    def test_n_qubits(self):
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        result = qsvt_resource_estimate(K, omega)
        assert result.n_qubits == 8

    def test_ancilla_positive(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = qsvt_resource_estimate(K, omega)
        assert result.n_ancilla_qsvt >= 2

    def test_scpn_8_resource(self):
        """Record QSVT resource estimate for 8 oscillators."""
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        result = qsvt_resource_estimate(K, omega, t=1.0, epsilon=0.01)
        print("\n  QSVT 8-osc resource estimate:")
        print(f"  α (1-norm) = {result.alpha:.4f}")
        print(f"  ||H|| (spectral) = {result.spectral_norm:.4f}")
        print(f"  QSVT queries: {result.qsvt_queries}")
        print(f"  Trotter-1 steps: {result.trotter1_steps}")
        print(f"  Trotter-2 steps: {result.trotter2_steps}")
        print(f"  Speedup vs T1: {result.speedup_vs_trotter1:.1f}x")
        print(f"  Speedup vs T2: {result.speedup_vs_trotter2:.1f}x")
        print(f"  Ancilla qubits: {result.n_ancilla_qsvt}")
        assert result.qsvt_queries > 0

    @pytest.mark.parametrize(
        ("K", "omega", "match"),
        [
            (np.ones((2, 3)), np.ones(2), "square"),
            (np.eye(3), np.ones(2), "omega"),
            (np.array([[0.0, np.nan], [np.nan, 0.0]]), np.ones(2), "finite"),
            (np.eye(2), np.array([0.0, np.inf]), "finite"),
        ],
    )
    def test_resource_estimate_rejects_invalid_problem_shapes_and_values(self, K, omega, match):
        with pytest.raises(ValueError, match=match):
            qsvt_resource_estimate(K, omega)

    def test_resource_estimate_rejects_string_coupling_coercion(self):
        K = [["0.0", "0.25"], ["0.25", "0.0"]]
        omega = np.array([0.1, 0.2])

        with pytest.raises(ValueError, match="K must contain real numeric scalars"):
            qsvt_resource_estimate(K, omega)

    def test_resource_estimate_rejects_boolean_frequency_coercion(self):
        K = np.zeros((2, 2))
        omega = [True, False]

        with pytest.raises(ValueError, match="omega must contain real numeric scalars"):
            qsvt_resource_estimate(K, omega)

    @pytest.mark.parametrize(
        ("time", "epsilon", "match"),
        [
            (-1.0, 0.01, "simulation time"),
            (1.0, 0.0, "epsilon"),
            (1.0, 1.0, "epsilon"),
        ],
    )
    def test_resource_estimate_rejects_invalid_budget_parameters(self, time, epsilon, match):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]

        with pytest.raises(ValueError, match=match):
            qsvt_resource_estimate(K, omega, t=time, epsilon=epsilon)

    def test_resource_estimate_propagates_dense_budget(self, monkeypatch):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]

        def fail_dense(*args, **kwargs):
            raise AssertionError("dense builder must not run after budget rejection")

        monkeypatch.setattr(qsvt_mod, "knm_to_dense_matrix", fail_dense)

        with pytest.raises(DenseAllocationError, match="QSVT dense spectral norm"):
            qsvt_resource_estimate(K, omega, max_dense_gib=1e-12)


class TestQSPPhaseAngles:
    def test_refuses_unoptimised_phase_claim(self):
        with pytest.raises(NotImplementedError, match="QSP phase synthesis"):
            qsp_phase_angles(10)

    @pytest.mark.parametrize("degree", [1.5, True, "4"])
    def test_rejects_non_integer_degree(self, degree):
        with pytest.raises(ValueError, match="degree"):
            qsp_phase_angles(degree, allow_initial_guess=True)

    def test_initial_guess_length_when_explicitly_requested(self):
        phases = qsp_phase_angles(10, allow_initial_guess=True)
        assert len(phases) == 11

    def test_initial_guess_symmetric_when_explicitly_requested(self):
        phases = qsp_phase_angles(8, allow_initial_guess=True)
        assert phases[0] == pytest.approx(phases[-1])


class TestQSVTCoverage:
    """Cover sparse eigensolver path for n >= 14."""

    def test_spectral_norm_large_system(self):
        """Cover lines 87-91: n=14 sparse eigsh path."""
        from scpn_quantum_control.bridge.knm_hamiltonian import (
            OMEGA_N_16,
            build_knm_paper27,
        )
        from scpn_quantum_control.phase.qsvt_evolution import hamiltonian_spectral_norm

        K = build_knm_paper27(L=14)
        omega = OMEGA_N_16[:14]
        alpha = hamiltonian_spectral_norm(K, omega)
        assert alpha > 0
        import numpy as np_

        assert np_.isfinite(alpha)
