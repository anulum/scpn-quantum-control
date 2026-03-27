# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Parametrized Sweep Tests
"""Sweep tests across system sizes, time steps, and coupling variants.

Existing tests typically use one hardcoded system size. These tests apply
the same assertions across {2, 3, 4, 6, 8} qubits and multiple dt/coupling
values, multiplying effective test coverage by 5-10x.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis.bkt_analysis import (
    coupling_laplacian,
    fiedler_eigenvalue,
)
from scpn_quantum_control.analysis.entanglement_spectrum import (
    entanglement_entropy_half_chain,
    entanglement_spectrum_half_chain,
)
from scpn_quantum_control.analysis.phase_diagram import (
    critical_coupling_mean_field,
    order_parameter_steady_state,
)
from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    build_kuramoto_ring,
    knm_to_ansatz,
    knm_to_hamiltonian,
    knm_to_xxz_hamiltonian,
)
from scpn_quantum_control.hardware.classical import (
    classical_brute_mpc,
    classical_exact_diag,
    classical_exact_evolution,
    classical_kuramoto_reference,
)
from scpn_quantum_control.phase.xy_kuramoto import QuantumKuramotoSolver

# ---------------------------------------------------------------------------
# Hamiltonian construction sweep
# ---------------------------------------------------------------------------

SIZES = [2, 3, 4, 6, 8]
SMALL_SIZES = [2, 3, 4]
EVEN_SIZES = [2, 4, 6, 8]


class TestHamiltonianSweep:
    @pytest.mark.parametrize("n", SIZES)
    def test_qubit_count(self, n):
        K = build_knm_paper27(L=n)
        H = knm_to_hamiltonian(K, OMEGA_N_16[:n])
        assert H.num_qubits == n

    @pytest.mark.parametrize("n", SIZES)
    def test_hermitian(self, n):
        K = build_knm_paper27(L=n)
        H = knm_to_hamiltonian(K, OMEGA_N_16[:n])
        H_mat = np.array(H.to_matrix())
        np.testing.assert_allclose(H_mat, H_mat.conj().T, atol=1e-14)

    @pytest.mark.parametrize("n", SIZES)
    def test_real_eigenvalues(self, n):
        K = build_knm_paper27(L=n)
        H = knm_to_hamiltonian(K, OMEGA_N_16[:n])
        evals = np.linalg.eigvalsh(np.array(H.to_matrix()))
        assert np.all(np.isreal(evals))

    @pytest.mark.parametrize("n", SIZES)
    @pytest.mark.parametrize("delta", [0.0, 0.5, 1.0])
    def test_xxz_hermitian(self, n, delta):
        K = build_knm_paper27(L=n)
        H = knm_to_xxz_hamiltonian(K, OMEGA_N_16[:n], delta=delta)
        H_mat = np.array(H.to_matrix())
        np.testing.assert_allclose(H_mat, H_mat.conj().T, atol=1e-14)

    @pytest.mark.parametrize("n", SIZES)
    @pytest.mark.parametrize("coupling", [0.1, 0.5, 1.0, 2.0])
    def test_ring_hermitian(self, n, coupling):
        K, omega = build_kuramoto_ring(n, coupling=coupling, rng_seed=42)
        H = knm_to_hamiltonian(K, omega)
        H_mat = np.array(H.to_matrix())
        np.testing.assert_allclose(H_mat, H_mat.conj().T, atol=1e-14)

    @pytest.mark.parametrize("n", SMALL_SIZES)
    @pytest.mark.parametrize("reps", [1, 2, 3])
    def test_ansatz_parameter_count(self, n, reps):
        K = build_knm_paper27(L=n)
        qc = knm_to_ansatz(K, reps=reps)
        assert qc.num_qubits == n
        assert qc.num_parameters == n * 2 * reps


# ---------------------------------------------------------------------------
# Classical Kuramoto sweep
# ---------------------------------------------------------------------------


class TestKuramotoSweep:
    @pytest.mark.parametrize("n", SIZES)
    @pytest.mark.parametrize("dt", [0.01, 0.05, 0.1])
    def test_R_bounded(self, n, dt):
        result = classical_kuramoto_reference(n, t_max=0.3, dt=dt)
        assert np.all(result["R"] >= -1e-10)
        assert np.all(result["R"] <= 1.0 + 1e-10)

    @pytest.mark.parametrize("n", SIZES)
    def test_shapes(self, n):
        dt = 0.05
        result = classical_kuramoto_reference(n, t_max=0.5, dt=dt)
        n_steps = max(1, round(0.5 / dt))
        assert result["times"].shape == (n_steps + 1,)
        assert result["theta"].shape == (n_steps + 1, n)
        assert result["R"].shape == (n_steps + 1,)

    @pytest.mark.parametrize("n", SMALL_SIZES)
    def test_zero_coupling_free_rotation(self, n):
        K = np.zeros((n, n))
        omega = OMEGA_N_16[:n]
        theta0 = np.zeros(n)
        result = classical_kuramoto_reference(
            n, t_max=0.1, dt=0.0001, K=K, omega=omega, theta0=theta0
        )
        np.testing.assert_allclose(result["theta"][-1], omega * 0.1, atol=0.01)


# ---------------------------------------------------------------------------
# Exact evolution sweep
# ---------------------------------------------------------------------------


class TestEvolutionSweep:
    @pytest.mark.parametrize("n", SIZES)
    @pytest.mark.parametrize("dt", [0.01, 0.05, 0.1])
    def test_R_bounded(self, n, dt):
        result = classical_exact_evolution(n, t_max=0.3, dt=dt)
        assert np.all(result["R"] >= -1e-10)
        assert np.all(result["R"] <= 1.0 + 1e-10)

    @pytest.mark.parametrize("n", SIZES)
    def test_shapes(self, n):
        dt = 0.05
        result = classical_exact_evolution(n, t_max=0.3, dt=dt)
        n_steps = max(1, round(0.3 / dt))
        assert result["times"].shape == (n_steps + 1,)
        assert result["R"].shape == (n_steps + 1,)

    @pytest.mark.parametrize("n", SMALL_SIZES)
    def test_single_step_matches_multi_step(self, n):
        """exp(-iH*0.1) should give same R whether computed in 1 or 2 steps."""
        r1 = classical_exact_evolution(n, t_max=0.1, dt=0.1)
        r2 = classical_exact_evolution(n, t_max=0.1, dt=0.05)
        np.testing.assert_allclose(r1["R"][-1], r2["R"][-1], atol=1e-10)


# ---------------------------------------------------------------------------
# Exact diag sweep
# ---------------------------------------------------------------------------


class TestDiagSweep:
    @pytest.mark.parametrize("n", SIZES)
    def test_spectral_gap_positive(self, n):
        result = classical_exact_diag(n)
        assert result["spectral_gap"] > 0

    @pytest.mark.parametrize("n", SIZES)
    def test_ground_state_normalised(self, n):
        result = classical_exact_diag(n)
        np.testing.assert_allclose(np.linalg.norm(result["ground_state"]), 1.0, atol=1e-12)

    @pytest.mark.parametrize("n", SIZES)
    def test_ground_is_minimum(self, n):
        result = classical_exact_diag(n)
        assert result["ground_energy"] == pytest.approx(result["eigenvalues"][0])

    @pytest.mark.parametrize("n", SMALL_SIZES)
    def test_ground_state_eigenvector(self, n):
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        H_mat = np.array(knm_to_hamiltonian(K, omega).to_matrix())
        result = classical_exact_diag(n)
        residual = (
            H_mat @ result["ground_state"] - result["ground_energy"] * result["ground_state"]
        )
        assert np.linalg.norm(residual) < 1e-10

    @pytest.mark.parametrize("n", [4, 6, 8])
    def test_sparse_path(self, n):
        """k_eigenvalues forces sparse solver. Needs k < 2^n - 1."""
        result = classical_exact_diag(n, k_eigenvalues=4)
        assert len(result["eigenvalues"]) == 4
        assert result["spectral_gap"] > 0


# ---------------------------------------------------------------------------
# Laplacian / BKT sweep
# ---------------------------------------------------------------------------


class TestLaplacianSweep:
    @pytest.mark.parametrize("n", SIZES)
    def test_symmetric(self, n):
        K = build_knm_paper27(L=n)
        L = coupling_laplacian(K)
        np.testing.assert_allclose(L, L.T, atol=1e-12)

    @pytest.mark.parametrize("n", SIZES)
    def test_row_sums_zero(self, n):
        K = build_knm_paper27(L=n)
        L = coupling_laplacian(K)
        np.testing.assert_allclose(L.sum(axis=1), 0.0, atol=1e-12)

    @pytest.mark.parametrize("n", SIZES)
    def test_positive_semidefinite(self, n):
        K = build_knm_paper27(L=n)
        L = coupling_laplacian(K)
        evals = np.linalg.eigvalsh(L)
        assert np.all(evals >= -1e-12)

    @pytest.mark.parametrize("n", SIZES)
    def test_fiedler_positive(self, n):
        K = build_knm_paper27(L=n)
        lam2 = fiedler_eigenvalue(K)
        assert lam2 > 0


# ---------------------------------------------------------------------------
# Entanglement sweep
# ---------------------------------------------------------------------------


class TestEntanglementSweep:
    @pytest.mark.parametrize("n", EVEN_SIZES)
    def test_half_chain_entropy_nonneg(self, n):
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        s = entanglement_entropy_half_chain(K, omega)
        assert s >= -1e-10

    @pytest.mark.parametrize("n", EVEN_SIZES)
    def test_half_chain_entropy_bounded(self, n):
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        s = entanglement_entropy_half_chain(K, omega)
        assert s <= n / 2 + 1e-10

    @pytest.mark.parametrize("n", EVEN_SIZES)
    def test_spectrum_sums_to_one(self, n):
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        spectrum = entanglement_spectrum_half_chain(K, omega)
        assert np.sum(spectrum) == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("n", EVEN_SIZES)
    def test_spectrum_nonneg(self, n):
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        spectrum = entanglement_spectrum_half_chain(K, omega)
        assert np.all(spectrum >= -1e-12)


# ---------------------------------------------------------------------------
# QuantumKuramotoSolver sweep
# ---------------------------------------------------------------------------


class TestSolverSweep:
    @pytest.mark.parametrize("n", SMALL_SIZES)
    def test_build_hamiltonian(self, n):
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        solver = QuantumKuramotoSolver(n, K, omega)
        H = solver.build_hamiltonian()
        assert H.num_qubits == n

    @pytest.mark.parametrize("n", SMALL_SIZES)
    def test_evolve_circuit(self, n):
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        solver = QuantumKuramotoSolver(n, K, omega)
        solver.build_hamiltonian()
        qc = solver.evolve(0.5, trotter_steps=3)
        assert qc.num_qubits == n

    @pytest.mark.parametrize("n", SMALL_SIZES)
    @pytest.mark.parametrize("dt", [0.05, 0.1])
    def test_run_R_bounded(self, n, dt):
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        solver = QuantumKuramotoSolver(n, K, omega)
        result = solver.run(t_max=0.3, dt=dt)
        for r in result["R"]:
            assert -1e-10 <= r <= 1.0 + 1e-10

    @pytest.mark.parametrize("n", SMALL_SIZES)
    @pytest.mark.parametrize("order", [1, 2])
    def test_trotter_orders(self, n, order):
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        solver = QuantumKuramotoSolver(n, K, omega, trotter_order=order)
        solver.build_hamiltonian()
        qc = solver.evolve(0.5, trotter_steps=2)
        assert qc.num_qubits == n


# ---------------------------------------------------------------------------
# Phase diagram sweep
# ---------------------------------------------------------------------------


class TestPhaseDiagramSweep:
    @pytest.mark.parametrize("n", [4, 8, 16])
    def test_critical_coupling_mean_field(self, n):
        omega = OMEGA_N_16[:n]
        k_c = critical_coupling_mean_field(omega)
        assert k_c >= 0.0

    @pytest.mark.parametrize(
        "K_coupling,expected_zero",
        [(0.5, True), (1.0, True), (2.0, False), (10.0, False)],
        ids=["sub", "at", "above", "far_above"],
    )
    def test_order_parameter_vs_critical(self, K_coupling, expected_zero):
        R = order_parameter_steady_state(K_coupling, k_critical=1.0)
        if expected_zero:
            assert R == 0.0
        else:
            assert R > 0.0
            assert R <= 1.0


# ---------------------------------------------------------------------------
# Brute MPC sweep
# ---------------------------------------------------------------------------


class TestMPCSweep:
    @pytest.mark.parametrize("dim", [1, 2, 3])
    @pytest.mark.parametrize("horizon", [1, 2, 3, 4])
    def test_enumerates_all(self, dim, horizon):
        B = np.eye(dim)
        target = np.ones(dim) * 0.5
        result = classical_brute_mpc(B, target, horizon)
        assert result["n_evaluated"] == 2**horizon

    @pytest.mark.parametrize("dim", [1, 2, 3])
    @pytest.mark.parametrize("horizon", [1, 2, 3])
    def test_optimal_is_minimum(self, dim, horizon):
        B = np.eye(dim)
        target = np.ones(dim) * 0.5
        result = classical_brute_mpc(B, target, horizon)
        assert result["optimal_cost"] <= np.min(result["all_costs"]) + 1e-12
