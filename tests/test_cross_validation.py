# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Cross-Validation Tests
"""Cross-validation between independent computation paths.

Tests that different implementations of the same quantity agree:
  - Sparse Krylov vs dense matrix exponential (R order parameter)
  - _state_order_param vs _state_order_param_sparse (per-qubit expectations)
  - Rust engine vs Python fallback (Kuramoto trajectory)
  - XY Hamiltonian vs XXZ(delta=0) (must be identical)
  - Statevector vs Trotter circuit (R convergence)
  - Sparse diag vs dense diag (eigenvalues)
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    knm_to_hamiltonian,
    knm_to_xxz_hamiltonian,
)
from scpn_quantum_control.hardware.classical import (
    _build_initial_state,
    _state_order_param,
    _state_order_param_sparse,
    classical_exact_diag,
    classical_exact_evolution,
    classical_kuramoto_reference,
)
from scpn_quantum_control.phase.xy_kuramoto import QuantumKuramotoSolver

SIZES = [2, 3, 4, 6, 8]
SMALL_SIZES = [2, 3, 4]
EVEN_SIZES = [2, 4, 6, 8]


# ---------------------------------------------------------------------------
# XY == XXZ(delta=0)
# ---------------------------------------------------------------------------


class TestXYvsXXZ:
    @pytest.mark.parametrize("n", SIZES)
    def test_xy_equals_xxz_delta0(self, n):
        """knm_to_hamiltonian(K,w) == knm_to_xxz_hamiltonian(K,w, delta=0)."""
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        H_xy = knm_to_hamiltonian(K, omega)
        H_xxz = knm_to_xxz_hamiltonian(K, omega, delta=0.0)
        # SparsePauliOp equality via matrix comparison
        M_xy = np.array(H_xy.to_matrix())
        M_xxz = np.array(H_xxz.to_matrix())
        np.testing.assert_allclose(M_xy, M_xxz, atol=1e-14)


# ---------------------------------------------------------------------------
# Sparse vs dense state order parameter
# ---------------------------------------------------------------------------


class TestSparseVsDenseR:
    @pytest.mark.parametrize("n", SIZES)
    def test_initial_state(self, n):
        omega = OMEGA_N_16[:n]
        psi = _build_initial_state(n, omega)
        R_d = _state_order_param(psi, n)
        R_s = _state_order_param_sparse(psi, n)
        np.testing.assert_allclose(R_s, R_d, atol=1e-14)

    @pytest.mark.parametrize("n", SIZES)
    def test_evolved_state(self, n):
        """After 5 evolution steps, sparse and dense R still agree."""
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        H = knm_to_hamiltonian(K, omega)
        H_mat = np.array(H.to_matrix())
        U = expm(-1j * H_mat * 0.05)
        psi = _build_initial_state(n, omega)
        for _ in range(5):
            psi = U @ psi
        R_d = _state_order_param(psi, n)
        R_s = _state_order_param_sparse(psi, n)
        np.testing.assert_allclose(R_s, R_d, atol=1e-13)

    @pytest.mark.parametrize("n", SMALL_SIZES)
    def test_computational_basis_states(self, n):
        """R on each computational basis state agrees between paths."""
        dim = 2**n
        for idx in range(min(8, dim)):
            psi = np.zeros(dim, dtype=complex)
            psi[idx] = 1.0
            R_d = _state_order_param(psi, n)
            R_s = _state_order_param_sparse(psi, n)
            np.testing.assert_allclose(R_s, R_d, atol=1e-14)

    @pytest.mark.parametrize("n", SMALL_SIZES)
    def test_random_states(self, n):
        """10 random normalised states, sparse == dense."""
        rng = np.random.default_rng(7777)
        dim = 2**n
        for _ in range(10):
            psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
            psi /= np.linalg.norm(psi)
            R_d = _state_order_param(psi, n)
            R_s = _state_order_param_sparse(psi, n)
            np.testing.assert_allclose(R_s, R_d, atol=1e-13)

    @pytest.mark.parametrize("n", SMALL_SIZES)
    def test_bell_states(self, n):
        """Bell-like entangled states: sparse == dense."""
        if n < 2:
            pytest.skip("Need at least 2 qubits for Bell state")
        dim = 2**n
        psi = np.zeros(dim, dtype=complex)
        psi[0] = 1.0 / np.sqrt(2)
        psi[-1] = 1.0 / np.sqrt(2)
        R_d = _state_order_param(psi, n)
        R_s = _state_order_param_sparse(psi, n)
        np.testing.assert_allclose(R_s, R_d, atol=1e-14)


# ---------------------------------------------------------------------------
# Dense evolution vs manual step-by-step
# ---------------------------------------------------------------------------


class TestEvolutionConsistency:
    @pytest.mark.parametrize("n", SMALL_SIZES)
    def test_evolution_R_matches_manual(self, n):
        """classical_exact_evolution R(t) should match manual U@psi loop."""
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        dt = 0.05
        n_steps = 5

        result = classical_exact_evolution(n, t_max=dt * n_steps, dt=dt)

        H = knm_to_hamiltonian(K, omega)
        H_mat = np.array(H.to_matrix())
        U = expm(-1j * H_mat * dt)
        psi = _build_initial_state(n, omega)

        for step in range(1, n_steps + 1):
            psi = U @ psi
            R_manual = _state_order_param(psi, n)
            np.testing.assert_allclose(
                result["R"][step],
                R_manual,
                atol=1e-10,
                err_msg=f"Step {step}: evolution R={result['R'][step]}, manual R={R_manual}",
            )

    @pytest.mark.parametrize("n", SMALL_SIZES)
    def test_dt_independence(self, n):
        """Total unitary is same regardless of dt subdivision."""
        t_total = 0.1
        r_coarse = classical_exact_evolution(n, t_max=t_total, dt=t_total)
        r_fine = classical_exact_evolution(n, t_max=t_total, dt=t_total / 10)
        np.testing.assert_allclose(r_coarse["R"][-1], r_fine["R"][-1], atol=1e-10)


# ---------------------------------------------------------------------------
# Sparse diag vs dense diag
# ---------------------------------------------------------------------------


class TestDiagConsistency:
    @pytest.mark.parametrize("n", [4, 6, 8])
    def test_sparse_vs_dense_ground_energy(self, n):
        """k_eigenvalues sparse solver agrees with full dense diag.

        n=2 excluded: eigsh requires k < 2^n - 1, so k=4 fails for dim=4.
        """
        result_dense = classical_exact_diag(n)
        result_sparse = classical_exact_diag(n, k_eigenvalues=4)
        np.testing.assert_allclose(
            result_sparse["ground_energy"], result_dense["ground_energy"], atol=1e-8
        )

    @pytest.mark.parametrize("n", [4, 6, 8])
    def test_sparse_vs_dense_spectral_gap(self, n):
        """n=2 excluded: eigsh k >= N-1 constraint."""
        result_dense = classical_exact_diag(n)
        result_sparse = classical_exact_diag(n, k_eigenvalues=4)
        np.testing.assert_allclose(
            result_sparse["spectral_gap"], result_dense["spectral_gap"], atol=1e-8
        )

    @pytest.mark.parametrize("n", SMALL_SIZES)
    def test_sparse_vs_dense_eigenvalues(self, n):
        """First k eigenvalues from sparse should match dense."""
        result_dense = classical_exact_diag(n)
        k = min(4, 2**n - 2)
        result_sparse = classical_exact_diag(n, k_eigenvalues=k)
        np.testing.assert_allclose(
            result_sparse["eigenvalues"], result_dense["eigenvalues"][:k], atol=1e-8
        )


# ---------------------------------------------------------------------------
# Rust vs Python Kuramoto (if engine available)
# ---------------------------------------------------------------------------


class TestRustVsPython:
    @pytest.mark.parametrize("n", SMALL_SIZES)
    def test_kuramoto_trajectory_agreement(self, n):
        """If Rust engine is available, trajectories should agree with Python."""
        try:
            import scpn_quantum_engine  # noqa: F401
        except ImportError:
            pytest.skip("scpn_quantum_engine not installed")

        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n].copy()
        theta0 = np.array([om % (2 * np.pi) for om in omega])
        dt = 0.01
        n_steps = 20

        # Rust path
        result_rust = classical_kuramoto_reference(n, t_max=dt * n_steps, dt=dt)

        # Force Python path
        theta = theta0.copy()
        R_python = np.zeros(n_steps + 1)
        z = np.mean(np.exp(1j * theta))
        R_python[0] = float(abs(z))

        for s in range(1, n_steps + 1):
            dtheta = omega.copy()
            for i in range(n):
                coupling = 0.0
                for j in range(n):
                    coupling += K[i, j] * np.sin(theta[j] - theta[i])
                dtheta[i] += coupling
            theta = theta + dt * dtheta
            z = np.mean(np.exp(1j * theta))
            R_python[s] = float(abs(z))

        # R trajectories should match to Euler integration precision
        np.testing.assert_allclose(result_rust["R"], R_python, atol=1e-6)


# ---------------------------------------------------------------------------
# Trotter vs exact evolution (convergence)
# ---------------------------------------------------------------------------


class TestTrotterConvergence:
    @pytest.mark.parametrize("n", SMALL_SIZES)
    def test_trotter_converges_to_exact(self, n):
        """Trotter R should approach exact R as reps increase."""
        from qiskit.quantum_info import Operator

        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        t = 0.5

        solver = QuantumKuramotoSolver(n, K, omega)
        solver.build_hamiltonian()

        H_mat = np.array(solver._hamiltonian.to_matrix())
        U_exact = expm(-1j * H_mat * t)

        errors = []
        for reps in [2, 5, 10]:
            qc = solver.evolve(t, trotter_steps=reps).decompose(reps=2)
            U_trotter = Operator(qc).data
            errors.append(np.linalg.norm(U_exact - U_trotter, "fro"))

        # Errors should decrease monotonically
        assert errors[1] < errors[0]
        assert errors[2] < errors[1]


class TestRustLanczosParity:
    """Rust lanczos_b_coefficients vs Python operator Lanczos."""

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_lanczos_coefficients_match(self, n: int) -> None:
        import scpn_quantum_engine as engine
        from scpn_quantum_control.analysis.krylov_complexity import (
            _liouvillian_action,
            _operator_inner_product,
        )

        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        H = knm_to_hamiltonian(K, omega).to_matrix()
        H = H.toarray() if hasattr(H, "toarray") else np.array(H)
        dim = H.shape[0]

        Z0 = np.zeros((dim, dim), dtype=complex)
        for i in range(dim):
            Z0[i, i] = 1.0 - 2.0 * ((i >> 0) & 1)

        # Python Lanczos
        tol = 1e-12
        norm_0 = np.sqrt(_operator_inner_product(Z0, Z0))
        O_prev = np.zeros_like(Z0)
        O_curr = Z0 / norm_0
        b_py: list[float] = []
        for _ in range(50):
            A = _liouvillian_action(H, O_curr)
            if b_py:
                A -= b_py[-1] * O_prev
            a_n = _operator_inner_product(O_curr, A)
            A -= a_n * O_curr
            b_n = np.sqrt(max(0, _operator_inner_product(A, A)))
            if b_n < tol:
                break
            b_py.append(b_n)
            O_prev = O_curr.copy()
            O_curr = A / b_n

        # Rust Lanczos
        H_c = np.asarray(H, dtype=complex)
        Z_c = np.asarray(Z0, dtype=complex)
        b_rs = engine.lanczos_b_coefficients(
            np.ascontiguousarray(H_c.real).ravel(),
            np.ascontiguousarray(H_c.imag).ravel(),
            np.ascontiguousarray(Z_c.real).ravel(),
            np.ascontiguousarray(Z_c.imag).ravel(),
            dim,
            50,
            tol,
        )

        assert len(b_rs) == len(b_py)
        # First ~40 coefficients match to machine precision; late ones diverge
        # due to Lanczos numerical instability (no reorthogonalization)
        n_check = min(len(b_rs), 40)
        np.testing.assert_allclose(b_rs[:n_check], b_py[:n_check], atol=1e-10)


class TestRustOTOCParity:
    """Rust otoc_from_eigendecomp vs scipy.expm loop."""

    @pytest.mark.parametrize("n", [2, 3])
    def test_otoc_values_match(self, n: int) -> None:
        import scpn_quantum_engine as engine

        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        H = knm_to_hamiltonian(K, omega).to_matrix()
        H = H.toarray() if hasattr(H, "toarray") else np.array(H)
        dim = H.shape[0]

        # Pauli Z_0 and X_1
        I2 = np.eye(2, dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)

        def _nqubit_pauli(P: np.ndarray, qubit: int, n_q: int) -> np.ndarray:
            result = np.eye(1, dtype=complex)
            for i in range(n_q):
                result = np.kron(result, P if i == qubit else I2)
            return result

        W = _nqubit_pauli(Z, 0, n)
        V = _nqubit_pauli(X, min(1, n - 1), n)
        psi = np.zeros(dim, dtype=complex)
        psi[0] = 1.0
        times = np.linspace(0, 1.0, 10)

        # Python reference via scipy.expm
        otoc_ref = np.zeros(len(times))
        for idx, t in enumerate(times):
            U = expm(-1j * H * t)
            U_dag = expm(1j * H * t)
            W_t = U_dag @ W @ U
            state = V @ psi
            state = W_t @ state
            state = V.conj().T @ state
            state = W_t.conj().T @ state
            otoc_ref[idx] = float(np.real(psi.conj() @ state))

        # Rust via eigendecomposition
        eigenvalues, eigvecs = np.linalg.eigh(H)
        otoc_rs = np.asarray(
            engine.otoc_from_eigendecomp(
                eigenvalues,
                np.ascontiguousarray(eigvecs.real).ravel(),
                np.ascontiguousarray(eigvecs.imag).ravel(),
                np.ascontiguousarray(W.real).ravel(),
                np.ascontiguousarray(W.imag).ravel(),
                np.ascontiguousarray(V.real).ravel(),
                np.ascontiguousarray(V.imag).ravel(),
                np.ascontiguousarray(psi.real),
                np.ascontiguousarray(psi.imag),
                times,
                dim,
            )
        )

        np.testing.assert_allclose(otoc_rs, otoc_ref, atol=1e-10)
