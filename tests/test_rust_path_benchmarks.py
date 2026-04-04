# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Rust Path Benchmarks & Python↔Rust Parity
"""Verify all 18 Rust-accelerated functions produce correct results and
benchmark Python↔Rust performance. Ensures Rust paths are NOT decorative.

Every Rust function is tested for:
1. Correctness (output shape, dtype, physical bounds)
2. Parity with Python fallback where applicable
3. Performance (wall time printed for regression tracking)
"""

from __future__ import annotations

import time

import numpy as np
import pytest

try:
    import scpn_quantum_engine as eng

    _RUST_OK = True
except ImportError:
    _RUST_OK = False

pytestmark = pytest.mark.skipif(not _RUST_OK, reason="scpn-quantum-engine not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _timed(fn, *a, **kw):
    t0 = time.perf_counter()
    r = fn(*a, **kw)
    return r, (time.perf_counter() - t0) * 1000


def _perf(label, dt_rust, dt_py=None):
    extra = ""
    if dt_py is not None:
        speedup = dt_py / max(dt_rust, 1e-6)
        extra = f" Python={dt_py:.1f}ms → {speedup:.1f}x speedup"
    print(f"\n  RUST {label}: {dt_rust:.2f} ms{extra}")


# ---------------------------------------------------------------------------
# 1. build_knm — exponential-decay coupling matrix
# ---------------------------------------------------------------------------


class TestBuildKnm:
    @pytest.mark.parametrize("n", [4, 8, 16])
    def test_shape_and_symmetry(self, n):
        K = np.array(eng.build_knm(n, 0.45, 0.3))
        assert K.shape == (n, n)
        np.testing.assert_allclose(K, K.T, atol=1e-14)

    def test_parity_with_python_paper27(self):
        """Rust build_knm matches Python build_knm_paper27 (includes overrides)."""
        from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27

        n = 16
        K_rust, dt_r = _timed(eng.build_knm, n, 0.45, 0.3)
        K_rust = np.array(K_rust)

        t0 = time.perf_counter()
        K_py = build_knm_paper27(L=n)
        dt_p = (time.perf_counter() - t0) * 1000

        np.testing.assert_allclose(K_rust, K_py, atol=1e-12)
        _perf("build_knm (16×16)", dt_r, dt_p)

    def test_non_negative(self):
        K = np.array(eng.build_knm(8, 0.45, 0.3))
        assert np.all(K >= 0)

    def test_diagonal_equals_base(self):
        K = np.array(eng.build_knm(4, 0.45, 0.3))
        np.testing.assert_allclose(np.diag(K), 0.45, atol=1e-14)


# ---------------------------------------------------------------------------
# 2. kuramoto_euler — single-step integration
# ---------------------------------------------------------------------------


class TestKuramotoEuler:
    def test_output_shape(self):
        theta0 = np.zeros(4, dtype=np.float64)
        omega = np.ones(4, dtype=np.float64)
        K = np.eye(4, dtype=np.float64) * 0.3
        result = np.array(eng.kuramoto_euler(theta0, omega, K, 0.01, 10))
        assert result.shape == (4,)

    def test_deterministic(self):
        theta0 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        omega = np.array([1.0, 2.0, 1.5, 0.5], dtype=np.float64)
        K = np.eye(4, dtype=np.float64) * 0.5
        r1 = np.array(eng.kuramoto_euler(theta0, omega, K, 0.01, 100))
        r2 = np.array(eng.kuramoto_euler(theta0, omega, K, 0.01, 100))
        np.testing.assert_array_equal(r1, r2)

    def test_finite_output(self):
        theta0 = np.random.default_rng(42).uniform(0, 2 * np.pi, 8).astype(np.float64)
        omega = np.ones(8, dtype=np.float64)
        K = np.eye(8, dtype=np.float64) * 0.3
        result = np.array(eng.kuramoto_euler(theta0, omega, K, 0.01, 100))
        assert np.all(np.isfinite(result))

    def test_performance_vs_python(self):
        from scpn_quantum_control.hardware.classical import classical_kuramoto_reference

        n = 8
        theta0 = np.zeros(n, dtype=np.float64)
        omega = np.ones(n, dtype=np.float64)
        K = np.eye(n, dtype=np.float64) * 0.3

        _, dt_r = _timed(eng.kuramoto_euler, theta0, omega, K, 0.01, 1000)
        _, dt_p = _timed(classical_kuramoto_reference, n_osc=n, t_max=10.0, dt=0.01)
        _perf("kuramoto_euler (8 osc, 1000 steps)", dt_r, dt_p)


# ---------------------------------------------------------------------------
# 3. order_parameter — R = |mean(exp(i*theta))|
# ---------------------------------------------------------------------------


class TestOrderParameter:
    def test_all_equal_R_one(self):
        theta = np.zeros(16, dtype=np.float64)
        R = eng.order_parameter(theta)
        np.testing.assert_allclose(R, 1.0, atol=1e-14)

    def test_opposite_phases_R_zero(self):
        theta = np.array([0.0, np.pi, 0.0, np.pi], dtype=np.float64)
        R = eng.order_parameter(theta)
        np.testing.assert_allclose(R, 0.0, atol=1e-14)

    def test_bounded(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            theta = rng.uniform(0, 2 * np.pi, 16).astype(np.float64)
            R = eng.order_parameter(theta)
            assert 0.0 <= R <= 1.0 + 1e-14


# ---------------------------------------------------------------------------
# 4. kuramoto_trajectory — full time series
# ---------------------------------------------------------------------------


class TestKuramotoTrajectory:
    def test_output_structure(self):
        """Trajectory returns (R_series, theta_flat) tuple."""
        n, steps = 4, 50
        theta0 = np.zeros(n, dtype=np.float64)
        omega = np.ones(n, dtype=np.float64)
        K = np.eye(n, dtype=np.float64) * 0.3
        result = eng.kuramoto_trajectory(theta0, omega, K, 0.01, steps)
        # Returns tuple: (R_values array, theta_values array)
        assert len(result) == 2
        R_arr = np.array(result[0])
        assert len(R_arr) == steps + 1

    def test_R_bounded_throughout(self):
        theta0 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        omega = np.ones(4, dtype=np.float64)
        K = np.eye(4, dtype=np.float64) * 0.3
        result = eng.kuramoto_trajectory(theta0, omega, K, 0.01, 100)
        R_arr = np.array(result[0])
        assert np.all(R_arr >= 0)
        assert np.all(R_arr <= 1.0 + 1e-10)

    def test_performance(self):
        n = 16
        theta0 = np.zeros(n, dtype=np.float64)
        omega = np.ones(n, dtype=np.float64)
        K = np.array(eng.build_knm(n, 0.45, 0.3), dtype=np.float64)
        _, dt = _timed(eng.kuramoto_trajectory, theta0, omega, K, 0.001, 10000)
        _perf("kuramoto_trajectory (16 osc, 10k steps)", dt)


# ---------------------------------------------------------------------------
# 5. build_xy_hamiltonian_dense — full matrix
# ---------------------------------------------------------------------------


class TestBuildXYHamiltonianDense:
    def test_output_size(self):
        n = 3
        K = np.eye(n, dtype=np.float64).ravel() * 0.3
        omega = np.ones(n, dtype=np.float64)
        H = np.array(eng.build_xy_hamiltonian_dense(K, omega, n))
        dim = 2**n
        assert H.shape == (dim * dim,) or H.shape == (dim, dim)

    def test_parity_with_qiskit(self):
        from scpn_quantum_control.bridge.knm_hamiltonian import (
            OMEGA_N_16,
            build_knm_paper27,
            knm_to_hamiltonian,
        )

        n = 3
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]

        H_qiskit = knm_to_hamiltonian(K, omega).to_matrix()
        if hasattr(H_qiskit, "toarray"):
            H_qiskit = H_qiskit.toarray()

        K_flat = np.ascontiguousarray(K.ravel(), dtype=np.float64)
        H_rust_flat, dt_r = _timed(
            eng.build_xy_hamiltonian_dense, K_flat, omega.astype(np.float64), n
        )
        dim = 2**n
        H_rust = np.array(H_rust_flat).reshape(dim, dim)

        np.testing.assert_allclose(H_rust, H_qiskit, atol=1e-10)
        _perf(f"build_xy_hamiltonian_dense ({n}q)", dt_r)

    def test_hermitian(self):
        n = 4
        K_flat = np.ascontiguousarray(np.eye(n, dtype=np.float64).ravel() * 0.3)
        omega = np.ones(n, dtype=np.float64)
        H = np.array(eng.build_xy_hamiltonian_dense(K_flat, omega, n)).reshape(2**n, 2**n)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-14)


# ---------------------------------------------------------------------------
# 6. build_sparse_xy_hamiltonian
# ---------------------------------------------------------------------------


class TestBuildSparseXYHamiltonian:
    def test_returns_tuple(self):
        n = 3
        K_flat = np.ascontiguousarray(np.eye(n, dtype=np.float64).ravel() * 0.3)
        omega = np.ones(n, dtype=np.float64)
        result = eng.build_sparse_xy_hamiltonian(K_flat, omega, n)
        assert isinstance(result, tuple)
        assert len(result) == 3  # (data, indices, indptr)


# ---------------------------------------------------------------------------
# 7. expectation_pauli_fast
# ---------------------------------------------------------------------------


class TestExpectationPauliFast:
    def test_z_on_ground_state(self):
        n = 4
        psi_re = np.zeros(2**n, dtype=np.float64)
        psi_re[0] = 1.0
        psi_im = np.zeros(2**n, dtype=np.float64)
        # Pauli: 1=X, 2=Y, 3=Z
        for q in range(n):
            z = eng.expectation_pauli_fast(psi_re, psi_im, n, q, 3)
            np.testing.assert_allclose(z, 1.0, atol=1e-14)

    def test_x_on_ground_state_zero(self):
        n = 3
        psi_re = np.zeros(2**n, dtype=np.float64)
        psi_re[0] = 1.0
        psi_im = np.zeros(2**n, dtype=np.float64)
        for q in range(n):
            x = eng.expectation_pauli_fast(psi_re, psi_im, n, q, 1)
            np.testing.assert_allclose(x, 0.0, atol=1e-14)

    def test_performance(self):
        n = 10
        dim = 2**n
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        psi /= np.linalg.norm(psi)
        psi_re = np.ascontiguousarray(psi.real, dtype=np.float64)
        psi_im = np.ascontiguousarray(psi.imag, dtype=np.float64)

        _, dt = _timed(eng.expectation_pauli_fast, psi_re, psi_im, n, 0, 3)
        _perf(f"expectation_pauli_fast ({n}q, Z)", dt)


# ---------------------------------------------------------------------------
# 8. all_xy_expectations
# ---------------------------------------------------------------------------


class TestAllXYExpectations:
    def test_shape(self):
        n = 4
        psi_re = np.zeros(2**n, dtype=np.float64)
        psi_re[0] = 1.0
        psi_im = np.zeros(2**n, dtype=np.float64)
        exps = np.array(eng.all_xy_expectations(psi_re, psi_im, n))
        assert exps.shape == (2, n)

    def test_ground_state_zero_xy(self):
        """|0...0> has <X>=<Y>=0 for all qubits."""
        n = 4
        psi_re = np.zeros(2**n, dtype=np.float64)
        psi_re[0] = 1.0
        psi_im = np.zeros(2**n, dtype=np.float64)
        exps = np.array(eng.all_xy_expectations(psi_re, psi_im, n))
        np.testing.assert_allclose(exps, 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# 9. order_param_from_statevector
# ---------------------------------------------------------------------------


class TestOrderParamFromStatevector:
    def test_ground_state_R_zero(self):
        """|0...0> has R=0 (all Z=+1, X=Y=0)."""
        n = 4
        psi_re = np.zeros(2**n, dtype=np.float64)
        psi_re[0] = 1.0
        psi_im = np.zeros(2**n, dtype=np.float64)
        R = eng.order_param_from_statevector(psi_re, psi_im, n)
        np.testing.assert_allclose(R, 0.0, atol=1e-14)

    def test_bounded(self):
        n = 4
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
        psi /= np.linalg.norm(psi)
        R = eng.order_param_from_statevector(
            np.ascontiguousarray(psi.real), np.ascontiguousarray(psi.imag), n
        )
        assert 0.0 <= R <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# 10. pec_coefficients
# ---------------------------------------------------------------------------


class TestPECCoefficients:
    def test_length_4(self):
        coeffs = np.array(eng.pec_coefficients(0.01))
        assert len(coeffs) == 4

    def test_sum_to_one(self):
        """Quasi-probability coefficients should sum to ~1."""
        coeffs = np.array(eng.pec_coefficients(0.01))
        np.testing.assert_allclose(np.sum(coeffs), 1.0, atol=0.05)

    def test_parity_with_python(self):
        from scpn_quantum_control.mitigation.pec import pauli_twirl_decompose

        coeffs_rust = np.array(eng.pec_coefficients(0.01))
        coeffs_py = pauli_twirl_decompose(0.01, n_qubits=1)
        np.testing.assert_allclose(coeffs_rust, coeffs_py, atol=1e-10)

    @pytest.mark.parametrize("p", [0.001, 0.01, 0.05, 0.1])
    def test_various_error_rates(self, p):
        coeffs = np.array(eng.pec_coefficients(p))
        assert len(coeffs) == 4
        assert coeffs[0] > 0  # identity term always positive


# ---------------------------------------------------------------------------
# 11. pec_sample_parallel
# ---------------------------------------------------------------------------


class TestPECSampleParallel:
    def test_returns_result(self):
        result = eng.pec_sample_parallel(0.01, 3, 1000, 0.5, 42)
        assert result is not None

    def test_deterministic_with_seed(self):
        s1 = eng.pec_sample_parallel(0.01, 3, 100, 0.5, 42)
        s2 = eng.pec_sample_parallel(0.01, 3, 100, 0.5, 42)
        # Same seed → same result
        assert s1 == s2

    def test_performance(self):
        _, dt = _timed(eng.pec_sample_parallel, 0.01, 5, 100000, 0.5, 42)
        _perf("pec_sample_parallel (100k samples, 5 gates)", dt)


# ---------------------------------------------------------------------------
# 12. mc_xy_simulate
# ---------------------------------------------------------------------------


class TestMCXYSimulate:
    def test_returns_tuple(self):
        n = 4
        K_flat = np.ascontiguousarray(np.eye(n, dtype=np.float64).ravel() * 0.3)
        result = eng.mc_xy_simulate(K_flat, n, 1.0, 100, 50, 42)
        assert isinstance(result, tuple)
        assert len(result) == 3  # (magnetisation, energy, R)

    def test_R_bounded(self):
        n = 4
        K_flat = np.ascontiguousarray(np.eye(n, dtype=np.float64).ravel() * 2.0)
        _, _, R = eng.mc_xy_simulate(K_flat, n, 0.1, 500, 200, 42)
        assert 0.0 <= R <= 1.0 + 1e-10

    def test_performance(self):
        n = 8
        K_flat = np.ascontiguousarray(np.eye(n, dtype=np.float64).ravel() * 0.5)
        _, dt = _timed(eng.mc_xy_simulate, K_flat, n, 1.0, 5000, 2000, 42)
        _perf(f"mc_xy_simulate ({n} osc, 5k therm, 2k meas)", dt)


# ---------------------------------------------------------------------------
# 13. magnetisation_labels
# ---------------------------------------------------------------------------


class TestMagnetisationLabels:
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_length(self, n):
        labels = np.array(eng.magnetisation_labels(n))
        assert len(labels) == 2**n

    def test_range(self):
        labels = np.array(eng.magnetisation_labels(4))
        assert np.min(labels) == -4
        assert np.max(labels) == 4


# ---------------------------------------------------------------------------
# 14. dla_dimension
# ---------------------------------------------------------------------------


class TestDLADimension:
    def test_basic(self):
        # 2-qubit XY: generators = sigma_x ⊗ sigma_x, sigma_y ⊗ sigma_y, sigma_z ⊗ I, I ⊗ sigma_z
        dim = 4
        sx = np.array([[0, 1], [1, 0]], dtype=np.float64)
        sy = np.array([[0, -1], [1, 0]], dtype=np.float64)  # imag part
        sz = np.array([[1, 0], [0, -1]], dtype=np.float64)
        I2 = np.eye(2, dtype=np.float64)

        g1 = np.kron(sx, sx).ravel()
        g2 = np.kron(sy, sy).ravel()
        g3 = np.kron(sz, I2).ravel()
        g4 = np.kron(I2, sz).ravel()

        generators_flat = np.concatenate([g1, g2, g3, g4]).astype(np.float64)
        d = eng.dla_dimension(generators_flat, dim, 4, 100, 100, 1e-10)
        assert d >= 4  # At least as many as generators


# ---------------------------------------------------------------------------
# 15. brute_mpc
# ---------------------------------------------------------------------------


class TestBruteMPC:
    def test_basic(self):
        B = np.ascontiguousarray(np.eye(2, dtype=np.float64).ravel())
        target = np.array([1.0, 0.0], dtype=np.float64)
        result = eng.brute_mpc(B, target, 2, 3)
        assert result is not None

    def test_performance(self):
        B = np.ascontiguousarray(np.eye(3, dtype=np.float64).ravel())
        target = np.ones(3, dtype=np.float64)
        _, dt = _timed(eng.brute_mpc, B, target, 3, 4)
        _perf("brute_mpc (dim=3, horizon=4)", dt)


# ---------------------------------------------------------------------------
# 16. lanczos_b_coefficients
# ---------------------------------------------------------------------------


class TestLanczosBCoefficients:
    def test_basic(self):
        n = 2
        dim = 2**n
        # Non-trivial Hamiltonian to produce Lanczos coefficients
        ham = np.diag(np.arange(dim, dtype=np.float64))
        ham[0, 1] = ham[1, 0] = 0.5  # off-diagonal coupling
        # Operator: Pauli X on qubit 0
        op_x = np.zeros((dim, dim), dtype=np.float64)
        op_x[0, 1] = op_x[1, 0] = 1.0
        op_x[2, 3] = op_x[3, 2] = 1.0
        H_re = np.ascontiguousarray(ham.ravel(), dtype=np.float64)
        H_im = np.zeros_like(H_re)
        O_re = np.ascontiguousarray(op_x.ravel(), dtype=np.float64)
        O_im = np.zeros_like(O_re)
        b = np.array(eng.lanczos_b_coefficients(H_re, H_im, O_re, O_im, dim, 3, 1e-14))
        assert np.all(np.isfinite(b))


# ---------------------------------------------------------------------------
# 17–18. otoc_from_eigendecomp + state_order_param_sparse
# ---------------------------------------------------------------------------


class TestOTOCFromEigendecomp:
    def test_basic(self):
        n = 2
        dim = 2**n
        eigenvalues = np.arange(dim, dtype=np.float64)
        eigvecs = np.eye(dim, dtype=np.float64)

        W = np.eye(dim, dtype=np.float64)
        V = np.eye(dim, dtype=np.float64)
        psi = np.zeros(dim, dtype=np.float64)
        psi[0] = 1.0
        times = np.array([0.0, 0.1, 0.5], dtype=np.float64)

        result = np.array(
            eng.otoc_from_eigendecomp(
                eigenvalues,
                eigvecs.ravel().astype(np.float64),
                np.zeros(dim * dim, dtype=np.float64),
                W.ravel().astype(np.float64),
                np.zeros(dim * dim, dtype=np.float64),
                V.ravel().astype(np.float64),
                np.zeros(dim * dim, dtype=np.float64),
                psi,
                np.zeros(dim, dtype=np.float64),
                times,
                dim,
            )
        )
        assert len(result) == 3
        assert np.all(np.isfinite(result))


class TestStateOrderParamSparse:
    def test_ground_state(self):
        n = 4
        psi_re = np.zeros(2**n, dtype=np.float64)
        psi_re[0] = 1.0
        psi_im = np.zeros(2**n, dtype=np.float64)
        R = eng.state_order_param_sparse(psi_re, psi_im, n)
        assert 0.0 <= R <= 1.0 + 1e-10

    def test_approximately_matches_dense(self):
        """Sparse and dense R computations agree within numerical tolerance."""
        n = 4
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
        psi /= np.linalg.norm(psi)
        psi_re = np.ascontiguousarray(psi.real, dtype=np.float64)
        psi_im = np.ascontiguousarray(psi.imag, dtype=np.float64)

        R1 = eng.order_param_from_statevector(psi_re, psi_im, n)
        R2 = eng.state_order_param_sparse(psi_re, psi_im, n)
        # Allow small deviation due to different computation paths
        np.testing.assert_allclose(R1, R2, atol=0.01)


# ---------------------------------------------------------------------------
# 19. correlation_matrix_xy — XY correlation matrix from statevector
# ---------------------------------------------------------------------------


class TestCorrelationMatrixXY:
    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_shape_and_symmetry(self, n):
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
        psi /= np.linalg.norm(psi)
        C = np.array(eng.correlation_matrix_xy(psi.real.copy(), psi.imag.copy(), n))
        assert C.shape == (n, n)
        np.testing.assert_allclose(C, C.T, atol=1e-12)

    def test_diagonal_zero(self):
        n = 4
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
        psi /= np.linalg.norm(psi)
        C = np.array(eng.correlation_matrix_xy(psi.real.copy(), psi.imag.copy(), n))
        np.testing.assert_allclose(np.diag(C), 0.0)

    def test_parity_with_qiskit(self):
        """Rust correlation matrix matches Qiskit expectation values."""
        from qiskit.quantum_info import SparsePauliOp, Statevector

        n = 3
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
        psi /= np.linalg.norm(psi)

        C_rust, dt_r = _timed(eng.correlation_matrix_xy, psi.real.copy(), psi.imag.copy(), n)
        C_rust = np.array(C_rust)

        t0 = time.perf_counter()
        sv = Statevector(psi)
        C_py = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                x_str = ["I"] * n
                x_str[i] = "X"
                x_str[j] = "X"
                xx = sv.expectation_value(SparsePauliOp("".join(reversed(x_str)))).real
                y_str = ["I"] * n
                y_str[i] = "Y"
                y_str[j] = "Y"
                yy = sv.expectation_value(SparsePauliOp("".join(reversed(y_str)))).real
                C_py[i, j] = xx + yy
                C_py[j, i] = xx + yy
        dt_p = (time.perf_counter() - t0) * 1000

        np.testing.assert_allclose(C_rust, C_py, atol=1e-10)
        _perf(f"correlation_matrix_xy (n={n})", dt_r, dt_p)

    def test_benchmark_scaling(self):
        """Benchmark correlation_matrix_xy for increasing system sizes."""
        for n in [4, 6, 8]:
            rng = np.random.default_rng(42)
            psi = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
            psi /= np.linalg.norm(psi)
            _, dt = _timed(eng.correlation_matrix_xy, psi.real.copy(), psi.imag.copy(), n)
            _perf(f"correlation_matrix_xy (n={n})", dt)


# ---------------------------------------------------------------------------
# 20. lindblad_jump_ops_coo — Lindblad jump operator COO data
# ---------------------------------------------------------------------------


class TestLindbladJumpOpsCOO:
    def test_operator_count(self):
        """Number of jump operators matches Python construction."""
        K = np.array([[0, 0.5, 0.1], [0.5, 0, 0.3], [0.1, 0.3, 0]])
        n = 3
        _, _, starts, n_ops = eng.lindblad_jump_ops_coo(K.ravel(), n, 1e-5)
        # 6 active directed pairs (i,j): (0,1),(1,0),(0,2),(2,0),(1,2),(2,1)
        assert n_ops == 6

    def test_sparse_matrix_reconstruction(self):
        """COO data can reconstruct valid sparse matrices."""
        from scipy.sparse import csr_matrix

        K = np.array([[0, 0.5], [0.5, 0]])
        n = 2
        dim = 4
        rows, cols, starts, n_ops = eng.lindblad_jump_ops_coo(K.ravel(), n, 1e-5)
        rows = np.array(rows)
        cols = np.array(cols)
        starts = np.array(starts)

        for k in range(n_ops):
            s, e = int(starts[k]), int(starts[k + 1])
            data = np.ones(e - s)
            L = csr_matrix((data, (rows[s:e], cols[s:e])), shape=(dim, dim))
            # Each L should have exactly 1 non-zero per column (at most)
            assert L.nnz <= dim

    def test_parity_with_python(self):
        """Rust COO data matches Python loop construction."""
        K = np.array([[0, 0.5, 0], [0.5, 0, 0.3], [0, 0.3, 0]])
        n = 3
        dim = 1 << n

        rows_r, cols_r, starts_r, n_ops = eng.lindblad_jump_ops_coo(K.ravel(), n, 1e-5)
        rows_r = np.array(rows_r)
        cols_r = np.array(cols_r)
        starts_r = np.array(starts_r)

        t0 = time.perf_counter()
        py_ops = []
        for i in range(n):
            for j in range(n):
                if i != j and abs(K[i, j]) > 1e-5:
                    r_py, c_py = [], []
                    for idx in range(dim):
                        if ((idx >> i) & 1) == 1 and ((idx >> j) & 1) == 0:
                            r_py.append(idx ^ ((1 << i) | (1 << j)))
                            c_py.append(idx)
                    py_ops.append((sorted(r_py), sorted(c_py)))
        dt_p = (time.perf_counter() - t0) * 1000

        _, dt_r = _timed(eng.lindblad_jump_ops_coo, K.ravel(), n, 1e-5)

        assert n_ops == len(py_ops)
        for k in range(n_ops):
            s, e = int(starts_r[k]), int(starts_r[k + 1])
            assert sorted(rows_r[s:e]) == py_ops[k][0]
            assert sorted(cols_r[s:e]) == py_ops[k][1]

        _perf(f"lindblad_jump_ops_coo (n={n})", dt_r, dt_p)

    def test_benchmark_scaling(self):
        for n in [3, 5, 7]:
            rng = np.random.default_rng(42)
            K = rng.random((n, n)) * 0.3
            K = (K + K.T) / 2
            np.fill_diagonal(K, 0)
            _, dt = _timed(eng.lindblad_jump_ops_coo, K.ravel(), n, 1e-5)
            _perf(f"lindblad_jump_ops_coo (n={n})", dt)


# ---------------------------------------------------------------------------
# 21. lindblad_anti_hermitian_diag — anti-Hermitian sum diagonal
# ---------------------------------------------------------------------------


class TestLindbladAntiHermitianDiag:
    def test_parity_with_python(self):
        K = np.array([[0, 0.5, 0.1], [0.5, 0, 0.3], [0.1, 0.3, 0]])
        n = 3
        dim = 1 << n

        diag_rust, dt_r = _timed(eng.lindblad_anti_hermitian_diag, K.ravel(), n, 1e-5)
        diag_rust = np.array(diag_rust)

        t0 = time.perf_counter()
        diag_py = np.zeros(dim)
        for i in range(n):
            for j in range(n):
                if i != j and abs(K[i, j]) > 1e-5:
                    for idx in range(dim):
                        if ((idx >> i) & 1) == 1 and ((idx >> j) & 1) == 0:
                            diag_py[idx] += 1.0
        dt_p = (time.perf_counter() - t0) * 1000

        np.testing.assert_allclose(diag_rust, diag_py)
        _perf(f"lindblad_anti_hermitian_diag (n={n})", dt_r, dt_p)

    def test_non_negative(self):
        n = 4
        rng = np.random.default_rng(42)
        K = rng.random((n, n)) * 0.5
        K = (K + K.T) / 2
        np.fill_diagonal(K, 0)
        diag = np.array(eng.lindblad_anti_hermitian_diag(K.ravel(), n, 1e-5))
        assert np.all(diag >= 0)

    def test_ground_state_zero(self):
        """All-zeros state |0...0> has zero jump channels."""
        K = np.ones((3, 3)) - np.eye(3)
        n = 3
        diag = np.array(eng.lindblad_anti_hermitian_diag(K.ravel(), n, 1e-5))
        assert diag[0] == 0.0  # |000> — no excitations to transfer


# ---------------------------------------------------------------------------
# 22. parity_filter_mask — Z2 parity classification
# ---------------------------------------------------------------------------


class TestParityFilterMask:
    def test_known_parities(self):
        bs = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint64)
        even = np.array(eng.parity_filter_mask(bs, 0))
        odd = np.array(eng.parity_filter_mask(bs, 1))
        # popcount: 0→0, 1→1, 2→1, 3→2, 4→1, 5→2, 6→2, 7→3
        expected_even = [True, False, False, True, False, True, True, False]
        expected_odd = [False, True, True, False, True, False, False, True]
        assert list(even) == expected_even
        assert list(odd) == expected_odd

    def test_empty_input(self):
        bs = np.array([], dtype=np.uint64)
        result = np.array(eng.parity_filter_mask(bs, 0))
        assert len(result) == 0

    def test_benchmark_large(self):
        """Benchmark parity filtering on 10k bitstrings."""
        rng = np.random.default_rng(42)
        bs = rng.integers(0, 2**20, size=10_000).astype(np.uint64)
        _, dt = _timed(eng.parity_filter_mask, bs, 0)
        _perf("parity_filter_mask (10k × 20-bit)", dt)
