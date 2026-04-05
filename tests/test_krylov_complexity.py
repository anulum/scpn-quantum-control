# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Krylov Complexity
"""Tests for Krylov complexity at the synchronization transition."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.analysis.krylov_complexity import (
    KrylovResult,
    krylov_complexity,
    krylov_vs_coupling,
    lanczos_coefficients,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, knm_to_hamiltonian


def _ring(n: int) -> np.ndarray:
    T = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        T[i, j] = T[j, i] = 1.0
    return T


class TestLanczosCoefficients:
    def test_returns_coefficients(self):
        K = 2.0 * _ring(3)
        omega = OMEGA_N_16[:3]
        H = knm_to_hamiltonian(K, omega).to_matrix()
        Z0 = np.diag([1, 1, 1, 1, -1, -1, -1, -1]).astype(complex)
        b, basis = lanczos_coefficients(H, Z0, max_steps=10)
        assert len(b) > 0
        assert len(basis) == len(b) + 1

    def test_b_positive(self):
        K = 2.0 * _ring(3)
        omega = OMEGA_N_16[:3]
        H = knm_to_hamiltonian(K, omega).to_matrix()
        Z0 = np.diag([1, 1, 1, 1, -1, -1, -1, -1]).astype(complex)
        b, _ = lanczos_coefficients(H, Z0)
        assert np.all(b > 0)

    def test_identity_operator_no_growth(self):
        """[H, I] = 0 → b_1 = 0, Krylov space is 1D."""
        K = 2.0 * _ring(2)
        omega = OMEGA_N_16[:2]
        H = knm_to_hamiltonian(K, omega).to_matrix()
        I_op = np.eye(4, dtype=complex)
        b, basis = lanczos_coefficients(H, I_op)
        assert len(b) == 0


class TestKrylovComplexity:
    def test_returns_result(self):
        K = 2.0 * _ring(3)
        omega = OMEGA_N_16[:3]
        H = knm_to_hamiltonian(K, omega).to_matrix()
        Z0 = np.diag([1, 1, 1, 1, -1, -1, -1, -1]).astype(complex)
        result = krylov_complexity(H, Z0, t_max=5.0, n_times=30)
        assert isinstance(result, KrylovResult)
        assert len(result.krylov_complexity) == 30

    def test_starts_at_zero(self):
        """K(0) = 0 (operator starts in first Krylov basis element)."""
        K = 2.0 * _ring(3)
        omega = OMEGA_N_16[:3]
        H = knm_to_hamiltonian(K, omega).to_matrix()
        Z0 = np.diag([1, 1, 1, 1, -1, -1, -1, -1]).astype(complex)
        result = krylov_complexity(H, Z0, t_max=5.0, n_times=30)
        assert result.krylov_complexity[0] < 1e-10

    def test_nonnegative(self):
        K = 2.0 * _ring(3)
        omega = OMEGA_N_16[:3]
        H = knm_to_hamiltonian(K, omega).to_matrix()
        Z0 = np.diag([1, 1, 1, 1, -1, -1, -1, -1]).astype(complex)
        result = krylov_complexity(H, Z0, t_max=10.0, n_times=50)
        assert np.all(result.krylov_complexity >= -1e-10)

    def test_4qubit(self):
        K = 1.5 * _ring(4)
        omega = OMEGA_N_16[:4]
        H = knm_to_hamiltonian(K, omega).to_matrix()
        Z0 = np.zeros((16, 16), dtype=complex)
        for i in range(16):
            Z0[i, i] = 1.0 - 2.0 * (i & 1)
        result = krylov_complexity(H, Z0, t_max=5.0, n_times=20)
        assert result.peak_complexity > 0


class TestKrylovVsCoupling:
    def test_returns_dict(self):
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = krylov_vs_coupling(omega, T, k_range=np.array([1.0, 3.0]))
        assert "peak_complexity" in result
        assert len(result["K_base"]) == 2

    def test_complexity_varies(self):
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = krylov_vs_coupling(omega, T, k_range=np.linspace(0.5, 5.0, 5))
        assert not all(c == result["peak_complexity"][0] for c in result["peak_complexity"])

    def test_all_finite(self):
        T = _ring(3)
        omega = OMEGA_N_16[:3]
        result = krylov_vs_coupling(omega, T, k_range=np.array([1.0, 2.0, 4.0]))
        for key in result:
            assert all(np.isfinite(v) for v in result[key])


# ---------------------------------------------------------------------------
# Krylov physics: Lanczos convergence and operator growth
# ---------------------------------------------------------------------------


class TestKrylovPhysics:
    def test_lanczos_b_coefficients_decay(self):
        """Lanczos b_n should eventually decrease (finite Hilbert space)."""
        K = 2.0 * _ring(3)
        omega = OMEGA_N_16[:3]
        H = knm_to_hamiltonian(K, omega).to_matrix()
        Z0 = np.diag([1, 1, 1, 1, -1, -1, -1, -1]).astype(complex)
        b, _ = lanczos_coefficients(H, Z0, max_steps=20)
        if len(b) >= 3:
            # In finite dim, b_n eventually hits zero (Krylov exhaustion)
            assert b[-1] <= b[0] + 1e-10 or len(b) < 20

    def test_complexity_bounded_by_dimension(self):
        """K(t) ≤ dim(Krylov space) ≤ d² where d=2^n."""
        K = 2.0 * _ring(3)
        omega = OMEGA_N_16[:3]
        H = knm_to_hamiltonian(K, omega).to_matrix()
        Z0 = np.diag([1, 1, 1, 1, -1, -1, -1, -1]).astype(complex)
        result = krylov_complexity(H, Z0, t_max=5.0, n_times=20)
        d_sq = 8**2  # d² for 3 qubits
        assert np.max(result.krylov_complexity) <= d_sq


# ---------------------------------------------------------------------------
# Rust path: Lanczos coefficients
# ---------------------------------------------------------------------------


class TestKrylovRust:
    def test_rust_lanczos_parity(self):
        """Rust lanczos_b_coefficients should produce similar results."""
        try:
            import scpn_quantum_engine as eng
        except ImportError:
            import pytest

            pytest.skip("scpn-quantum-engine not available")

        K = 2.0 * _ring(2)
        omega = OMEGA_N_16[:2]
        H = np.array(knm_to_hamiltonian(K, omega).to_matrix())
        # Z_0 on qubit 0
        Z0 = np.zeros((4, 4), dtype=complex)
        Z0[0, 0] = Z0[1, 1] = 1.0
        Z0[2, 2] = Z0[3, 3] = -1.0

        dim = 4
        H_re = np.ascontiguousarray(H.real.ravel(), dtype=np.float64)
        H_im = np.ascontiguousarray(H.imag.ravel(), dtype=np.float64)
        O_re = np.ascontiguousarray(Z0.real.ravel(), dtype=np.float64)
        O_im = np.ascontiguousarray(Z0.imag.ravel(), dtype=np.float64)

        b_rust = np.array(eng.lanczos_b_coefficients(H_re, H_im, O_re, O_im, dim, 10, 1e-14))
        b_py, _ = lanczos_coefficients(H, Z0, max_steps=10)

        # Rust and Python should agree on at least first few coefficients
        n_compare = min(len(b_rust), len(b_py))
        if n_compare > 0:
            np.testing.assert_allclose(b_rust[:n_compare], b_py[:n_compare], atol=1e-6)


# ---------------------------------------------------------------------------
# Pipeline: Knm → Krylov → peak complexity → wired
# ---------------------------------------------------------------------------


class TestKrylovPipeline:
    def test_pipeline_knm_to_krylov(self):
        """Full pipeline: Knm → Hamiltonian → Krylov complexity → peak.
        Verifies Krylov module is wired end-to-end, not decorative.
        """
        import time

        from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27

        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        H = knm_to_hamiltonian(K, omega).to_matrix()
        Z0 = np.diag([1, 1, 1, 1, -1, -1, -1, -1]).astype(complex)

        t0 = time.perf_counter()
        result = krylov_complexity(H, Z0, t_max=5.0, n_times=30)
        dt = (time.perf_counter() - t0) * 1000

        assert result.krylov_complexity[0] < 1e-10
        assert result.peak_complexity > 0

        print(f"\n  PIPELINE Knm→Krylov (3q, 30 times): {dt:.1f} ms")
        print(f"  Peak K(t) = {result.peak_complexity:.4f}")


# ---------------------------------------------------------------------------
# Coverage: internal helpers, edge cases, default parameters
# ---------------------------------------------------------------------------


class TestInternalHelpers:
    def test_liouvillian_action_commutator(self):
        from scpn_quantum_control.analysis.krylov_complexity import _liouvillian_action

        H = np.array([[1, 0], [0, -1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        L_X = _liouvillian_action(H, X)
        expected = H @ X - X @ H
        np.testing.assert_allclose(L_X, expected, atol=1e-12)

    def test_liouvillian_identity_is_zero(self):
        from scpn_quantum_control.analysis.krylov_complexity import _liouvillian_action

        H = np.diag([1.0, 2.0, 3.0, 4.0]).astype(complex)
        I_op = np.eye(4, dtype=complex)
        L_I = _liouvillian_action(H, I_op)
        np.testing.assert_allclose(L_I, 0.0, atol=1e-12)

    def test_operator_inner_product_self(self):
        from scpn_quantum_control.analysis.krylov_complexity import (
            _operator_inner_product,
        )

        A = np.array([[1, 0], [0, -1]], dtype=complex)
        ip = _operator_inner_product(A, A)
        assert ip > 0

    def test_operator_inner_product_orthogonal(self):
        from scpn_quantum_control.analysis.krylov_complexity import (
            _operator_inner_product,
        )

        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        ip = _operator_inner_product(X, Z)
        assert abs(ip) < 1e-10  # Tr(X†Z)/d = 0

    def test_operator_inner_product_normalisation(self):
        from scpn_quantum_control.analysis.krylov_complexity import (
            _operator_inner_product,
        )

        I2 = np.eye(2, dtype=complex)
        ip = _operator_inner_product(I2, I2)
        np.testing.assert_allclose(ip, 1.0, atol=1e-12)  # Tr(I)/2 = 1


class TestLanczosEdgeCases:
    def test_zero_operator(self):
        K = 2.0 * _ring(2)
        omega = OMEGA_N_16[:2]
        H = knm_to_hamiltonian(K, omega).to_matrix()
        zero_op = np.zeros((4, 4), dtype=complex)
        b, basis = lanczos_coefficients(H, zero_op)
        assert len(b) == 1
        assert b[0] == 0.0

    def test_max_steps_respected(self):
        K = 2.0 * _ring(3)
        omega = OMEGA_N_16[:3]
        H = knm_to_hamiltonian(K, omega).to_matrix()
        Z0 = np.diag([1, 1, 1, 1, -1, -1, -1, -1]).astype(complex)
        b, _ = lanczos_coefficients(H, Z0, max_steps=3)
        assert len(b) <= 3


class TestKrylovComplexityEdgeCases:
    def test_trivial_operator(self):
        """Operator that commutes with H → constant zero complexity."""
        H = np.diag([1.0, 2.0, 3.0, 4.0]).astype(complex)
        D = np.diag([0.5, 0.3, 0.1, 0.7]).astype(complex)
        result = krylov_complexity(H, D, t_max=5.0, n_times=10)
        assert result.peak_complexity < 1e-6


class TestLanczosPythonFallback:
    """Cover lines 98-133: Python Lanczos when Rust unavailable."""

    def test_python_lanczos_no_rust(self):
        """Mock Rust import to fail → Python Lanczos executes."""
        from unittest.mock import patch

        T = _ring(2)
        omega = OMEGA_N_16[:2]
        K = 2.0 * T
        H = knm_to_hamiltonian(K, omega).to_matrix()
        # X_0 operator: off-diagonal, doesn't commute with H
        X0 = np.zeros((4, 4), dtype=complex)
        X0[0, 2] = X0[2, 0] = 1.0
        X0[1, 3] = X0[3, 1] = 1.0

        with patch.dict("sys.modules", {"scpn_quantum_engine": None}):
            b, basis = lanczos_coefficients(H, X0, max_steps=10)

        assert len(b) > 0
        assert all(bi >= 0 for bi in b)
        # Python path returns actual basis vectors
        assert len(basis) == len(b) + 1

    def test_python_lanczos_zero_operator(self):
        """Zero initial operator → b=[0], basis=[zero]."""
        from unittest.mock import patch

        H = np.eye(4, dtype=complex)
        O_zero = np.zeros((4, 4), dtype=complex)

        with patch.dict("sys.modules", {"scpn_quantum_engine": None}):
            b, basis = lanczos_coefficients(H, O_zero, max_steps=5)

        assert len(b) == 1
        assert b[0] == 0.0

    def test_krylov_complexity_python_fallback(self):
        """Full krylov_complexity via Python Lanczos path."""
        from unittest.mock import patch

        T = _ring(2)
        omega = OMEGA_N_16[:2]
        K = 2.0 * T
        H = knm_to_hamiltonian(K, omega).to_matrix()
        X0 = np.zeros((4, 4), dtype=complex)
        X0[0, 2] = X0[2, 0] = 1.0
        X0[1, 3] = X0[3, 1] = 1.0

        with patch.dict("sys.modules", {"scpn_quantum_engine": None}):
            result = krylov_complexity(H, X0, t_max=2.0, n_times=10, max_lanczos=10)

        assert isinstance(result, KrylovResult)
        assert result.peak_complexity >= 0
        assert len(result.times) == 10


class TestKrylovVsCouplingDefaults:
    def test_default_k_range(self):
        T = _ring(2)
        omega = OMEGA_N_16[:2]
        result = krylov_vs_coupling(omega, T)
        assert len(result["K_base"]) == 10  # default linspace 0.5-5.0, 10 pts
        assert "mean_b" in result
