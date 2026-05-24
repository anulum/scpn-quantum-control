# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Contraction optimiser contract tests
"""Contract tests for tensor contraction correctness and path metadata."""

from __future__ import annotations

import numpy as np


def _system(n: int = 4):
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    np.fill_diagonal(K, 0.0)
    omega = np.linspace(0.8, 1.2, n)
    return K, omega


def _homogeneous_system(n: int = 4):
    """Circulant K + uniform omega for translation symmetry tests."""
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = min(abs(i - j), n - abs(i - j))
            K[i, j] = 0.5 * np.exp(-0.3 * d) if d > 0 else 0
    omega = np.ones(n) * 1.0
    return K, omega


class TestContractionOptimiser:
    """Tensor contraction path optimiser tests."""

    def test_contract_matches_einsum_matmul(self):
        from scpn_quantum_control.phase.contraction_optimiser import contract

        rng = np.random.default_rng(42)
        A = rng.standard_normal((10, 20))
        B = rng.standard_normal((20, 15))
        result = contract("ij,jk->ik", A, B)
        expected = np.einsum("ij,jk->ik", A, B)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_contract_matches_einsum_chain(self):
        """Three-matrix chain contraction."""
        from scpn_quantum_control.phase.contraction_optimiser import contract

        rng = np.random.default_rng(42)
        A = rng.standard_normal((8, 12))
        B = rng.standard_normal((12, 10))
        C = rng.standard_normal((10, 6))
        result = contract("ij,jk,kl->il", A, B, C)
        expected = A @ B @ C
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_contract_trace(self):
        """Trace via einsum: Tr(A) = einsum('ii->', A)."""
        from scpn_quantum_control.phase.contraction_optimiser import contract

        A = np.random.randn(10, 10)
        result = contract("ii->", A)
        np.testing.assert_allclose(result, np.trace(A), atol=1e-10)

    def test_contract_outer_product(self):
        from scpn_quantum_control.phase.contraction_optimiser import contract

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0])
        result = contract("i,j->ij", a, b)
        expected = np.outer(a, b)
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_optimal_path_returns_valid_info(self):
        from scpn_quantum_control.phase.contraction_optimiser import (
            optimal_contraction_path,
        )

        A = np.random.randn(5, 10)
        B = np.random.randn(10, 8)
        path, info = optimal_contraction_path("ij,jk->ik", A, B)
        assert "method" in info
        assert isinstance(path, list)

    def test_benchmark_returns_valid_results(self):
        from scpn_quantum_control.phase.contraction_optimiser import (
            benchmark_contraction,
        )

        A = np.random.randn(30, 30)
        B = np.random.randn(30, 30)
        result = benchmark_contraction("ij,jk->ik", A, B, n_repeats=3)
        assert result["naive_ms"] > 0
        assert result["optimised_ms"] > 0
        assert result["speedup"] > 0
        assert np.isfinite(result["speedup"])

    def test_cotengra_availability_is_bool(self):
        from scpn_quantum_control.phase.contraction_optimiser import (
            is_cotengra_available,
        )

        assert isinstance(is_cotengra_available(), bool)

    def test_contract_identity(self):
        """Contracting with identity should preserve the matrix."""
        from scpn_quantum_control.phase.contraction_optimiser import contract

        A = np.random.randn(5, 5)
        eye = np.eye(5)
        result = contract("ij,jk->ik", A, eye)
        np.testing.assert_allclose(result, A, atol=1e-12)
