# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Tensor Contraction Optimiser
"""Tests for tensor contraction path optimiser.

Covers:
    - is_cotengra_available flag
    - optimal_contraction_path with numpy fallback
    - contract function correctness against np.einsum
    - benchmark_contraction output structure
    - Matrix multiply, trace, outer product contractions
    - cotengra path (if available) and fallback on exception
    - Edge case: single tensor, identity contraction
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from scpn_quantum_control.phase.contraction_optimiser import (
    benchmark_contraction,
    contract,
    is_cotengra_available,
    optimal_contraction_path,
)

# ── Availability flag ─────────────────────────────────────────────────


class TestCotengraAvailability:
    def test_returns_bool(self):
        result = is_cotengra_available()
        assert isinstance(result, bool)


# ── optimal_contraction_path ──────────────────────────────────────────


class TestOptimalContractionPath:
    def test_matmul_path(self):
        """Matrix multiply contraction returns valid path."""
        A = np.random.default_rng(42).random((4, 5))
        B = np.random.default_rng(43).random((5, 3))
        path, info = optimal_contraction_path("ij,jk->ik", A, B)
        assert isinstance(path, list)
        assert isinstance(info, dict)

    def test_numpy_fallback(self):
        """With optimiser='greedy', always uses numpy path."""
        A = np.random.default_rng(42).random((3, 4))
        B = np.random.default_rng(43).random((4, 2))
        path, info = optimal_contraction_path("ij,jk->ik", A, B, optimiser="greedy")
        assert info["method"] == "numpy_optimal"
        assert "info_string" in info

    def test_trace_path(self):
        """Trace contraction: ii->."""
        A = np.random.default_rng(42).random((5, 5))
        path, info = optimal_contraction_path("ii->", A)
        assert isinstance(path, list)

    def test_cotengra_exception_falls_through(self):
        """If cotengra raises, falls back to numpy."""
        import scpn_quantum_control.phase.contraction_optimiser as mod

        original = mod._COTENGRA_AVAILABLE
        try:
            mod._COTENGRA_AVAILABLE = True
            # Mock cotengra to raise
            mock_cotengra = type(
                "MockCotengra",
                (),
                {
                    "einsum_path": staticmethod(
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("test"))
                    )
                },
            )()
            with patch.object(mod, "cotengra", mock_cotengra):
                A = np.eye(3)
                B = np.eye(3)
                path, info = optimal_contraction_path("ij,jk->ik", A, B)
                assert info["method"] == "numpy_optimal"
        finally:
            mod._COTENGRA_AVAILABLE = original


# ── contract ──────────────────────────────────────────────────────────


class TestContract:
    def test_matmul_correctness(self):
        """contract('ij,jk->ik', A, B) == A @ B."""
        rng = np.random.default_rng(42)
        A = rng.random((4, 5))
        B = rng.random((5, 3))
        result = contract("ij,jk->ik", A, B)
        expected = A @ B
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_trace(self):
        """contract('ii->', A) == np.trace(A)."""
        A = np.random.default_rng(42).random((5, 5))
        result = contract("ii->", A)
        np.testing.assert_allclose(result, np.trace(A), atol=1e-12)

    def test_outer_product(self):
        """contract('i,j->ij', a, b) == np.outer(a, b)."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0])
        result = contract("i,j->ij", a, b)
        np.testing.assert_allclose(result, np.outer(a, b))

    def test_returns_ndarray(self):
        """Result is always numpy array."""
        A = np.eye(3)
        result = contract("ij->ij", A)
        assert isinstance(result, np.ndarray)

    def test_batch_matmul(self):
        """Batched contraction: bij,bjk->bik."""
        rng = np.random.default_rng(42)
        A = rng.random((2, 3, 4))
        B = rng.random((2, 4, 5))
        result = contract("bij,bjk->bik", A, B)
        expected = np.einsum("bij,bjk->bik", A, B)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_explicit_numpy_optimiser(self):
        """optimiser='greedy' forces numpy path."""
        A = np.eye(3)
        B = np.eye(3)
        result = contract("ij,jk->ik", A, B, optimiser="greedy")
        np.testing.assert_allclose(result, np.eye(3))

    def test_cotengra_path_if_available(self):
        """If cotengra is available, auto uses it."""
        if not is_cotengra_available():
            pytest.skip("cotengra not installed")
        A = np.random.default_rng(42).random((4, 4))
        B = np.random.default_rng(43).random((4, 4))
        result = contract("ij,jk->ik", A, B)
        np.testing.assert_allclose(result, A @ B, atol=1e-10)


# ── benchmark_contraction ────────────────────────────────────────────


class TestBenchmarkContraction:
    def test_output_keys(self):
        A = np.random.default_rng(42).random((3, 4))
        B = np.random.default_rng(43).random((4, 2))
        result = benchmark_contraction("ij,jk->ik", A, B, n_repeats=3)
        assert set(result.keys()) == {"naive_ms", "optimised_ms", "speedup"}

    def test_positive_times(self):
        A = np.random.default_rng(42).random((5, 5))
        B = np.random.default_rng(43).random((5, 5))
        result = benchmark_contraction("ij,jk->ik", A, B, n_repeats=3)
        assert result["naive_ms"] >= 0
        assert result["optimised_ms"] >= 0

    def test_speedup_is_ratio(self):
        A = np.eye(4)
        B = np.eye(4)
        result = benchmark_contraction("ij,jk->ik", A, B, n_repeats=5)
        # speedup can be 0.0 for tiny matrices where both paths round to ~0 ms
        assert result["speedup"] >= 0

    def test_single_repeat(self):
        A = np.eye(3)
        result = benchmark_contraction("ii->", A, n_repeats=1)
        assert "speedup" in result


# ── Mocked cotengra paths ─────────────────────────────────────────────


class TestMockedCotengraPath:
    def test_cotengra_path_used(self):
        """When cotengra available, optimal_contraction_path uses it."""
        from unittest.mock import MagicMock

        import scpn_quantum_control.phase.contraction_optimiser as mod

        original = mod._COTENGRA_AVAILABLE
        try:
            mod._COTENGRA_AVAILABLE = True
            mock_cotengra = MagicMock()
            mock_cotengra.einsum_path.return_value = (
                [(0, 1)],
                "mock_info",
            )
            with patch.object(mod, "cotengra", mock_cotengra):
                A = np.eye(3)
                B = np.eye(3)
                path, info = optimal_contraction_path("ij,jk->ik", A, B)
                assert info["method"] == "cotengra"
                mock_cotengra.einsum_path.assert_called_once()
        finally:
            mod._COTENGRA_AVAILABLE = original

    def test_cotengra_contract_used(self):
        """When cotengra available, contract uses cotengra.einsum."""
        from unittest.mock import MagicMock

        import scpn_quantum_control.phase.contraction_optimiser as mod

        original = mod._COTENGRA_AVAILABLE
        try:
            mod._COTENGRA_AVAILABLE = True
            mock_cotengra = MagicMock()
            mock_cotengra.einsum.return_value = np.eye(3)
            with patch.object(mod, "cotengra", mock_cotengra):
                A = np.eye(3)
                B = np.eye(3)
                result = contract("ij,jk->ik", A, B)
                np.testing.assert_allclose(result, np.eye(3))
                mock_cotengra.einsum.assert_called_once()
        finally:
            mod._COTENGRA_AVAILABLE = original
