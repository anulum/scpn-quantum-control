# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Sparse Hamiltonian
"""Tests for sparse XY Hamiltonian construction and eigensolvers."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.bridge.sparse_hamiltonian import (
    build_sparse_hamiltonian,
    build_sparse_sector_hamiltonian,
    sparse_eigsh,
    sparsity_stats,
)
from scpn_quantum_control.dense_budget import DenseAllocationError


def _system(n: int = 4) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a deterministic dense-coupling test system."""
    K = np.asarray(
        0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n)))),
        dtype=np.float64,
    )
    omega = np.linspace(0.8, 1.2, n, dtype=np.float64)
    return K, omega


def _asymmetric_system(n: int = 4) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Learned couplings may be directed; XY Hamiltonians use the Hermitian part."""
    K, omega = _system(n)
    K = K.copy()
    K[0, 1] += 0.17
    K[1, 0] -= 0.11
    K[0, 2] -= 0.09
    K[2, 0] += 0.05
    return K, omega


class TestSparseConstruction:
    """Exercise full-basis sparse Hamiltonian construction."""

    def test_shape(self) -> None:
        """Build the expected Hilbert-space matrix shape."""
        K, omega = _system(4)
        H = build_sparse_hamiltonian(K, omega)
        assert H.shape == (16, 16)

    def test_hermitian(self) -> None:
        """Build a Hermitian sparse Hamiltonian."""
        K, omega = _system(4)
        H = build_sparse_hamiltonian(K, omega)
        diff = (H - H.T).toarray()
        np.testing.assert_allclose(diff, 0, atol=1e-12)

    def test_matches_dense(self) -> None:
        """Match the dense bridge for four oscillators."""
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        K, omega = _system(4)
        H_dense = knm_to_dense_matrix(K, omega)
        H_sparse = build_sparse_hamiltonian(K, omega).toarray()
        np.testing.assert_allclose(H_sparse, H_dense, atol=1e-12)

    def test_matches_dense_n6(self) -> None:
        """Match the dense bridge for six oscillators."""
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        K, omega = _system(6)
        H_dense = knm_to_dense_matrix(K, omega)
        H_sparse = build_sparse_hamiltonian(K, omega).toarray()
        np.testing.assert_allclose(H_sparse, H_dense, atol=1e-12)

    def test_asymmetric_coupling_matches_dense_canonicalisation(self) -> None:
        """Canonicalise directed residue identically to the dense bridge."""
        from unittest.mock import patch

        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        K, omega = _asymmetric_system(4)
        H_dense = knm_to_dense_matrix(K, omega)
        with patch(
            "scpn_quantum_control.bridge.sparse_hamiltonian._try_rust_sparse",
            return_value=None,
        ):
            H_sparse = build_sparse_hamiltonian(K, omega).toarray()
        np.testing.assert_allclose(H_sparse, H_dense, atol=1e-12)

    def test_nnz_reasonable(self) -> None:
        """Retain nontrivial sparsity for an eight-oscillator system."""
        K, omega = _system(8)
        H = build_sparse_hamiltonian(K, omega)
        dim = 2**8
        assert H.nnz < dim * dim  # sparse, not dense
        assert H.nnz > dim  # more than just diagonal


class TestSparseSector:
    """Exercise magnetisation-sector sparse construction."""

    def test_sector_matches_dense_sector(self) -> None:
        """Match the dense sector builder and its basis ordering."""
        from scpn_quantum_control.analysis.magnetisation_sectors import build_sector_hamiltonian

        K, omega = _system(4)
        H_dense, idx_dense = build_sector_hamiltonian(K, omega, M=0)
        H_sparse, idx_sparse = build_sparse_sector_hamiltonian(K, omega, M=0)
        np.testing.assert_array_equal(idx_dense, idx_sparse)
        np.testing.assert_allclose(H_sparse.toarray(), H_dense, atol=1e-12)

    def test_asymmetric_sector_matches_full_dense_canonicalisation(self) -> None:
        """Match the canonical dense projection for asymmetric input."""
        from scpn_quantum_control.analysis.magnetisation_sectors import basis_by_magnetisation
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        K, omega = _asymmetric_system(4)
        H_dense = knm_to_dense_matrix(K, omega)
        H_sparse, idx_sparse = build_sparse_sector_hamiltonian(K, omega, M=0)
        idx_dense = basis_by_magnetisation(4)[0]
        np.testing.assert_array_equal(idx_sparse, idx_dense)
        np.testing.assert_allclose(
            H_sparse.toarray(),
            H_dense[np.ix_(idx_dense, idx_dense)],
            atol=1e-12,
        )

    def test_sector_smaller_than_full(self) -> None:
        """Reduce the full Hilbert space in the zero-magnetisation sector."""
        K, omega = _system(6)
        H_full = build_sparse_hamiltonian(K, omega)
        H_m0, _ = build_sparse_sector_hamiltonian(K, omega, M=0)
        assert H_m0.shape[0] < H_full.shape[0]


class TestSparseEigsh:
    """Exercise sparse and dense-fallback eigenvalue solving."""

    def test_ground_energy_matches_dense(self) -> None:
        """Match the dense ground-state energy."""
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        K, omega = _system(6)
        H = knm_to_dense_matrix(K, omega)
        e_exact = np.linalg.eigvalsh(H)[0]
        result = sparse_eigsh(K, omega, k=5)
        np.testing.assert_allclose(result["eigvals"][0], e_exact, atol=1e-8)

    def test_sector_ground_energy(self) -> None:
        """Recover the full ground energy across all sectors."""
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        K, omega = _system(6)
        H = knm_to_dense_matrix(K, omega)
        e_exact = np.linalg.eigvalsh(H)[0]
        # All sectors
        from scpn_quantum_control.analysis.magnetisation_sectors import sector_dimensions

        dims = sector_dimensions(6)
        best_e = float("inf")
        for m in dims:
            result = sparse_eigsh(K, omega, k=3, M=m)
            if result["eigvals"][0] < best_e:
                best_e = result["eigvals"][0]
        np.testing.assert_allclose(best_e, e_exact, atol=1e-8)

    def test_eigsh_output_keys(self) -> None:
        """Return the documented eigensolver metadata."""
        K, omega = _system(4)
        result = sparse_eigsh(K, omega, k=3)
        assert set(result.keys()) >= {"eigvals", "eigvecs", "nnz", "dim", "method"}

    def test_n8_sparse_feasible(self) -> None:
        """Use sparse ARPACK for a bounded eight-oscillator system."""
        K, omega = _system(8)
        result = sparse_eigsh(K, omega, k=5)
        assert len(result["eigvals"]) == 5
        assert result["method"] == "sparse_arpack"

    def test_small_n_falls_back_to_dense(self) -> None:
        """Use dense eigendecomposition when ARPACK cannot request one mode."""
        K, omega = _system(2)
        result = sparse_eigsh(K, omega, k=3, M=0)
        assert result["method"] == "dense_fallback"


class TestPythonFallback:
    """Cover Python fallback when Rust is unavailable."""

    def test_python_fallback_matches_dense(self) -> None:
        """Force Python path and verify against dense reference."""
        from unittest.mock import patch

        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        K, omega = _system(4)
        H_dense = knm_to_dense_matrix(K, omega)

        with patch(
            "scpn_quantum_control.bridge.sparse_hamiltonian._try_rust_sparse",
            return_value=None,
        ):
            H_sparse = build_sparse_hamiltonian(K, omega).toarray()
            np.testing.assert_allclose(H_sparse, H_dense, atol=1e-12)

    def test_python_fallback_hermitian(self) -> None:
        """Keep the forced Python construction Hermitian."""
        from unittest.mock import patch

        K, omega = _system(4)
        with patch(
            "scpn_quantum_control.bridge.sparse_hamiltonian._try_rust_sparse",
            return_value=None,
        ):
            H = build_sparse_hamiltonian(K, omega)
            diff = (H - H.T).toarray()
            np.testing.assert_allclose(diff, 0, atol=1e-12)

    def test_python_fallback_zero_coupling_skip(self) -> None:
        """Zero coupling entries should be skipped (fewer nnz)."""
        from unittest.mock import patch

        n = 4
        K = np.zeros((n, n))
        K[0, 1] = K[1, 0] = 0.5  # only one coupling
        omega = np.ones(n)
        with patch(
            "scpn_quantum_control.bridge.sparse_hamiltonian._try_rust_sparse",
            return_value=None,
        ):
            H = build_sparse_hamiltonian(K, omega)
            assert H.nnz < 2**n * 2**n

    def test_python_fallback_budget_guard_rejects_before_basis_loops(self) -> None:
        """The Python fallback fails closed before allocating COO basis arrays."""
        from unittest.mock import patch

        K, omega = _system(4)
        with (
            patch(
                "scpn_quantum_control.bridge.sparse_hamiltonian._try_rust_sparse",
                return_value=None,
            ),
            pytest.raises(DenseAllocationError, match="sparse XY Python builder COO workspace"),
        ):
            build_sparse_hamiltonian(K, omega, max_sparse_gib=1e-12)

    def test_missing_rust_symbol_falls_back(self) -> None:
        """_try_rust_sparse returns None when the optional symbol is absent."""
        # Mock scpn_quantum_engine to raise inside
        from unittest.mock import MagicMock, patch

        from scpn_quantum_control.bridge.sparse_hamiltonian import _try_rust_sparse

        mock_eng = MagicMock()
        del mock_eng.build_sparse_xy_hamiltonian
        with patch.dict("sys.modules", {"scpn_quantum_engine": mock_eng}):
            result = _try_rust_sparse(np.eye(2), np.ones(2), 2)
            assert result is None

    def test_rust_runtime_failure_is_not_silently_downgraded(self) -> None:
        """A present but failing Rust sparse builder must not be hidden."""
        from unittest.mock import MagicMock, patch

        from scpn_quantum_control.bridge.sparse_hamiltonian import _try_rust_sparse

        mock_eng = MagicMock()
        mock_eng.build_sparse_xy_hamiltonian.side_effect = RuntimeError("accelerator failed")
        with (
            patch.dict("sys.modules", {"scpn_quantum_engine": mock_eng}),
            pytest.raises(RuntimeError, match="accelerator failed"),
        ):
            _try_rust_sparse(np.eye(2), np.ones(2), 2)


class TestSectorErrors:
    """Exercise invalid and degenerate sector inputs."""

    def test_invalid_m_raises(self) -> None:
        """Reject a magnetisation absent from the sector catalogue."""
        K, omega = _system(4)
        with pytest.raises(ValueError, match="not valid"):
            build_sparse_sector_hamiltonian(K, omega, M=3)

    def test_sector_zero_coupling_skip(self) -> None:
        """Sector with zero coupling has fewer off-diagonal entries."""
        n = 4
        K = np.zeros((n, n))  # no coupling
        omega = np.ones(n)
        H, indices = build_sparse_sector_hamiltonian(K, omega, M=0)
        # Without coupling, only diagonal entries
        assert H.nnz == H.shape[0]


class TestSparsityStats:
    """Exercise analytical sparse-storage estimates."""

    def test_n16_reduction(self) -> None:
        """Estimate a material storage reduction at sixteen oscillators."""
        n = 16
        K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
        stats = sparsity_stats(n, K)
        assert stats["reduction_factor"] > 10
        assert stats["fill_pct"] < 1.0

    def test_n8_stats(self) -> None:
        """Report the expected dimension and sparse-memory advantage."""
        K, omega = _system(8)
        stats = sparsity_stats(8, K)
        assert stats["dim"] == 256
        assert stats["memory_sparse_mb"] < stats["memory_dense_mb"]
