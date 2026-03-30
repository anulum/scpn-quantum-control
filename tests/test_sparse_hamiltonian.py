# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
"""Tests for sparse XY Hamiltonian construction and eigensolvers."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.bridge.sparse_hamiltonian import (
    build_sparse_hamiltonian,
    build_sparse_sector_hamiltonian,
    sparse_eigsh,
    sparsity_stats,
)


def _system(n: int = 4):
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    omega = np.linspace(0.8, 1.2, n)
    return K, omega


class TestSparseConstruction:
    def test_shape(self):
        K, omega = _system(4)
        H = build_sparse_hamiltonian(K, omega)
        assert H.shape == (16, 16)

    def test_hermitian(self):
        K, omega = _system(4)
        H = build_sparse_hamiltonian(K, omega)
        diff = (H - H.T).toarray()
        np.testing.assert_allclose(diff, 0, atol=1e-12)

    def test_matches_dense(self):
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        K, omega = _system(4)
        H_dense = knm_to_dense_matrix(K, omega)
        H_sparse = build_sparse_hamiltonian(K, omega).toarray()
        np.testing.assert_allclose(H_sparse, H_dense, atol=1e-12)

    def test_matches_dense_n6(self):
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        K, omega = _system(6)
        H_dense = knm_to_dense_matrix(K, omega)
        H_sparse = build_sparse_hamiltonian(K, omega).toarray()
        np.testing.assert_allclose(H_sparse, H_dense, atol=1e-12)

    def test_nnz_reasonable(self):
        K, omega = _system(8)
        H = build_sparse_hamiltonian(K, omega)
        dim = 2**8
        assert H.nnz < dim * dim  # sparse, not dense
        assert H.nnz > dim  # more than just diagonal


class TestSparseSector:
    def test_sector_matches_dense_sector(self):
        from scpn_quantum_control.analysis.magnetisation_sectors import build_sector_hamiltonian

        K, omega = _system(4)
        H_dense, idx_dense = build_sector_hamiltonian(K, omega, M=0)
        H_sparse, idx_sparse = build_sparse_sector_hamiltonian(K, omega, M=0)
        np.testing.assert_array_equal(idx_dense, idx_sparse)
        np.testing.assert_allclose(H_sparse.toarray(), H_dense, atol=1e-12)

    def test_sector_smaller_than_full(self):
        K, omega = _system(6)
        H_full = build_sparse_hamiltonian(K, omega)
        H_m0, _ = build_sparse_sector_hamiltonian(K, omega, M=0)
        assert H_m0.shape[0] < H_full.shape[0]


class TestSparseEigsh:
    def test_ground_energy_matches_dense(self):
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        K, omega = _system(6)
        H = knm_to_dense_matrix(K, omega)
        e_exact = np.linalg.eigvalsh(H)[0]
        result = sparse_eigsh(K, omega, k=5)
        np.testing.assert_allclose(result["eigvals"][0], e_exact, atol=1e-8)

    def test_sector_ground_energy(self):
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

    def test_eigsh_output_keys(self):
        K, omega = _system(4)
        result = sparse_eigsh(K, omega, k=3)
        assert set(result.keys()) >= {"eigvals", "eigvecs", "nnz", "dim", "method"}

    def test_n8_sparse_feasible(self):
        K, omega = _system(8)
        result = sparse_eigsh(K, omega, k=5)
        assert len(result["eigvals"]) == 5
        assert result["method"] == "sparse_arpack"

    def test_small_n_falls_back_to_dense(self):
        K, omega = _system(2)
        result = sparse_eigsh(K, omega, k=3, M=0)
        assert result["method"] == "dense_fallback"


class TestSparsityStats:
    def test_n16_reduction(self):
        n = 16
        K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
        stats = sparsity_stats(n, K)
        assert stats["reduction_factor"] > 10
        assert stats["fill_pct"] < 1.0

    def test_n8_stats(self):
        K, omega = _system(8)
        stats = sparsity_stats(8, K)
        assert stats["dim"] == 256
        assert stats["memory_sparse_mb"] < stats["memory_dense_mb"]
