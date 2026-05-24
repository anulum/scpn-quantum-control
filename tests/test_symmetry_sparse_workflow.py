# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Symmetry and sparse-solver workflow contract tests
"""Workflow tests for Z2, U1, sparse, dense, and translation-symmetry consistency."""

from __future__ import annotations

import numpy as np
import pytest


def _system(n: int = 4):
    """Standard heterogeneous Kuramoto-XY system."""
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    np.fill_diagonal(K, 0.0)
    omega = np.linspace(0.8, 1.2, n)
    return n, K, omega


def _homogeneous_system(n: int = 4):
    """Circulant K + uniform omega for translation symmetry."""
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = min(abs(i - j), n - abs(i - j))
            K[i, j] = 0.5 * np.exp(-0.3 * d) if d > 0 else 0
    omega = np.ones(n) * 1.0
    return n, K, omega


class TestSymmetryPipeline:
    """Symmetry sectors and sparse methods should give consistent eigenvalues
    as the pipeline progresses from full ED to sector ED to sparse."""

    def test_z2_eigenvalues_match_full_ed(self):
        """Z₂ sector eigenvalues should reconstruct the full spectrum."""
        from scpn_quantum_control.analysis.symmetry_sectors import eigh_by_sector
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        _, K, omega = _system(6)
        H = knm_to_dense_matrix(K, omega)
        eigvals_full = np.sort(np.linalg.eigvalsh(H))

        result = eigh_by_sector(K, omega)
        eigvals_sectors = np.sort(result["eigvals_all"])

        np.testing.assert_allclose(
            eigvals_sectors,
            eigvals_full,
            atol=1e-10,
            err_msg="Z₂ sector eigenvalues should match full ED",
        )

    def test_u1_eigenvalues_match_full_ed(self):
        """U(1) sector eigenvalues should reconstruct the full spectrum."""
        from scpn_quantum_control.analysis.magnetisation_sectors import eigh_by_magnetisation
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        _, K, omega = _system(6)
        H = knm_to_dense_matrix(K, omega)
        eigvals_full = np.sort(np.linalg.eigvalsh(H))

        result = eigh_by_magnetisation(K, omega)
        eigvals_u1 = np.sort(result["eigvals_all"])

        np.testing.assert_allclose(
            eigvals_u1,
            eigvals_full,
            atol=1e-10,
            err_msg="U(1) sector eigenvalues should match full ED",
        )

    def test_z2_u1_ground_energies_agree(self):
        """Z₂ and U(1) decompositions should find the same ground energy."""
        from scpn_quantum_control.analysis.magnetisation_sectors import eigh_by_magnetisation
        from scpn_quantum_control.analysis.symmetry_sectors import eigh_by_sector

        _, K, omega = _system(8)

        z2 = eigh_by_sector(K, omega)
        u1 = eigh_by_magnetisation(K, omega)

        np.testing.assert_allclose(
            z2["ground_energy"],
            u1["ground_energy"],
            atol=1e-10,
            err_msg="Z₂ and U(1) ground energies must agree",
        )

    def test_sparse_matches_dense_ground_energy(self):
        """Sparse eigsh ground energy should match dense eigh."""
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix
        from scpn_quantum_control.bridge.sparse_hamiltonian import sparse_eigsh

        _, K, omega = _system(6)
        H = knm_to_dense_matrix(K, omega)
        E_dense = np.linalg.eigvalsh(H)[0]

        result = sparse_eigsh(K, omega, k=5)
        E_sparse = result["eigvals"][0]

        np.testing.assert_allclose(
            E_sparse,
            E_dense,
            atol=1e-8,
            err_msg="Sparse eigsh should match dense ground energy",
        )

    def test_sparse_sector_matches_u1_sector(self):
        """Sparse eigsh within M=0 should match U(1) sector ED for M=0."""
        from scpn_quantum_control.analysis.magnetisation_sectors import eigh_by_magnetisation
        from scpn_quantum_control.bridge.sparse_hamiltonian import sparse_eigsh

        _, K, omega = _system(6)

        u1 = eigh_by_magnetisation(K, omega, sectors=[0])
        E_u1 = np.sort(u1["results"][0]["eigvals"])[:5]

        sparse = sparse_eigsh(K, omega, k=5, M=0)
        E_sparse = np.sort(sparse["eigvals"])

        np.testing.assert_allclose(
            E_sparse,
            E_u1,
            atol=1e-8,
            err_msg="Sparse M=0 eigsh should match dense U(1) M=0 sector",
        )

    def test_full_symmetry_pipeline_n8(self):
        """Full pipeline: Z₂ → U(1) → sparse at n=8, all giving same ground."""
        from scpn_quantum_control.analysis.magnetisation_sectors import eigh_by_magnetisation
        from scpn_quantum_control.analysis.symmetry_sectors import eigh_by_sector
        from scpn_quantum_control.bridge.sparse_hamiltonian import sparse_eigsh

        _, K, omega = _system(8)

        E_z2 = eigh_by_sector(K, omega)["ground_energy"]
        E_u1 = eigh_by_magnetisation(K, omega)["ground_energy"]
        E_sparse = sparse_eigsh(K, omega, k=3)["eigvals"][0]

        np.testing.assert_allclose(E_z2, E_u1, atol=1e-10)
        np.testing.assert_allclose(E_z2, E_sparse, atol=1e-8)

    def test_translation_within_full_spectrum(self):
        """Translation symmetry k=0 ground energy ≥ full ground energy."""
        from scpn_quantum_control.analysis.translation_symmetry import eigh_with_translation
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        _, K, omega = _homogeneous_system(6)
        H = knm_to_dense_matrix(K, omega)
        E_full = np.linalg.eigvalsh(H)[0]

        result = eigh_with_translation(K, omega, momentum=0)
        E_k0 = result["eigvals"][0]

        # k=0 ground may or may not be the global ground
        assert E_k0 >= E_full - 1e-8, f"k=0 ground {E_k0:.6f} below full ground {E_full:.6f}"

    def test_memory_estimates_consistent(self):
        """Memory estimates should decrease: full > Z₂ > U(1)."""
        from scpn_quantum_control.analysis.magnetisation_sectors import memory_estimate
        from scpn_quantum_control.analysis.symmetry_sectors import memory_estimate_mb

        n = 12
        full_mb = memory_estimate_mb(n, use_sectors=False)
        z2_mb = memory_estimate_mb(n, use_sectors=True)
        u1_est = memory_estimate(n)

        assert full_mb > z2_mb, "Z₂ should reduce memory vs full"
        assert z2_mb > u1_est["u1_largest_mb"], "U(1) should reduce vs Z₂"


class TestCrossModuleConsistency:
    """Verify that different modules agree on shared computations."""

    def test_sparse_hermiticity(self):
        """Sparse Hamiltonian should be Hermitian."""
        from scpn_quantum_control.bridge.sparse_hamiltonian import (
            build_sparse_hamiltonian,
        )

        _, K, omega = _system(6)
        H = build_sparse_hamiltonian(K, omega)
        diff = H - H.T.conj()
        assert diff.nnz == 0 or abs(diff).max() < 1e-12

    def test_sparse_vs_dense_matrix(self):
        """Sparse and dense Hamiltonians should be identical."""
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix
        from scpn_quantum_control.bridge.sparse_hamiltonian import (
            build_sparse_hamiltonian,
        )

        _, K, omega = _system(6)
        H_dense = knm_to_dense_matrix(K, omega)
        H_sparse = build_sparse_hamiltonian(K, omega).toarray()

        np.testing.assert_allclose(
            H_sparse,
            H_dense,
            atol=1e-12,
            err_msg="Sparse and dense Hamiltonians must match",
        )

    def test_sparsity_stats_consistent(self):
        """Sparsity stats should match actual sparse matrix properties."""
        from scpn_quantum_control.bridge.sparse_hamiltonian import (
            build_sparse_hamiltonian,
            sparsity_stats,
        )

        _, K, omega = _system(6)
        stats = sparsity_stats(6, K)
        H = build_sparse_hamiltonian(K, omega)

        assert stats["dim"] == H.shape[0]
        # NNZ estimate should be in the right ballpark
        assert stats["nnz_estimate"] > 0
        assert H.nnz > 0

    def test_lindblad_order_parameter_bounded(self):
        """Lindblad R(t) should stay in [0, 1] for all time steps."""
        from scpn_quantum_control.phase.lindblad import LindbladKuramotoSolver

        _, K, omega = _system(4)
        solver = LindbladKuramotoSolver(4, K, omega, gamma_amp=0.1, gamma_deph=0.05)
        result = solver.run(t_max=1.0, dt=0.05)

        assert all(0 <= r <= 1.01 for r in result["R"]), (
            f"R out of bounds: {[r for r in result['R'] if r < 0 or r > 1.01]}"
        )

    def test_mcwf_ensemble_produces_valid_statistics(self):
        """MCWF ensemble should produce bounded R with finite std."""
        from scpn_quantum_control.phase.tensor_jump import mcwf_ensemble

        _, K, omega = _system(4)

        result = mcwf_ensemble(
            K,
            omega,
            gamma_amp=0.1,
            t_max=0.5,
            dt=0.1,
            n_trajectories=50,
            seed=42,
        )

        # R_mean should be bounded [0, 1]
        assert all(0 <= r <= 1.01 for r in result["R_mean"]), "MCWF ensemble R_mean out of bounds"
        # R_std should be non-negative and finite
        assert all(s >= 0 for s in result["R_std"]), "R_std must be non-negative"
        assert all(np.isfinite(s) for s in result["R_std"]), "R_std must be finite"
        # Shape consistency
        assert result["R_trajectories"].shape == (50, len(result["times"]))

    @pytest.mark.parametrize("n", [4, 6, 8])
    def test_all_ed_methods_agree(self, n):
        """Full ED, Z₂, U(1), and sparse should all agree on ground energy."""
        from scpn_quantum_control.analysis.magnetisation_sectors import (
            eigh_by_magnetisation,
        )
        from scpn_quantum_control.analysis.symmetry_sectors import eigh_by_sector
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix
        from scpn_quantum_control.bridge.sparse_hamiltonian import sparse_eigsh

        _, K, omega = _system(n)

        H = knm_to_dense_matrix(K, omega)
        E_full = np.linalg.eigvalsh(H)[0]
        E_z2 = eigh_by_sector(K, omega)["ground_energy"]
        E_u1 = eigh_by_magnetisation(K, omega)["ground_energy"]
        E_sparse = sparse_eigsh(K, omega, k=1)["eigvals"][0]

        np.testing.assert_allclose(E_z2, E_full, atol=1e-10)
        np.testing.assert_allclose(E_u1, E_full, atol=1e-10)
        np.testing.assert_allclose(E_sparse, E_full, atol=1e-8)
