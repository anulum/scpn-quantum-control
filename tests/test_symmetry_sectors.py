# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Symmetry Sectors
"""Tests for Z2 parity sector decomposition."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.analysis.symmetry_sectors import (
    basis_indices_by_parity,
    build_sector_hamiltonian,
    eigh_by_sector,
    level_spacing_by_sector,
    memory_estimate_mb,
)


class TestBasisPartition:
    def test_partition_n2(self):
        even, odd = basis_indices_by_parity(2)
        assert set(even) == {0, 3}  # |00>, |11>
        assert set(odd) == {1, 2}  # |01>, |10>

    def test_partition_n3(self):
        even, odd = basis_indices_by_parity(3)
        assert len(even) == 4
        assert len(odd) == 4

    def test_partition_covers_all(self):
        for n in [2, 4, 6, 8]:
            even, odd = basis_indices_by_parity(n)
            assert len(even) + len(odd) == 2**n
            assert len(set(even) & set(odd)) == 0

    def test_partition_equal_size(self):
        for n in [2, 4, 6, 8]:
            even, odd = basis_indices_by_parity(n)
            assert len(even) == len(odd) == 2 ** (n - 1)


class TestProjection:
    def test_sector_dimensions(self):
        n = 4
        K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
        omega = np.linspace(0.8, 1.2, n)
        H_even, idx_even = build_sector_hamiltonian(K, omega, parity=0)
        H_odd, idx_odd = build_sector_hamiltonian(K, omega, parity=1)
        assert H_even.shape == (8, 8)
        assert H_odd.shape == (8, 8)

    def test_sector_hermitian(self):
        n = 4
        K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
        omega = np.linspace(0.8, 1.2, n)
        H_even, _ = build_sector_hamiltonian(K, omega, parity=0)
        np.testing.assert_allclose(H_even, H_even.T, atol=1e-12)


class TestEighBySector:
    def setup_method(self):
        self.n = 4
        self.K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(self.n), range(self.n))))
        self.omega = np.linspace(0.8, 1.2, self.n)

    def test_ground_energy_matches_full(self):
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        H_full = knm_to_dense_matrix(self.K, self.omega)
        e_full = np.linalg.eigvalsh(H_full)[0]

        result = eigh_by_sector(self.K, self.omega)
        np.testing.assert_allclose(
            result["ground_energy"],
            e_full,
            atol=1e-10,
            err_msg="Sector ED ground energy must match full ED",
        )

    def test_all_eigenvalues_match(self):
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        H_full = knm_to_dense_matrix(self.K, self.omega)
        e_full = np.sort(np.linalg.eigvalsh(H_full))

        result = eigh_by_sector(self.K, self.omega)
        np.testing.assert_allclose(
            result["eigvals_all"],
            e_full,
            atol=1e-10,
            err_msg="Sector eigenvalues combined must match full spectrum",
        )

    def test_correct_count(self):
        result = eigh_by_sector(self.K, self.omega)
        total = len(result["eigvals_even"]) + len(result["eigvals_odd"])
        assert total == 2**self.n


class TestLevelSpacing:
    def test_r_bar_bounded(self):
        n = 6
        K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
        omega = np.linspace(0.8, 1.2, n)
        result = level_spacing_by_sector(K, omega)
        assert 0 < result["r_bar_even"] < 1
        assert 0 < result["r_bar_odd"] < 1

    def test_dim_per_sector(self):
        n = 6
        K = np.eye(n)
        omega = np.ones(n)
        result = level_spacing_by_sector(K, omega)
        assert result["dim_per_sector"] == 2 ** (n - 1)


class TestMemory:
    def test_sector_halves_memory(self):
        n = 16
        full = memory_estimate_mb(n, use_sectors=False)
        sector = memory_estimate_mb(n, use_sectors=True)
        assert sector < full / 3  # sector is dim/2 → memory/4

    def test_n16_sector_fits_32gb(self):
        assert memory_estimate_mb(16, use_sectors=True) < 32000


class TestLevelSpacingSmallSector:
    def test_n2_has_nan_sector(self):
        """n=2: each Z2 sector has only 2 eigenvalues → 1 gap → r̄=nan."""
        from scpn_quantum_control.analysis.symmetry_sectors import level_spacing_by_sector

        K = np.array([[0.0, 0.5], [0.5, 0.0]])
        omega = np.array([1.0, 1.0])
        result = level_spacing_by_sector(K, omega)
        # With 2 eigenvalues per sector, only 1 gap, so r_bar = nan
        assert np.isnan(result["r_bar_even"]) or np.isnan(result["r_bar_odd"])
