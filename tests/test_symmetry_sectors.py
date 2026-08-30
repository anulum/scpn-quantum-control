# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Symmetry Sectors
"""Tests for Z2 parity sector decomposition."""

from __future__ import annotations

from typing import NoReturn

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.analysis import symmetry_sectors as symmetry_module
from scpn_quantum_control.analysis.symmetry_sectors import (
    basis_indices_by_parity,
    build_sector_hamiltonian,
    eigh_by_sector,
    level_spacing_by_sector,
    memory_estimate_mb,
)
from scpn_quantum_control.dense_budget import DenseAllocationError


def _system(n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build the deterministic test coupling and frequency arrays."""
    sites = np.arange(n)
    distances = np.abs(np.subtract.outer(sites, sites))
    K: NDArray[np.float64] = 0.45 * np.exp(-0.3 * distances)
    omega = np.linspace(0.8, 1.2, n, dtype=np.float64)
    return K, omega


class TestBasisPartition:
    """Verify computational-basis parity partitions."""

    def test_partition_n2(self) -> None:
        """Split the two-qubit basis into expected parity indices."""
        even, odd = basis_indices_by_parity(2)
        assert set(even) == {0, 3}  # |00>, |11>
        assert set(odd) == {1, 2}  # |01>, |10>

    def test_partition_n3(self) -> None:
        """Split the three-qubit basis evenly by parity."""
        even, odd = basis_indices_by_parity(3)
        assert len(even) == 4
        assert len(odd) == 4

    def test_partition_covers_all(self) -> None:
        """Cover every basis state exactly once."""
        for n in [2, 4, 6, 8]:
            even, odd = basis_indices_by_parity(n)
            assert len(even) + len(odd) == 2**n
            assert len(set(even) & set(odd)) == 0

    def test_partition_equal_size(self) -> None:
        """Keep even and odd sectors equal in dimension."""
        for n in [2, 4, 6, 8]:
            even, odd = basis_indices_by_parity(n)
            assert len(even) == len(odd) == 2 ** (n - 1)


class TestProjection:
    """Verify sparse construction of dense parity-sector matrices."""

    def test_sector_dimensions(self) -> None:
        """Return half-Hilbert-space matrices for both parities."""
        n = 4
        K, omega = _system(n)
        H_even, idx_even = build_sector_hamiltonian(K, omega, parity=0)
        H_odd, idx_odd = build_sector_hamiltonian(K, omega, parity=1)
        assert H_even.shape == (8, 8)
        assert H_odd.shape == (8, 8)

    def test_sector_hermitian(self) -> None:
        """Preserve Hermiticity after sector restriction."""
        n = 4
        K, omega = _system(n)
        H_even, _ = build_sector_hamiltonian(K, omega, parity=0)
        np.testing.assert_allclose(H_even, H_even.T, atol=1e-12)

    def test_sector_builder_does_not_use_full_dense_hamiltonian(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Build from the sparse Hamiltonian without a full dense allocation."""
        n = 4
        K, omega = _system(n)

        def fail_dense(*_args: object, **_kwargs: object) -> NoReturn:
            raise AssertionError("Z2 sector builder must not build full dense Hamiltonian")

        monkeypatch.setattr(symmetry_module, "knm_to_dense_matrix", fail_dense, raising=False)

        H_even, idx_even = build_sector_hamiltonian(K, omega, parity=0)

        assert H_even.shape == (len(idx_even), len(idx_even))
        np.testing.assert_allclose(H_even, H_even.T, atol=1e-12)

    def test_sector_builder_rejects_dense_budget_before_sparse_build(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reject an undersized budget before sparse Hamiltonian construction."""
        n = 4
        K, omega = _system(n)

        def fail_sparse(*_args: object, **_kwargs: object) -> NoReturn:
            raise AssertionError("sparse Hamiltonian must not build after budget rejection")

        monkeypatch.setattr(symmetry_module, "build_sparse_hamiltonian", fail_sparse)

        with pytest.raises(DenseAllocationError, match="Z2 parity sector dense workspace"):
            build_sector_hamiltonian(K, omega, parity=0, max_dense_gib=1e-12)


class TestEighBySector:
    """Verify independent parity-sector diagonalisation."""

    n: int
    K: NDArray[np.float64]
    omega: NDArray[np.float64]

    def setup_method(self) -> None:
        """Create a deterministic four-qubit system for each test."""
        self.n = 4
        self.K, self.omega = _system(self.n)

    def test_ground_energy_matches_full(self) -> None:
        """Match the full-Hamiltonian ground energy."""
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

    def test_all_eigenvalues_match(self) -> None:
        """Recover the complete sorted full-spectrum eigenvalues."""
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

    def test_correct_count(self) -> None:
        """Return one sector eigenvalue for every basis state."""
        result = eigh_by_sector(self.K, self.omega)
        total = len(result["eigvals_even"]) + len(result["eigvals_odd"])
        assert total == 2**self.n

    def test_eigh_by_sector_rejects_dense_budget_before_sparse_build(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reject an undersized eigensolver budget before sparse construction."""

        def fail_sparse(*_args: object, **_kwargs: object) -> NoReturn:
            raise AssertionError("sparse Hamiltonian must not build after budget rejection")

        monkeypatch.setattr(symmetry_module, "build_sparse_hamiltonian", fail_sparse)

        with pytest.raises(DenseAllocationError, match="Z2 parity sector dense workspace"):
            eigh_by_sector(self.K, self.omega, max_dense_gib=1e-12)


class TestLevelSpacing:
    """Verify parity-resolved adjacent-gap statistics."""

    def test_r_bar_bounded(self) -> None:
        """Keep both nondegenerate sector ratios within unit bounds."""
        n = 6
        K, omega = _system(n)
        result = level_spacing_by_sector(K, omega)
        assert 0 < result["r_bar_even"] < 1
        assert 0 < result["r_bar_odd"] < 1

    def test_dim_per_sector(self) -> None:
        """Report the expected half-Hilbert-space sector dimension."""
        n = 6
        K = np.eye(n, dtype=np.float64)
        omega = np.ones(n, dtype=np.float64)
        result = level_spacing_by_sector(K, omega)
        assert result["dim_per_sector"] == 2 ** (n - 1)


class TestMemory:
    """Verify dense exact-diagonalisation memory estimates."""

    def test_sector_halves_memory(self) -> None:
        """Reduce matrix storage fourfold after halving its dimension."""
        n = 16
        full = memory_estimate_mb(n, use_sectors=False)
        sector = memory_estimate_mb(n, use_sectors=True)
        assert sector < full / 3  # sector is dim/2 → memory/4

    def test_n16_sector_fits_32gb(self) -> None:
        """Keep the documented 16-qubit sector estimate below 32 GB."""
        assert memory_estimate_mb(16, use_sectors=True) < 32000


class TestLevelSpacingSmallSector:
    """Verify adjacent-gap behavior in undersized sectors."""

    def test_n2_has_nan_sector(self) -> None:
        """n=2: each Z2 sector has only 2 eigenvalues → 1 gap → r̄=nan."""
        from scpn_quantum_control.analysis.symmetry_sectors import level_spacing_by_sector

        K = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64)
        omega = np.array([1.0, 1.0], dtype=np.float64)
        result = level_spacing_by_sector(K, omega)
        # With 2 eigenvalues per sector, only 1 gap, so r_bar = nan
        assert np.isnan(result["r_bar_even"]) or np.isnan(result["r_bar_odd"])
