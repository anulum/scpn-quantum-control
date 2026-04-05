# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Magnetisation Sectors
"""Tests for U(1) magnetisation sector decomposition."""

from __future__ import annotations

from math import comb

import numpy as np

from scpn_quantum_control.analysis.magnetisation_sectors import (
    basis_by_magnetisation,
    eigh_by_magnetisation,
    largest_sector_dim,
    level_spacing_by_magnetisation,
    memory_estimate,
    sector_dimensions,
)


def _system(n: int = 4):
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    omega = np.linspace(0.8, 1.2, n)
    return K, omega


class TestBasisPartition:
    def test_total_count(self):
        for n in [2, 4, 6, 8]:
            sectors = basis_by_magnetisation(n)
            total = sum(len(v) for v in sectors.values())
            assert total == 2**n, f"n={n}: {total} != {2**n}"

    def test_no_overlap(self):
        sectors = basis_by_magnetisation(4)
        all_indices = []
        for v in sectors.values():
            all_indices.extend(v.tolist())
        assert len(all_indices) == len(set(all_indices))

    def test_correct_M_values(self):
        sectors = basis_by_magnetisation(4)
        assert set(sectors.keys()) == {-4, -2, 0, 2, 4}

    def test_sector_sizes_match_binomial(self):
        n = 6
        sectors = basis_by_magnetisation(n)
        dims = sector_dimensions(n)
        for m, indices in sectors.items():
            assert len(indices) == dims[m], f"M={m}: {len(indices)} != {dims[m]}"

    def test_n2_explicit(self):
        sectors = basis_by_magnetisation(2)
        # |00⟩=M+2, |01⟩=|10⟩=M0, |11⟩=M-2
        assert len(sectors[2]) == 1  # |00⟩
        assert len(sectors[0]) == 2  # |01⟩, |10⟩
        assert len(sectors[-2]) == 1  # |11⟩


class TestSectorDimensions:
    def test_sum_equals_hilbert_dim(self):
        for n in [4, 6, 8, 10, 12]:
            dims = sector_dimensions(n)
            assert sum(dims.values()) == 2**n

    def test_largest_is_central(self):
        for n in [4, 6, 8]:
            assert largest_sector_dim(n) == comb(n, n // 2)

    def test_n16_largest(self):
        assert largest_sector_dim(16) == comb(16, 8)
        assert largest_sector_dim(16) == 12870


class TestEighByMagnetisation:
    def test_ground_energy_matches_full_ed(self):
        K, omega = _system(4)
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        H = knm_to_dense_matrix(K, omega)
        e_exact = np.linalg.eigvalsh(H)[0]

        result = eigh_by_magnetisation(K, omega)
        np.testing.assert_allclose(
            result["ground_energy"],
            e_exact,
            atol=1e-10,
            err_msg="U(1) sector ED must match full ED ground energy",
        )

    def test_all_eigenvalues_match_full(self):
        K, omega = _system(4)
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        H = knm_to_dense_matrix(K, omega)
        e_full = np.sort(np.linalg.eigvalsh(H))

        result = eigh_by_magnetisation(K, omega)
        np.testing.assert_allclose(
            result["eigvals_all"],
            e_full,
            atol=1e-10,
            err_msg="All U(1) sector eigenvalues must match full spectrum",
        )

    def test_selective_sectors(self):
        K, omega = _system(4)
        result = eigh_by_magnetisation(K, omega, sectors=[0, 2])
        assert set(result["results"].keys()) == {0, 2}
        assert result["n_sectors_computed"] == 2

    def test_ground_sector_identified(self):
        K, omega = _system(4)
        result = eigh_by_magnetisation(K, omega)
        gs = result["ground_sector"]
        assert gs in result["results"]

    def test_n6_all_eigenvalues(self):
        K, omega = _system(6)
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        H = knm_to_dense_matrix(K, omega)
        e_full = np.sort(np.linalg.eigvalsh(H))

        result = eigh_by_magnetisation(K, omega)
        np.testing.assert_allclose(result["eigvals_all"], e_full, atol=1e-10)

    def test_n8_ground_energy(self):
        K, omega = _system(8)
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        H = knm_to_dense_matrix(K, omega)
        e_exact = np.linalg.eigvalsh(H)[0]

        # All sectors — must match exactly
        result = eigh_by_magnetisation(K, omega)
        np.testing.assert_allclose(result["ground_energy"], e_exact, atol=1e-10)


class TestLevelSpacing:
    def test_r_bar_bounded(self):
        K, omega = _system(6)
        result = level_spacing_by_magnetisation(K, omega, M=0)
        assert 0 < result["r_bar"] < 1

    def test_dimension_correct(self):
        K, omega = _system(6)
        result = level_spacing_by_magnetisation(K, omega, M=0)
        assert result["dim"] == comb(6, 3)  # C(6,3) = 20

    def test_default_m_is_zero_for_even_n(self):
        K, omega = _system(4)
        result = level_spacing_by_magnetisation(K, omega)
        assert result["M"] == 0


class TestMemoryEstimate:
    def test_u1_smaller_than_z2(self):
        est = memory_estimate(16)
        assert est["u1_largest_mb"] < est["z2_sector_mb"]

    def test_reduction_factor(self):
        est = memory_estimate(16)
        assert est["reduction_factor"] > 4  # 65536/12870 ≈ 5.1

    def test_n16_u1_fits_32gb(self):
        est = memory_estimate(16)
        assert est["u1_largest_mb"] < 32000

    def test_n20_dimensions(self):
        est = memory_estimate(20)
        assert est["full_dim"] == 2**20
        assert est["u1_largest_dim"] == comb(20, 10)


# ---------------------------------------------------------------------------
# Coverage: internal helpers, project_to_sector, build_sector_hamiltonian,
# edge cases, odd-N, Rust fallback
# ---------------------------------------------------------------------------


class TestMagnetisationFunction:
    def test_all_up(self):
        from scpn_quantum_control.analysis.magnetisation_sectors import _magnetisation

        # |0000⟩ = k=0, all spin-up → M = +N
        assert _magnetisation(0, 4) == 4

    def test_all_down(self):
        from scpn_quantum_control.analysis.magnetisation_sectors import _magnetisation

        # |1111⟩ = k=15 → M = -4
        assert _magnetisation(15, 4) == -4

    def test_single_flip(self):
        from scpn_quantum_control.analysis.magnetisation_sectors import _magnetisation

        # |0001⟩ = k=1 → 1 one → M = 4-2 = 2
        assert _magnetisation(1, 4) == 2

    def test_half_filled(self):
        from scpn_quantum_control.analysis.magnetisation_sectors import _magnetisation

        # |0011⟩ = k=3 → 2 ones → M = 4-4 = 0
        assert _magnetisation(3, 4) == 0


class TestProjectToSector:
    def test_sector_matrix_shape(self):
        from scpn_quantum_control.analysis.magnetisation_sectors import project_to_sector

        H = np.random.default_rng(42).random((8, 8))
        H = (H + H.T) / 2
        indices = np.array([1, 2, 4])  # M=0 sector for 3 qubits
        H_sec = project_to_sector(H, indices)
        assert H_sec.shape == (3, 3)

    def test_full_hilbert_identity(self):
        from scpn_quantum_control.analysis.magnetisation_sectors import project_to_sector

        H = np.eye(4)
        indices = np.array([0, 1, 2, 3])
        H_sec = project_to_sector(H, indices)
        np.testing.assert_allclose(H_sec, np.eye(4))


class TestBuildSectorHamiltonian:
    def test_m0_sector_4qubit(self):
        from scpn_quantum_control.analysis.magnetisation_sectors import (
            build_sector_hamiltonian,
        )

        K, omega = _system(4)
        H_sec, indices = build_sector_hamiltonian(K, omega, M=0)
        assert H_sec.shape[0] == comb(4, 2)  # C(4,2) = 6
        assert len(indices) == 6

    def test_invalid_m_raises(self):
        import pytest

        from scpn_quantum_control.analysis.magnetisation_sectors import (
            build_sector_hamiltonian,
        )

        K, omega = _system(4)
        with pytest.raises(ValueError, match="not valid"):
            build_sector_hamiltonian(K, omega, M=3)  # 3 not in {-4,-2,0,2,4}

    def test_sector_hermitian(self):
        from scpn_quantum_control.analysis.magnetisation_sectors import (
            build_sector_hamiltonian,
        )

        K, omega = _system(4)
        H_sec, _ = build_sector_hamiltonian(K, omega, M=0)
        np.testing.assert_allclose(H_sec, H_sec.conj().T, atol=1e-12)


class TestOddN:
    def test_odd_n_default_m_is_one(self):
        K, omega = _system(3)
        result = level_spacing_by_magnetisation(K, omega)
        assert result["M"] == 1  # odd N → default M=1

    def test_odd_n_sectors(self):
        sectors = basis_by_magnetisation(3)
        assert set(sectors.keys()) == {-3, -1, 1, 3}

    def test_odd_n_eigenvalues(self):
        K, omega = _system(3)
        result = eigh_by_magnetisation(K, omega)
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        H = knm_to_dense_matrix(K, omega)
        e_full = np.sort(np.linalg.eigvalsh(H))
        np.testing.assert_allclose(result["eigvals_all"], e_full, atol=1e-10)


class TestLevelSpacingEdgeCases:
    def test_invalid_sector_nan(self):
        K, omega = _system(4)
        result = level_spacing_by_magnetisation(K, omega, M=4)
        # M=4 sector has only 1 state → not enough gaps → r_bar = nan
        assert np.isnan(result["r_bar"]) or result["dim"] == 1

    def test_n_gaps_present(self):
        K, omega = _system(6)
        result = level_spacing_by_magnetisation(K, omega, M=0)
        assert "n_gaps" in result
        assert result["n_gaps"] > 0

    # NOTE: line 231 (M not in result["results"]) is dead code — when
    # level_spacing_by_magnetisation passes sectors=[invalid_M] to
    # eigh_by_magnetisation, the latter crashes with IndexError on
    # empty all_eigvals_sorted before control returns to line 230.


class TestPythonFallback:
    def test_basis_by_magnetisation_python_path(self):
        """Force Python fallback by mocking Rust import failure."""
        from unittest.mock import patch

        with patch.dict("sys.modules", {"scpn_quantum_engine": None}):
            # Re-import to trigger fallback — but module is already loaded.
            # Instead, call the function directly with Rust unavailable.
            import importlib

            import scpn_quantum_control.analysis.magnetisation_sectors as _mod

            importlib.reload(_mod)
            sectors = _mod.basis_by_magnetisation(4)
            total = sum(len(v) for v in sectors.values())
            assert total == 16
            assert set(sectors.keys()) == {-4, -2, 0, 2, 4}

            # Reload back to normal
            importlib.reload(_mod)


class TestEighInvalidSector:
    def test_invalid_sector_skipped(self):
        """Requesting a non-existent M value is silently skipped."""
        K, omega = _system(4)
        result = eigh_by_magnetisation(K, omega, sectors=[0, 99])
        # M=99 doesn't exist, only M=0 computed
        assert 0 in result["results"]
        assert 99 not in result["results"]
        assert result["n_sectors_computed"] == 1
