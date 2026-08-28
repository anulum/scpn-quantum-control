# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Magnetisation Sectors
"""Tests for U(1) magnetisation sector decomposition."""

from __future__ import annotations

import sys
from math import comb

import numpy as np
import pytest

from scpn_quantum_control.analysis import magnetisation_sectors as magnetisation_module
from scpn_quantum_control.analysis.magnetisation_sectors import (
    basis_by_magnetisation,
    build_sector_hamiltonian,
    eigh_by_magnetisation,
    largest_sector_dim,
    level_spacing_by_magnetisation,
    memory_estimate,
    sector_dimensions,
)
from scpn_quantum_control.dense_budget import DenseAllocationError


def _system(n: int = 4):
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    omega = np.linspace(0.8, 1.2, n)
    return K, omega


class TestBasisPartition:
    """Verify exhaustive, disjoint basis partitioning by magnetisation."""

    def test_total_count(self):
        """Keep every computational-basis state in exactly one sector count."""
        for n in [2, 4, 6, 8]:
            sectors = basis_by_magnetisation(n)
            total = sum(len(v) for v in sectors.values())
            assert total == 2**n, f"n={n}: {total} != {2**n}"

    def test_no_overlap(self):
        """Keep sector index sets disjoint."""
        sectors = basis_by_magnetisation(4)
        all_indices = []
        for v in sectors.values():
            all_indices.extend(v.tolist())
        assert len(all_indices) == len(set(all_indices))

    def test_correct_M_values(self):
        """Enumerate the allowed even four-spin magnetisations."""
        sectors = basis_by_magnetisation(4)
        assert set(sectors.keys()) == {-4, -2, 0, 2, 4}

    def test_sector_sizes_match_binomial(self):
        """Match enumerated sector sizes to their binomial dimensions."""
        n = 6
        sectors = basis_by_magnetisation(n)
        dims = sector_dimensions(n)
        for m, indices in sectors.items():
            assert len(indices) == dims[m], f"M={m}: {len(indices)} != {dims[m]}"

    def test_n2_explicit(self):
        """Match every two-spin basis state to its explicit sector."""
        sectors = basis_by_magnetisation(2)
        # |00⟩=M+2, |01⟩=|10⟩=M0, |11⟩=M-2
        assert len(sectors[2]) == 1  # |00⟩
        assert len(sectors[0]) == 2  # |01⟩, |10⟩
        assert len(sectors[-2]) == 1  # |11⟩


class TestSectorDimensions:
    """Verify analytic magnetisation-sector dimensions."""

    def test_sum_equals_hilbert_dim(self):
        """Recover the full Hilbert dimension by summing all sectors."""
        for n in [4, 6, 8, 10, 12]:
            dims = sector_dimensions(n)
            assert sum(dims.values()) == 2**n

    def test_largest_is_central(self):
        """Identify the central binomial sector as the largest."""
        for n in [4, 6, 8]:
            assert largest_sector_dim(n) == comb(n, n // 2)

    def test_n16_largest(self):
        """Lock the exact largest-sector size for sixteen spins."""
        assert largest_sector_dim(16) == comb(16, 8)
        assert largest_sector_dim(16) == 12870


class TestEighByMagnetisation:
    """Verify sector eigensolvers, budgets, and sparse-only allocation."""

    def test_ground_energy_matches_full_ed(self):
        """Match the sector ground energy to full exact diagonalisation."""
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
        """Recover the complete full-space spectrum across all sectors."""
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
        """Diagonalise only explicitly requested valid sectors."""
        K, omega = _system(4)
        result = eigh_by_magnetisation(K, omega, sectors=[0, 2])
        assert set(result["results"].keys()) == {0, 2}
        assert result["n_sectors_computed"] == 2

    def test_invalid_requested_sector_returns_empty_result(self):
        """Return a typed empty summary for an entirely invalid selection."""
        K, omega = _system(4)
        result = eigh_by_magnetisation(K, omega, sectors=[3])

        assert result["results"] == {}
        assert result["eigvals_all"].size == 0
        assert np.isnan(result["ground_energy"])
        assert result["ground_sector"] is None
        assert result["n_sectors_computed"] == 0

    def test_rejects_sector_budget_before_sparse_sector_hamiltonian(self, monkeypatch):
        """Reject oversized eigensolver workspaces before sparse construction."""
        K, omega = _system(4)

        def fail_if_sparse_sector_hamiltonian_is_requested(*args, **kwargs):  # noqa: ARG001
            raise AssertionError(
                "sparse sector Hamiltonian allocation happened before budget gate"
            )

        monkeypatch.setattr(
            magnetisation_module,
            "build_sparse_sector_hamiltonian",
            fail_if_sparse_sector_hamiltonian_is_requested,
        )

        with pytest.raises(DenseAllocationError, match="magnetisation sector"):
            eigh_by_magnetisation(K, omega, max_dense_gib=1e-12)

    def test_ground_sector_identified(self):
        """Report a ground sector present in the computed result map."""
        K, omega = _system(4)
        result = eigh_by_magnetisation(K, omega)
        gs = result["ground_sector"]
        assert gs in result["results"]

    def test_build_sector_rejects_budget_before_sparse_sector_hamiltonian(self, monkeypatch):
        """Reject oversized dense conversion before sparse construction."""
        K, omega = _system(4)

        def fail_if_sparse_sector_hamiltonian_is_requested(*args, **kwargs):  # noqa: ARG001
            raise AssertionError(
                "sparse sector Hamiltonian allocation happened before budget gate"
            )

        monkeypatch.setattr(
            magnetisation_module,
            "build_sparse_sector_hamiltonian",
            fail_if_sparse_sector_hamiltonian_is_requested,
        )

        with pytest.raises(DenseAllocationError, match="magnetisation sector"):
            build_sector_hamiltonian(K, omega, M=0, max_dense_gib=1e-12)

    def test_sector_builder_does_not_use_full_dense_hamiltonian(self, monkeypatch):
        """Build a sector without allocating the full dense Hamiltonian."""
        K, omega = _system(4)

        def fail_if_full_dense_hamiltonian_is_requested(*args, **kwargs):  # noqa: ARG001
            raise AssertionError("full dense Hamiltonian path was used")

        monkeypatch.setattr(
            magnetisation_module,
            "knm_to_dense_matrix",
            fail_if_full_dense_hamiltonian_is_requested,
            raising=False,
        )

        H_sec, indices = build_sector_hamiltonian(K, omega, M=0)

        assert H_sec.shape == (len(indices), len(indices))
        np.testing.assert_allclose(H_sec, H_sec.conj().T, atol=1e-12)

    def test_sector_builder_does_not_use_full_sparse_hamiltonian(self, monkeypatch):
        """Build a sector without constructing the full sparse Hamiltonian."""
        K, omega = _system(4)

        def fail_if_full_sparse_hamiltonian_is_requested(*args, **kwargs):  # noqa: ARG001
            raise AssertionError("full sparse Hamiltonian path was used")

        monkeypatch.setattr(
            magnetisation_module,
            "build_sparse_hamiltonian",
            fail_if_full_sparse_hamiltonian_is_requested,
            raising=False,
        )

        H_sec, indices = build_sector_hamiltonian(K, omega, M=0)

        assert H_sec.shape == (len(indices), len(indices))
        np.testing.assert_allclose(H_sec, H_sec.conj().T, atol=1e-12)

    def test_eigh_by_magnetisation_does_not_use_full_dense_hamiltonian(self, monkeypatch):
        """Diagonalise a sector without allocating the full dense matrix."""
        K, omega = _system(4)

        def fail_if_full_dense_hamiltonian_is_requested(*args, **kwargs):  # noqa: ARG001
            raise AssertionError("full dense Hamiltonian path was used")

        monkeypatch.setattr(
            magnetisation_module,
            "knm_to_dense_matrix",
            fail_if_full_dense_hamiltonian_is_requested,
            raising=False,
        )

        result = eigh_by_magnetisation(K, omega, sectors=[0])

        assert result["n_sectors_computed"] == 1
        assert 0 in result["results"]

    def test_eigh_by_magnetisation_does_not_use_full_sparse_hamiltonian(self, monkeypatch):
        """Diagonalise a sector without constructing the full sparse matrix."""
        K, omega = _system(4)

        def fail_if_full_sparse_hamiltonian_is_requested(*args, **kwargs):  # noqa: ARG001
            raise AssertionError("full sparse Hamiltonian path was used")

        monkeypatch.setattr(
            magnetisation_module,
            "build_sparse_hamiltonian",
            fail_if_full_sparse_hamiltonian_is_requested,
            raising=False,
        )

        result = eigh_by_magnetisation(K, omega, sectors=[0])

        assert result["n_sectors_computed"] == 1
        assert 0 in result["results"]

    def test_n6_all_eigenvalues(self):
        """Match the full six-spin spectrum across sector blocks."""
        K, omega = _system(6)
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        H = knm_to_dense_matrix(K, omega)
        e_full = np.sort(np.linalg.eigvalsh(H))

        result = eigh_by_magnetisation(K, omega)
        np.testing.assert_allclose(result["eigvals_all"], e_full, atol=1e-10)

    def test_n8_ground_energy(self):
        """Match the full eight-spin ground energy across all sectors."""
        K, omega = _system(8)
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        H = knm_to_dense_matrix(K, omega)
        e_exact = np.linalg.eigvalsh(H)[0]

        # All sectors — must match exactly
        result = eigh_by_magnetisation(K, omega)
        np.testing.assert_allclose(result["ground_energy"], e_exact, atol=1e-10)


class TestLevelSpacing:
    """Verify within-sector level-spacing summaries."""

    def test_r_bar_bounded(self):
        """Keep the mean adjacent-gap ratio inside its mathematical bounds."""
        K, omega = _system(6)
        result = level_spacing_by_magnetisation(K, omega, M=0)
        assert 0 < result["r_bar"] < 1

    def test_dimension_correct(self):
        """Report the central sector's binomial dimension."""
        K, omega = _system(6)
        result = level_spacing_by_magnetisation(K, omega, M=0)
        assert result["dim"] == comb(6, 3)  # C(6,3) = 20

    def test_default_m_is_zero_for_even_n(self):
        """Default even systems to their zero-magnetisation sector."""
        K, omega = _system(4)
        result = level_spacing_by_magnetisation(K, omega)
        assert result["M"] == 0


class TestMemoryEstimate:
    """Verify analytic memory and dimension estimates."""

    def test_u1_smaller_than_z2(self):
        """Keep the largest U(1) block smaller than the Z2 block."""
        est = memory_estimate(16)
        assert est["u1_largest_mb"] < est["z2_sector_mb"]

    def test_reduction_factor(self):
        """Expose a material full-space reduction for sixteen spins."""
        est = memory_estimate(16)
        assert est["reduction_factor"] > 4  # 65536/12870 ≈ 5.1

    def test_n16_u1_fits_32gb(self):
        """Estimate the sixteen-spin U(1) block below 32 GB."""
        est = memory_estimate(16)
        assert est["u1_largest_mb"] < 32000

    def test_n20_dimensions(self):
        """Lock full and largest-sector dimensions for twenty spins."""
        est = memory_estimate(20)
        assert est["full_dim"] == 2**20
        assert est["u1_largest_dim"] == comb(20, 10)


# ---------------------------------------------------------------------------
# Coverage: internal helpers, project_to_sector, build_sector_hamiltonian,
# edge cases, odd-N, Rust fallback
# ---------------------------------------------------------------------------


class TestMagnetisationFunction:
    """Verify computational-basis magnetisation labels."""

    def test_all_up(self):
        """Assign maximal magnetisation to the all-up state."""
        from scpn_quantum_control.analysis.magnetisation_sectors import _magnetisation

        # |0000⟩ = k=0, all spin-up → M = +N
        assert _magnetisation(0, 4) == 4

    def test_all_down(self):
        """Assign minimal magnetisation to the all-down state."""
        from scpn_quantum_control.analysis.magnetisation_sectors import _magnetisation

        # |1111⟩ = k=15 → M = -4
        assert _magnetisation(15, 4) == -4

    def test_single_flip(self):
        """Reduce magnetisation by two for one flipped spin."""
        from scpn_quantum_control.analysis.magnetisation_sectors import _magnetisation

        # |0001⟩ = k=1 → 1 one → M = 4-2 = 2
        assert _magnetisation(1, 4) == 2

    def test_half_filled(self):
        """Assign zero magnetisation to a half-filled even system."""
        from scpn_quantum_control.analysis.magnetisation_sectors import _magnetisation

        # |0011⟩ = k=3 → 2 ones → M = 4-4 = 0
        assert _magnetisation(3, 4) == 0


class TestProjectToSector:
    """Verify dense indexing into selected sector subspaces."""

    def test_sector_matrix_shape(self):
        """Return a square block sized by the selected indices."""
        from scpn_quantum_control.analysis.magnetisation_sectors import project_to_sector

        H = np.random.default_rng(42).random((8, 8))
        H = (H + H.T) / 2
        indices = np.array([1, 2, 4])  # M=0 sector for 3 qubits
        H_sec = project_to_sector(H, indices)
        assert H_sec.shape == (3, 3)

    def test_full_hilbert_identity(self):
        """Preserve an identity matrix when every index is selected."""
        from scpn_quantum_control.analysis.magnetisation_sectors import project_to_sector

        H = np.eye(4)
        indices = np.array([0, 1, 2, 3])
        H_sec = project_to_sector(H, indices)
        np.testing.assert_allclose(H_sec, np.eye(4))


class TestBuildSectorHamiltonian:
    """Verify direct dense-sector Hamiltonian construction."""

    def test_m0_sector_4qubit(self):
        """Build the six-dimensional central block for four spins."""
        from scpn_quantum_control.analysis.magnetisation_sectors import (
            build_sector_hamiltonian,
        )

        K, omega = _system(4)
        H_sec, indices = build_sector_hamiltonian(K, omega, M=0)
        assert H_sec.shape[0] == comb(4, 2)  # C(4,2) = 6
        assert len(indices) == 6

    def test_invalid_m_raises(self):
        """Reject a magnetisation outside the system's parity ladder."""
        import pytest

        from scpn_quantum_control.analysis.magnetisation_sectors import (
            build_sector_hamiltonian,
        )

        K, omega = _system(4)
        with pytest.raises(ValueError, match="not valid"):
            build_sector_hamiltonian(K, omega, M=3)  # 3 not in {-4,-2,0,2,4}

    def test_sector_hermitian(self):
        """Preserve Hermiticity after sector construction."""
        from scpn_quantum_control.analysis.magnetisation_sectors import (
            build_sector_hamiltonian,
        )

        K, omega = _system(4)
        H_sec, _ = build_sector_hamiltonian(K, omega, M=0)
        np.testing.assert_allclose(H_sec, H_sec.conj().T, atol=1e-12)


class TestOddN:
    """Verify odd-system sector enumeration and defaults."""

    def test_odd_n_default_m_is_one(self):
        """Default odd systems to positive unit magnetisation."""
        K, omega = _system(3)
        result = level_spacing_by_magnetisation(K, omega)
        assert result["M"] == 1  # odd N → default M=1

    def test_odd_n_sectors(self):
        """Enumerate only odd magnetisations for an odd system."""
        sectors = basis_by_magnetisation(3)
        assert set(sectors.keys()) == {-3, -1, 1, 3}

    def test_odd_n_eigenvalues(self):
        """Recover the full odd-system spectrum from its sectors."""
        K, omega = _system(3)
        result = eigh_by_magnetisation(K, omega)
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        H = knm_to_dense_matrix(K, omega)
        e_full = np.sort(np.linalg.eigvalsh(H))
        np.testing.assert_allclose(result["eigvals_all"], e_full, atol=1e-10)


class TestLevelSpacingEdgeCases:
    """Verify level-spacing refusal and small-spectrum summaries."""

    def test_invalid_sector_nan(self):
        """Return NaN and zero dimension for an invalid sector."""
        K, omega = _system(4)
        result = level_spacing_by_magnetisation(K, omega, M=3)

        assert np.isnan(result["r_bar"])
        assert result["M"] == 3
        assert result["dim"] == 0

    def test_one_state_sector_has_nan_spacing(self):
        """Return NaN when a one-state sector has no adjacent gaps."""
        K, omega = _system(4)
        result = level_spacing_by_magnetisation(K, omega, M=4)

        assert np.isnan(result["r_bar"])
        assert result["dim"] == 1

    def test_n_gaps_present(self):
        """Report a positive gap count for a nontrivial central sector."""
        K, omega = _system(6)
        result = level_spacing_by_magnetisation(K, omega, M=0)
        assert "n_gaps" in result
        assert result["n_gaps"] > 0


class TestPythonFallback:
    """Verify accelerator fallback and runtime-error boundaries."""

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

    def test_basis_by_magnetisation_does_not_swallow_engine_runtime_errors(self, monkeypatch):
        """Propagate accelerator runtime failures instead of falling back."""

        class BrokenEngine:
            @staticmethod
            def magnetisation_labels(n):  # noqa: ARG004
                raise RuntimeError("accelerator failed")

        monkeypatch.setitem(sys.modules, "scpn_quantum_engine", BrokenEngine())

        with pytest.raises(RuntimeError, match="accelerator failed"):
            basis_by_magnetisation(4)


class TestEighInvalidSector:
    """Verify mixed valid and invalid sector selections."""

    def test_invalid_sector_skipped(self):
        """Requesting a non-existent M value is silently skipped."""
        K, omega = _system(4)
        result = eigh_by_magnetisation(K, omega, sectors=[0, 99])
        # M=99 doesn't exist, only M=0 computed
        assert 0 in result["results"]
        assert 99 not in result["results"]
        assert result["n_sectors_computed"] == 1
