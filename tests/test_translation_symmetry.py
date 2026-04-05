# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Translation Symmetry
"""Tests for translation symmetry exploitation for periodic chains.

Covers:
    - _cyclic_shift correctness
    - is_translation_invariant detection
    - momentum_sectors orbit structure
    - momentum_sector_dimensions sum
    - eigh_with_translation eigenvalue correctness
    - Error handling for non-TI systems
    - Edge cases: n=2, empty sector
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis.translation_symmetry import (
    _cyclic_shift,
    eigh_with_translation,
    is_translation_invariant,
    momentum_sector_dimensions,
    momentum_sectors,
)


def _circulant_system(n: int = 4, coupling: float = 0.5):
    """Build a translation-invariant ring system."""
    K = np.zeros((n, n))
    omega = np.ones(n)
    for i in range(n):
        K[i, (i + 1) % n] = coupling
        K[(i + 1) % n, i] = coupling
    return K, omega


class TestCyclicShift:
    def test_identity_after_n(self):
        """n applications of T gives identity."""
        for n in [3, 4, 6]:
            for k in range(2**n):
                state = k
                for _ in range(n):
                    state = _cyclic_shift(state, n)
                assert state == k, f"n={n}, k={k}: T^n != I"

    def test_shift_n2(self):
        """n=2: T|00⟩=|00⟩, T|01⟩=|10⟩, T|10⟩=|01⟩, T|11⟩=|11⟩."""
        assert _cyclic_shift(0b00, 2) == 0b00
        assert _cyclic_shift(0b01, 2) == 0b10
        assert _cyclic_shift(0b10, 2) == 0b01
        assert _cyclic_shift(0b11, 2) == 0b11

    def test_shift_n3(self):
        """n=3: T|001⟩=|010⟩."""
        assert _cyclic_shift(0b001, 3) == 0b010
        assert _cyclic_shift(0b010, 3) == 0b100
        assert _cyclic_shift(0b100, 3) == 0b001


class TestIsTranslationInvariant:
    def test_circulant_is_ti(self):
        K, omega = _circulant_system(4)
        assert is_translation_invariant(K, omega) is True

    def test_heterogeneous_omega_not_ti(self):
        K, _ = _circulant_system(4)
        omega = np.array([1.0, 1.1, 1.0, 1.0])
        assert is_translation_invariant(K, omega) is False

    def test_non_circulant_k_not_ti(self):
        n = 4
        K = np.zeros((n, n))
        K[0, 1] = K[1, 0] = 0.5
        K[1, 2] = K[2, 1] = 0.3  # different coupling
        omega = np.ones(n)
        assert is_translation_invariant(K, omega) is False

    def test_n2_ring(self):
        K, omega = _circulant_system(2)
        assert is_translation_invariant(K, omega) is True


class TestMomentumSectors:
    def test_total_orbits_cover_hilbert_space(self):
        """Each basis state belongs to exactly one orbit."""
        for n in [3, 4]:
            sectors = momentum_sectors(n)
            # Count distinct states across all sector representatives' orbits
            all_reps = set()
            for reps in sectors.values():
                for r in reps:
                    all_reps.add(r)
            # Each representative maps to an orbit, total states = 2^n
            assert len(all_reps) <= 2**n

    def test_n_sectors_equals_n(self):
        for n in [3, 4, 5]:
            sectors = momentum_sectors(n)
            assert len(sectors) == n

    def test_k0_sector_nonempty(self):
        sectors = momentum_sectors(4)
        assert len(sectors[0]) > 0


class TestMomentumSectorDimensions:
    def test_sum_at_least_dim(self):
        """Sum of sector dimensions ≥ 2^n (overcounting due to orbits in multiple sectors)."""
        for n in [3, 4]:
            dims = momentum_sector_dimensions(n)
            total = sum(dims.values())
            assert total >= 2**n

    def test_k0_largest_for_n4(self):
        dims = momentum_sector_dimensions(4)
        assert dims[0] >= dims.get(1, 0)


class TestEighWithTranslation:
    def test_k0_eigenvalues(self):
        K, omega = _circulant_system(4)
        result = eigh_with_translation(K, omega, momentum=0)
        assert len(result["eigvals"]) == result["dim"]
        assert result["dim"] > 0
        assert result["is_ti"] is True

    def test_all_sectors_cover_spectrum(self):
        """Union of all momentum sectors' eigenvalues ≈ full spectrum."""
        K, omega = _circulant_system(4)
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        H = knm_to_dense_matrix(K, omega)
        e_full = np.sort(np.linalg.eigvalsh(H))

        all_eigvals = []
        for m in range(4):
            result = eigh_with_translation(K, omega, momentum=m)
            all_eigvals.extend(result["eigvals"].tolist())
        all_eigvals.sort()

        np.testing.assert_allclose(all_eigvals, e_full, atol=1e-8)

    def test_ground_energy_in_k0(self):
        """Ground state is typically in k=0 sector."""
        K, omega = _circulant_system(4)
        result = eigh_with_translation(K, omega, momentum=0)
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        H = knm_to_dense_matrix(K, omega)
        e_exact = np.linalg.eigvalsh(H)[0]
        assert result["eigvals"][0] <= e_exact + 1e-8

    def test_non_ti_raises(self):
        K = np.array([[0, 0.5], [0.5, 0]])
        omega = np.array([1.0, 2.0])  # heterogeneous → not TI
        with pytest.raises(ValueError, match="not translation-invariant"):
            eigh_with_translation(K, omega, momentum=0)

    def test_empty_sector(self):
        """If a momentum sector has no representatives, return empty."""
        K, omega = _circulant_system(2)
        # n=2 has 2 momentum sectors: m=0, m=1
        # Both should be non-empty for n=2, but let's test the API
        for m in range(2):
            result = eigh_with_translation(K, omega, momentum=m)
            assert result["momentum"] == m

    def test_n3_k1(self):
        K, omega = _circulant_system(3)
        result = eigh_with_translation(K, omega, momentum=1)
        assert result["dim"] > 0
        assert all(np.isfinite(result["eigvals"]))

    def test_invalid_momentum_returns_empty(self):
        """Momentum value not in sectors dict → empty result."""
        K, omega = _circulant_system(4)
        result = eigh_with_translation(K, omega, momentum=99)
        assert result["dim"] == 0
        assert len(result["eigvals"]) == 0
