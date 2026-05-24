# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Translation symmetry contract tests
"""Contract tests for translation-invariant sectors, spectrum ordering, and invalid heterogeneous systems."""

from __future__ import annotations

import numpy as np
import pytest


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


class TestTranslationSymmetry:
    """Cyclic translation symmetry tests."""

    @pytest.mark.parametrize("n", [4, 6, 8])
    def test_is_ti_homogeneous(self, n):
        from scpn_quantum_control.analysis.translation_symmetry import (
            is_translation_invariant,
        )

        K, omega = _homogeneous_system(n)
        assert is_translation_invariant(K, omega)

    @pytest.mark.parametrize("n", [4, 6, 8])
    def test_is_not_ti_heterogeneous(self, n):
        from scpn_quantum_control.analysis.translation_symmetry import (
            is_translation_invariant,
        )

        K, omega = _system(n)
        assert not is_translation_invariant(K, omega)

    def test_momentum_sectors_dimensions(self):
        from scpn_quantum_control.analysis.translation_symmetry import (
            momentum_sector_dimensions,
        )

        dims = momentum_sector_dimensions(4)
        assert sum(dims.values()) >= 2**4
        assert all(d > 0 for d in dims.values())

    @pytest.mark.parametrize("momentum", [0, 1, 2, 3])
    def test_eigh_various_momentum_sectors(self, momentum):
        from scpn_quantum_control.analysis.translation_symmetry import (
            eigh_with_translation,
        )

        K, omega = _homogeneous_system(4)
        result = eigh_with_translation(K, omega, momentum=momentum)
        assert len(result["eigvals"]) > 0
        assert result["momentum"] == momentum
        assert result["is_ti"]

    def test_heterogeneous_raises_valueerror(self):
        from scpn_quantum_control.analysis.translation_symmetry import (
            eigh_with_translation,
        )

        K, omega = _system(4)
        with pytest.raises(ValueError):
            eigh_with_translation(K, omega, momentum=0)

    @pytest.mark.parametrize("n", [4, 6])
    def test_k0_ground_within_full_spectrum(self, n):
        """k=0 sector ground energy ≥ full ground energy."""
        from scpn_quantum_control.analysis.translation_symmetry import (
            eigh_with_translation,
        )
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        K, omega = _homogeneous_system(n)
        H = knm_to_dense_matrix(K, omega)
        e_full = np.linalg.eigvalsh(H)

        result = eigh_with_translation(K, omega, momentum=0)
        assert result["eigvals"][0] >= e_full[0] - 1e-8

    def test_eigenvalues_are_real(self):
        from scpn_quantum_control.analysis.translation_symmetry import (
            eigh_with_translation,
        )

        K, omega = _homogeneous_system(4)
        result = eigh_with_translation(K, omega, momentum=0)
        assert all(np.isreal(e) for e in result["eigvals"])

    def test_eigenvalues_sorted(self):
        from scpn_quantum_control.analysis.translation_symmetry import (
            eigh_with_translation,
        )

        K, omega = _homogeneous_system(6)
        result = eigh_with_translation(K, omega, momentum=0)
        eigvals = result["eigvals"]
        np.testing.assert_array_equal(eigvals, np.sort(eigvals))
