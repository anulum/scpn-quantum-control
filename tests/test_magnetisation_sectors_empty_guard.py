# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Empty-spectrum guard test for magnetisation sectors
"""Defensive empty-spectrum guard test for the magnetisation-sector eigensolver."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import sparse

from scpn_quantum_control.analysis import magnetisation_sectors


def test_eigh_returns_empty_summary_when_sectors_yield_no_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If every selected sector builds an empty Hamiltonian, an empty summary is returned.

    Patching the sparse-sector builder to emit a zero-dimensional Hamiltonian
    simulates a divergence between the basis enumeration and the Hamiltonian
    construction, exercising the defensive guard that returns an empty spectrum
    rather than diagonalising nothing.
    """

    def _empty_builder(
        _k: NDArray[np.float64], _omega: NDArray[np.float64], _m: int
    ) -> tuple[sparse.csr_matrix, NDArray[np.intp]]:
        return sparse.csr_matrix((0, 0), dtype=np.float64), np.array([], dtype=np.intp)

    monkeypatch.setattr(magnetisation_sectors, "build_sparse_sector_hamiltonian", _empty_builder)

    k = np.array([[0.0, 0.3], [0.3, 0.0]], dtype=np.float64)
    omega = np.array([0.2, -0.2], dtype=np.float64)

    result = magnetisation_sectors.eigh_by_magnetisation(k, omega)

    assert result["n_sectors_computed"] == 0
    assert result["eigvals_all"].size == 0
    assert result["ground_sector"] is None
    assert np.isnan(result["ground_energy"])
