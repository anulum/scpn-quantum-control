# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch test for the symmetry-sector projection
"""Projection test for the parity-sector Hamiltonian restriction."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.analysis.symmetry_sectors import project_hamiltonian


def test_project_hamiltonian_restricts_to_indices() -> None:
    """Projection restricts the Hamiltonian to the sector index sub-block."""
    H = np.arange(16, dtype=np.complex128).reshape(4, 4)
    indices = np.array([0, 2], dtype=np.intp)
    projected = project_hamiltonian(H, indices)
    assert projected.shape == (2, 2)
    assert projected[0, 0] == H[0, 0]
    assert projected[1, 1] == H[2, 2]
