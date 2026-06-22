# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the SSGF spectral bridge
"""Coverage tests for the SSGF spectral bridge analysis and coupling scan."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.ssgf.quantum_spectral import (
    spectral_bridge_analysis,
    spectral_bridge_vs_coupling,
)

_K = np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.float64)
_OMEGA = np.array([0.1, 0.2], dtype=np.float64)


def test_spectral_bridge_analysis_runs() -> None:
    """The full spectral bridge analysis computes a Laplacian spectrum."""
    result = spectral_bridge_analysis(_K, _OMEGA)
    assert result is not None


def test_spectral_bridge_vs_coupling_default_grid() -> None:
    """The coupling scan uses the default grid when none is supplied."""
    scan = spectral_bridge_vs_coupling(_OMEGA)
    assert len(scan["fiedler"]) == 20
