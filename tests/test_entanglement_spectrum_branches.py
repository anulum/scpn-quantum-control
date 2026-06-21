# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the entanglement spectrum analysis
"""Branch tests for the entanglement-spectrum CFT fit and coupling scan.

Covers the dimensionality guard of the central-charge fit and the default
coupling grid of the entropy-versus-coupling scan.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis.entanglement_spectrum import (
    entropy_vs_coupling_scan,
    fit_cft_central_charge,
)


def test_cft_fit_rejects_non_one_dimensional_inputs() -> None:
    """The central-charge fit requires one-dimensional input arrays."""
    with pytest.raises(ValueError, match="one-dimensional arrays"):
        fit_cft_central_charge(
            np.zeros((2, 2), dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            10,
        )


def test_coupling_scan_uses_default_grid() -> None:
    """Omitting the coupling grid scans the built-in default range."""
    omega = np.zeros(4, dtype=np.float64)
    results = entropy_vs_coupling_scan(omega)
    assert len(results["k_base"]) == 30
    assert len(results["half_chain_entropy"]) == 30
    assert len(results["cft_central_charge"]) == 30
