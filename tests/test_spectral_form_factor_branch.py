# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch test for the spectral form factor parity guard
"""Parity-guard test for the sector level-spacing ratio."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis.spectral_form_factor import _sector_level_spacing_ratio


def test_sector_level_spacing_rejects_invalid_parity() -> None:
    """A parity outside {0, 1, None} is rejected for parity-basis selection."""
    with pytest.raises(ValueError, match="parity must be 0, 1, or None"):
        _sector_level_spacing_ratio(
            np.eye(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            basis="parity",
            parity=2,
            full_eigenvalues=np.zeros(2, dtype=np.float64),
        )
