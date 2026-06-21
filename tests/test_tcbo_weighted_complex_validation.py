# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Validation test for the TCBO weighted-complex replay
"""Target-probability guard test for the TCBO weighted-complex uncertainty replay."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis.tcbo_weighted_complex import (
    tcbo_weighted_uncertainty_replay,
)


def test_replay_rejects_out_of_range_target_probability() -> None:
    """A target H1 probability outside [0, 1] is rejected before any replay."""
    coupling = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match=r"target_p_h1 must be in \[0, 1\]"):
        tcbo_weighted_uncertainty_replay(coupling, target_p_h1=1.5)
