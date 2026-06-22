# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the quantum EVS enhancer
"""Type-guard tests for the quantum EVS feature enhancer."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control.applications.quantum_evs import (
    _validated_n_osc,
    quantum_evs_enhance,
)


def test_validated_n_osc_rejects_non_integer() -> None:
    """A non-integer oscillator count is rejected."""
    with pytest.raises(TypeError, match="n_osc must be an integer"):
        _validated_n_osc(cast(Any, 2.5))


def test_enhance_rejects_non_integer_trotter_reps() -> None:
    """A non-integer Trotter repetition count is rejected."""
    with pytest.raises(TypeError, match="trotter_reps must be an integer"):
        quantum_evs_enhance(np.array([0.1, 0.2], dtype=np.float64), trotter_reps=cast(Any, "two"))
