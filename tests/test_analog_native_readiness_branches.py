# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for analog-native readiness inputs
"""Guard tests for the analog-native readiness problem validation."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.hardware.analog_native_readiness import _validate_problem_arrays


def test_validate_rejects_non_finite_inputs() -> None:
    """A non-finite coupling matrix is rejected."""
    k = np.array([[0.0, np.inf], [np.inf, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="K_nm and omega must contain finite values"):
        _validate_problem_arrays(k, np.zeros(2, dtype=np.float64))


def test_validate_rejects_asymmetric_coupling() -> None:
    """An asymmetric coupling matrix is rejected."""
    k = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="K_nm must be symmetric for analog-native readiness"):
        _validate_problem_arrays(k, np.zeros(2, dtype=np.float64))
