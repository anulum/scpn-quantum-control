# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Array-validation tests for the PennyLane HAL adapter
"""Array-shape and finiteness guard tests for the PennyLane HAL adapter."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.hardware.pennylane_adapter import (
    _as_finite_matrix,
    _as_finite_vector,
)


def test_matrix_rejects_non_two_dimensional() -> None:
    """A non-2-D matrix argument is rejected."""
    with pytest.raises(ValueError, match="two-dimensional array"):
        _as_finite_matrix("coupling", np.zeros(3, dtype=np.float64))


def test_matrix_rejects_empty() -> None:
    """An empty matrix argument is rejected."""
    with pytest.raises(ValueError, match="must not be empty"):
        _as_finite_matrix("coupling", np.zeros((0, 2), dtype=np.float64))


def test_vector_rejects_non_one_dimensional() -> None:
    """A non-1-D vector argument is rejected."""
    with pytest.raises(ValueError, match="one-dimensional array"):
        _as_finite_vector("omega", np.zeros((2, 2), dtype=np.float64))


def test_vector_rejects_non_finite() -> None:
    """A non-finite vector entry is rejected."""
    with pytest.raises(ValueError, match="must contain only finite values"):
        _as_finite_vector("omega", np.array([1.0, np.inf], dtype=np.float64))
