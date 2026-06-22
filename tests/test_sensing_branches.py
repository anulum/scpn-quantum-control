# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the quantum-sensing readiness inputs
"""Validation guard tests for the quantum-sensing readiness input checks."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis.sensing import _require_positive, _validate_inputs

_OMEGA = np.array([0.1, 0.2], dtype=np.float64)
_TOPOLOGY = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
_GRID = np.array([1.0, 2.0], dtype=np.float64)


def test_validate_inputs_rejects_non_finite_omega() -> None:
    """A non-finite frequency vector is rejected."""
    with pytest.raises(ValueError, match="omega and topology must contain finite values"):
        _validate_inputs(np.array([np.inf, 0.2], dtype=np.float64), _TOPOLOGY, _GRID)


def test_validate_inputs_rejects_non_positive_grid() -> None:
    """A coupling grid with a non-positive value is rejected."""
    with pytest.raises(ValueError, match="k_grid must contain finite positive values"):
        _validate_inputs(_OMEGA, _TOPOLOGY, np.array([0.0, 1.0], dtype=np.float64))


def test_validate_inputs_rejects_asymmetric_topology() -> None:
    """An asymmetric topology matrix is rejected."""
    asymmetric = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="topology must be symmetric"):
        _validate_inputs(_OMEGA, asymmetric, _GRID)


def test_require_positive_rejects_non_positive() -> None:
    """A non-positive scalar is rejected."""
    with pytest.raises(ValueError, match="delta must be finite and positive"):
        _require_positive(-1.0, "delta")
