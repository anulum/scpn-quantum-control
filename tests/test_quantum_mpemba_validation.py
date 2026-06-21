# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Validation tests for the quantum Mpemba experiment
"""Input-validation tests for the quantum Mpemba experiment helper."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis.quantum_mpemba import _validate_mpemba_inputs


def test_rejects_non_one_dimensional_omega() -> None:
    """A non-1-D natural-frequency array is rejected."""
    with pytest.raises(ValueError, match="one-dimensional"):
        _validate_mpemba_inputs(
            np.zeros((2, 2), dtype=np.float64),
            np.eye(2, dtype=np.float64),
            1.0,
            0.1,
            1.0,
            10,
        )


def test_rejects_non_finite_inputs() -> None:
    """Non-finite omega or topology entries are rejected."""
    with pytest.raises(ValueError, match="finite"):
        _validate_mpemba_inputs(
            np.array([np.inf, 1.0], dtype=np.float64),
            np.eye(2, dtype=np.float64),
            1.0,
            0.1,
            1.0,
            10,
        )


def test_rejects_asymmetric_topology() -> None:
    """An asymmetric coupling topology is rejected."""
    with pytest.raises(ValueError, match="symmetric"):
        _validate_mpemba_inputs(
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64),
            1.0,
            0.1,
            1.0,
            10,
        )
