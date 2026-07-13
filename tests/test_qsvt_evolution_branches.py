# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — QSVT public validation branch contracts
"""Exercise fail-closed QSVT validation branches through public functions."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.phase.qsvt_evolution import (
    qsp_phase_angles,
    qsvt_resource_estimate,
)

FloatArray = NDArray[np.float64]


def _invalid_float_array(value: object) -> FloatArray:
    """Expose an intentionally invalid runtime payload to the typed public API."""
    return cast(FloatArray, value)


def test_resource_estimate_rejects_ragged_coupling() -> None:
    """Reject a ragged coupling payload before object-array coercion."""
    ragged = _invalid_float_array([[0.0], [1.0, 0.0]])

    with pytest.raises(ValueError, match="rectangular numeric array"):
        qsvt_resource_estimate(ragged, np.zeros(2, dtype=np.float64))


def test_resource_estimate_rejects_structured_coupling() -> None:
    """Reject a structured coupling dtype at the final numeric conversion guard."""
    structured = np.array([(1, 2)], dtype=[("left", "i4"), ("right", "i4")])

    with pytest.raises(ValueError, match="real numeric scalars"):
        qsvt_resource_estimate(
            _invalid_float_array(structured),
            np.zeros(1, dtype=np.float64),
        )


def test_resource_estimate_rejects_non_vector_frequencies() -> None:
    """Reject a frequency column even when its element count matches the graph."""
    with pytest.raises(ValueError, match="one-dimensional"):
        qsvt_resource_estimate(
            np.eye(2, dtype=np.float64),
            np.zeros((2, 1), dtype=np.float64),
        )


def test_resource_estimate_rejects_asymmetric_coupling() -> None:
    """Reject a real square coupling that cannot define a Hermitian XY model."""
    coupling = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="symmetric"):
        qsvt_resource_estimate(coupling, np.zeros(2, dtype=np.float64))


def test_qsp_phase_angles_rejects_negative_degree() -> None:
    """Reject negative integer degrees before exposing seed angles."""
    with pytest.raises(ValueError, match="non-negative"):
        qsp_phase_angles(-1, allow_initial_guess=True)
