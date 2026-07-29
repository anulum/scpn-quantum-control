# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — BL-54 parity-projector tests
"""Exact forward, JVP, VJP, leakage, and custody tests."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from scpn_quantum_control.analysis.dla_parity_theorem import project_to_parity_sector
from scpn_quantum_control.dla_topology_control.parity import (
    ParityLeakageEvaluation,
    ParitySectorProjector,
)
from scpn_quantum_control.dla_topology_control.schema import ParitySector


def _complex_vector(size: int) -> np.ndarray:
    return np.arange(size, dtype=np.float64) + 1j * np.arange(size, dtype=np.float64)[::-1]


def test_projector_delegates_to_existing_parity_projection_and_is_immutable() -> None:
    """Match the existing DLA projector while returning read-only custody."""
    projector = ParitySectorProjector(3, ParitySector.ODD)
    state = _complex_vector(8)
    projected = projector.project(state)
    np.testing.assert_array_equal(projected, project_to_parity_sector(state, 1, 3))
    assert projector.dimension == 8
    assert int(np.sum(projector.mask)) == 4
    assert not projector.mask.flags.writeable
    assert not projected.flags.writeable
    state[1] = 999.0
    assert projected[1] != 999.0


def test_projector_jvp_vjp_match_central_difference_and_adjoint_identity() -> None:
    """Verify exact linear derivatives and self-adjoint parity projection."""
    rng = np.random.default_rng(54)
    projector = ParitySectorProjector(4, ParitySector.EVEN)
    state = rng.normal(size=16) + 1j * rng.normal(size=16)
    tangent = rng.normal(size=16) + 1j * rng.normal(size=16)
    cotangent = rng.normal(size=16) + 1j * rng.normal(size=16)
    epsilon = 1.0e-6
    central = (
        projector.project(state + epsilon * tangent) - projector.project(state - epsilon * tangent)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(projector.jvp(tangent), central, atol=1.0e-9)
    lhs = float(np.vdot(projector.jvp(tangent), cotangent).real)
    rhs = float(np.vdot(tangent, projector.vjp(cotangent)).real)
    assert lhs == pytest.approx(rhs, abs=1.0e-12)


def test_absolute_leakage_gradient_matches_real_and_imaginary_directions() -> None:
    """Match the absolute outside-sector mass gradient to finite differences."""
    projector = ParitySectorProjector(2, ParitySector.EVEN)
    state = np.array([1.0 + 0.2j, 0.4 - 0.7j, -0.3 + 0.5j, 0.8 - 0.1j])
    evaluation = projector.leakage_value_and_gradient(state)
    assert evaluation.value == pytest.approx(abs(state[1]) ** 2 + abs(state[2]) ** 2)
    epsilon = 1.0e-6
    for index in range(4):
        real = np.zeros(4, dtype=np.complex128)
        real[index] = 1.0
        imag = 1j * real
        real_fd = (
            projector.leakage_value_and_gradient(state + epsilon * real).value
            - projector.leakage_value_and_gradient(state - epsilon * real).value
        ) / (2.0 * epsilon)
        imag_fd = (
            projector.leakage_value_and_gradient(state + epsilon * imag).value
            - projector.leakage_value_and_gradient(state - epsilon * imag).value
        ) / (2.0 * epsilon)
        assert evaluation.gradient[index].real == pytest.approx(real_fd, abs=1.0e-9)
        assert evaluation.gradient[index].imag == pytest.approx(imag_fd, abs=1.0e-9)
    assert not evaluation.gradient.flags.writeable


def test_normalised_leakage_gradient_matches_directional_difference() -> None:
    """Differentiate the outside-sector fraction through its norm quotient."""
    projector = ParitySectorProjector(2, ParitySector.ODD)
    state = np.array([0.7 + 0.2j, 0.5 - 0.3j, -0.4 + 0.1j, 0.2 + 0.6j])
    direction = np.array([0.3j, -0.2, 0.4 + 0.1j, -0.1j])
    evaluation = projector.leakage_value_and_gradient(state, normalised=True)
    epsilon = 1.0e-6
    central = (
        projector.leakage_value_and_gradient(state + epsilon * direction, normalised=True).value
        - projector.leakage_value_and_gradient(state - epsilon * direction, normalised=True).value
    ) / (2.0 * epsilon)
    analytic = float(np.vdot(evaluation.gradient, direction).real)
    assert evaluation.normalised
    assert 0.0 <= evaluation.value <= 1.0
    assert analytic == pytest.approx(central, abs=1.0e-9)


@pytest.mark.parametrize("n_qubits", [0, 21, True, 2.5])
def test_projector_rejects_invalid_dense_qubit_counts(n_qubits: object) -> None:
    """Reject unsafe or non-integer dense Hilbert-space sizes."""
    with pytest.raises(ValueError, match="n_qubits"):
        ParitySectorProjector(cast(int, n_qubits), ParitySector.EVEN)


def test_projector_rejects_invalid_sector_and_boundary() -> None:
    """Require closed parity enum and explicit claim-boundary text."""
    with pytest.raises(ValueError, match="sector"):
        ParitySectorProjector(2, cast(ParitySector, 0))
    with pytest.raises(ValueError, match="claim_boundary"):
        ParitySectorProjector(2, ParitySector.EVEN, claim_boundary=" ")


@pytest.mark.parametrize(
    ("state", "message"),
    [
        (np.zeros((2, 2), dtype=np.complex128), "shape"),
        (np.zeros(3, dtype=np.complex128), "shape"),
        (np.array([0.0, np.nan, 0.0, 0.0]), "finite"),
    ],
)
def test_state_validation_rejects_wrong_shape_and_non_finite_values(
    state: np.ndarray, message: str
) -> None:
    """Reject malformed state, tangent, and cotangent arrays."""
    projector = ParitySectorProjector(2, ParitySector.EVEN)
    with pytest.raises(ValueError, match=message):
        projector.as_state(state)


def test_leakage_rejects_zero_state() -> None:
    """Refuse an undefined normalised or absolute zero-state evaluation."""
    projector = ParitySectorProjector(2, ParitySector.EVEN)
    with pytest.raises(ValueError, match="positive norm"):
        projector.leakage_value_and_gradient(np.zeros(4, dtype=np.complex128))


def test_leakage_evaluation_contract_rejects_invalid_custody() -> None:
    """Reject invalid leakage scalars and malformed gradient custody."""
    valid = {
        "value": 0.2,
        "gradient": np.ones(4, dtype=np.complex128),
        "normalised": False,
        "state_norm_squared": 1.0,
    }
    for key in ("value", "state_norm_squared"):
        with pytest.raises(ValueError, match=key):
            ParityLeakageEvaluation(**(valid | {key: -1.0}))
    with pytest.raises(ValueError, match="gradient"):
        ParityLeakageEvaluation(**(valid | {"gradient": np.ones((2, 2))}))
    with pytest.raises(ValueError, match="gradient"):
        ParityLeakageEvaluation(
            **(valid | {"gradient": np.array([1.0, np.nan], dtype=np.complex128)})
        )
