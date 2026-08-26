# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — topology-control objective tests
"""Analytic parity-protected objective tests against finite differences."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from scpn_quantum_control.dla_topology_control.objectives import (
    ParityProtectedObjectiveEvaluation,
    ParityProtectedQuadraticObjective,
)
from scpn_quantum_control.dla_topology_control.parity import ParitySectorProjector
from scpn_quantum_control.dla_topology_control.schema import ParitySector


def _objective() -> ParityProtectedQuadraticObjective:
    projector = ParitySectorProjector(2, ParitySector.EVEN)
    target = np.array([1.0, 0.0, 0.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
    return ParityProtectedQuadraticObjective(projector, target, leakage_weight=1.5)


def test_objective_gradient_matches_every_real_and_imaginary_coordinate() -> None:
    """Match the analytic Euclidean complex gradient to central differences."""
    objective = _objective()
    state = np.array([0.4 + 0.1j, -0.3 + 0.2j, 0.6 - 0.4j, 0.2 + 0.5j])
    evaluation = objective.evaluate(state)
    epsilon = 1.0e-6
    for index in range(state.size):
        real = np.zeros_like(state)
        real[index] = 1.0
        imag = 1j * real
        real_fd = (objective(state + epsilon * real) - objective(state - epsilon * real)) / (
            2.0 * epsilon
        )
        imag_fd = (objective(state + epsilon * imag) - objective(state - epsilon * imag)) / (
            2.0 * epsilon
        )
        assert evaluation.gradient[index].real == pytest.approx(real_fd, abs=1.0e-9)
        assert evaluation.gradient[index].imag == pytest.approx(imag_fd, abs=1.0e-9)
    assert evaluation.value == pytest.approx(
        evaluation.target_distance + 1.5 * evaluation.leakage_mass
    )
    assert not evaluation.state.flags.writeable
    assert not evaluation.gradient.flags.writeable


def test_objective_zero_weight_retains_target_distance_only() -> None:
    """Allow a zero leakage weight without changing the target-distance contract."""
    base = _objective()
    objective = ParityProtectedQuadraticObjective(base.projector, base.target_state, 0.0)
    state = np.ones(4, dtype=np.complex128)
    evaluation = objective.evaluate(state)
    assert evaluation.value == pytest.approx(evaluation.target_distance)
    np.testing.assert_allclose(evaluation.gradient, state - base.target_state)


def test_objective_rejects_wrong_projector_target_and_weight() -> None:
    """Reject wrong owner types, malformed targets, and invalid penalty weights."""
    projector = ParitySectorProjector(2, ParitySector.EVEN)
    target = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    with pytest.raises(ValueError, match="projector"):
        ParityProtectedQuadraticObjective(cast(ParitySectorProjector, object()), target)
    with pytest.raises(ValueError, match="positive norm"):
        ParityProtectedQuadraticObjective(projector, np.zeros(4, dtype=np.complex128))
    with pytest.raises(ValueError, match="selected parity sector"):
        ParityProtectedQuadraticObjective(
            projector, np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex128)
        )
    with pytest.raises(ValueError, match="leakage_weight"):
        ParityProtectedQuadraticObjective(projector, target, leakage_weight=-1.0)
    with pytest.raises(ValueError, match="leakage_weight"):
        ParityProtectedQuadraticObjective(projector, target, leakage_weight=np.nan)
    with pytest.raises(ValueError, match="claim_boundary"):
        ParityProtectedQuadraticObjective(projector, target, claim_boundary=" ")


def test_objective_evaluation_rejects_malformed_state_through_projector() -> None:
    """Preserve projector shape and finite-value guards at objective entry."""
    objective = _objective()
    with pytest.raises(ValueError, match="shape"):
        objective.evaluate(np.zeros(3, dtype=np.complex128))
    with pytest.raises(ValueError, match="finite"):
        objective.evaluate(np.array([0.0, np.nan, 0.0, 0.0]))


def test_evaluation_contract_rejects_invalid_scalars_arrays_and_boundary() -> None:
    """Reject contradictory or malformed objective evaluation custody."""
    valid: dict[str, object] = {
        "value": 1.0,
        "target_distance": 0.5,
        "leakage_mass": 0.25,
        "state": np.ones(4, dtype=np.complex128),
        "gradient": np.ones(4, dtype=np.complex128),
    }
    for key in ("value", "target_distance", "leakage_mass"):
        with pytest.raises(ValueError, match=key):
            ParityProtectedObjectiveEvaluation(**(valid | {key: -1.0}))
    with pytest.raises(ValueError, match="state"):
        ParityProtectedObjectiveEvaluation(**(valid | {"state": np.ones((2, 2))}))
    with pytest.raises(ValueError, match="gradient"):
        ParityProtectedObjectiveEvaluation(**(valid | {"gradient": np.ones(3)}))
    with pytest.raises(ValueError, match="gradient"):
        ParityProtectedObjectiveEvaluation(**(valid | {"gradient": np.array([np.nan] * 4)}))
    with pytest.raises(ValueError, match="claim_boundary"):
        ParityProtectedObjectiveEvaluation(**(valid | {"claim_boundary": " "}))
