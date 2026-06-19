# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase Coupling Learning
"""Tests for phase/coupling_learning.py differentiable coupling training."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from scpn_quantum_control.differentiable import ParameterShiftRule
from scpn_quantum_control.phase import (
    CouplingGradientVerificationResult,
    CouplingLearningResult,
    coupling_matrix_from_edge_vector,
    learn_couplings_from_observations,
    multi_frequency_parameter_shift_rule,
    verify_coupling_parameter_shift_gradient,
)

FloatArray: TypeAlias = NDArray[np.float64]
ObservationModel: TypeAlias = Callable[[FloatArray], ArrayLike]


def _sin_observation(couplings: FloatArray) -> FloatArray:
    return np.array([np.sin(couplings[0, 1])], dtype=np.float64)


def test_learn_couplings_from_observations_converges_on_sinusoidal_edge() -> None:
    initial = np.array([[0.0, 0.8], [0.8, 0.0]], dtype=np.float64)
    rule = multi_frequency_parameter_shift_rule([2.0])

    result = learn_couplings_from_observations(
        _sin_observation,
        np.array([0.0], dtype=np.float64),
        initial,
        rule=rule,
        learning_rate=0.35,
        max_steps=80,
        gradient_tolerance=1e-7,
        min_loss_decrease=0.1,
    )

    assert isinstance(result, CouplingLearningResult)
    assert result.backend == "statevector_simulator"
    assert result.edges == ((0, 1),)
    assert result.training.accepted_steps > 0
    assert result.best_loss < 1e-8
    assert result.max_abs_residual < 1e-4
    assert result.certificate.monotone_accepted_values
    assert result.certificate.min_decrease_satisfied
    payload = result.to_dict()
    assert payload["backend"] == "statevector_simulator"
    assert payload["edges"] == [[0, 1]]
    np.testing.assert_allclose(
        result.learned_coupling_matrix,
        result.learned_coupling_matrix.T,
        atol=1e-14,
    )
    np.testing.assert_allclose(np.diag(result.learned_coupling_matrix), 0.0, atol=1e-14)


def test_coupling_matrix_from_edge_vector_preserves_static_edges() -> None:
    matrix = coupling_matrix_from_edge_vector(
        np.array([0.2, -0.4], dtype=np.float64),
        n_nodes=3,
        edges=((0, 2), (1, 2)),
    )

    expected = np.array(
        [
            [0.0, 0.0, 0.2],
            [0.0, 0.0, -0.4],
            [0.2, -0.4, 0.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(matrix, expected, atol=1e-14)


def test_verify_coupling_parameter_shift_gradient_matches_finite_difference() -> None:
    initial = np.array([[0.0, 0.8], [0.8, 0.0]], dtype=np.float64)
    rule = multi_frequency_parameter_shift_rule([2.0])

    certificate = verify_coupling_parameter_shift_gradient(
        _sin_observation,
        np.array([0.0], dtype=np.float64),
        initial,
        rule=rule,
        finite_difference_step=1e-6,
        tolerance=1e-5,
    )

    assert isinstance(certificate, CouplingGradientVerificationResult)
    assert certificate.passed
    assert certificate.edges == ((0, 1),)
    assert certificate.max_abs_error < 1e-5
    assert certificate.parameter_shift_evaluations >= 3
    assert certificate.finite_difference_evaluations >= 3
    np.testing.assert_allclose(
        certificate.parameter_shift_gradient,
        certificate.finite_difference_gradient,
        atol=1e-5,
    )
    payload = certificate.to_dict()
    assert payload["passed"] is True
    assert payload["edges"] == [[0, 1]]
    assert payload["method"] == "parameter_shift_vs_central_finite_difference"


def test_verify_coupling_parameter_shift_gradient_rejects_bad_step() -> None:
    with pytest.raises(ValueError, match="finite_difference_step"):
        verify_coupling_parameter_shift_gradient(
            _sin_observation,
            np.array([0.0], dtype=np.float64),
            np.array([[0.0, 0.8], [0.8, 0.0]], dtype=np.float64),
            finite_difference_step=0.0,
        )


def test_learn_couplings_from_observations_fails_closed_for_hardware() -> None:
    with pytest.raises(ValueError, match="hardware gradient execution requires"):
        learn_couplings_from_observations(
            _sin_observation,
            np.array([0.0], dtype=np.float64),
            np.array([[0.0, 0.8], [0.8, 0.0]], dtype=np.float64),
            backend="hardware",
            max_steps=2,
        )


def test_learn_couplings_from_observations_rejects_invalid_matrices() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        learn_couplings_from_observations(
            _sin_observation,
            np.array([0.0], dtype=np.float64),
            np.array([[0.0, 0.8], [0.2, 0.0]], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="diagonal"):
        learn_couplings_from_observations(
            _sin_observation,
            np.array([0.0], dtype=np.float64),
            np.array([[1.0, 0.8], [0.8, 0.0]], dtype=np.float64),
        )


def test_learn_couplings_from_observations_rejects_shape_drift() -> None:
    def bad_observation_model(couplings: FloatArray) -> FloatArray:
        return np.array([couplings[0, 1], couplings[0, 1]], dtype=np.float64)

    with pytest.raises(ValueError, match="output shape"):
        learn_couplings_from_observations(
            bad_observation_model,
            np.array([0.0], dtype=np.float64),
            np.array([[0.0, 0.8], [0.8, 0.0]], dtype=np.float64),
            max_steps=2,
        )


def test_default_rule_learning_and_verification_cover_json_contracts() -> None:
    initial = np.array([[0.0, 0.6], [0.6, 0.0]], dtype=np.float64)
    rule = ParameterShiftRule()

    result = learn_couplings_from_observations(
        _sin_observation,
        np.array([0.0], dtype=np.float64),
        initial,
        rule=rule,
        learning_rate=0.25,
        max_steps=40,
        gradient_tolerance=1e-7,
        min_loss_decrease=0.01,
    )
    certificate = verify_coupling_parameter_shift_gradient(
        _sin_observation,
        np.array([0.0], dtype=np.float64),
        initial,
        rule=rule,
        finite_difference_step=1e-6,
        tolerance=1e-5,
    )

    assert result.to_dict()["claim_boundary"]
    assert certificate.to_dict()["claim_boundary"]
    assert result.best_loss <= result.training.initial_value
    assert isinstance(certificate.passed, bool)


def test_vector_initialisation_and_complete_graph_edges() -> None:
    matrix = coupling_matrix_from_edge_vector(
        np.array([0.1, 0.2, 0.3], dtype=np.float64),
        n_nodes=3,
    )

    result = learn_couplings_from_observations(
        lambda coupling_matrix: np.array(
            [coupling_matrix[0, 1], coupling_matrix[0, 2], coupling_matrix[1, 2]],
            dtype=np.float64,
        ),
        np.array([0.1, 0.2, 0.3], dtype=np.float64),
        np.array([0.1, 0.2, 0.3], dtype=np.float64),
        n_nodes=3,
        max_steps=1,
    )

    np.testing.assert_allclose(np.diag(matrix), 0.0, atol=1e-14)
    assert result.edges == ((0, 1), (0, 2), (1, 2))


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.array([], dtype=np.float64), "non-empty one-dimensional"),
        (np.array([[1.0]], dtype=np.float64), "one-dimensional"),
        (np.array([np.nan], dtype=np.float64), "finite couplings"),
    ],
)
def test_coupling_matrix_from_edge_vector_rejects_invalid_values(
    values: FloatArray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        coupling_matrix_from_edge_vector(values, n_nodes=2)


def test_coupling_matrix_from_edge_vector_rejects_bad_node_count_and_length() -> None:
    with pytest.raises(ValueError, match="at least two"):
        coupling_matrix_from_edge_vector(np.array([0.1], dtype=np.float64), n_nodes=1)
    with pytest.raises(ValueError, match="values length"):
        coupling_matrix_from_edge_vector(np.array([0.1], dtype=np.float64), n_nodes=3)


@pytest.mark.parametrize(
    ("edges", "message"),
    [
        (((0, 1, 2),), "exactly two"),
        (((1, 1),), "self edges"),
        (((0, 3),), "out of bounds"),
        (((0, 1), (1, 0)), "unique"),
        ((), "at least one"),
    ],
)
def test_coupling_matrix_from_edge_vector_rejects_invalid_edges(
    edges: tuple[tuple[int, ...], ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        coupling_matrix_from_edge_vector(
            np.array([0.1], dtype=np.float64),
            n_nodes=3,
            edges=edges,
        )


def test_initial_couplings_reject_invalid_shapes_and_sizes() -> None:
    with pytest.raises(ValueError, match="vector or square matrix"):
        learn_couplings_from_observations(
            _sin_observation,
            np.array([0.0], dtype=np.float64),
            np.zeros((1, 1, 1), dtype=np.float64),
        )
    with pytest.raises(ValueError, match="square"):
        learn_couplings_from_observations(
            _sin_observation,
            np.array([0.0], dtype=np.float64),
            np.zeros((2, 3), dtype=np.float64),
        )
    with pytest.raises(ValueError, match="at least two nodes"):
        learn_couplings_from_observations(
            _sin_observation,
            np.array([0.0], dtype=np.float64),
            np.zeros((1, 1), dtype=np.float64),
        )
    with pytest.raises(ValueError, match="n_nodes must match"):
        learn_couplings_from_observations(
            _sin_observation,
            np.array([0.0], dtype=np.float64),
            np.array([[0.0, 0.1], [0.1, 0.0]], dtype=np.float64),
            n_nodes=3,
        )
    with pytest.raises(ValueError, match="n_nodes is required"):
        learn_couplings_from_observations(
            _sin_observation,
            np.array([0.0], dtype=np.float64),
            np.array([0.1], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="finite values"):
        learn_couplings_from_observations(
            _sin_observation,
            np.array([0.0], dtype=np.float64),
            np.array([[0.0, np.nan], [np.nan, 0.0]], dtype=np.float64),
        )


def test_target_and_observation_values_must_be_finite() -> None:
    with pytest.raises(ValueError, match="target_observations"):
        learn_couplings_from_observations(
            _sin_observation,
            np.array([[0.0]], dtype=np.float64),
            np.array([[0.0, 0.1], [0.1, 0.0]], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="target_observations"):
        learn_couplings_from_observations(
            _sin_observation,
            np.array([np.nan], dtype=np.float64),
            np.array([[0.0, 0.1], [0.1, 0.0]], dtype=np.float64),
        )

    def nonfinite_model(_couplings: FloatArray) -> FloatArray:
        return np.array([np.inf], dtype=np.float64)

    with pytest.raises(ValueError, match="observation_model"):
        learn_couplings_from_observations(
            nonfinite_model,
            np.array([0.0], dtype=np.float64),
            np.array([[0.0, 0.1], [0.1, 0.0]], dtype=np.float64),
            max_steps=1,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_abs_error": -1.0}, "max_abs_error"),
        ({"finite_difference_step": -1.0}, "finite_difference_step"),
        ({"tolerance": -1.0}, "tolerance"),
        ({"passed": 1}, "passed"),
        ({"method": ""}, "method"),
        ({"parameter_shift_evaluations": 0}, "parameter_shift_evaluations"),
        ({"finite_difference_evaluations": 0}, "finite_difference_evaluations"),
    ],
)
def test_gradient_verification_result_rejects_invalid_scalar_metadata(
    kwargs: dict[str, object],
    message: str,
) -> None:
    payload: dict[str, object] = {
        "parameters": np.array([0.1], dtype=np.float64),
        "parameter_shift_gradient": np.array([0.1], dtype=np.float64),
        "finite_difference_gradient": np.array([0.1], dtype=np.float64),
        "abs_error": np.array([0.0], dtype=np.float64),
        "max_abs_error": 0.0,
        "objective_value": 0.1,
        "value_delta": 0.0,
        "passed": True,
        "method": "check",
        "finite_difference_step": 1e-6,
        "tolerance": 1e-5,
        "parameter_shift_evaluations": 1,
        "finite_difference_evaluations": 1,
        "edges": ((0, 1),),
        "claim_boundary": "small smooth diagnostic",
    }
    payload.update(kwargs)

    with pytest.raises(ValueError, match=message):
        CouplingGradientVerificationResult(**payload)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"parameter_shift_gradient": np.array([0.1, 0.2], dtype=np.float64)}, "shape"),
        ({"finite_difference_gradient": np.array([0.1, 0.2], dtype=np.float64)}, "shape"),
        ({"abs_error": np.array([0.0, 0.1], dtype=np.float64)}, "shape"),
        ({"abs_error": np.array([-0.1], dtype=np.float64)}, "non-negative"),
        ({"parameters": np.array([[0.1]], dtype=np.float64)}, "one-dimensional"),
        ({"parameters": np.array([np.nan], dtype=np.float64)}, "finite"),
        ({"max_abs_error": "bad"}, "finite real scalar"),
    ],
)
def test_gradient_verification_result_rejects_invalid_arrays(
    kwargs: dict[str, object],
    message: str,
) -> None:
    payload: dict[str, object] = {
        "parameters": np.array([0.1], dtype=np.float64),
        "parameter_shift_gradient": np.array([0.1], dtype=np.float64),
        "finite_difference_gradient": np.array([0.1], dtype=np.float64),
        "abs_error": np.array([0.0], dtype=np.float64),
        "max_abs_error": 0.0,
        "objective_value": 0.1,
        "value_delta": 0.0,
        "passed": True,
        "method": "check",
        "finite_difference_step": 1e-6,
        "tolerance": 1e-5,
        "parameter_shift_evaluations": 1,
        "finite_difference_evaluations": 1,
        "edges": ((0, 1),),
        "claim_boundary": "small smooth diagnostic",
    }
    payload.update(kwargs)

    with pytest.raises(ValueError, match=message):
        CouplingGradientVerificationResult(**payload)


def test_verify_coupling_parameter_shift_gradient_rejects_bad_tolerance() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        verify_coupling_parameter_shift_gradient(
            _sin_observation,
            np.array([0.0], dtype=np.float64),
            np.array([[0.0, 0.8], [0.8, 0.0]], dtype=np.float64),
            tolerance=-1.0,
        )
    with pytest.raises(ValueError, match="finite real scalar"):
        verify_coupling_parameter_shift_gradient(
            _sin_observation,
            np.array([0.0], dtype=np.float64),
            np.array([[0.0, 0.8], [0.8, 0.0]], dtype=np.float64),
            finite_difference_step=np.nan,
        )
