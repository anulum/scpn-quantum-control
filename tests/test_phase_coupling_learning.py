# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase Coupling Learning
"""Tests for phase/coupling_learning.py differentiable coupling training."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.phase import (
    CouplingLearningResult,
    coupling_matrix_from_edge_vector,
    learn_couplings_from_observations,
    multi_frequency_parameter_shift_rule,
)


def test_learn_couplings_from_observations_converges_on_sinusoidal_edge() -> None:
    def observation_model(couplings: np.ndarray) -> np.ndarray:
        return np.array([np.sin(couplings[0, 1])], dtype=float)

    initial = np.array([[0.0, 0.8], [0.8, 0.0]], dtype=float)
    rule = multi_frequency_parameter_shift_rule([2.0])

    result = learn_couplings_from_observations(
        observation_model,
        np.array([0.0], dtype=float),
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
    np.testing.assert_allclose(
        result.learned_coupling_matrix,
        result.learned_coupling_matrix.T,
        atol=1e-14,
    )
    np.testing.assert_allclose(np.diag(result.learned_coupling_matrix), 0.0, atol=1e-14)


def test_coupling_matrix_from_edge_vector_preserves_static_edges() -> None:
    matrix = coupling_matrix_from_edge_vector(
        np.array([0.2, -0.4], dtype=float),
        n_nodes=3,
        edges=((0, 2), (1, 2)),
    )

    expected = np.array(
        [
            [0.0, 0.0, 0.2],
            [0.0, 0.0, -0.4],
            [0.2, -0.4, 0.0],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(matrix, expected, atol=1e-14)


def test_learn_couplings_from_observations_fails_closed_for_hardware() -> None:
    def observation_model(couplings: np.ndarray) -> np.ndarray:
        return np.array([np.sin(couplings[0, 1])], dtype=float)

    with pytest.raises(ValueError, match="hardware gradient execution requires"):
        learn_couplings_from_observations(
            observation_model,
            np.array([0.0], dtype=float),
            np.array([[0.0, 0.8], [0.8, 0.0]], dtype=float),
            backend="hardware",
            max_steps=2,
        )


def test_learn_couplings_from_observations_rejects_invalid_matrices() -> None:
    def observation_model(couplings: np.ndarray) -> np.ndarray:
        return np.array([np.sin(couplings[0, 1])], dtype=float)

    with pytest.raises(ValueError, match="symmetric"):
        learn_couplings_from_observations(
            observation_model,
            np.array([0.0], dtype=float),
            np.array([[0.0, 0.8], [0.2, 0.0]], dtype=float),
        )

    with pytest.raises(ValueError, match="diagonal"):
        learn_couplings_from_observations(
            observation_model,
            np.array([0.0], dtype=float),
            np.array([[1.0, 0.8], [0.8, 0.0]], dtype=float),
        )


def test_learn_couplings_from_observations_rejects_shape_drift() -> None:
    def bad_observation_model(couplings: np.ndarray) -> np.ndarray:
        return np.array([couplings[0, 1], couplings[0, 1]], dtype=float)

    with pytest.raises(ValueError, match="output shape"):
        learn_couplings_from_observations(
            bad_observation_model,
            np.array([0.0], dtype=float),
            np.array([[0.0, 0.8], [0.8, 0.0]], dtype=float),
            max_steps=2,
        )
