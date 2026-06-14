# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — SPSA Gradient Tests
"""Tests for backend-neutral SPSA gradient estimation."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_quantum_control as scpn
from scpn_quantum_control import Parameter
from scpn_quantum_control.differentiable import (
    SPSAGradientResult,
    SPSAObjectiveSample,
    spsa_gradient_estimate,
)


def _linear_objective(params: np.ndarray) -> float:
    return float(1.2 * params[0] + 0.5)


def test_spsa_gradient_estimate_is_seeded_and_records_probe_pairs() -> None:
    params = np.array([0.4, -0.2, 0.8], dtype=np.float64)

    first = spsa_gradient_estimate(
        _linear_objective,
        params,
        perturbation_radius=0.125,
        repetitions=6,
        seed=123,
        parameters=(
            Parameter("theta_0"),
            Parameter("theta_1", trainable=False),
            Parameter("frozen", trainable=False),
        ),
    )
    second = spsa_gradient_estimate(
        _linear_objective,
        params,
        perturbation_radius=0.125,
        repetitions=6,
        seed=123,
        parameters=(
            Parameter("theta_0"),
            Parameter("theta_1", trainable=False),
            Parameter("frozen", trainable=False),
        ),
    )

    assert isinstance(first, SPSAGradientResult)
    assert first.method == "spsa"
    assert first.parameter_names == ("theta_0", "theta_1", "frozen")
    assert first.trainable == (True, False, False)
    assert first.evaluations == 12
    assert len(first.records) == 6
    assert first.total_shots is None
    assert first.hardware_execution is False
    np.testing.assert_allclose(first.gradient, [1.2, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(first.gradient, second.gradient, atol=0.0)
    np.testing.assert_allclose(first.records[0].perturbation[-1], 0.0, atol=0.0)
    assert first.to_dict()["method"] == "spsa"


def test_spsa_gradient_estimate_propagates_finite_shot_uncertainty() -> None:
    params = np.array([0.4, -0.2], dtype=np.float64)

    def sample_objective(values: np.ndarray, shots: int | None) -> SPSAObjectiveSample:
        assert shots == 400
        return SPSAObjectiveSample(
            value=float(0.5 * values[0] - 0.25 * values[1]),
            variance=0.04,
            shots=shots,
            metadata={"fixture": "finite-shot"},
        )

    result = spsa_gradient_estimate(
        sample_objective,
        params,
        perturbation_radius=0.25,
        repetitions=4,
        seed=11,
        shots=400,
        confidence_z=2.0,
    )

    assert result.method == "finite_shot_spsa"
    assert result.total_shots == 3200
    assert result.records[0].plus.metadata["fixture"] == "finite-shot"
    assert np.all(result.standard_error > 0.0)
    np.testing.assert_allclose(result.confidence_radius, 2.0 * result.standard_error)


def test_spsa_gradient_estimate_fails_closed_for_invalid_contracts() -> None:
    with pytest.raises(ValueError, match="perturbation_radius"):
        spsa_gradient_estimate(_linear_objective, [0.1], perturbation_radius=0.0)
    with pytest.raises(ValueError, match="repetitions"):
        spsa_gradient_estimate(_linear_objective, [0.1], repetitions=0)
    with pytest.raises(ValueError, match="seed"):
        spsa_gradient_estimate(_linear_objective, [0.1], seed=-1)
    with pytest.raises(ValueError, match="variance"):
        spsa_gradient_estimate(
            lambda values, shots: SPSAObjectiveSample(value=float(values[0]), shots=shots),
            [0.1],
            shots=100,
        )


def test_spsa_exports_from_package_root() -> None:
    assert scpn.SPSAObjectiveSample is SPSAObjectiveSample
    assert scpn.SPSAGradientResult is SPSAGradientResult
    assert scpn.spsa_gradient_estimate is spsa_gradient_estimate
