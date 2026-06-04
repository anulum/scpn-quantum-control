# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase Provider Gradients
"""Tests for provider-safe parameter-shift gradient execution."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_quantum_control.phase import (
    ProviderExpectationSample,
    ProviderGradientExecutionResult,
    execute_provider_parameter_shift_gradient,
    multi_frequency_parameter_shift_rule,
)


def _objective(values: np.ndarray) -> float:
    return float(np.cos(values[0]) + 0.25 * np.sin(values[1]))


def test_provider_gradient_executes_statevector_parameter_shift() -> None:
    values = np.array([0.2, -0.4], dtype=float)

    def sampler(params: np.ndarray, shots: int | None) -> ProviderExpectationSample:
        assert shots is None
        return ProviderExpectationSample(value=_objective(params))

    result = execute_provider_parameter_shift_gradient(
        sampler,
        values,
        backend="statevector",
    )

    expected = np.array([-np.sin(values[0]), 0.25 * np.cos(values[1])], dtype=float)
    assert isinstance(result, ProviderGradientExecutionResult)
    assert result.backend == "statevector_simulator"
    assert result.method == "parameter_shift"
    assert result.total_evaluations == 4
    assert result.total_shots is None
    assert result.claim_boundary.startswith("provider callback")
    np.testing.assert_allclose(result.gradient, expected, atol=1e-12)
    np.testing.assert_allclose(result.standard_error, np.zeros(2), atol=0.0)
    assert result.to_dict()["total_evaluations"] == 4


def test_provider_gradient_executes_finite_shot_parameter_shift_with_uncertainty() -> None:
    values = np.array([0.2, -0.4], dtype=float)
    observed_shots: list[int | None] = []

    def sampler(params: np.ndarray, shots: int | None) -> ProviderExpectationSample:
        observed_shots.append(shots)
        return ProviderExpectationSample(
            value=_objective(params),
            variance=0.04,
            shots=shots,
            metadata={"source": "finite-shot fixture"},
        )

    result = execute_provider_parameter_shift_gradient(
        sampler,
        values,
        backend="qasm_simulator",
        shots=400,
    )

    expected_se = 0.5 * math.sqrt(0.04 / 400 + 0.04 / 400)
    assert observed_shots == [400, 400, 400, 400]
    assert result.backend == "finite_shot_simulator"
    assert result.method == "stochastic_parameter_shift"
    assert result.total_shots == 1600
    assert result.records[0].plus.metadata["source"] == "finite-shot fixture"
    np.testing.assert_allclose(result.standard_error, np.array([expected_se, expected_se]))
    np.testing.assert_allclose(result.confidence_radius, 1.959963984540054 * result.standard_error)


def test_provider_gradient_executes_multi_frequency_finite_shot_records() -> None:
    values = np.array([0.4], dtype=float)
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])
    observed_shots: list[int | None] = []

    def objective(params: np.ndarray) -> float:
        return float(np.sin(params[0]) + 0.1 * np.cos(2.0 * params[0]))

    def sampler(params: np.ndarray, shots: int | None) -> ProviderExpectationSample:
        observed_shots.append(shots)
        return ProviderExpectationSample(
            value=objective(params),
            variance=0.05,
            shots=shots,
            metadata={"term_safe": True},
        )

    result = execute_provider_parameter_shift_gradient(
        sampler,
        values,
        backend="qasm_simulator",
        shots=300,
        rule=rule,
    )

    expected_gradient = np.cos(values[0]) - 0.2 * np.sin(2.0 * values[0])
    expected_variance = sum(
        coefficient**2 * (0.05 / 300 + 0.05 / 300) for _, coefficient in rule.terms
    )

    assert observed_shots == [300] * (2 * len(rule.terms))
    assert result.method == "multi_frequency_stochastic_parameter_shift"
    assert result.plan.shift_terms == len(rule.terms)
    assert result.plan.evaluations == 2 * len(rule.terms)
    assert result.total_evaluations == 2 * len(rule.terms)
    assert result.total_shots == 2 * len(rule.terms) * 300
    assert [record.shift_index for record in result.records] == [0, 1]
    assert result.records[0].plus.metadata["term_safe"] is True
    np.testing.assert_allclose(result.gradient, np.array([expected_gradient]), atol=1e-12)
    np.testing.assert_allclose(result.standard_error, np.array([math.sqrt(expected_variance)]))
    np.testing.assert_allclose(result.confidence_radius, 1.959963984540054 * result.standard_error)


def test_provider_gradient_fails_closed_for_hardware_without_policy() -> None:
    def sampler(params: np.ndarray, shots: int | None) -> ProviderExpectationSample:
        return ProviderExpectationSample(value=_objective(params), variance=0.04, shots=shots)

    with pytest.raises(ValueError, match="hardware gradient execution requires"):
        execute_provider_parameter_shift_gradient(
            sampler,
            np.array([0.2, -0.4], dtype=float),
            backend="ibm_quantum",
            shots=1024,
        )


def test_provider_gradient_rejects_invalid_samples() -> None:
    def non_finite_sampler(params: np.ndarray, shots: int | None) -> ProviderExpectationSample:
        return ProviderExpectationSample(value=float("nan"))

    with pytest.raises(ValueError, match="sample value"):
        execute_provider_parameter_shift_gradient(
            non_finite_sampler,
            np.array([0.2], dtype=float),
            backend="statevector",
        )

    def missing_variance_sampler(
        params: np.ndarray, shots: int | None
    ) -> ProviderExpectationSample:
        return ProviderExpectationSample(value=_objective(params), shots=shots)

    with pytest.raises(ValueError, match="variance"):
        execute_provider_parameter_shift_gradient(
            missing_variance_sampler,
            np.array([0.2, -0.4], dtype=float),
            backend="qasm_simulator",
            shots=400,
        )
