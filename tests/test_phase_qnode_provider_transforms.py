# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Provider QNode Transforms
"""Tests for provider-callback QNode transform execution evidence."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_quantum_control.phase import (
    ProviderExpectationSample,
    ProviderQNodeTransformResult,
    execute_provider_qnode_transform,
    execute_provider_qnode_vmap_grad,
    run_provider_qnode_transform_readiness_suite,
)


def _objective(values: np.ndarray) -> float:
    return float(np.cos(values[0]) + 0.25 * np.sin(values[1]))


def _gradient(values: np.ndarray) -> np.ndarray:
    return np.array([-np.sin(values[0]), 0.25 * np.cos(values[1])], dtype=float)


def _sampler(values: np.ndarray, shots: int | None) -> ProviderExpectationSample:
    return ProviderExpectationSample(
        value=_objective(values),
        variance=None if shots is None else 0.04,
        shots=shots,
        metadata={"route": "provider-qnode-fixture"},
    )


def test_provider_qnode_transform_executes_scalar_transform_family() -> None:
    values = np.array([0.2, -0.4], dtype=float)
    tangent = np.array([0.5, -2.0], dtype=float)
    cotangent = np.array([3.0], dtype=float)
    expected_gradient = _gradient(values)

    grad = execute_provider_qnode_transform("grad", _sampler, values)
    value_grad = execute_provider_qnode_transform("value_and_grad", _sampler, values)
    jvp = execute_provider_qnode_transform("jvp", _sampler, values, tangent=tangent)
    vjp = execute_provider_qnode_transform("vjp", _sampler, values, cotangent=cotangent)
    jacfwd = execute_provider_qnode_transform("jacfwd", _sampler, values)

    assert isinstance(grad, ProviderQNodeTransformResult)
    assert grad.supported
    assert grad.gradient is not None
    np.testing.assert_allclose(grad.gradient, expected_gradient, atol=1e-12)
    assert grad.value is None
    assert grad.provider_gradient_result is not None
    assert grad.total_evaluations == 4
    assert "provider callback QNode transform" in grad.claim_boundary

    assert value_grad.supported
    assert value_grad.value == _objective(values)
    assert value_grad.gradient is not None
    np.testing.assert_allclose(value_grad.gradient, expected_gradient, atol=1e-12)

    assert jvp.supported
    assert jvp.jvp == pytest.approx(float(np.dot(expected_gradient, tangent)))
    assert vjp.supported
    assert vjp.vjp is not None
    np.testing.assert_allclose(vjp.vjp, cotangent[0] * expected_gradient, atol=1e-12)

    assert jacfwd.supported
    assert jacfwd.jacobian is not None
    np.testing.assert_allclose(jacfwd.jacobian, expected_gradient.reshape(1, -1), atol=1e-12)


def test_provider_qnode_transform_executes_finite_shot_with_uncertainty() -> None:
    values = np.array([0.2, -0.4], dtype=float)

    result = execute_provider_qnode_transform(
        "value_and_grad",
        _sampler,
        values,
        backend="qasm_simulator",
        shots=400,
    )

    expected_se = 0.5 * math.sqrt(0.04 / 400 + 0.04 / 400)
    assert result.supported
    assert result.value == _objective(values)
    assert result.total_shots == 2000
    assert result.standard_error is not None
    assert result.confidence_radius is not None
    np.testing.assert_allclose(result.standard_error, np.array([expected_se, expected_se]))
    np.testing.assert_allclose(result.confidence_radius, 1.959963984540054 * result.standard_error)


def test_provider_qnode_vmap_grad_executes_rowwise_callback_gradients() -> None:
    batched_values = np.array(
        [[0.2, -0.4], [0.7, 0.1], [-0.3, 0.6]],
        dtype=float,
    )

    result = execute_provider_qnode_vmap_grad(_sampler, batched_values)

    assert result.supported
    assert result.transform == "vmap.grad"
    assert result.batched_values is not None
    assert result.batched_gradients is not None
    np.testing.assert_allclose(
        result.batched_values,
        np.array([_objective(row) for row in batched_values]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.batched_gradients,
        np.vstack([_gradient(row) for row in batched_values]),
        atol=1e-12,
    )
    assert len(result.provider_gradient_results) == 3
    assert result.total_evaluations == 12


def test_provider_qnode_transforms_fail_closed_for_unsafe_routes() -> None:
    values = np.array([0.2, -0.4], dtype=float)
    batched_values = np.array([[0.2, -0.4], [0.7, 0.1]], dtype=float)

    hardware = execute_provider_qnode_transform(
        "grad",
        _sampler,
        values,
        backend="ibm_quantum",
        shots=1024,
    )
    curvature = execute_provider_qnode_transform("hessian", _sampler, values)
    finite_vmap = execute_provider_qnode_vmap_grad(
        lambda shifted, shots: ProviderExpectationSample(value=_objective(shifted), shots=shots),
        batched_values,
        backend="qasm_simulator",
        shots=256,
    )

    assert hardware.fail_closed
    assert "hardware gradient execution requires" in hardware.failure_reason
    assert hardware.gradient is None

    assert curvature.fail_closed
    assert "provider QNode curvature transforms are not implemented" in curvature.failure_reason

    assert finite_vmap.fail_closed
    assert "variance" in finite_vmap.failure_reason
    assert finite_vmap.batched_gradients is None


def test_provider_qnode_transform_validates_directional_inputs() -> None:
    values = np.array([0.2, -0.4], dtype=float)

    missing_tangent = execute_provider_qnode_transform("jvp", _sampler, values)
    bad_cotangent = execute_provider_qnode_transform(
        "vjp",
        _sampler,
        values,
        cotangent=np.array([1.0, 2.0], dtype=float),
    )

    assert missing_tangent.fail_closed
    assert "tangent" in missing_tangent.failure_reason
    assert bad_cotangent.fail_closed
    assert "cotangent" in bad_cotangent.failure_reason


def test_provider_qnode_transform_readiness_suite_records_boundaries() -> None:
    suite = run_provider_qnode_transform_readiness_suite()
    payload = suite.to_dict()

    assert suite.passed
    assert suite.record_count == 8
    assert suite.supported_count == 5
    assert suite.fail_closed_count == 3
    assert suite.total_parameter_shift_evaluations > 0
    assert not suite.hardware_execution
    assert {"grad", "value_and_grad", "jvp", "jacfwd", "vmap.grad"}.issubset(
        {record.transform for record in suite.records if record.supported}
    )
    assert payload["passed"] is True
