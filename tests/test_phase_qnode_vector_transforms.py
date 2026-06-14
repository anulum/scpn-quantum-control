# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Vector Phase QNode Transforms
"""Tests for vector-output QNode Jacobians and native manual vmap gradients."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.phase import (
    PhaseQNodeVectorTransformResult,
    execute_phase_qnode_vector_jacobian,
    execute_phase_qnode_vector_jvp,
    execute_phase_qnode_vector_vjp,
    execute_phase_qnode_vmap_grad,
    plan_gradient_transform_nesting,
    run_phase_qnode_vector_transform_readiness_suite,
)


def _scalar_objective(params: np.ndarray) -> float:
    return float(np.cos(params[0]) + 0.25 * np.sin(params[1]))


def _scalar_gradient(params: np.ndarray) -> np.ndarray:
    return np.array([-np.sin(params[0]), 0.25 * np.cos(params[1])], dtype=float)


def _vector_objective(params: np.ndarray) -> np.ndarray:
    return np.array(
        [
            np.cos(params[0]) + 0.1 * np.sin(params[1]),
            np.sin(params[0]) - 0.25 * np.cos(params[1]),
        ],
        dtype=float,
    )


def _vector_jacobian(params: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [-np.sin(params[0]), 0.1 * np.cos(params[1])],
            [np.cos(params[0]), 0.25 * np.sin(params[1])],
        ],
        dtype=float,
    )


def test_phase_qnode_vector_jacobians_match_analytic_reference() -> None:
    params = np.array([0.31, -0.17], dtype=float)

    jacfwd = execute_phase_qnode_vector_jacobian("jacfwd", _vector_objective, params)
    jacrev = execute_phase_qnode_vector_jacobian("jacrev", _vector_objective, params)

    assert isinstance(jacfwd, PhaseQNodeVectorTransformResult)
    assert jacfwd.supported
    assert jacrev.supported
    assert jacfwd.values is not None
    assert jacfwd.jacobian is not None
    assert jacrev.jacobian is not None
    np.testing.assert_allclose(jacfwd.values, _vector_objective(params), atol=1e-12)
    np.testing.assert_allclose(jacfwd.jacobian, _vector_jacobian(params), atol=1e-12)
    np.testing.assert_allclose(jacrev.jacobian, _vector_jacobian(params), atol=1e-12)
    assert jacfwd.output_dim == 2
    assert jacfwd.parameter_shift_evaluations == 8
    assert "vector-output phase-QNode Jacobian" in jacfwd.claim_boundary


def test_phase_qnode_vector_jvp_and_vjp_match_jacobian_reference() -> None:
    params = np.array([0.31, -0.17], dtype=float)
    tangent = np.array([0.5, -1.25], dtype=float)
    cotangent = np.array([2.0, -0.75], dtype=float)
    expected_jacobian = _vector_jacobian(params)

    jvp = execute_phase_qnode_vector_jvp(_vector_objective, params, tangent)
    vjp = execute_phase_qnode_vector_vjp(_vector_objective, params, cotangent)

    assert jvp.supported
    assert vjp.supported
    assert jvp.transform == "jvp"
    assert vjp.transform == "vjp"
    assert jvp.jvp is not None
    assert vjp.vjp is not None
    np.testing.assert_allclose(jvp.jvp, expected_jacobian @ tangent, atol=1e-12)
    np.testing.assert_allclose(vjp.vjp, expected_jacobian.T @ cotangent, atol=1e-12)
    np.testing.assert_allclose(jvp.tangent, tangent, atol=1e-12)
    np.testing.assert_allclose(vjp.cotangent, cotangent, atol=1e-12)
    assert jvp.parameter_shift_evaluations == 8
    assert vjp.parameter_shift_evaluations == 8
    assert "vector-output phase-QNode directional" in jvp.claim_boundary


def test_phase_qnode_vector_jvp_vjp_validate_direction_shapes() -> None:
    params = np.array([0.31, -0.17], dtype=float)

    with pytest.raises(ValueError, match="tangent"):
        execute_phase_qnode_vector_jvp(_vector_objective, params, np.array([1.0], dtype=float))
    with pytest.raises(ValueError, match="cotangent"):
        execute_phase_qnode_vector_vjp(
            _vector_objective,
            params,
            np.array([1.0], dtype=float),
        )


def test_phase_qnode_vector_jvp_vjp_fail_closed_for_unsafe_routes() -> None:
    params = np.array([0.31, -0.17], dtype=float)

    hardware = execute_phase_qnode_vector_jvp(
        _vector_objective,
        params,
        np.array([0.5, -1.25], dtype=float),
        backend="hardware",
        shots=1024,
    )
    adapter = execute_phase_qnode_vector_vjp(
        _vector_objective,
        params,
        np.array([2.0, -0.75], dtype=float),
        adapter="jax",
    )

    assert hardware.fail_closed
    assert hardware.jvp is None
    assert "hardware gradient execution requires" in hardware.failure_reason
    assert adapter.fail_closed
    assert adapter.vjp is None
    assert "native local route" in adapter.failure_reason


def test_phase_qnode_vmap_grad_matches_rowwise_analytic_reference() -> None:
    batched_params = np.array(
        [[0.2, -0.4], [0.7, 0.1], [-0.3, 0.6]],
        dtype=float,
    )
    expected_values = np.array([_scalar_objective(row) for row in batched_params])
    expected_gradients = np.vstack([_scalar_gradient(row) for row in batched_params])

    result = execute_phase_qnode_vmap_grad(_scalar_objective, batched_params)

    assert result.supported
    assert result.transform == "vmap.grad"
    assert result.batch_size == 3
    assert result.batched_values is not None
    assert result.batched_gradients is not None
    np.testing.assert_allclose(result.batched_values, expected_values, atol=1e-12)
    np.testing.assert_allclose(result.batched_gradients, expected_gradients, atol=1e-12)
    assert result.parameter_shift_evaluations == 12
    assert result.plan.strategy == "native_manual_vmap_parameter_shift_grad"
    assert result.plan.requires_deterministic_backend


def test_phase_qnode_vector_transforms_fail_closed_for_unsafe_routes() -> None:
    params = np.array([0.2, -0.4], dtype=float)
    batched_params = np.array([[0.2, -0.4], [0.7, 0.1]], dtype=float)

    hardware = execute_phase_qnode_vector_jacobian(
        "jacfwd",
        _vector_objective,
        params,
        backend="hardware",
        shots=1024,
    )
    adapter = execute_phase_qnode_vector_jacobian(
        "jacrev",
        _vector_objective,
        params,
        adapter="jax",
    )
    finite_shot_vmap = execute_phase_qnode_vmap_grad(
        _scalar_objective,
        batched_params,
        backend="qasm_simulator",
        shots=256,
    )

    assert hardware.fail_closed
    assert hardware.jacobian is None
    assert "hardware gradient execution requires" in hardware.failure_reason
    assert adapter.fail_closed
    assert "native local route" in adapter.failure_reason
    assert finite_shot_vmap.fail_closed
    assert finite_shot_vmap.batched_gradients is None
    assert "deterministic local expectations" in finite_shot_vmap.failure_reason


def test_phase_qnode_vector_transforms_validate_shapes_and_finiteness() -> None:
    params = np.array([0.2, -0.4], dtype=float)

    with pytest.raises(ValueError, match="vector output"):
        execute_phase_qnode_vector_jacobian("jacfwd", lambda _: 1.0, params)
    with pytest.raises(ValueError, match="output shape"):
        execute_phase_qnode_vector_jacobian(
            "jacfwd",
            lambda shifted: np.array([shifted[0]]) if shifted[0] > 0 else np.array([1.0, 2.0]),
            params,
        )
    with pytest.raises(ValueError, match="two-dimensional"):
        execute_phase_qnode_vmap_grad(_scalar_objective, params)
    with pytest.raises(ValueError, match="finite"):
        execute_phase_qnode_vmap_grad(
            lambda _: float("nan"),
            np.array([[0.2, -0.4]], dtype=float),
        )


def test_phase_qnode_vector_transforms_reject_complex_derivative_inputs() -> None:
    real_params = np.array([0.2, -0.4], dtype=float)

    with pytest.raises(ValueError, match="real-valued.*complex"):
        execute_phase_qnode_vector_jacobian(
            "jacfwd",
            _vector_objective,
            np.array([0.2 + 0.1j, -0.4], dtype=np.complex128),
        )
    with pytest.raises(ValueError, match="real-valued.*complex"):
        execute_phase_qnode_vector_jvp(
            _vector_objective,
            real_params,
            np.array([0.5 + 0.1j, -1.0], dtype=np.complex128),
        )
    with pytest.raises(ValueError, match="real-valued.*complex"):
        execute_phase_qnode_vector_vjp(
            _vector_objective,
            real_params,
            np.array([1.0 + 0.5j, -0.25], dtype=np.complex128),
        )
    with pytest.raises(ValueError, match="real-valued.*complex"):
        execute_phase_qnode_vmap_grad(
            _scalar_objective,
            np.array([[0.2 + 0.1j, -0.4]], dtype=np.complex128),
        )
    with pytest.raises(ValueError, match="real-valued.*complex"):
        execute_phase_qnode_vector_jacobian(
            "jacfwd",
            lambda _: np.array([1.0 + 0.1j], dtype=np.complex128),
            real_params,
        )


def test_phase_qnode_vector_transform_readiness_suite_records_boundaries() -> None:
    suite = run_phase_qnode_vector_transform_readiness_suite()
    payload = suite.to_dict()

    assert suite.passed
    assert suite.record_count == 8
    assert suite.supported_count == 5
    assert suite.fail_closed_count == 3
    assert suite.total_parameter_shift_evaluations > 0
    assert not suite.hardware_execution
    assert {record.transform for record in suite.records if record.supported} == {
        "jacfwd",
        "jacrev",
        "jvp",
        "vjp",
        "vmap.grad",
    }
    assert payload["passed"] is True


def test_transform_nesting_supports_native_manual_vmap_grad_only() -> None:
    native = plan_gradient_transform_nesting(("vmap", "grad"), n_params=2)
    jax = plan_gradient_transform_nesting(("vmap", "grad"), adapter="jax", n_params=2)

    assert native.supported
    assert native.strategy == "native_manual_vmap_parameter_shift_grad"
    assert native.requires_deterministic_backend
    assert "manual batch loop" in native.warnings[0]
    assert "framework vmap" in native.claim_boundary

    assert jax.fail_closed
    assert (
        "vmap over quantum-gradient executions is supported only for native manual vmap(grad)"
        in jax.blocked_reasons
    )
