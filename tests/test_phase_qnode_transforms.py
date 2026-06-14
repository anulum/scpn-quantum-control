# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase QNode Transforms
"""Tests for executable scalar phase-QNode transform evidence."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.phase import (
    PhaseQNodeComplexDerivativeContract,
    PhaseQNodeTransformResult,
    execute_phase_qnode_hessian_vector_product,
    execute_phase_qnode_transform,
    phase_qnode_complex_derivative_contract,
    plan_gradient_transform_nesting,
    run_phase_qnode_transform_readiness_suite,
)


def _objective(params: np.ndarray) -> float:
    return float(np.cos(params[0]) + 0.25 * np.sin(params[1]))


def _gradient(params: np.ndarray) -> np.ndarray:
    return np.array([-np.sin(params[0]), 0.25 * np.cos(params[1])], dtype=float)


def _hessian(params: np.ndarray) -> np.ndarray:
    return np.array(
        [[-np.cos(params[0]), 0.0], [0.0, -0.25 * np.sin(params[1])]],
        dtype=float,
    )


def test_phase_qnode_transform_executes_grad_value_and_hessian() -> None:
    params = np.array([0.2, -0.4], dtype=float)

    grad = execute_phase_qnode_transform("grad", _objective, params)
    value_grad = execute_phase_qnode_transform("value_and_grad", _objective, params)
    hessian = execute_phase_qnode_transform("hessian", _objective, params)

    assert isinstance(grad, PhaseQNodeTransformResult)
    assert grad.supported
    assert grad.transform == "grad"
    assert grad.value is None
    assert grad.gradient is not None
    np.testing.assert_allclose(grad.gradient, _gradient(params), atol=1e-12)
    assert grad.hessian is None
    assert grad.parameter_shift_evaluations == 4
    assert "not hardware" in grad.claim_boundary

    assert value_grad.supported
    assert value_grad.value == pytest.approx(_objective(params))
    assert value_grad.gradient is not None
    np.testing.assert_allclose(value_grad.gradient, _gradient(params), atol=1e-12)

    assert hessian.supported
    assert hessian.hessian is not None
    np.testing.assert_allclose(hessian.hessian, _hessian(params), atol=1e-12)
    assert hessian.plan.requires_deterministic_backend
    assert hessian.parameter_shift_evaluations > grad.parameter_shift_evaluations


def test_phase_qnode_transform_executes_hessian_vector_product() -> None:
    params = np.array([0.31, -0.17], dtype=float)
    vector = np.array([0.5, -1.25], dtype=float)

    hvp = execute_phase_qnode_hessian_vector_product(_objective, params, vector)

    assert hvp.supported
    assert hvp.transform == "hessian_vector_product"
    assert hvp.hessian is not None
    assert hvp.hessian_vector_product is not None
    np.testing.assert_allclose(hvp.hessian, _hessian(params), atol=1e-12)
    np.testing.assert_allclose(hvp.hessian_vector_product, _hessian(params) @ vector, atol=1e-12)
    np.testing.assert_allclose(hvp.tangent, vector, atol=1e-12)
    assert hvp.plan.requires_deterministic_backend
    assert hvp.parameter_shift_evaluations > 0
    assert "Hessian-vector" in hvp.claim_boundary


def test_phase_qnode_transform_executes_jvp_vjp_and_scalar_jacobians() -> None:
    params = np.array([0.31, -0.17], dtype=float)
    tangent = np.array([0.5, -2.0], dtype=float)
    cotangent = np.array([3.0], dtype=float)
    expected_gradient = _gradient(params)

    jvp = execute_phase_qnode_transform("jvp", _objective, params, tangent=tangent)
    vjp = execute_phase_qnode_transform("vjp", _objective, params, cotangent=cotangent)
    jacfwd = execute_phase_qnode_transform("jacfwd", _objective, params)
    jacrev = execute_phase_qnode_transform("jacrev", _objective, params)

    assert jvp.supported
    assert jvp.jvp == pytest.approx(float(np.dot(expected_gradient, tangent)))
    assert jvp.gradient is not None
    np.testing.assert_allclose(jvp.gradient, expected_gradient, atol=1e-12)
    assert jvp.value == pytest.approx(_objective(params))

    assert vjp.supported
    assert vjp.vjp is not None
    np.testing.assert_allclose(vjp.vjp, cotangent[0] * expected_gradient, atol=1e-12)
    assert vjp.value == pytest.approx(_objective(params))

    assert jacfwd.supported
    assert jacrev.supported
    assert jacfwd.jacobian is not None
    assert jacrev.jacobian is not None
    np.testing.assert_allclose(jacfwd.jacobian, expected_gradient.reshape(1, -1), atol=1e-12)
    np.testing.assert_allclose(jacrev.jacobian, expected_gradient.reshape(1, -1), atol=1e-12)
    assert jacfwd.claim_boundary == jacrev.claim_boundary


def test_phase_qnode_transform_fails_closed_for_unsupported_routes() -> None:
    params = np.array([0.2, -0.4], dtype=float)

    hardware = execute_phase_qnode_transform(
        "grad",
        _objective,
        params,
        backend="hardware",
        shots=1024,
    )
    finite_hessian = execute_phase_qnode_transform(
        "hessian",
        _objective,
        params,
        backend="finite_shot_simulator",
        shots=256,
    )
    vectorized = execute_phase_qnode_transform(("vmap", "grad"), _objective, params)

    assert not hardware.supported
    assert hardware.fail_closed
    assert hardware.gradient is None
    assert "hardware gradient execution requires" in hardware.failure_reason

    assert not finite_hessian.supported
    assert finite_hessian.hessian is None
    assert "deterministic local expectations" in finite_hessian.failure_reason

    assert not vectorized.supported
    assert "use execute_phase_qnode_vmap_grad" in vectorized.failure_reason


def test_phase_qnode_transform_validates_directional_inputs() -> None:
    params = np.array([0.2, -0.4], dtype=float)

    with pytest.raises(ValueError, match="tangent"):
        execute_phase_qnode_transform("jvp", _objective, params)
    with pytest.raises(ValueError, match="cotangent"):
        execute_phase_qnode_transform("vjp", _objective, params)
    with pytest.raises(ValueError, match="tangent"):
        execute_phase_qnode_transform(
            "jvp",
            _objective,
            params,
            tangent=np.array([1.0], dtype=float),
        )
    with pytest.raises(ValueError, match="vector"):
        execute_phase_qnode_hessian_vector_product(
            _objective,
            params,
            np.array([1.0], dtype=float),
        )


def test_phase_qnode_complex_and_wirtinger_contract_is_explicit_fail_closed() -> None:
    contract = phase_qnode_complex_derivative_contract()

    assert isinstance(contract, PhaseQNodeComplexDerivativeContract)
    assert not contract.supported
    assert contract.parameter_domain == "real"
    assert contract.requested_derivative in {"complex", "wirtinger"}
    assert "Wirtinger" in contract.failure_reason
    assert "real-valued parameter vectors" in contract.claim_boundary
    assert contract.to_dict()["supported"] is False


def test_phase_qnode_transform_rejects_complex_derivative_inputs() -> None:
    params = np.array([0.2 + 0.1j, -0.4], dtype=np.complex128)
    real_params = np.array([0.2, -0.4], dtype=float)

    with pytest.raises(ValueError, match="real-valued.*complex"):
        execute_phase_qnode_transform("grad", _objective, params)
    with pytest.raises(ValueError, match="real-valued.*complex"):
        execute_phase_qnode_transform(
            "jvp",
            _objective,
            real_params,
            tangent=np.array([0.5 + 0.1j, -1.0], dtype=np.complex128),
        )
    with pytest.raises(ValueError, match="real-valued.*complex"):
        execute_phase_qnode_transform(
            "vjp",
            _objective,
            real_params,
            cotangent=np.array([1.0 + 0.5j], dtype=np.complex128),
        )
    with pytest.raises(ValueError, match="real-valued.*complex"):
        execute_phase_qnode_hessian_vector_product(
            _objective,
            real_params,
            np.array([0.5 + 0.1j, -1.0], dtype=np.complex128),
        )


def test_phase_qnode_hessian_vector_product_fails_closed_for_unsafe_routes() -> None:
    params = np.array([0.2, -0.4], dtype=float)
    vector = np.array([0.5, -1.25], dtype=float)

    finite_shot = execute_phase_qnode_hessian_vector_product(
        _objective,
        params,
        vector,
        backend="finite_shot_simulator",
        shots=256,
    )
    adapter = execute_phase_qnode_hessian_vector_product(
        _objective,
        params,
        vector,
        adapter="jax",
    )

    assert finite_shot.fail_closed
    assert finite_shot.hessian_vector_product is None
    assert "deterministic local expectations" in finite_shot.failure_reason
    assert adapter.fail_closed
    assert adapter.hessian_vector_product is None
    assert (
        "hessian transform is supported only on the native local route" in adapter.failure_reason
    )


def test_phase_qnode_transform_readiness_suite_records_supported_and_blocked() -> None:
    suite = run_phase_qnode_transform_readiness_suite()

    assert suite.passed
    assert suite.record_count >= 9
    assert suite.supported_count >= 6
    assert suite.fail_closed_count >= 3
    assert suite.total_parameter_shift_evaluations > 0
    assert not suite.hardware_execution
    assert {record.transform for record in suite.records if record.supported} >= {
        "grad",
        "value_and_grad",
        "hessian",
        "hessian_vector_product",
        "jvp",
        "vjp",
        "jacfwd",
        "jacrev",
    }
    assert suite.to_dict()["passed"] is True


def test_transform_nesting_now_supports_local_directional_and_scalar_jacobian_routes() -> None:
    jvp = plan_gradient_transform_nesting("jvp", n_params=2)
    vjp = plan_gradient_transform_nesting("vjp", n_params=2)
    jacfwd = plan_gradient_transform_nesting("jacfwd", n_params=2)
    jacrev = plan_gradient_transform_nesting("jacrev", n_params=2)

    assert jvp.supported
    assert jvp.strategy == "native_parameter_shift_jvp"
    assert vjp.supported
    assert vjp.strategy == "native_parameter_shift_vjp"
    assert jacfwd.supported
    assert jacfwd.strategy == "native_parameter_shift_scalar_jacobian"
    assert jacrev.supported
    assert jacrev.strategy == "native_parameter_shift_scalar_jacobian"
