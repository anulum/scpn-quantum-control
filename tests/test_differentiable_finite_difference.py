# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable finite difference tests
# scpn-quantum-control -- finite-difference differentiable transform tests
"""Tests for finite-difference and complex-step differentiable diagnostics."""

from __future__ import annotations

import inspect
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control as scpn
import scpn_quantum_control.differentiable_finite_difference as finite_difference_module
from scpn_quantum_control import differentiable as differentiable_module
from scpn_quantum_control.differentiable import (
    GradientResult,
    HessianResult,
    HVPResult,
    JacobianResult,
    JVPResult,
    Parameter,
    VJPResult,
    batch_complex_step_gradient,
    batch_finite_difference_hvp,
    batch_finite_difference_jvp,
    batch_finite_difference_vjp,
    batch_value_and_complex_step_grad,
    batch_value_and_finite_difference_grad,
    batch_value_and_finite_difference_hvp,
    batch_value_and_finite_difference_jvp,
    batch_value_and_finite_difference_vjp,
    batch_vector_jacobian_product,
    complex_step_gradient,
    finite_difference_gradient,
    finite_difference_hessian,
    finite_difference_hvp,
    finite_difference_jacobian,
    finite_difference_jvp,
    finite_difference_vjp,
    hessian,
    jacfwd,
    jacobian,
    jacrev,
    jvp,
    value_and_complex_step_grad,
    value_and_finite_difference_grad,
    value_and_finite_difference_hessian,
    value_and_finite_difference_hvp,
    value_and_finite_difference_jacobian,
    value_and_finite_difference_jvp,
    value_and_finite_difference_vjp,
    value_and_hessian,
    value_and_jacfwd,
    value_and_jacobian,
    value_and_jacrev,
    value_and_jvp,
    value_and_vjp,
    vector_jacobian_product,
    vjp,
)
from scpn_quantum_control.differentiable_finite_difference import (
    batch_complex_step_gradient as extracted_batch_complex_step_gradient,
)
from scpn_quantum_control.differentiable_finite_difference import (
    batch_finite_difference_hvp as extracted_batch_finite_difference_hvp,
)
from scpn_quantum_control.differentiable_finite_difference import (
    batch_finite_difference_jvp as extracted_batch_finite_difference_jvp,
)
from scpn_quantum_control.differentiable_finite_difference import (
    batch_finite_difference_vjp as extracted_batch_finite_difference_vjp,
)
from scpn_quantum_control.differentiable_finite_difference import (
    batch_value_and_complex_step_grad as extracted_batch_value_and_complex_step_grad,
)
from scpn_quantum_control.differentiable_finite_difference import (
    batch_value_and_finite_difference_grad as extracted_batch_value_and_finite_difference_grad,
)
from scpn_quantum_control.differentiable_finite_difference import (
    batch_value_and_finite_difference_hvp as extracted_batch_value_and_finite_difference_hvp,
)
from scpn_quantum_control.differentiable_finite_difference import (
    batch_value_and_finite_difference_jvp as extracted_batch_value_and_finite_difference_jvp,
)
from scpn_quantum_control.differentiable_finite_difference import (
    batch_value_and_finite_difference_vjp as extracted_batch_value_and_finite_difference_vjp,
)
from scpn_quantum_control.differentiable_finite_difference import (
    batch_vector_jacobian_product as extracted_batch_vector_jacobian_product,
)
from scpn_quantum_control.differentiable_finite_difference import (
    complex_step_gradient as extracted_complex_step_gradient,
)
from scpn_quantum_control.differentiable_finite_difference import (
    finite_difference_gradient as extracted_finite_difference_gradient,
)
from scpn_quantum_control.differentiable_finite_difference import (
    finite_difference_hessian as extracted_finite_difference_hessian,
)
from scpn_quantum_control.differentiable_finite_difference import (
    finite_difference_hvp as extracted_finite_difference_hvp,
)
from scpn_quantum_control.differentiable_finite_difference import (
    finite_difference_jacobian as extracted_finite_difference_jacobian,
)
from scpn_quantum_control.differentiable_finite_difference import (
    finite_difference_jvp as extracted_finite_difference_jvp,
)
from scpn_quantum_control.differentiable_finite_difference import (
    finite_difference_vjp as extracted_finite_difference_vjp,
)
from scpn_quantum_control.differentiable_finite_difference import (
    hessian as extracted_hessian,
)
from scpn_quantum_control.differentiable_finite_difference import (
    jacfwd as extracted_jacfwd,
)
from scpn_quantum_control.differentiable_finite_difference import (
    jacobian as extracted_jacobian,
)
from scpn_quantum_control.differentiable_finite_difference import (
    jacrev as extracted_jacrev,
)
from scpn_quantum_control.differentiable_finite_difference import jvp as extracted_jvp
from scpn_quantum_control.differentiable_finite_difference import (
    value_and_complex_step_grad as extracted_value_and_complex_step_grad,
)
from scpn_quantum_control.differentiable_finite_difference import (
    value_and_finite_difference_grad as extracted_value_and_finite_difference_grad,
)
from scpn_quantum_control.differentiable_finite_difference import (
    value_and_finite_difference_hessian as extracted_value_and_finite_difference_hessian,
)
from scpn_quantum_control.differentiable_finite_difference import (
    value_and_finite_difference_hvp as extracted_value_and_finite_difference_hvp,
)
from scpn_quantum_control.differentiable_finite_difference import (
    value_and_finite_difference_jacobian as extracted_value_and_finite_difference_jacobian,
)
from scpn_quantum_control.differentiable_finite_difference import (
    value_and_finite_difference_jvp as extracted_value_and_finite_difference_jvp,
)
from scpn_quantum_control.differentiable_finite_difference import (
    value_and_finite_difference_vjp as extracted_value_and_finite_difference_vjp,
)
from scpn_quantum_control.differentiable_finite_difference import (
    value_and_hessian as extracted_value_and_hessian,
)
from scpn_quantum_control.differentiable_finite_difference import (
    value_and_jacfwd as extracted_value_and_jacfwd,
)
from scpn_quantum_control.differentiable_finite_difference import (
    value_and_jacobian as extracted_value_and_jacobian,
)
from scpn_quantum_control.differentiable_finite_difference import (
    value_and_jacrev as extracted_value_and_jacrev,
)
from scpn_quantum_control.differentiable_finite_difference import (
    value_and_jvp as extracted_value_and_jvp,
)
from scpn_quantum_control.differentiable_finite_difference import (
    value_and_vjp as extracted_value_and_vjp,
)
from scpn_quantum_control.differentiable_finite_difference import (
    vector_jacobian_product as extracted_vector_jacobian_product,
)
from scpn_quantum_control.differentiable_finite_difference import vjp as extracted_vjp

FloatArray = NDArray[np.float64]

PUBLIC_FINITE_DIFFERENCE_TRANSFORMS = (
    "finite_difference_gradient",
    "complex_step_gradient",
    "batch_complex_step_gradient",
    "batch_value_and_complex_step_grad",
    "value_and_jacobian",
    "jacobian",
    "value_and_jacfwd",
    "jacfwd",
    "value_and_jacrev",
    "jacrev",
    "value_and_hessian",
    "hessian",
    "batch_value_and_finite_difference_grad",
    "value_and_complex_step_grad",
    "value_and_finite_difference_grad",
    "finite_difference_jacobian",
    "value_and_finite_difference_jacobian",
    "finite_difference_jvp",
    "value_and_jvp",
    "jvp",
    "value_and_finite_difference_jvp",
    "batch_finite_difference_jvp",
    "batch_value_and_finite_difference_jvp",
    "vector_jacobian_product",
    "finite_difference_vjp",
    "value_and_finite_difference_vjp",
    "value_and_vjp",
    "vjp",
    "batch_vector_jacobian_product",
    "batch_finite_difference_vjp",
    "batch_value_and_finite_difference_vjp",
    "finite_difference_hessian",
    "value_and_finite_difference_hessian",
    "finite_difference_hvp",
    "value_and_finite_difference_hvp",
    "batch_finite_difference_hvp",
    "batch_value_and_finite_difference_hvp",
)


def _assert_allclose(
    actual: object, expected: object, *, rtol: float = 1.0e-7, atol: float = 0.0
) -> None:
    """Assert NumPy closeness across differentiable diagnostic payloads."""
    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def test_public_finite_difference_transforms_document_contracts() -> None:
    """Public finite-difference transforms must expose NumPy API contracts."""
    for name in PUBLIC_FINITE_DIFFERENCE_TRANSFORMS:
        transform = getattr(finite_difference_module, name)
        doc = inspect.getdoc(transform)
        assert doc is not None
        assert "Parameters" in doc, name
        assert "Returns" in doc, name


def test_facade_and_package_root_reuse_extracted_diagnostic_helpers() -> None:
    """Facade and package-root exports should point at extracted diagnostics."""
    expected = {
        "batch_complex_step_gradient": extracted_batch_complex_step_gradient,
        "batch_finite_difference_hvp": extracted_batch_finite_difference_hvp,
        "batch_finite_difference_jvp": extracted_batch_finite_difference_jvp,
        "batch_finite_difference_vjp": extracted_batch_finite_difference_vjp,
        "batch_value_and_complex_step_grad": extracted_batch_value_and_complex_step_grad,
        "batch_value_and_finite_difference_grad": extracted_batch_value_and_finite_difference_grad,
        "batch_value_and_finite_difference_hvp": extracted_batch_value_and_finite_difference_hvp,
        "batch_value_and_finite_difference_jvp": extracted_batch_value_and_finite_difference_jvp,
        "batch_value_and_finite_difference_vjp": extracted_batch_value_and_finite_difference_vjp,
        "batch_vector_jacobian_product": extracted_batch_vector_jacobian_product,
        "complex_step_gradient": extracted_complex_step_gradient,
        "finite_difference_gradient": extracted_finite_difference_gradient,
        "finite_difference_hessian": extracted_finite_difference_hessian,
        "finite_difference_hvp": extracted_finite_difference_hvp,
        "finite_difference_jacobian": extracted_finite_difference_jacobian,
        "finite_difference_jvp": extracted_finite_difference_jvp,
        "finite_difference_vjp": extracted_finite_difference_vjp,
        "hessian": extracted_hessian,
        "jacfwd": extracted_jacfwd,
        "jacobian": extracted_jacobian,
        "jacrev": extracted_jacrev,
        "jvp": extracted_jvp,
        "value_and_complex_step_grad": extracted_value_and_complex_step_grad,
        "value_and_finite_difference_grad": extracted_value_and_finite_difference_grad,
        "value_and_finite_difference_hessian": extracted_value_and_finite_difference_hessian,
        "value_and_finite_difference_hvp": extracted_value_and_finite_difference_hvp,
        "value_and_finite_difference_jacobian": extracted_value_and_finite_difference_jacobian,
        "value_and_finite_difference_jvp": extracted_value_and_finite_difference_jvp,
        "value_and_finite_difference_vjp": extracted_value_and_finite_difference_vjp,
        "value_and_hessian": extracted_value_and_hessian,
        "value_and_jacfwd": extracted_value_and_jacfwd,
        "value_and_jacobian": extracted_value_and_jacobian,
        "value_and_jacrev": extracted_value_and_jacrev,
        "value_and_jvp": extracted_value_and_jvp,
        "value_and_vjp": extracted_value_and_vjp,
        "vector_jacobian_product": extracted_vector_jacobian_product,
        "vjp": extracted_vjp,
    }
    for name, helper in expected.items():
        assert getattr(differentiable_module, name) is helper
        assert getattr(scpn, name) is helper


def test_finite_difference_gradient_matches_quadratic_derivative() -> None:
    """Finite-difference backend should support non-parameter-shift diagnostics."""
    result = value_and_finite_difference_grad(
        lambda values: values[0] ** 2 + 3.0 * values[1],
        [2.0, -1.0],
        parameters=[Parameter("x"), Parameter("bias", trainable=False)],
        step=1.0e-6,
    )

    assert result.method == "finite_difference_central"
    assert (
        result.claim_boundary == differentiable_module.FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY
    )
    assert "diagnostic only" in result.claim_boundary
    assert "not analytic" in result.claim_boundary
    assert "whole-program AD" in result.claim_boundary
    assert "production benchmark evidence" in result.claim_boundary
    assert result.evaluations == 3
    _assert_allclose(result.gradient, [4.0, 0.0], rtol=1.0e-6, atol=1.0e-6)
    _assert_allclose(
        finite_difference_gradient(lambda values: values[0] ** 2, [1.5]),
        [3.0],
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_finite_difference_gradient_rejects_invalid_step() -> None:
    """Finite-difference step size must be explicit finite positive real data."""
    invalid_step = cast(Any, "1e-6")
    with pytest.raises(ValueError, match="finite difference step must be a real numeric scalar"):
        finite_difference_gradient(lambda values: values[0] ** 2, [1.0], step=invalid_step)
    with pytest.raises(ValueError, match="finite difference step must be finite and positive"):
        finite_difference_gradient(lambda values: values[0] ** 2, [1.0], step=0.0)


def test_batch_finite_difference_gradient_helpers_cover_values_and_empty_inputs() -> None:
    """Batched finite-difference gradient helpers should return one result per objective."""
    objectives = [
        lambda values: float(values[0] ** 2),
        lambda values: float(3.0 * values[0]),
    ]

    results = batch_value_and_finite_difference_grad(objectives, [2.0])

    assert len(results) == 2
    _assert_allclose([result.value for result in results], [4.0, 6.0])
    _assert_allclose([result.gradient for result in results], [[4.0], [3.0]], atol=1.0e-6)
    with pytest.raises(ValueError, match="at least one scalar objective"):
        batch_value_and_finite_difference_grad([], [2.0])


def test_derivative_result_claim_boundary_must_be_explicit() -> None:
    """Derivative result artefacts should reject blank claim boundaries."""
    with pytest.raises(ValueError, match="claim_boundary must be non-empty"):
        GradientResult(
            value=1.0,
            gradient=np.array([1.0]),
            method="finite_difference_central",
            shift=1.0e-6,
            coefficient=5.0e5,
            evaluations=3,
            parameter_names=("x",),
            trainable=(True,),
            claim_boundary=" ",
        )


def test_complex_step_gradient_matches_analytic_derivative() -> None:
    """Complex-step gradients should avoid finite-difference cancellation."""
    result = value_and_complex_step_grad(
        lambda values: np.sin(values[0]) + values[1] ** 3,
        [0.4, -0.2],
        parameters=[Parameter("x"), Parameter("y")],
    )

    assert isinstance(result, GradientResult)
    assert result.method == "complex_step"
    assert result.evaluations == 3
    assert result.parameter_names == ("x", "y")
    _assert_allclose(
        result.gradient,
        [np.cos(0.4), 3.0 * (-0.2) ** 2],
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    _assert_allclose(
        complex_step_gradient(lambda values: values[0] ** 2, [3.0]),
        [6.0],
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_complex_step_gradient_respects_frozen_parameters() -> None:
    """Complex-step gradients must preserve trainable masks exactly."""
    result = value_and_complex_step_grad(
        lambda values: values[0] ** 2 + np.exp(values[1]),
        [2.0, 0.5],
        parameters=[Parameter("active"), Parameter("frozen", trainable=False)],
    )

    assert result.evaluations == 2
    assert result.trainable == (True, False)
    _assert_allclose(result.gradient, [4.0, 0.0], rtol=1.0e-14, atol=1.0e-14)


def test_batch_complex_step_helpers_cover_values_and_empty_inputs() -> None:
    """Batched complex-step helpers should stack gradients and full results."""
    objectives = [
        lambda values: values[0] ** 2,
        lambda values: np.exp(values[0]),
    ]

    results = batch_value_and_complex_step_grad(objectives, [0.5])
    stacked = batch_complex_step_gradient(objectives, [0.5])

    assert len(results) == 2
    _assert_allclose([result.value for result in results], [0.25, np.exp(0.5)])
    _assert_allclose(stacked, [[1.0], [np.exp(0.5)]], rtol=1.0e-14, atol=1.0e-14)
    with pytest.raises(ValueError, match="at least one scalar objective"):
        batch_complex_step_gradient([], [0.5])
    with pytest.raises(ValueError, match="at least one scalar objective"):
        batch_value_and_complex_step_grad([], [0.5])


def test_complex_step_gradient_rejects_invalid_inputs() -> None:
    """Complex-step gradients should fail closed on invalid scalar contracts."""
    invalid_step = cast(Any, "1e-30")
    with pytest.raises(ValueError, match="complex-step step must be a real numeric scalar"):
        complex_step_gradient(lambda values: values[0] ** 2, [1.0], step=invalid_step)
    with pytest.raises(ValueError, match="complex-step step must be finite and positive"):
        complex_step_gradient(lambda values: values[0] ** 2, [1.0], step=0.0)
    with pytest.raises(ValueError, match="complex-step objective must return a scalar"):
        complex_step_gradient(lambda values: np.array([values[0], values[0]]), [1.0])
    with pytest.raises(ValueError, match="complex-step objective must return a scalar"):
        complex_step_gradient(lambda _values: "not numeric", [1.0])
    with pytest.raises(ValueError, match="complex-step objective returned a non-finite scalar"):
        complex_step_gradient(lambda values: values[0] * complex(np.nan, 1.0), [1.0])
    with pytest.raises(ValueError, match="non-real base scalar"):
        complex_step_gradient(lambda values: values[0] + 1j, [1.0])


def test_finite_difference_jacobian_matches_vector_objective() -> None:
    """Vector-valued differentiable diagnostics should expose Jacobians."""
    result = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0] ** 2, values[0] + 2.0 * values[1]]),
        [3.0, -1.0],
        parameters=[Parameter("x"), Parameter("frozen", trainable=False)],
        step=1.0e-6,
    )

    assert isinstance(result, JacobianResult)
    assert result.method == "finite_difference_central"
    assert (
        result.claim_boundary == differentiable_module.FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY
    )
    assert result.evaluations == 3
    _assert_allclose(result.value, [9.0, 1.0])
    _assert_allclose(result.jacobian, [[6.0, 0.0], [1.0, 0.0]], atol=1.0e-6)
    _assert_allclose(
        finite_difference_jacobian(lambda values: np.array([values[0] ** 2]), [2.0]),
        [[4.0]],
        atol=1.0e-6,
    )


def test_finite_difference_jacobian_rejects_unstable_vector_shape() -> None:
    """Vector objectives must keep output shape stable across perturbations."""

    def unstable(values: FloatArray) -> FloatArray:
        if values[0] > 0.0:
            return np.array([values[0], values[0] ** 2], dtype=np.float64)
        return np.array([values[0]], dtype=np.float64)

    with pytest.raises(ValueError, match="shape must remain stable"):
        value_and_finite_difference_jacobian(unstable, [0.0])


def test_finite_difference_jacobian_rejects_non_vector_output() -> None:
    """Jacobian objectives must return explicit finite one-dimensional arrays."""
    with pytest.raises(ValueError, match="one-dimensional"):
        value_and_finite_difference_jacobian(lambda _values: np.array([[1.0]]), [0.0])
    with pytest.raises(ValueError, match="real numeric"):
        value_and_finite_difference_jacobian(lambda _values: np.array(["1.0"]), [0.0])
    with pytest.raises(ValueError, match="finite difference step"):
        value_and_finite_difference_jacobian(lambda values: np.array([values[0]]), [1.0], step=0.0)


def test_canonical_jacobian_and_hessian_wrappers_dispatch_and_reject_methods() -> None:
    """Canonical finite-difference wrappers should preserve method boundaries."""

    def vector_objective(values: FloatArray) -> FloatArray:
        return np.array([values[0] ** 2], dtype=np.float64)

    def scalar_objective(values: FloatArray) -> float:
        return float(values[0] ** 2)

    jacobian_result = value_and_jacobian(vector_objective, [2.0])
    jacfwd_result = value_and_jacfwd(vector_objective, [2.0])
    jacrev_result = value_and_jacrev(vector_objective, [2.0])
    hessian_result = value_and_hessian(scalar_objective, [2.0])

    _assert_allclose(jacobian_result.jacobian, [[4.0]], atol=1.0e-6)
    _assert_allclose(jacfwd_result.jacobian, [[4.0]], atol=1.0e-6)
    _assert_allclose(jacrev_result.jacobian, [[4.0]], atol=1.0e-6)
    _assert_allclose(jacobian(vector_objective, [2.0]), [[4.0]], atol=1.0e-6)
    _assert_allclose(jacfwd(vector_objective, [2.0]), [[4.0]], atol=1.0e-6)
    _assert_allclose(jacrev(vector_objective, [2.0]), [[4.0]], atol=1.0e-6)
    _assert_allclose(hessian_result.hessian, [[2.0]], atol=1.0e-3)
    _assert_allclose(hessian(scalar_objective, [2.0]), [[2.0]], atol=1.0e-3)

    with pytest.raises(ValueError, match="Jacobian method"):
        value_and_jacobian(vector_objective, [2.0], method="reverse")
    with pytest.raises(ValueError, match="Hessian method"):
        value_and_hessian(scalar_objective, [2.0], method="reverse")


def test_finite_difference_jvp_matches_jacobian_directional_product() -> None:
    """Directional finite differences should expose native forward-mode JVPs."""

    def objective(values: FloatArray) -> FloatArray:
        return np.array([values[0] ** 2 + values[1], values[0] * values[1]], dtype=np.float64)

    result = value_and_finite_difference_jvp(
        objective,
        [2.0, 3.0],
        [0.5, -1.0],
        parameters=[Parameter("x"), Parameter("y")],
    )

    assert isinstance(result, JVPResult)
    assert result.method == "finite_difference_directional"
    assert (
        result.claim_boundary == differentiable_module.FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY
    )
    assert result.evaluations == 3
    _assert_allclose(result.value, [7.0, 6.0])
    _assert_allclose(result.tangent, [0.5, -1.0])
    _assert_allclose(result.jvp, [1.0, -0.5], atol=1.0e-6)
    _assert_allclose(
        finite_difference_jvp(objective, [2.0, 3.0], [0.5, -1.0]),
        [1.0, -0.5],
        atol=1.0e-6,
    )


def test_finite_difference_jvp_respects_frozen_parameters() -> None:
    """Frozen parameters must be removed from directional tangents."""
    result = value_and_finite_difference_jvp(
        lambda values: np.array([values[0] + 10.0 * values[1]]),
        [1.0, 2.0],
        [0.25, 100.0],
        parameters=[Parameter("x"), Parameter("frozen", trainable=False)],
    )

    assert result.evaluations == 3
    _assert_allclose(result.tangent, [0.25, 0.0])
    _assert_allclose(result.jvp, [0.25], atol=1.0e-6)


def test_finite_difference_jvp_rejects_invalid_inputs() -> None:
    """JVP tangents and vector outputs must be finite and shape-stable."""
    invalid_tangent = cast(Any, ["1.0"])
    with pytest.raises(ValueError, match="JVP tangent length"):
        value_and_finite_difference_jvp(lambda values: np.array([values[0]]), [1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="real numeric"):
        value_and_finite_difference_jvp(
            lambda values: np.array([values[0]]), [1.0], invalid_tangent
        )

    def unstable(values: FloatArray) -> FloatArray:
        if values[0] > 0.0:
            return np.array([values[0], values[0] ** 2], dtype=np.float64)
        return np.array([values[0]], dtype=np.float64)

    with pytest.raises(ValueError, match="shape must remain stable"):
        value_and_finite_difference_jvp(unstable, [0.0], [1.0])


def test_canonical_jvp_wrappers_and_zero_direction_branch() -> None:
    """JVP wrappers should expose finite-difference dispatch and zero tangents."""

    def objective(values: FloatArray) -> FloatArray:
        return np.array([values[0] ** 2], dtype=np.float64)

    result = value_and_jvp(objective, [2.0], [0.0])

    assert result.evaluations == 1
    _assert_allclose(result.jvp, [0.0])
    _assert_allclose(jvp(objective, [2.0], [1.0]), [4.0], atol=1.0e-6)
    with pytest.raises(ValueError, match="JVP method"):
        value_and_jvp(objective, [2.0], [1.0], method="reverse")
    with pytest.raises(ValueError, match="finite difference step"):
        value_and_finite_difference_jvp(objective, [2.0], [1.0], step=0.0)


def test_vector_jacobian_product_contracts_cotangent() -> None:
    """Reverse-mode VJP should contract cotangents against validated Jacobians."""
    jacobian_result = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0] ** 2, values[0] + 2.0 * values[1]]),
        [3.0, -1.0],
        parameters=[Parameter("x"), Parameter("y")],
    )
    result = vector_jacobian_product(jacobian_result, [0.5, -2.0])

    assert isinstance(result, VJPResult)
    assert result.method == "vjp:finite_difference_central"
    assert (
        result.claim_boundary == differentiable_module.FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY
    )
    _assert_allclose(result.value, [9.0, 1.0])
    _assert_allclose(result.cotangent, [0.5, -2.0])
    _assert_allclose(result.vjp, [1.0, -4.0], atol=1.0e-6)
    _assert_allclose(
        finite_difference_vjp(
            lambda values: np.array([values[0] ** 2, values[0] + 2.0 * values[1]]),
            [3.0, -1.0],
            [0.5, -2.0],
        ).vjp,
        [1.0, -4.0],
        atol=1.0e-6,
    )


def test_value_and_canonical_vjp_wrappers_dispatch_and_reject_methods() -> None:
    """VJP wrappers should route through finite-difference Jacobian products."""

    def objective(values: FloatArray) -> FloatArray:
        return np.array([values[0] ** 2], dtype=np.float64)

    result = value_and_finite_difference_vjp(objective, [3.0], [0.5])

    _assert_allclose(result.vjp, [3.0], atol=1.0e-6)
    _assert_allclose(value_and_vjp(objective, [3.0], [0.5]).vjp, [3.0], atol=1.0e-6)
    _assert_allclose(vjp(objective, [3.0], [0.5]), [3.0], atol=1.0e-6)
    with pytest.raises(ValueError, match="VJP method"):
        value_and_vjp(objective, [3.0], [0.5], method="forward")


def test_vector_jacobian_product_respects_frozen_parameters_and_validation() -> None:
    """VJP products should zero frozen columns and reject malformed cotangents."""
    jacobian_result = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0] + values[1], values[1] ** 2]),
        [1.0, 2.0],
        parameters=[Parameter("x"), Parameter("frozen", trainable=False)],
    )
    result = vector_jacobian_product(jacobian_result, [1.0, 10.0])

    _assert_allclose(result.vjp, [1.0, 0.0], atol=1.0e-6)
    with pytest.raises(ValueError, match="JacobianResult"):
        vector_jacobian_product(cast(Any, np.eye(2)), [1.0, 2.0])
    with pytest.raises(ValueError, match="cotangent shape"):
        vector_jacobian_product(jacobian_result, [1.0])
    with pytest.raises(ValueError, match="JacobianResult"):
        batch_vector_jacobian_product(cast(Any, np.eye(2)), [[1.0, 2.0]])


def test_batch_finite_difference_jvp_returns_stacked_products_and_results() -> None:
    """Batched JVP helpers should preserve one result per tangent row."""

    def objective(values: FloatArray) -> FloatArray:
        return np.array([values[0] ** 2 + values[1], values[0] * values[1]], dtype=np.float64)

    tangents = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    results = batch_value_and_finite_difference_jvp(objective, [2.0, 3.0], tangents)
    stacked = batch_finite_difference_jvp(objective, [2.0, 3.0], tangents)

    assert len(results) == 2
    assert all(isinstance(result, JVPResult) for result in results)
    _assert_allclose(stacked, [[4.0, 3.0], [1.0, 2.0]], atol=1.0e-6)
    _assert_allclose(np.vstack([result.jvp for result in results]), stacked)


def test_batch_finite_difference_vjp_reuses_single_jacobian() -> None:
    """Batched VJP helpers should contract multiple cotangent rows."""

    def objective(values: FloatArray) -> FloatArray:
        return np.array([values[0] ** 2, values[0] + 2.0 * values[1]], dtype=np.float64)

    cotangents = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    results = batch_value_and_finite_difference_vjp(objective, [3.0, -1.0], cotangents)
    stacked = batch_finite_difference_vjp(objective, [3.0, -1.0], cotangents)

    assert len(results) == 2
    assert all(isinstance(result, VJPResult) for result in results)
    _assert_allclose(stacked, [[6.0, 0.0], [1.0, 2.0]], atol=1.0e-6)


def test_batch_vector_jacobian_product_contracts_existing_jacobian() -> None:
    """Batched VJP contraction should work from an existing validated Jacobian."""
    jacobian_result = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0] ** 2, values[0] + 2.0 * values[1]]),
        [3.0, -1.0],
    )
    results = batch_vector_jacobian_product(
        jacobian_result,
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
    )

    assert len(results) == 2
    _assert_allclose([result.vjp for result in results], [[6.0, 0.0], [1.0, 2.0]], atol=1.0e-6)


def test_batch_finite_difference_hvp_returns_stacked_products_and_results() -> None:
    """Batched HVP helpers should preserve one result per tangent row."""

    def objective(values: FloatArray) -> float:
        return float(values[0] ** 2 + 3.0 * values[0] * values[1] + 2.0 * values[1] ** 2)

    tangents = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    results = batch_value_and_finite_difference_hvp(objective, [1.0, -1.0], tangents)
    stacked = batch_finite_difference_hvp(objective, [1.0, -1.0], tangents)

    assert len(results) == 2
    assert all(isinstance(result, HVPResult) for result in results)
    _assert_allclose(stacked, [[2.0, 3.0], [3.0, 4.0]], atol=1.0e-4)


def test_batch_transform_helpers_reject_malformed_batches() -> None:
    """Batched transform helpers should require explicit two-dimensional batches."""
    with pytest.raises(ValueError, match="two-dimensional batch"):
        batch_finite_difference_jvp(lambda values: np.array([values[0]]), [1.0], [1.0])
    with pytest.raises(ValueError, match="row length"):
        batch_finite_difference_jvp(lambda values: np.array([values[0]]), [1.0], [[1.0, 2.0]])
    jacobian_result = value_and_finite_difference_jacobian(
        lambda values: np.array([values[0], values[1]]),
        [1.0, 2.0],
    )
    with pytest.raises(ValueError, match="two-dimensional batch"):
        batch_vector_jacobian_product(jacobian_result, [1.0, 2.0])
    with pytest.raises(ValueError, match="row length"):
        batch_vector_jacobian_product(jacobian_result, [[1.0]])
    with pytest.raises(ValueError, match="two-dimensional batch"):
        batch_finite_difference_hvp(lambda values: float(values[0] ** 2), [1.0], [1.0])


def test_finite_difference_hessian_matches_quadratic_curvature() -> None:
    """Scalar differentiable diagnostics should expose second-order curvature."""
    result = value_and_finite_difference_hessian(
        lambda values: values[0] ** 2 + 3.0 * values[0] * values[1] + 2.0 * values[1] ** 2,
        [1.0, -1.0],
        step=1.0e-4,
    )

    assert isinstance(result, HessianResult)
    assert result.method == "finite_difference_central"
    assert (
        result.claim_boundary == differentiable_module.FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY
    )
    _assert_allclose(result.hessian, [[2.0, 3.0], [3.0, 4.0]], atol=1.0e-5)
    _assert_allclose(
        finite_difference_hessian(lambda values: values[0] ** 2, [1.0]),
        [[2.0]],
        atol=1.0e-5,
    )


def test_finite_difference_hessian_respects_frozen_parameters() -> None:
    """Frozen parameters should have zero Hessian rows and columns."""
    result = value_and_finite_difference_hessian(
        lambda values: values[0] ** 2 + values[1] ** 2,
        [1.0, 2.0],
        parameters=[Parameter("x"), Parameter("frozen", trainable=False)],
    )

    _assert_allclose(result.hessian[:, 1], [0.0, 0.0])
    _assert_allclose(result.hessian[1, :], [0.0, 0.0])


def test_finite_difference_hessian_rejects_invalid_step() -> None:
    """Hessian step size must be explicit finite positive real data."""
    invalid_step = cast(Any, "1e-4")
    with pytest.raises(ValueError, match="finite difference step must be a real numeric scalar"):
        finite_difference_hessian(lambda values: values[0] ** 2, [1.0], step=invalid_step)
    with pytest.raises(ValueError, match="finite difference step must be finite and positive"):
        finite_difference_hessian(lambda values: values[0] ** 2, [1.0], step=0.0)


def test_finite_difference_hvp_matches_quadratic_curvature_product() -> None:
    """Hessian-vector products should match full Hessian multiplication."""

    def objective(values: FloatArray) -> float:
        return float(values[0] ** 2 + 3.0 * values[0] * values[1] + 2.0 * values[1] ** 2)

    result = value_and_finite_difference_hvp(objective, [1.0, -1.0], [0.5, -2.0])

    assert isinstance(result, HVPResult)
    assert result.method == "finite_difference_hvp"
    assert (
        result.claim_boundary == differentiable_module.FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY
    )
    assert result.value == pytest.approx(0.0)
    _assert_allclose(result.tangent, [0.5, -2.0])
    _assert_allclose(result.hvp, [-5.0, -6.5], atol=1.0e-4)
    _assert_allclose(
        finite_difference_hvp(objective, [1.0, -1.0], [0.5, -2.0]),
        [-5.0, -6.5],
        atol=1.0e-4,
    )


def test_finite_difference_hvp_respects_frozen_parameters() -> None:
    """Frozen parameters should not contribute to HVP tangents or products."""
    result = value_and_finite_difference_hvp(
        lambda values: float(values[0] ** 2 + 100.0 * values[1] ** 2),
        [1.0, 2.0],
        [3.0, 999.0],
        parameters=[Parameter("x"), Parameter("frozen", trainable=False)],
    )

    _assert_allclose(result.tangent, [3.0, 0.0])
    _assert_allclose(result.hvp, [6.0, 0.0], atol=1.0e-4)


def test_finite_difference_hvp_zero_direction_branch() -> None:
    """HVP wrappers should avoid gradient probes for zero trainable tangents."""
    result = value_and_finite_difference_hvp(
        lambda values: float(values[0] ** 2),
        [2.0],
        [0.0],
    )

    assert result.evaluations == 1
    _assert_allclose(result.hvp, [0.0])


def test_finite_difference_hvp_rejects_invalid_inputs() -> None:
    """HVP tangents and controls must be finite, real, and shape-consistent."""
    invalid_tangent = cast(Any, ["1.0"])
    with pytest.raises(ValueError, match="HVP tangent length"):
        value_and_finite_difference_hvp(lambda values: float(values[0] ** 2), [1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="real numeric"):
        value_and_finite_difference_hvp(
            lambda values: float(values[0] ** 2),
            [1.0],
            invalid_tangent,
        )
    with pytest.raises(ValueError, match="finite difference step"):
        value_and_finite_difference_hvp(
            lambda values: float(values[0] ** 2), [1.0], [1.0], step=0.0
        )
