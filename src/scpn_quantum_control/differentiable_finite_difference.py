# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- finite-difference diagnostic transforms
"""Finite-difference and complex-step diagnostic differentiable transforms."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .differentiable_batch_helpers import _as_batch_parameter_array, _as_batch_vector_array
from .differentiable_parameter_contracts import Parameter, _as_parameter_array, _as_real_scalar
from .differentiable_result_contracts import (
    FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY,
    GradientResult,
    HessianResult,
    HVPResult,
    JacobianResult,
    JVPResult,
    VJPResult,
)
from .differentiable_transform_helpers import (
    _as_complex_step_scalar,
    _as_scalar,
    _as_vector_output,
    _normalise_parameters,
)

ScalarObjective = Callable[[NDArray[np.float64]], float | int | np.floating[Any]]
VectorObjective = Callable[[NDArray[np.float64]], ArrayLike]
ComplexStepObjective = Callable[[NDArray[np.complex128]], object]


def finite_difference_gradient(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Return a central finite-difference gradient for scalar diagnostics.

    Parameters
    ----------
    objective
        Scalar objective evaluated on a real parameter vector.
    values
        Real parameter vector for the diagnostic probe.
    parameters
        Optional parameter metadata. Non-trainable entries receive zero
        gradient components and are not perturbed.
    step
        Positive central-difference displacement.

    Returns
    -------
    numpy.ndarray
        Gradient vector with the same length as ``values``.

    Raises
    ------
    ValueError
        If parameters, objective values, or the finite-difference step violate
        the diagnostic contract.
    """
    result = value_and_finite_difference_grad(
        objective,
        values,
        parameters=parameters,
        step=step,
    )
    return result.gradient


def complex_step_gradient(
    objective: ComplexStepObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-30,
) -> NDArray[np.float64]:
    """Return a complex-step gradient for real-analytic scalar objectives.

    Parameters
    ----------
    objective
        Scalar objective that accepts complex-valued perturbations and returns a
        real base value.
    values
        Real parameter vector that seeds the complex-step probes.
    parameters
        Optional parameter metadata. Non-trainable entries receive zero
        gradient components and are not perturbed.
    step
        Positive imaginary displacement for each trainable component.

    Returns
    -------
    numpy.ndarray
        Complex-step gradient vector with the same length as ``values``.

    Raises
    ------
    ValueError
        If the step is invalid, the objective is not scalar, or the base value
        has a non-zero imaginary component.
    """
    result = value_and_complex_step_grad(
        objective,
        values,
        parameters=parameters,
        step=step,
    )
    return result.gradient


def batch_complex_step_gradient(
    objectives: Sequence[ComplexStepObjective],
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-30,
) -> NDArray[np.float64]:
    """Return stacked complex-step gradients for real-analytic objectives.

    Parameters
    ----------
    objectives
        Non-empty sequence of scalar objectives that support complex-step
        perturbations.
    values
        Real parameter vector shared by every objective.
    parameters
        Optional parameter metadata applied to each objective.
    step
        Positive imaginary displacement for trainable components.

    Returns
    -------
    numpy.ndarray
        Matrix whose rows are objective gradients.

    Raises
    ------
    ValueError
        If no objectives are provided or a delegated complex-step evaluation
        fails validation.
    """
    if not objectives:
        raise ValueError("objectives must contain at least one scalar objective")
    rows = [
        complex_step_gradient(
            objective,
            values,
            parameters=parameters,
            step=step,
        )
        for objective in objectives
    ]
    return np.vstack(rows)


def batch_value_and_complex_step_grad(
    objectives: Sequence[ComplexStepObjective],
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-30,
) -> tuple[GradientResult, ...]:
    """Return full complex-step results for multiple scalar objectives.

    Parameters
    ----------
    objectives
        Non-empty sequence of scalar objectives that support complex-step
        perturbations.
    values
        Real parameter vector shared by every objective.
    parameters
        Optional parameter metadata applied to each objective.
    step
        Positive imaginary displacement for trainable components.

    Returns
    -------
    tuple[GradientResult, ...]
        One value-and-gradient result per objective.

    Raises
    ------
    ValueError
        If no objectives are provided or a delegated complex-step evaluation
        fails validation.
    """
    if not objectives:
        raise ValueError("objectives must contain at least one scalar objective")
    return tuple(
        value_and_complex_step_grad(
            objective,
            values,
            parameters=parameters,
            step=step,
        )
        for objective in objectives
    )


def value_and_jacobian(
    objective: VectorObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    method: str = "finite_difference",
    step: float = 1.0e-6,
) -> JacobianResult:
    """Evaluate a vector objective and Jacobian through the canonical transform API.

    Parameters
    ----------
    objective
        Vector-valued objective evaluated on a real parameter vector.
    values
        Real parameter vector for the diagnostic probes.
    parameters
        Optional parameter metadata. Non-trainable columns are zeroed.
    method
        Jacobian backend selector. Only ``"finite_difference"`` is currently
        accepted.
    step
        Positive central-difference displacement.

    Returns
    -------
    JacobianResult
        Objective value, Jacobian matrix, metadata, and claim boundary.

    Raises
    ------
    ValueError
        If ``method`` is unsupported or the vector objective violates the
        finite-difference contract.
    """
    if method != "finite_difference":
        raise ValueError("Jacobian method must be finite_difference")
    return value_and_finite_difference_jacobian(
        objective,
        values,
        parameters=parameters,
        step=step,
    )


def jacobian(
    objective: VectorObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    method: str = "finite_difference",
    step: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Return a vector-objective Jacobian through the canonical transform API.

    Parameters
    ----------
    objective
        Vector-valued objective evaluated on a real parameter vector.
    values
        Real parameter vector for the diagnostic probes.
    parameters
        Optional parameter metadata. Non-trainable columns are zeroed.
    method
        Jacobian backend selector. Only ``"finite_difference"`` is currently
        accepted.
    step
        Positive central-difference displacement.

    Returns
    -------
    numpy.ndarray
        Dense Jacobian with shape ``(output_size, parameter_count)``.

    Raises
    ------
    ValueError
        If ``method`` is unsupported or the delegated Jacobian evaluation fails
        validation.
    """
    return value_and_jacobian(
        objective,
        values,
        parameters=parameters,
        method=method,
        step=step,
    ).jacobian


def value_and_jacfwd(
    objective: VectorObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    method: str = "finite_difference",
    step: float = 1.0e-6,
) -> JacobianResult:
    """Evaluate a vector objective and Jacobian through forward-Jacobian semantics.

    Parameters
    ----------
    objective
        Vector-valued objective evaluated on a real parameter vector.
    values
        Real parameter vector for the diagnostic probes.
    parameters
        Optional parameter metadata. Non-trainable columns are zeroed.
    method
        Jacobian backend selector. Only ``"finite_difference"`` is currently
        accepted.
    step
        Positive central-difference displacement.

    Returns
    -------
    JacobianResult
        Objective value, Jacobian matrix, metadata, and claim boundary.

    Raises
    ------
    ValueError
        If ``method`` is unsupported or the delegated Jacobian evaluation fails
        validation.

    Notes
    -----
    The current backend is the same central finite-difference Jacobian used by
    ``jacobian``. The separate name establishes transform algebra semantics for
    callers and tests while leaving room for a future true forward-mode Jacobian
    implementation behind the same contract.
    """
    return value_and_jacobian(
        objective,
        values,
        parameters=parameters,
        method=method,
        step=step,
    )


def jacfwd(
    objective: VectorObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    method: str = "finite_difference",
    step: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Return a vector-objective Jacobian using forward-Jacobian semantics.

    Parameters
    ----------
    objective
        Vector-valued objective evaluated on a real parameter vector.
    values
        Real parameter vector for the diagnostic probes.
    parameters
        Optional parameter metadata. Non-trainable columns are zeroed.
    method
        Jacobian backend selector. Only ``"finite_difference"`` is currently
        accepted.
    step
        Positive central-difference displacement.

    Returns
    -------
    numpy.ndarray
        Dense Jacobian with shape ``(output_size, parameter_count)``.

    Raises
    ------
    ValueError
        If ``method`` is unsupported or the delegated Jacobian evaluation fails
        validation.
    """
    return value_and_jacfwd(
        objective,
        values,
        parameters=parameters,
        method=method,
        step=step,
    ).jacobian


def value_and_jacrev(
    objective: VectorObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    method: str = "finite_difference",
    step: float = 1.0e-6,
) -> JacobianResult:
    """Evaluate a vector objective and Jacobian through reverse-Jacobian semantics.

    Parameters
    ----------
    objective
        Vector-valued objective evaluated on a real parameter vector.
    values
        Real parameter vector for the diagnostic probes.
    parameters
        Optional parameter metadata. Non-trainable columns are zeroed.
    method
        Jacobian backend selector. Only ``"finite_difference"`` is currently
        accepted.
    step
        Positive central-difference displacement.

    Returns
    -------
    JacobianResult
        Objective value, Jacobian matrix, metadata, and claim boundary.

    Raises
    ------
    ValueError
        If ``method`` is unsupported or the delegated Jacobian evaluation fails
        validation.

    Notes
    -----
    Until a true reverse-over-vector backend exists, this is an explicit alias to
    the finite-difference Jacobian contract. It preserves API and composition
    semantics without overclaiming reverse compiler AD.
    """
    return value_and_jacobian(
        objective,
        values,
        parameters=parameters,
        method=method,
        step=step,
    )


def jacrev(
    objective: VectorObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    method: str = "finite_difference",
    step: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Return a vector-objective Jacobian using reverse-Jacobian semantics.

    Parameters
    ----------
    objective
        Vector-valued objective evaluated on a real parameter vector.
    values
        Real parameter vector for the diagnostic probes.
    parameters
        Optional parameter metadata. Non-trainable columns are zeroed.
    method
        Jacobian backend selector. Only ``"finite_difference"`` is currently
        accepted.
    step
        Positive central-difference displacement.

    Returns
    -------
    numpy.ndarray
        Dense Jacobian with shape ``(output_size, parameter_count)``.

    Raises
    ------
    ValueError
        If ``method`` is unsupported or the delegated Jacobian evaluation fails
        validation.
    """
    return value_and_jacrev(
        objective,
        values,
        parameters=parameters,
        method=method,
        step=step,
    ).jacobian


def value_and_hessian(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    method: str = "finite_difference",
    step: float = 1.0e-4,
) -> HessianResult:
    """Evaluate a scalar objective and Hessian through the canonical transform API.

    Parameters
    ----------
    objective
        Scalar objective evaluated on a real parameter vector.
    values
        Real parameter vector for the diagnostic probes.
    parameters
        Optional parameter metadata. Non-trainable rows and columns are zeroed.
    method
        Hessian backend selector. Only ``"finite_difference"`` is currently
        accepted.
    step
        Positive central-difference displacement.

    Returns
    -------
    HessianResult
        Objective value, Hessian matrix, metadata, and claim boundary.

    Raises
    ------
    ValueError
        If ``method`` is unsupported or the delegated Hessian evaluation fails
        validation.
    """
    if method != "finite_difference":
        raise ValueError("Hessian method must be finite_difference")
    return value_and_finite_difference_hessian(
        objective,
        values,
        parameters=parameters,
        step=step,
    )


def hessian(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    method: str = "finite_difference",
    step: float = 1.0e-4,
) -> NDArray[np.float64]:
    """Return a scalar-objective Hessian through the canonical transform API.

    Parameters
    ----------
    objective
        Scalar objective evaluated on a real parameter vector.
    values
        Real parameter vector for the diagnostic probes.
    parameters
        Optional parameter metadata. Non-trainable rows and columns are zeroed.
    method
        Hessian backend selector. Only ``"finite_difference"`` is currently
        accepted.
    step
        Positive central-difference displacement.

    Returns
    -------
    numpy.ndarray
        Dense Hessian with shape ``(parameter_count, parameter_count)``.

    Raises
    ------
    ValueError
        If ``method`` is unsupported or the delegated Hessian evaluation fails
        validation.
    """
    return value_and_hessian(
        objective,
        values,
        parameters=parameters,
        method=method,
        step=step,
    ).hessian


def batch_value_and_finite_difference_grad(
    objectives: Sequence[ScalarObjective],
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> tuple[GradientResult, ...]:
    """Return full finite-difference results for multiple scalar objectives.

    Parameters
    ----------
    objectives
        Non-empty sequence of scalar objectives evaluated against the same
        parameter vector.
    values
        Real parameter vector for every objective.
    parameters
        Optional parameter metadata applied to each objective.
    step
        Positive central-difference displacement.

    Returns
    -------
    tuple[GradientResult, ...]
        One value-and-gradient result per objective.

    Raises
    ------
    ValueError
        If no objectives are provided or a delegated gradient evaluation fails
        validation.
    """
    if not objectives:
        raise ValueError("objectives must contain at least one scalar objective")
    return tuple(
        value_and_finite_difference_grad(
            objective,
            values,
            parameters=parameters,
            step=step,
        )
        for objective in objectives
    )


def value_and_complex_step_grad(
    objective: ComplexStepObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-30,
) -> GradientResult:
    """Evaluate a real-analytic scalar objective and complex-step gradient.

    Parameters
    ----------
    objective
        Scalar objective that accepts complex-valued perturbations and returns a
        real base value.
    values
        Real parameter vector that seeds the complex-step probes.
    parameters
        Optional parameter metadata. Non-trainable entries receive zero
        gradient components and are not perturbed.
    step
        Positive imaginary displacement for trainable components.

    Returns
    -------
    GradientResult
        Objective value, complex-step gradient, metadata, and claim boundary.

    Raises
    ------
    ValueError
        If the step is invalid, parameters are malformed, the objective is not
        scalar, or the base value has a non-zero imaginary component.
    """
    step_value = _as_real_scalar("complex-step step", step)
    if step_value <= 0.0:
        raise ValueError("complex-step step must be finite and positive")
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    gradient = np.zeros_like(parameter_values)
    base_values = cast(NDArray[np.complex128], parameter_values.astype(np.complex128))
    base_scalar = _as_complex_step_scalar(objective(base_values))
    if base_scalar.imag != 0.0:
        raise ValueError("complex-step objective returned a non-real base scalar")
    base_value = float(base_scalar.real)
    evaluations = 1

    for index, parameter in enumerate(parameter_meta):
        if not parameter.trainable:
            continue
        perturbed = cast(NDArray[np.complex128], parameter_values.astype(np.complex128))
        perturbed[index] += 1j * step_value
        perturbed_value = _as_complex_step_scalar(objective(perturbed))
        evaluations += 1
        gradient[index] = perturbed_value.imag / step_value

    return GradientResult(
        value=base_value,
        gradient=gradient,
        method="complex_step",
        shift=step_value,
        coefficient=1.0 / step_value,
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
    )


def value_and_finite_difference_grad(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> GradientResult:
    """Evaluate a scalar objective and central finite-difference gradient.

    Parameters
    ----------
    objective
        Scalar objective evaluated on a real parameter vector.
    values
        Real parameter vector for the central-difference probes.
    parameters
        Optional parameter metadata. Non-trainable entries receive zero
        gradient components and are not perturbed.
    step
        Positive central-difference displacement.

    Returns
    -------
    GradientResult
        Objective value, gradient, metadata, and diagnostic claim boundary.

    Raises
    ------
    ValueError
        If the step, parameters, or objective result violate the scalar
        diagnostic contract.
    """
    step_value = _as_real_scalar("finite difference step", step)
    if step_value <= 0.0:
        raise ValueError("finite difference step must be finite and positive")
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    gradient = np.zeros_like(parameter_values)
    base_value = _as_scalar(objective(parameter_values.copy()))
    evaluations = 1

    for index, parameter in enumerate(parameter_meta):
        if not parameter.trainable:
            continue
        plus = parameter_values.copy()
        minus = parameter_values.copy()
        plus[index] += step_value
        minus[index] -= step_value
        plus_value = _as_scalar(objective(plus))
        minus_value = _as_scalar(objective(minus))
        evaluations += 2
        gradient[index] = (plus_value - minus_value) / (2.0 * step_value)

    return GradientResult(
        value=base_value,
        gradient=gradient,
        method="finite_difference_central",
        shift=step_value,
        coefficient=1.0 / (2.0 * step_value),
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        claim_boundary=FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY,
    )


def finite_difference_jacobian(
    objective: VectorObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Return a central finite-difference Jacobian for vector objectives.

    Parameters
    ----------
    objective
        Vector-valued objective evaluated on a real parameter vector.
    values
        Real parameter vector for the central-difference probes.
    parameters
        Optional parameter metadata. Non-trainable columns are zeroed.
    step
        Positive central-difference displacement.

    Returns
    -------
    numpy.ndarray
        Dense Jacobian with shape ``(output_size, parameter_count)``.

    Raises
    ------
    ValueError
        If parameters, objective values, output shape, or the step violate the
        vector diagnostic contract.
    """
    return value_and_finite_difference_jacobian(
        objective,
        values,
        parameters=parameters,
        step=step,
    ).jacobian


def value_and_finite_difference_jacobian(
    objective: VectorObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> JacobianResult:
    """Evaluate a vector objective and its central finite-difference Jacobian.

    Parameters
    ----------
    objective
        Vector-valued objective evaluated on a real parameter vector.
    values
        Real parameter vector for the central-difference probes.
    parameters
        Optional parameter metadata. Non-trainable columns are zeroed.
    step
        Positive central-difference displacement.

    Returns
    -------
    JacobianResult
        Objective value, Jacobian matrix, metadata, and diagnostic claim
        boundary.

    Raises
    ------
    ValueError
        If the objective output is non-vector, non-finite, or shape-unstable, or
        if parameter and step validation fails.
    """
    step_value = _as_real_scalar("finite difference step", step)
    if step_value <= 0.0:
        raise ValueError("finite difference step must be finite and positive")
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    base_value = _as_vector_output(objective(parameter_values.copy()))
    jacobian = np.zeros((base_value.size, parameter_values.size), dtype=np.float64)
    evaluations = 1

    for index, parameter in enumerate(parameter_meta):
        if not parameter.trainable:
            continue
        plus = parameter_values.copy()
        minus = parameter_values.copy()
        plus[index] += step_value
        minus[index] -= step_value
        plus_value = _as_vector_output(objective(plus))
        minus_value = _as_vector_output(objective(minus))
        if plus_value.shape != base_value.shape or minus_value.shape != base_value.shape:
            raise ValueError("vector objective output shape must remain stable")
        evaluations += 2
        jacobian[:, index] = (plus_value - minus_value) / (2.0 * step_value)

    return JacobianResult(
        value=base_value,
        jacobian=jacobian,
        method="finite_difference_central",
        step=step_value,
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        claim_boundary=FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY,
    )


def finite_difference_jvp(
    objective: VectorObjective,
    values: ArrayLike,
    tangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Return a central finite-difference Jacobian-vector product.

    Parameters
    ----------
    objective
        Vector-valued objective evaluated on a real parameter vector.
    values
        Real parameter vector for the directional probes.
    tangent
        Direction vector. Non-trainable entries are masked to zero.
    parameters
        Optional parameter metadata applied to the direction mask.
    step
        Positive directional finite-difference displacement.

    Returns
    -------
    numpy.ndarray
        Directional output derivative with the same shape as the vector
        objective value.

    Raises
    ------
    ValueError
        If the tangent, objective output, parameters, or step violate the JVP
        contract.
    """
    return value_and_finite_difference_jvp(
        objective,
        values,
        tangent,
        parameters=parameters,
        step=step,
    ).jvp


def value_and_jvp(
    objective: VectorObjective,
    values: ArrayLike,
    tangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    method: str = "finite_difference",
    step: float = 1.0e-6,
) -> JVPResult:
    """Evaluate a vector objective and canonical Jacobian-vector product transform.

    Parameters
    ----------
    objective
        Vector-valued objective evaluated on a real parameter vector.
    values
        Real parameter vector for the directional probes.
    tangent
        Direction vector. Non-trainable entries are masked to zero.
    parameters
        Optional parameter metadata applied to the direction mask.
    method
        JVP backend selector. Only ``"finite_difference"`` is currently
        accepted.
    step
        Positive directional finite-difference displacement.

    Returns
    -------
    JVPResult
        Objective value, directional derivative, masked tangent, metadata, and
        claim boundary.

    Raises
    ------
    ValueError
        If ``method`` is unsupported or the delegated JVP evaluation fails
        validation.
    """
    if method != "finite_difference":
        raise ValueError("JVP method must be finite_difference")
    return value_and_finite_difference_jvp(
        objective,
        values,
        tangent,
        parameters=parameters,
        step=step,
    )


def jvp(
    objective: VectorObjective,
    values: ArrayLike,
    tangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    method: str = "finite_difference",
    step: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Return a canonical Jacobian-vector product transform.

    Parameters
    ----------
    objective
        Vector-valued objective evaluated on a real parameter vector.
    values
        Real parameter vector for the directional probes.
    tangent
        Direction vector. Non-trainable entries are masked to zero.
    parameters
        Optional parameter metadata applied to the direction mask.
    method
        JVP backend selector. Only ``"finite_difference"`` is currently
        accepted.
    step
        Positive directional finite-difference displacement.

    Returns
    -------
    numpy.ndarray
        Directional output derivative with the same shape as the vector
        objective value.

    Raises
    ------
    ValueError
        If ``method`` is unsupported or the delegated JVP evaluation fails
        validation.
    """
    return value_and_jvp(
        objective,
        values,
        tangent,
        parameters=parameters,
        method=method,
        step=step,
    ).jvp


def value_and_finite_difference_jvp(
    objective: VectorObjective,
    values: ArrayLike,
    tangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> JVPResult:
    """Evaluate a vector objective and a directional finite-difference JVP.

    Parameters
    ----------
    objective
        Vector-valued objective evaluated on a real parameter vector.
    values
        Real parameter vector for the directional probes.
    tangent
        Direction vector. Non-trainable entries are masked to zero.
    parameters
        Optional parameter metadata applied to the direction mask.
    step
        Positive directional finite-difference displacement.

    Returns
    -------
    JVPResult
        Objective value, directional derivative, masked tangent, metadata, and
        diagnostic claim boundary.

    Raises
    ------
    ValueError
        If tangent length, objective shape stability, parameter validation, or
        step validation fails.
    """
    step_value = _as_real_scalar("finite difference step", step)
    if step_value <= 0.0:
        raise ValueError("finite difference step must be finite and positive")
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    tangent_values = _as_parameter_array(tangent)
    if tangent_values.shape != parameter_values.shape:
        raise ValueError("JVP tangent length must match parameter length")
    trainable = np.array([parameter.trainable for parameter in parameter_meta], dtype=bool)
    masked_tangent = tangent_values.copy()
    masked_tangent[~trainable] = 0.0
    base_value = _as_vector_output(objective(parameter_values.copy()))
    if not np.any(masked_tangent):
        jvp = np.zeros_like(base_value)
        evaluations = 1
    else:
        plus = parameter_values + step_value * masked_tangent
        minus = parameter_values - step_value * masked_tangent
        plus_value = _as_vector_output(objective(plus))
        minus_value = _as_vector_output(objective(minus))
        if plus_value.shape != base_value.shape or minus_value.shape != base_value.shape:
            raise ValueError("vector objective output shape must remain stable")
        jvp = (plus_value - minus_value) / (2.0 * step_value)
        evaluations = 3
    return JVPResult(
        value=base_value,
        jvp=jvp,
        tangent=masked_tangent,
        method="finite_difference_directional",
        step=step_value,
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        claim_boundary=FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY,
    )


def batch_finite_difference_jvp(
    objective: VectorObjective,
    values: ArrayLike,
    tangents: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Return stacked finite-difference JVPs for a batch of tangents.

    Parameters
    ----------
    objective
        Vector-valued objective evaluated on a real parameter vector.
    values
        Real parameter vector for every directional probe.
    tangents
        Two-dimensional tangent matrix. Each row defines one direction.
    parameters
        Optional parameter metadata applied to each direction mask.
    step
        Positive directional finite-difference displacement.

    Returns
    -------
    numpy.ndarray
        Matrix whose rows are directional output derivatives.

    Raises
    ------
    ValueError
        If the tangent batch is malformed or a delegated JVP evaluation fails.
    """
    results = batch_value_and_finite_difference_jvp(
        objective,
        values,
        tangents,
        parameters=parameters,
        step=step,
    )
    return cast(NDArray[np.float64], np.vstack([result.jvp for result in results]))


def batch_value_and_finite_difference_jvp(
    objective: VectorObjective,
    values: ArrayLike,
    tangents: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> tuple[JVPResult, ...]:
    """Return one finite-difference JVP result per tangent row.

    Parameters
    ----------
    objective
        Vector-valued objective evaluated on a real parameter vector.
    values
        Real parameter vector for every directional probe.
    tangents
        Two-dimensional tangent matrix. Each row defines one direction.
    parameters
        Optional parameter metadata applied to each direction mask.
    step
        Positive directional finite-difference displacement.

    Returns
    -------
    tuple[JVPResult, ...]
        One value-and-JVP result per tangent row.

    Raises
    ------
    ValueError
        If the tangent batch is malformed or a delegated JVP evaluation fails.
    """
    parameter_values = _as_parameter_array(values)
    tangent_batch = _as_batch_parameter_array("JVP tangents", tangents, parameter_values.size)
    return tuple(
        value_and_finite_difference_jvp(
            objective,
            parameter_values,
            tangent,
            parameters=parameters,
            step=step,
        )
        for tangent in tangent_batch
    )


def vector_jacobian_product(
    jacobian: JacobianResult,
    cotangent: ArrayLike,
) -> VJPResult:
    """Contract a validated cotangent with a vector-objective Jacobian.

    Parameters
    ----------
    jacobian
        Previously validated value-and-Jacobian result.
    cotangent
        Vector cotangent whose length must match the objective output.

    Returns
    -------
    VJPResult
        Contracted vector-Jacobian product with inherited Jacobian metadata.

    Raises
    ------
    ValueError
        If ``jacobian`` is not a ``JacobianResult`` or the cotangent shape does
        not match the Jacobian value.
    """
    if not isinstance(jacobian, JacobianResult):
        raise ValueError("vector_jacobian_product requires a JacobianResult")
    cotangent_values = _as_vector_output(cotangent)
    if cotangent_values.shape != jacobian.value.shape:
        raise ValueError("VJP cotangent shape must match Jacobian value shape")
    vjp = jacobian.jacobian.T @ cotangent_values
    trainable = np.asarray(jacobian.trainable, dtype=bool)
    vjp[~trainable] = 0.0
    return VJPResult(
        value=jacobian.value,
        cotangent=cotangent_values,
        vjp=vjp,
        method=f"vjp:{jacobian.method}",
        step=jacobian.step,
        evaluations=jacobian.evaluations,
        parameter_names=jacobian.parameter_names,
        trainable=jacobian.trainable,
        claim_boundary=jacobian.claim_boundary,
    )


def finite_difference_vjp(
    objective: VectorObjective,
    values: ArrayLike,
    cotangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> VJPResult:
    """Return a finite-difference vector-Jacobian product for a vector objective.

    Parameters
    ----------
    objective
        Vector-valued objective evaluated on a real parameter vector.
    values
        Real parameter vector for the central-difference Jacobian probes.
    cotangent
        Vector cotangent contracted with the Jacobian.
    parameters
        Optional parameter metadata. Non-trainable columns are zeroed.
    step
        Positive central-difference displacement.

    Returns
    -------
    VJPResult
        Objective value, cotangent, contracted VJP, metadata, and claim
        boundary.

    Raises
    ------
    ValueError
        If Jacobian construction fails validation or the cotangent shape is
        incompatible.
    """
    jacobian = value_and_finite_difference_jacobian(
        objective,
        values,
        parameters=parameters,
        step=step,
    )
    return vector_jacobian_product(jacobian, cotangent)


def value_and_finite_difference_vjp(
    objective: VectorObjective,
    values: ArrayLike,
    cotangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> VJPResult:
    """Evaluate a vector objective and one finite-difference VJP result.

    Parameters
    ----------
    objective
        Vector-valued objective evaluated on a real parameter vector.
    values
        Real parameter vector for the central-difference Jacobian probes.
    cotangent
        Vector cotangent contracted with the Jacobian.
    parameters
        Optional parameter metadata. Non-trainable columns are zeroed.
    step
        Positive central-difference displacement.

    Returns
    -------
    VJPResult
        Objective value, cotangent, contracted VJP, metadata, and claim
        boundary.

    Raises
    ------
    ValueError
        If Jacobian construction fails validation or the cotangent shape is
        incompatible.
    """
    jacobian = value_and_finite_difference_jacobian(
        objective,
        values,
        parameters=parameters,
        step=step,
    )
    return vector_jacobian_product(jacobian, cotangent)


def value_and_vjp(
    objective: VectorObjective,
    values: ArrayLike,
    cotangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    method: str = "finite_difference",
    step: float = 1.0e-6,
) -> VJPResult:
    """Evaluate a vector objective and canonical vector-Jacobian product transform.

    Parameters
    ----------
    objective
        Vector-valued objective evaluated on a real parameter vector.
    values
        Real parameter vector for the central-difference Jacobian probes.
    cotangent
        Vector cotangent contracted with the Jacobian.
    parameters
        Optional parameter metadata. Non-trainable columns are zeroed.
    method
        VJP backend selector. Only ``"finite_difference"`` is currently
        accepted.
    step
        Positive central-difference displacement.

    Returns
    -------
    VJPResult
        Objective value, cotangent, contracted VJP, metadata, and claim
        boundary.

    Raises
    ------
    ValueError
        If ``method`` is unsupported or the delegated VJP evaluation fails
        validation.
    """
    if method != "finite_difference":
        raise ValueError("VJP method must be finite_difference")
    return value_and_finite_difference_vjp(
        objective,
        values,
        cotangent,
        parameters=parameters,
        step=step,
    )


def vjp(
    objective: VectorObjective,
    values: ArrayLike,
    cotangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    method: str = "finite_difference",
    step: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Return a canonical vector-Jacobian product transform.

    Parameters
    ----------
    objective
        Vector-valued objective evaluated on a real parameter vector.
    values
        Real parameter vector for the central-difference Jacobian probes.
    cotangent
        Vector cotangent contracted with the Jacobian.
    parameters
        Optional parameter metadata. Non-trainable columns are zeroed.
    method
        VJP backend selector. Only ``"finite_difference"`` is currently
        accepted.
    step
        Positive central-difference displacement.

    Returns
    -------
    numpy.ndarray
        Contracted parameter-space VJP vector.

    Raises
    ------
    ValueError
        If ``method`` is unsupported or the delegated VJP evaluation fails
        validation.
    """
    return value_and_vjp(
        objective,
        values,
        cotangent,
        parameters=parameters,
        method=method,
        step=step,
    ).vjp


def batch_vector_jacobian_product(
    jacobian: JacobianResult,
    cotangents: ArrayLike,
) -> tuple[VJPResult, ...]:
    """Return one vector-Jacobian product per cotangent row.

    Parameters
    ----------
    jacobian
        Previously validated value-and-Jacobian result.
    cotangents
        Two-dimensional cotangent matrix. Each row must match the objective
        output length.

    Returns
    -------
    tuple[VJPResult, ...]
        One contracted VJP result per cotangent row.

    Raises
    ------
    ValueError
        If ``jacobian`` is not a ``JacobianResult`` or the cotangent batch is
        malformed.
    """
    if not isinstance(jacobian, JacobianResult):
        raise ValueError("batch_vector_jacobian_product requires a JacobianResult")
    cotangent_batch = _as_batch_vector_array("VJP cotangents", cotangents, jacobian.value.size)
    return tuple(vector_jacobian_product(jacobian, cotangent) for cotangent in cotangent_batch)


def batch_finite_difference_vjp(
    objective: VectorObjective,
    values: ArrayLike,
    cotangents: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Return stacked finite-difference VJPs for a batch of cotangents.

    Parameters
    ----------
    objective
        Vector-valued objective evaluated on a real parameter vector.
    values
        Real parameter vector for the shared Jacobian probes.
    cotangents
        Two-dimensional cotangent matrix. Each row defines one VJP.
    parameters
        Optional parameter metadata. Non-trainable columns are zeroed.
    step
        Positive central-difference displacement.

    Returns
    -------
    numpy.ndarray
        Matrix whose rows are contracted parameter-space VJP vectors.

    Raises
    ------
    ValueError
        If Jacobian construction fails validation or the cotangent batch is
        malformed.
    """
    results = batch_value_and_finite_difference_vjp(
        objective,
        values,
        cotangents,
        parameters=parameters,
        step=step,
    )
    return cast(NDArray[np.float64], np.vstack([result.vjp for result in results]))


def batch_value_and_finite_difference_vjp(
    objective: VectorObjective,
    values: ArrayLike,
    cotangents: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-6,
) -> tuple[VJPResult, ...]:
    """Return one finite-difference VJP result per cotangent row.

    Parameters
    ----------
    objective
        Vector-valued objective evaluated on a real parameter vector.
    values
        Real parameter vector for the shared Jacobian probes.
    cotangents
        Two-dimensional cotangent matrix. Each row defines one VJP.
    parameters
        Optional parameter metadata. Non-trainable columns are zeroed.
    step
        Positive central-difference displacement.

    Returns
    -------
    tuple[VJPResult, ...]
        One contracted VJP result per cotangent row.

    Raises
    ------
    ValueError
        If Jacobian construction fails validation or the cotangent batch is
        malformed.
    """
    jacobian = value_and_finite_difference_jacobian(
        objective,
        values,
        parameters=parameters,
        step=step,
    )
    return batch_vector_jacobian_product(jacobian, cotangents)


def finite_difference_hessian(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-4,
) -> NDArray[np.float64]:
    """Return a central finite-difference Hessian for scalar objectives.

    Parameters
    ----------
    objective
        Scalar objective evaluated on a real parameter vector.
    values
        Real parameter vector for the second-order probes.
    parameters
        Optional parameter metadata. Non-trainable rows and columns are zeroed.
    step
        Positive central-difference displacement for curvature probes.

    Returns
    -------
    numpy.ndarray
        Dense Hessian with shape ``(parameter_count, parameter_count)``.

    Raises
    ------
    ValueError
        If parameter validation, scalar objective validation, or step validation
        fails.
    """
    return value_and_finite_difference_hessian(
        objective,
        values,
        parameters=parameters,
        step=step,
    ).hessian


def value_and_finite_difference_hessian(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-4,
) -> HessianResult:
    """Evaluate a scalar objective and central finite-difference Hessian.

    Parameters
    ----------
    objective
        Scalar objective evaluated on a real parameter vector.
    values
        Real parameter vector for the second-order probes.
    parameters
        Optional parameter metadata. Non-trainable rows and columns are zeroed.
    step
        Positive central-difference displacement for curvature probes.

    Returns
    -------
    HessianResult
        Objective value, Hessian matrix, metadata, and diagnostic claim
        boundary.

    Raises
    ------
    ValueError
        If parameter validation, scalar objective validation, or step validation
        fails.
    """
    step_value = _as_real_scalar("finite difference step", step)
    if step_value <= 0.0:
        raise ValueError("finite difference step must be finite and positive")
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    trainable = np.array([parameter.trainable for parameter in parameter_meta], dtype=bool)
    base_value = _as_scalar(objective(parameter_values.copy()))
    hessian = np.zeros((parameter_values.size, parameter_values.size), dtype=np.float64)
    evaluations = 1

    for row in range(parameter_values.size):
        if not trainable[row]:
            continue
        plus = parameter_values.copy()
        minus = parameter_values.copy()
        plus[row] += step_value
        minus[row] -= step_value
        plus_value = _as_scalar(objective(plus))
        minus_value = _as_scalar(objective(minus))
        evaluations += 2
        hessian[row, row] = (plus_value - 2.0 * base_value + minus_value) / (step_value**2)

        for column in range(row + 1, parameter_values.size):
            if not trainable[column]:
                continue
            plus_plus = parameter_values.copy()
            plus_minus = parameter_values.copy()
            minus_plus = parameter_values.copy()
            minus_minus = parameter_values.copy()
            plus_plus[row] += step_value
            plus_plus[column] += step_value
            plus_minus[row] += step_value
            plus_minus[column] -= step_value
            minus_plus[row] -= step_value
            minus_plus[column] += step_value
            minus_minus[row] -= step_value
            minus_minus[column] -= step_value
            mixed = (
                _as_scalar(objective(plus_plus))
                - _as_scalar(objective(plus_minus))
                - _as_scalar(objective(minus_plus))
                + _as_scalar(objective(minus_minus))
            ) / (4.0 * step_value**2)
            evaluations += 4
            hessian[row, column] = mixed
            hessian[column, row] = mixed

    return HessianResult(
        value=base_value,
        hessian=hessian,
        method="finite_difference_central",
        step=step_value,
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        claim_boundary=FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY,
    )


def finite_difference_hvp(
    objective: ScalarObjective,
    values: ArrayLike,
    tangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-5,
) -> NDArray[np.float64]:
    """Return a central finite-difference Hessian-vector product.

    Parameters
    ----------
    objective
        Scalar objective evaluated on a real parameter vector.
    values
        Real parameter vector for the directional curvature probes.
    tangent
        Direction vector. Non-trainable entries are masked to zero.
    parameters
        Optional parameter metadata applied to the direction mask.
    step
        Positive displacement for the nested finite-difference probes.

    Returns
    -------
    numpy.ndarray
        Parameter-space Hessian-vector product.

    Raises
    ------
    ValueError
        If the tangent, parameters, scalar objective result, or step violate the
        HVP contract.
    """
    return value_and_finite_difference_hvp(
        objective,
        values,
        tangent,
        parameters=parameters,
        step=step,
    ).hvp


def value_and_finite_difference_hvp(
    objective: ScalarObjective,
    values: ArrayLike,
    tangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-5,
) -> HVPResult:
    """Evaluate a scalar objective and a directional Hessian-vector product.

    Parameters
    ----------
    objective
        Scalar objective evaluated on a real parameter vector.
    values
        Real parameter vector for the directional curvature probes.
    tangent
        Direction vector. Non-trainable entries are masked to zero.
    parameters
        Optional parameter metadata applied to the direction mask.
    step
        Positive displacement for the nested finite-difference probes.

    Returns
    -------
    HVPResult
        Objective value, Hessian-vector product, masked tangent, metadata, and
        diagnostic claim boundary.

    Raises
    ------
    ValueError
        If tangent length, parameter validation, scalar objective validation, or
        step validation fails.
    """
    step_value = _as_real_scalar("finite difference step", step)
    if step_value <= 0.0:
        raise ValueError("finite difference step must be finite and positive")
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    tangent_values = _as_parameter_array(tangent)
    if tangent_values.shape != parameter_values.shape:
        raise ValueError("HVP tangent length must match parameter length")
    trainable = np.array([parameter.trainable for parameter in parameter_meta], dtype=bool)
    masked_tangent = tangent_values.copy()
    masked_tangent[~trainable] = 0.0
    base_value = _as_scalar(objective(parameter_values.copy()))
    if not np.any(masked_tangent):
        hvp = np.zeros_like(parameter_values)
        evaluations = 1
    else:
        plus = parameter_values + step_value * masked_tangent
        minus = parameter_values - step_value * masked_tangent
        plus_gradient = value_and_finite_difference_grad(
            objective,
            plus,
            parameters=parameter_meta,
            step=step_value,
        )
        minus_gradient = value_and_finite_difference_grad(
            objective,
            minus,
            parameters=parameter_meta,
            step=step_value,
        )
        hvp = (plus_gradient.gradient - minus_gradient.gradient) / (2.0 * step_value)
        hvp[~trainable] = 0.0
        evaluations = 1 + plus_gradient.evaluations + minus_gradient.evaluations
    return HVPResult(
        value=base_value,
        hvp=hvp,
        tangent=masked_tangent,
        method="finite_difference_hvp",
        step=step_value,
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        claim_boundary=FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY,
    )


def batch_finite_difference_hvp(
    objective: ScalarObjective,
    values: ArrayLike,
    tangents: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-5,
) -> NDArray[np.float64]:
    """Return stacked finite-difference HVPs for a batch of tangents.

    Parameters
    ----------
    objective
        Scalar objective evaluated on a real parameter vector.
    values
        Real parameter vector for every directional curvature probe.
    tangents
        Two-dimensional tangent matrix. Each row defines one HVP direction.
    parameters
        Optional parameter metadata applied to each direction mask.
    step
        Positive displacement for nested finite-difference probes.

    Returns
    -------
    numpy.ndarray
        Matrix whose rows are parameter-space Hessian-vector products.

    Raises
    ------
    ValueError
        If the tangent batch is malformed or a delegated HVP evaluation fails.
    """
    results = batch_value_and_finite_difference_hvp(
        objective,
        values,
        tangents,
        parameters=parameters,
        step=step,
    )
    return cast(NDArray[np.float64], np.vstack([result.hvp for result in results]))


def batch_value_and_finite_difference_hvp(
    objective: ScalarObjective,
    values: ArrayLike,
    tangents: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    step: float = 1.0e-5,
) -> tuple[HVPResult, ...]:
    """Return one finite-difference HVP result per tangent row.

    Parameters
    ----------
    objective
        Scalar objective evaluated on a real parameter vector.
    values
        Real parameter vector for every directional curvature probe.
    tangents
        Two-dimensional tangent matrix. Each row defines one HVP direction.
    parameters
        Optional parameter metadata applied to each direction mask.
    step
        Positive displacement for nested finite-difference probes.

    Returns
    -------
    tuple[HVPResult, ...]
        One value-and-HVP result per tangent row.

    Raises
    ------
    ValueError
        If the tangent batch is malformed or a delegated HVP evaluation fails.
    """
    parameter_values = _as_parameter_array(values)
    tangent_batch = _as_batch_parameter_array("HVP tangents", tangents, parameter_values.size)
    return tuple(
        value_and_finite_difference_hvp(
            objective,
            parameter_values,
            tangent,
            parameters=parameters,
            step=step,
        )
        for tangent in tangent_batch
    )
