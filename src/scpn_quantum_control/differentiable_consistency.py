# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- differentiable consistency diagnostics
"""Consistency diagnostics for native differentiable transforms."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .differentiable_parameter_contracts import (
    Parameter,
    ParameterShiftRule,
    _as_parameter_array,
    _as_real_scalar,
)
from .differentiable_result_contracts import (
    CustomDerivativeCheckResult,
    GradientCheckResult,
)
from .program_ad_registry import CustomDerivativeRule

ScalarObjective = Callable[[NDArray[np.float64]], float | int | np.floating[Any]]


def check_parameter_shift_consistency(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
    finite_difference_step: float = 1.0e-6,
    tolerance: float = 1.0e-5,
) -> GradientCheckResult:
    """Compare parameter-shift gradients against central finite differences.

    Parameters
    ----------
    objective:
        Scalar differentiable objective evaluated by both gradient estimators.
    values:
        Real parameter vector supplied to the objective.
    parameters:
        Optional parameter metadata and trainable mask.
    rule:
        Optional parameter-shift rule. Defaults to the standard two-point
        generator rule used by the facade.
    finite_difference_step:
        Positive central-difference probe spacing used for the reference
        gradient.
    tolerance:
        Non-negative maximum absolute gradient error allowed for a passing
        diagnostic.

    Returns
    -------
    GradientCheckResult
        Candidate parameter-shift gradient, finite-difference reference, error
        metrics, tolerance, and pass/fail status.
    """

    from . import differentiable as differentiable_facade

    tolerance_value = _as_real_scalar("gradient check tolerance", tolerance)
    if tolerance_value < 0.0:
        raise ValueError("gradient check tolerance must be finite and non-negative")
    candidate = differentiable_facade.value_and_parameter_shift_grad(
        objective,
        values,
        parameters=parameters,
        rule=rule,
    )
    reference = differentiable_facade.value_and_finite_difference_grad(
        objective,
        values,
        parameters=parameters,
        step=finite_difference_step,
    )
    delta = candidate.gradient - reference.gradient
    max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    l2_error = float(np.linalg.norm(delta, ord=2))
    value_delta = float(abs(candidate.value - reference.value))
    return GradientCheckResult(
        reference=reference,
        candidate=candidate,
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        value_delta=value_delta,
        tolerance=tolerance_value,
        passed=max_abs_error <= tolerance_value,
    )


def check_custom_derivative_consistency(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    tangent: ArrayLike,
    cotangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    finite_difference_step: float = 1.0e-6,
    tolerance: float = 1.0e-5,
) -> CustomDerivativeCheckResult:
    """Check custom derivative rules against adjoint and finite-difference identities.

    Parameters
    ----------
    rule:
        Custom derivative rule supplying exact JVP and VJP callbacks.
    values:
        Real parameter vector supplied to the rule.
    tangent:
        Tangent vector used for the JVP identity check.
    cotangent:
        Cotangent vector used for the VJP identity check.
    parameters:
        Optional parameter metadata and trainable mask.
    finite_difference_step:
        Positive probe spacing for finite-difference JVP/VJP references.
    tolerance:
        Non-negative maximum allowed error for adjoint, JVP, and VJP checks.

    Returns
    -------
    CustomDerivativeCheckResult
        Exact-rule outputs, finite-difference references, error metrics, and
        pass/fail status for the custom derivative rule.
    """

    from . import differentiable as differentiable_facade

    tolerance_value = _as_real_scalar("custom derivative tolerance", tolerance)
    if tolerance_value < 0.0:
        raise ValueError("custom derivative tolerance must be finite and non-negative")
    step_value = _as_real_scalar(
        "custom derivative finite_difference_step", finite_difference_step
    )
    if step_value <= 0.0:
        raise ValueError("custom derivative finite_difference_step must be finite and positive")
    custom_jvp_result = differentiable_facade.value_and_custom_jvp(
        rule,
        values,
        tangent,
        parameters=parameters,
    )
    custom_vjp_result = differentiable_facade.value_and_custom_vjp(
        rule,
        values,
        cotangent,
        parameters=parameters,
    )
    parameter_values = _as_parameter_array(values)
    reference_parameters = tuple(
        Parameter(name, trainable=flag)
        for name, flag in zip(
            custom_jvp_result.parameter_names,
            custom_jvp_result.trainable,
            strict=True,
        )
    )
    reference_jvp = differentiable_facade.value_and_finite_difference_jvp(
        rule.value_fn,
        parameter_values,
        custom_jvp_result.tangent,
        parameters=reference_parameters,
        step=step_value,
    )
    reference_vjp = differentiable_facade.finite_difference_vjp(
        rule.value_fn,
        parameter_values,
        custom_vjp_result.cotangent,
        parameters=reference_parameters,
        step=step_value,
    )
    primal_inner = float(np.dot(custom_jvp_result.jvp, custom_vjp_result.cotangent))
    adjoint_inner = float(np.dot(custom_jvp_result.tangent, custom_vjp_result.vjp))
    adjoint_inner_error = abs(primal_inner - adjoint_inner)
    jvp_l2_error = float(np.linalg.norm(custom_jvp_result.jvp - reference_jvp.jvp))
    vjp_l2_error = float(np.linalg.norm(custom_vjp_result.vjp - reference_vjp.vjp))
    passed = (
        adjoint_inner_error <= tolerance_value
        and jvp_l2_error <= tolerance_value
        and vjp_l2_error <= tolerance_value
    )
    return CustomDerivativeCheckResult(
        custom_jvp=custom_jvp_result,
        custom_vjp=custom_vjp_result,
        reference_jvp=reference_jvp,
        reference_vjp=reference_vjp,
        adjoint_inner_error=adjoint_inner_error,
        jvp_l2_error=jvp_l2_error,
        vjp_l2_error=vjp_l2_error,
        tolerance=tolerance_value,
        passed=passed,
    )


__all__ = [
    "check_custom_derivative_consistency",
    "check_parameter_shift_consistency",
]
