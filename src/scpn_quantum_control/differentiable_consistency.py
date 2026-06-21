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
    _as_real_scalar,
)
from .differentiable_result_contracts import GradientCheckResult

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


__all__ = [
    "check_parameter_shift_consistency",
]
