# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- canonical differentiable transform dispatchers
"""Canonical differentiable transform dispatchers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .differentiable_exact_modes import (
    value_and_forward_mode_grad,
    value_and_reverse_mode_grad,
)
from .differentiable_finite_difference import (
    value_and_complex_step_grad,
    value_and_finite_difference_grad,
)
from .differentiable_parameter_contracts import Parameter, ParameterShiftRule
from .differentiable_parameter_shift import value_and_parameter_shift_grad
from .differentiable_result_contracts import GradientResult
from .differentiable_scalar_kernels import DualNumber, ReverseNode
from .whole_program_ad_result import WholeProgramADResult

ScalarObjective = Callable[[NDArray[np.float64]], float | int | np.floating[Any]]
ComplexStepObjective = Callable[[NDArray[np.complex128]], object]


def value_and_grad(
    objective: Callable[[Any], Any],
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    method: str = "parameter_shift",
    rule: ParameterShiftRule | None = None,
    step: float | None = None,
) -> GradientResult | WholeProgramADResult:
    """Evaluate a scalar objective and gradient through the canonical transform API.

    Parameters
    ----------
    objective:
        Objective compatible with the selected differentiation method.
    values:
        Initial parameter values.
    parameters:
        Optional metadata that marks trainable parameters and supplies names.
    method:
        Differentiation backend. Supported values are ``parameter_shift``,
        ``finite_difference``, ``complex_step``, ``forward_mode``,
        ``reverse_mode``, and ``whole_program``.
    rule:
        Optional parameter-shift rule for ``parameter_shift``.
    step:
        Optional finite-difference or complex-step perturbation.

    Returns
    -------
    GradientResult | WholeProgramADResult
        Objective value and gradient, including whole-program trace metadata
        when ``method`` is ``whole_program``.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported canonical backends.
    """

    if method == "parameter_shift":
        return value_and_parameter_shift_grad(
            cast(ScalarObjective, objective),
            values,
            parameters=parameters,
            rule=rule,
        )
    if method == "finite_difference":
        return value_and_finite_difference_grad(
            cast(ScalarObjective, objective),
            values,
            parameters=parameters,
            step=1.0e-6 if step is None else step,
        )
    if method == "complex_step":
        return value_and_complex_step_grad(
            cast(ComplexStepObjective, objective),
            values,
            parameters=parameters,
            step=1.0e-30 if step is None else step,
        )
    if method == "forward_mode":
        return value_and_forward_mode_grad(
            cast(Callable[[tuple[DualNumber, ...]], object], objective),
            values,
            parameters=parameters,
        )
    if method == "reverse_mode":
        return value_and_reverse_mode_grad(
            cast(Callable[[tuple[ReverseNode, ...]], object], objective),
            values,
            parameters=parameters,
        )
    if method == "whole_program":
        from .differentiable import whole_program_value_and_grad

        return whole_program_value_and_grad(
            objective,
            values,
            parameters=parameters,
            trace=True,
        )
    raise ValueError(
        "gradient method must be one of: parameter_shift, finite_difference, complex_step, "
        "forward_mode, reverse_mode, whole_program"
    )


def grad(
    objective: Callable[[Any], Any],
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    method: str = "parameter_shift",
    rule: ParameterShiftRule | None = None,
    step: float | None = None,
) -> NDArray[np.float64]:
    """Return a scalar-objective gradient through the canonical transform API.

    Parameters
    ----------
    objective:
        Objective compatible with the selected differentiation method.
    values:
        Initial parameter values.
    parameters:
        Optional metadata that marks trainable parameters and supplies names.
    method:
        Differentiation backend passed to :func:`value_and_grad`.
    rule:
        Optional parameter-shift rule for ``parameter_shift``.
    step:
        Optional finite-difference or complex-step perturbation.

    Returns
    -------
    numpy.ndarray
        Gradient vector as ``float64`` values.
    """

    result = value_and_grad(
        objective,
        values,
        parameters=parameters,
        method=method,
        rule=rule,
        step=step,
    )
    return result.gradient


__all__ = ["grad", "value_and_grad"]
