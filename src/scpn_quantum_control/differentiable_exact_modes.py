# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- exact scalar AD mode wrappers
"""Exact scalar forward- and reverse-mode gradient wrappers."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .differentiable_parameter_contracts import Parameter, _as_parameter_array
from .differentiable_result_contracts import GradientResult
from .differentiable_scalar_kernels import DualNumber, ReverseNode
from .differentiable_transform_helpers import (
    _as_forward_mode_scalar,
    _as_reverse_mode_scalar,
    _normalise_parameters,
    _reverse_topological_order,
)


def value_and_forward_mode_grad(
    objective: Callable[[tuple[DualNumber, ...]], object],
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> GradientResult:
    """Evaluate a scalar objective and exact forward-mode dual gradient.

    Parameters
    ----------
    objective
        Scalar objective expressed in terms of ``DualNumber`` inputs.
    values
        Real parameter vector at which the objective is evaluated.
    parameters
        Optional parameter metadata. Frozen parameters are evaluated in the
        base objective but receive no tangent seed and therefore report a zero
        gradient entry.

    Returns
    -------
    GradientResult
        Objective value, exact forward-mode gradient, parameter metadata, and
        evaluation count.
    """
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    base_duals = tuple(DualNumber(float(value), 0.0) for value in parameter_values)
    base_value = _as_forward_mode_scalar(objective(base_duals)).primal
    gradient = np.zeros_like(parameter_values)
    evaluations = 1

    for index, parameter in enumerate(parameter_meta):
        if not parameter.trainable:
            continue
        dual_values = tuple(
            DualNumber(float(value), 1.0 if basis_index == index else 0.0)
            for basis_index, value in enumerate(parameter_values)
        )
        gradient[index] = _as_forward_mode_scalar(objective(dual_values)).tangent
        evaluations += 1

    return GradientResult(
        value=base_value,
        gradient=gradient,
        method="forward_mode_dual",
        shift=None,
        coefficient=None,
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
    )


def forward_mode_gradient(
    objective: Callable[[tuple[DualNumber, ...]], object],
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> NDArray[np.float64]:
    """Return an exact forward-mode dual gradient for scalar objectives.

    Parameters
    ----------
    objective
        Scalar objective expressed in terms of ``DualNumber`` inputs.
    values
        Real parameter vector at which the objective is evaluated.
    parameters
        Optional parameter metadata used to mask frozen tangent lanes.

    Returns
    -------
    numpy.ndarray
        One exact gradient entry per input parameter.
    """
    return value_and_forward_mode_grad(
        objective,
        values,
        parameters=parameters,
    ).gradient


def value_and_reverse_mode_grad(
    objective: Callable[[tuple[ReverseNode, ...]], object],
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> GradientResult:
    """Evaluate a scalar objective and exact reverse-mode tape gradient.

    Parameters
    ----------
    objective
        Scalar objective expressed in terms of ``ReverseNode`` inputs.
    values
        Real parameter vector at which the objective is evaluated.
    parameters
        Optional parameter metadata. Frozen parameters participate in the tape
        but are masked to zero in the returned gradient.

    Returns
    -------
    GradientResult
        Objective value, exact reverse-mode gradient, parameter metadata, and
        evaluation count.
    """
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    reverse_values = tuple(ReverseNode(float(value)) for value in parameter_values)
    output = _as_reverse_mode_scalar(objective(reverse_values))
    tape = _reverse_topological_order(output)
    for node in tape:
        node.adjoint = 0.0
    output.adjoint = 1.0
    for node in reversed(tape):
        for parent, local_derivative in node.parents:
            parent.adjoint += node.adjoint * local_derivative
    gradient = np.array(
        [
            node.adjoint if parameter.trainable else 0.0
            for node, parameter in zip(reverse_values, parameter_meta)
        ],
        dtype=np.float64,
    )
    return GradientResult(
        value=output.primal,
        gradient=gradient,
        method="reverse_mode_tape",
        shift=None,
        coefficient=None,
        evaluations=1,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
    )


def reverse_mode_gradient(
    objective: Callable[[tuple[ReverseNode, ...]], object],
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> NDArray[np.float64]:
    """Return an exact reverse-mode tape gradient for scalar objectives.

    Parameters
    ----------
    objective
        Scalar objective expressed in terms of ``ReverseNode`` inputs.
    values
        Real parameter vector at which the objective is evaluated.
    parameters
        Optional parameter metadata used to mask frozen gradient entries.

    Returns
    -------
    numpy.ndarray
        One exact gradient entry per input parameter.
    """
    return value_and_reverse_mode_grad(
        objective,
        values,
        parameters=parameters,
    ).gradient
