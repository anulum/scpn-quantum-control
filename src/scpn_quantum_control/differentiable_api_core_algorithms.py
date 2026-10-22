# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable API module
"""Construct numerical API envelopes from supplied AD providers and primitives."""

from __future__ import annotations

from collections.abc import Callable as Callable
from typing import Any as Any

from numpy.typing import ArrayLike as ArrayLike

from scpn_quantum_control.differentiable import ScalarObjective as ScalarObjective
from scpn_quantum_control.differentiable import VectorObjective as VectorObjective
from scpn_quantum_control.differentiable_api_contracts import (
    UnifiedDifferentiableAPIResult as UnifiedDifferentiableAPIResult,
)
from scpn_quantum_control.differentiable_api_core_bindings import DifferentiableValueBindings


def differentiable_value(
    objective: Callable[[Any], Any],
    values: ArrayLike,
    *,
    method: str = "parameter_shift",
    step: float | None = None,
    _bindings: DifferentiableValueBindings,
) -> UnifiedDifferentiableAPIResult:
    """Evaluate a scalar objective without exposing its computed gradient.

    Parameters
    ----------
    objective:
        Scalar objective accepted by the registered differentiation route.
    values:
        Parameter values supplied to the objective.
    method:
        Registered scalar differentiation method.
    step:
        Optional finite-difference step forwarded to the selected method.
    _bindings
        Typed live AD providers, result type and numerical/metadata primitives.

    Returns
    -------
    UnifiedDifferentiableAPIResult
        Supported ``value`` envelope with evaluation provenance.

    """
    result = _bindings.read_value_and_grad()(objective, values, method=method, step=step)
    return _bindings.read_UnifiedDifferentiableAPIResult()(
        operation="value",
        supported=True,
        method=result.method,
        value=_bindings.read_float()(result.value),
        gradient=None,
        jacobian=None,
        hessian=None,
        payload={
            "evaluations": result.evaluations,
            "parameter_names": _bindings.read_list()(result.parameter_names),
            "trainable": _bindings.read_list()(result.trainable),
        },
    )


def differentiable_gradient(
    objective: Callable[[Any], Any],
    values: ArrayLike,
    *,
    method: str = "parameter_shift",
    step: float | None = None,
    _bindings: DifferentiableValueBindings,
) -> UnifiedDifferentiableAPIResult:
    """Evaluate a scalar objective and its gradient.

    Parameters
    ----------
    objective:
        Scalar objective accepted by the registered differentiation route.
    values:
        Parameter values supplied to the objective.
    method:
        Registered scalar differentiation method.
    step:
        Optional finite-difference step forwarded to the selected method.
    _bindings
        Typed live AD providers, result type and numerical/metadata primitives.

    Returns
    -------
    UnifiedDifferentiableAPIResult
        Supported ``gradient`` envelope with a copied ``float64`` gradient.

    """
    result = _bindings.read_value_and_grad()(objective, values, method=method, step=step)
    return _bindings.read_UnifiedDifferentiableAPIResult()(
        operation="gradient",
        supported=True,
        method=result.method,
        value=_bindings.read_float()(result.value),
        gradient=result.gradient.astype(_bindings.read_float64(), copy=True),
        jacobian=None,
        hessian=None,
        payload={
            "evaluations": result.evaluations,
            "parameter_names": _bindings.read_list()(result.parameter_names),
            "trainable": _bindings.read_list()(result.trainable),
        },
    )


def differentiable_jacobian(
    objective: VectorObjective,
    values: ArrayLike,
    *,
    method: str = "finite_difference",
    step: float = 1e-06,
    _bindings: DifferentiableValueBindings,
) -> UnifiedDifferentiableAPIResult:
    """Evaluate a vector objective and its Jacobian.

    Parameters
    ----------
    objective:
        Vector objective accepted by the registered differentiation route.
    values:
        Parameter values supplied to the objective.
    method:
        Registered Jacobian method.
    step:
        Positive finite-difference step used by diagnostic routes.
    _bindings
        Typed live AD providers, result type and numerical/metadata primitives.

    Returns
    -------
    UnifiedDifferentiableAPIResult
        Supported ``jacobian`` envelope with objective and evaluation data.

    """
    result = _bindings.read_value_and_jacobian()(objective, values, method=method, step=step)
    return _bindings.read_UnifiedDifferentiableAPIResult()(
        operation="jacobian",
        supported=True,
        method=result.method,
        value=None,
        gradient=None,
        jacobian=result.jacobian.astype(_bindings.read_float64(), copy=True),
        hessian=None,
        payload={
            "objective_value": result.value.tolist(),
            "evaluations": result.evaluations,
            "parameter_names": _bindings.read_list()(result.parameter_names),
            "trainable": _bindings.read_list()(result.trainable),
        },
    )


def differentiable_hessian(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    method: str = "finite_difference",
    step: float = 0.0001,
    _bindings: DifferentiableValueBindings,
) -> UnifiedDifferentiableAPIResult:
    """Evaluate a scalar objective and its Hessian.

    Parameters
    ----------
    objective:
        Scalar objective accepted by the registered differentiation route.
    values:
        Parameter values supplied to the objective.
    method:
        Registered Hessian method.
    step:
        Positive finite-difference step used by diagnostic routes.
    _bindings
        Typed live AD providers, result type and numerical/metadata primitives.

    Returns
    -------
    UnifiedDifferentiableAPIResult
        Supported ``hessian`` envelope with evaluation provenance.

    """
    result = _bindings.read_value_and_hessian()(objective, values, method=method, step=step)
    return _bindings.read_UnifiedDifferentiableAPIResult()(
        operation="hessian",
        supported=True,
        method=result.method,
        value=_bindings.read_float()(result.value),
        gradient=None,
        jacobian=None,
        hessian=result.hessian.astype(_bindings.read_float64(), copy=True),
        payload={
            "evaluations": result.evaluations,
            "parameter_names": _bindings.read_list()(result.parameter_names),
            "trainable": _bindings.read_list()(result.trainable),
        },
    )


def _require_objective(
    objective: Callable[[Any], Any] | None, *, _bindings: DifferentiableValueBindings
) -> Callable[[Any], Any]:
    """Require an actual objective for the selected numerical operation."""
    if objective is None:
        raise _bindings.read_error()("objective is required for this differentiable operation")
    return objective


def _require_values(
    values: ArrayLike | None, *, _bindings: DifferentiableValueBindings
) -> ArrayLike:
    """Require the actual parameter values for the selected numerical operation."""
    if values is None:
        raise _bindings.read_error()("values are required for this differentiable operation")
    return values
