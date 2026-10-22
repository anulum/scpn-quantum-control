# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable API module
"""Public numerical value and derivative routes with live AD providers."""

from __future__ import annotations

from collections.abc import Callable as Callable
from typing import Any as Any

import numpy as np
from numpy.typing import ArrayLike as ArrayLike

import scpn_quantum_control.differentiable_api_core_algorithms as _algorithms
from scpn_quantum_control.differentiable import ScalarObjective as ScalarObjective
from scpn_quantum_control.differentiable import VectorObjective as VectorObjective
from scpn_quantum_control.differentiable import value_and_grad as value_and_grad
from scpn_quantum_control.differentiable import value_and_hessian as value_and_hessian
from scpn_quantum_control.differentiable import value_and_jacobian as value_and_jacobian
from scpn_quantum_control.differentiable_api_contracts import (
    UnifiedDifferentiableAPIResult as UnifiedDifferentiableAPIResult,
)
from scpn_quantum_control.differentiable_api_core_bindings import (
    DifferentiableValueBindings as _BindingReaders,
)

_readers = _BindingReaders(
    read_UnifiedDifferentiableAPIResult=lambda: UnifiedDifferentiableAPIResult,
    read_value_and_grad=lambda: value_and_grad,
    read_value_and_hessian=lambda: value_and_hessian,
    read_value_and_jacobian=lambda: value_and_jacobian,
    read_float64=lambda: np.float64,
    read_float=lambda: float,
    read_list=lambda: list,
    read_error=lambda: ValueError,
)


def differentiable_value(
    objective: Callable[[Any], Any],
    values: ArrayLike,
    *,
    method: str = "parameter_shift",
    step: float | None = None,
    _bindings: _BindingReaders = _readers,
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
        Typed numerical providers and AD result/primitives, defaulting to this
        module's live AD suppliers. Legacy composition retains its own globals.

    Returns
    -------
    UnifiedDifferentiableAPIResult
        Supported ``value`` envelope with evaluation provenance.

    """
    return _algorithms.differentiable_value(
        objective, values, method=method, step=step, _bindings=_bindings
    )


def differentiable_gradient(
    objective: Callable[[Any], Any],
    values: ArrayLike,
    *,
    method: str = "parameter_shift",
    step: float | None = None,
    _bindings: _BindingReaders = _readers,
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
        Typed numerical providers and AD result/primitives, defaulting to this
        module's live AD suppliers. Legacy composition retains its own globals.

    Returns
    -------
    UnifiedDifferentiableAPIResult
        Supported ``gradient`` envelope with a copied ``float64`` gradient.

    """
    return _algorithms.differentiable_gradient(
        objective, values, method=method, step=step, _bindings=_bindings
    )


def differentiable_jacobian(
    objective: VectorObjective,
    values: ArrayLike,
    *,
    method: str = "finite_difference",
    step: float = 1e-06,
    _bindings: _BindingReaders = _readers,
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
        Typed numerical providers and AD result/primitives, defaulting to this
        module's live AD suppliers. Legacy composition retains its own globals.

    Returns
    -------
    UnifiedDifferentiableAPIResult
        Supported ``jacobian`` envelope with objective and evaluation data.

    """
    return _algorithms.differentiable_jacobian(
        objective, values, method=method, step=step, _bindings=_bindings
    )


def differentiable_hessian(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    method: str = "finite_difference",
    step: float = 0.0001,
    _bindings: _BindingReaders = _readers,
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
        Typed numerical providers and AD result/primitives, defaulting to this
        module's live AD suppliers. Legacy composition retains its own globals.

    Returns
    -------
    UnifiedDifferentiableAPIResult
        Supported ``hessian`` envelope with evaluation provenance.

    """
    return _algorithms.differentiable_hessian(
        objective, values, method=method, step=step, _bindings=_bindings
    )


def _require_objective(
    objective: Callable[[Any], Any] | None, *, _bindings: _BindingReaders = _readers
) -> Callable[[Any], Any]:
    """Require an actual objective for the selected numerical operation."""
    return _algorithms._require_objective(objective, _bindings=_bindings)


def _require_values(
    values: ArrayLike | None, *, _bindings: _BindingReaders = _readers
) -> ArrayLike:
    """Require the actual parameter values for the selected numerical operation."""
    return _algorithms._require_values(values, _bindings=_bindings)


__all__ = [
    "differentiable_value",
    "differentiable_gradient",
    "differentiable_jacobian",
    "differentiable_hessian",
]
