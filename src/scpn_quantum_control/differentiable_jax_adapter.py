# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- optional JAX differentiable adapter
"""Optional JAX autodiff adapter for native differentiable objectives."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .differentiable_parameter_contracts import (
    _as_parameter_array,
    _as_real_numeric_array,
    _as_real_scalar,
)


def _load_jax_modules() -> tuple[ModuleType, ModuleType]:
    try:
        import jax
        import jax.numpy as jnp
    except Exception as exc:
        raise ImportError("JAX autodiff is unavailable; install the [jax] extra") from exc
    return jax, jnp


def is_jax_autodiff_available() -> bool:
    """Return whether JAX autodiff imports in the active environment.

    Returns
    -------
    bool
        ``True`` when both ``jax`` and ``jax.numpy`` import successfully,
        otherwise ``False``.
    """
    try:
        _load_jax_modules()
    except ImportError:
        return False
    return True


def jax_value_and_grad(
    objective: Callable[[Any], Any],
    values: ArrayLike,
) -> tuple[float, NDArray[np.float64]]:
    """Evaluate a JAX scalar objective and gradient.

    Parameters
    ----------
    objective:
        Callable receiving a JAX array and returning a scalar objective value.
    values:
        Real numeric parameter vector.

    Returns
    -------
    tuple[float, numpy.ndarray]
        Objective value and gradient converted to finite ``float64`` NumPy
        values.

    Raises
    ------
    ImportError
        If the optional JAX dependency is unavailable.
    ValueError
        If the input, objective value, or gradient violates the native
        differentiable contract.
    """
    parameter_values = _as_parameter_array(values)

    jax, jnp = _load_jax_modules()

    def wrapped(raw_values: Any) -> Any:
        return objective(raw_values)

    value, gradient = jax.value_and_grad(wrapped)(jnp.asarray(parameter_values))
    result_value = _as_real_scalar("JAX objective value", value)
    result_gradient = _as_real_numeric_array("JAX gradient", gradient)
    if result_gradient.shape != parameter_values.shape:
        raise ValueError("JAX gradient shape must match parameter shape")
    if not np.all(np.isfinite(result_gradient)):
        raise ValueError("JAX gradient must contain only finite values")
    return result_value, result_gradient


__all__ = [
    "is_jax_autodiff_available",
    "jax_value_and_grad",
]
