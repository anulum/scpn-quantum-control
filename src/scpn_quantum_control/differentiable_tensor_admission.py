# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Real numerical input admission
"""Admit numerical differentiation inputs without SDK or model dependencies.

These operations check the supplied numerical values. Allocation and copy policy
remain with the caller: an admitted array can share storage with the input.
Legacy adapters keep their own final dtype/copy expression and natural globals.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


class AdmissionNumpy(Protocol):
    """Array conversion and finiteness operations used by real input admission."""

    def asarray(self, values: object, *, dtype: type[float]) -> FloatArray:
        """Convert real input values using the caller's scalar dtype."""
        ...

    def isfinite(self, values: FloatArray | float, /) -> BoolArray | np.bool_:
        """Check every actual array value or the supplied scalar."""
        ...

    def all(self, values: BoolArray | np.bool_, /) -> np.bool_:
        """Require all entries of a finiteness result to hold."""
        ...


@dataclass(frozen=True)
class AdmissionPrimitives:
    """Read the numerical and error globals of the original caller.

    Parameters
    ----------
    numpy
        Read the live array namespace at each conversion or finiteness lookup.
        The outer all operation is resolved before its inner isfinite operand.
    scalar
        Read the active float constructor separately for dtype and scalar
        conversion, at the original expression boundaries.
    invalid
        Read the ValueError constructor immediately before error arguments
        are evaluated. No error type or message policy is selected here.

    """

    numpy: Callable[[], object]
    scalar: Callable[[], type[float]]
    invalid: Callable[[], type[ValueError]]


def require_real_parameter_vector(
    name: str,
    values: object,
    *,
    width: int | None = None,
    primitives: AdmissionPrimitives,
) -> FloatArray:
    """Admit a finite real parameter vector without applying a copy policy.

    Parameters
    ----------
    name
        Original parameter name used in errors.
    values
        Actual numerical array or array-like input.
    width
        Required number of parameters, or None when width is unconstrained.
    primitives
        Live numerical, scalar and error readers owned by the caller.

    Returns
    -------
    FloatArray
        Admitted one-dimensional array. An empty vector is accepted when its
        width is unconstrained or zero; it contains no non-finite values.
        The returned array may share storage with the supplied input.

    Raises
    ------
    ValueError
        If the array rank, declared width or finiteness is invalid.

    """
    vector = cast(AdmissionNumpy, primitives.numpy()).asarray(values, dtype=primitives.scalar())
    if vector.ndim != 1:
        raise primitives.invalid()(f"{name} must be a one-dimensional array")
    if width is not None and vector.shape != (width,):
        raise primitives.invalid()(f"{name} must have shape ({width},), got {vector.shape}")
    if not cast(AdmissionNumpy, primitives.numpy()).all(
        cast(AdmissionNumpy, primitives.numpy()).isfinite(vector)
    ):
        raise primitives.invalid()(f"{name} must contain only finite values")
    return vector


def require_real_scalar(
    name: str,
    value: object,
    *,
    primitives: AdmissionPrimitives,
) -> float:
    """Admit a finite scalar while retaining the caller's conversion order.

    Parameters
    ----------
    name
        Original parameter name used in errors.
    value
        Scalar or zero-dimensional array.
    primitives
        Live numerical, scalar and error readers owned by the caller.

    Returns
    -------
    float
        The finite value returned by the caller's actual scalar constructor.

    Raises
    ------
    ValueError
        If the converted input is not scalar-shaped or is non-finite.

    """
    array = cast(AdmissionNumpy, primitives.numpy()).asarray(value, dtype=primitives.scalar())
    if array.shape != ():
        raise primitives.invalid()(f"{name} must be a scalar")
    scalar = primitives.scalar()(array)
    if not cast(AdmissionNumpy, primitives.numpy()).isfinite(scalar):
        raise primitives.invalid()(f"{name} must be finite")
    return scalar


def require_nonnegative_tolerance(
    value: float,
    *,
    error_text: str,
    negative_first: bool = False,
    primitives: AdmissionPrimitives,
) -> float:
    """Admit a finite non-negative tolerance with the original short circuit.

    Parameters
    ----------
    value
        Supplied numerical tolerance.
    error_text
        Existing error message owned by the caller.
    negative_first
        Test negativity before finiteness when True. Legacy adapters have both
        orders; a negative value must not trigger an originally skipped lookup.
    primitives
        Live numerical, scalar and error readers owned by the caller.

    Returns
    -------
    float
        Finite non-negative tolerance, including the original signed zero.

    Raises
    ------
    ValueError
        If the converted value is negative or non-finite.

    """
    tolerance = primitives.scalar()(value)
    if negative_first:
        invalid = tolerance < 0.0 or not cast(AdmissionNumpy, primitives.numpy()).isfinite(
            tolerance
        )
    else:
        invalid = (
            not cast(AdmissionNumpy, primitives.numpy()).isfinite(tolerance) or tolerance < 0.0
        )
    if invalid:
        raise primitives.invalid()(error_text)
    return tolerance


__all__ = [
    "AdmissionNumpy",
    "AdmissionPrimitives",
    "require_nonnegative_tolerance",
    "require_real_parameter_vector",
    "require_real_scalar",
]
