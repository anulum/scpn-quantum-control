# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Numerical transform comparison
"""Pure finite-array normalization and residuals for AD transform comparisons."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]


class _ComparisonNumpy(Protocol):
    """Numerical operations required by finite-array comparison."""

    @property
    def float64(self) -> type[np.float64]:
        """The actual array scalar dtype."""
        ...

    def asarray(self, values: ArrayLike, *, dtype: type[np.float64]) -> FloatArray:
        """Convert supplied values to a real array."""
        ...

    def isfinite(self, values: FloatArray, /) -> NDArray[np.bool_]:
        """Identify finite scalar entries."""
        ...

    def all(self, values: NDArray[np.bool_], /) -> np.bool_:
        """Require every scalar admission to pass."""
        ...

    def abs(self, values: FloatArray, /) -> FloatArray:
        """Take absolute scalar differences."""
        ...

    def max(self, values: FloatArray, /) -> np.float64:
        """Reduce a nonempty array to its largest value."""
        ...


@dataclass(frozen=True)
class TransformComparisonPrimitives:
    """Numerical primitive readers for an explicitly supplied execution context.

    Parameters
    ----------
    numpy
        Read the numerical namespace at each operation, including after array
        conversion invokes user-defined array protocols.
    scalar
        Read the float constructor before evaluating its numeric argument.
    error
        Read the finite-value admission error constructor when needed.
    array_cast
        Read the nominal array cast immediately before evaluating its arguments.
    array_type
        Read the array type supplied to that cast.

    """

    numpy: Callable[[], object]
    scalar: Callable[[], type[float]]
    error: Callable[[], type[ValueError]]
    array_cast: Callable[[], object]
    array_type: Callable[[], object]


_NUMPY_PRIMITIVES = TransformComparisonPrimitives(
    numpy=lambda: np,
    scalar=lambda: float,
    error=lambda: ValueError,
    array_cast=lambda: cast,
    array_type=lambda: FloatArray,
)


def as_finite_flat_array(
    values: ArrayLike,
    *,
    primitives: TransformComparisonPrimitives = _NUMPY_PRIMITIVES,
) -> FloatArray:
    """Return an independent finite float64 vector for numeric comparisons.

    Parameters
    ----------
    values
        Scalars or arrays in the caller's observable units. Shape is flattened
        in C order; no physical units or derivative qualification are inferred.
    primitives
        Explicit numeric context, defaulting to this module's NumPy context.
        This interface contains no planner, policy or report constructor.

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 values with independent storage, including an
        empty vector when the supplied array has no entries.

    Raises
    ------
    ValueError
        If conversion fails or any converted entry is nonfinite.

    """
    array = cast(_ComparisonNumpy, primitives.numpy()).asarray(
        values, dtype=cast(_ComparisonNumpy, primitives.numpy()).float64
    )
    if not cast(_ComparisonNumpy, primitives.numpy()).all(
        cast(_ComparisonNumpy, primitives.numpy()).isfinite(array)
    ):
        raise primitives.error()("transform algebra values must be finite")
    array_cast = cast(Callable[[object, FloatArray], FloatArray], primitives.array_cast())
    return array_cast(
        primitives.array_type(),
        array.reshape(-1).astype(cast(_ComparisonNumpy, primitives.numpy()).float64, copy=True),
    )


def maximum_absolute_residual(
    lhs: FloatArray,
    rhs: FloatArray,
    *,
    primitives: TransformComparisonPrimitives = _NUMPY_PRIMITIVES,
) -> float:
    """Compare equally shaped nonempty arrays using their maximum absolute error.

    Parameters
    ----------
    lhs
        Finite reference or computed entries already admitted by the caller.
    rhs
        Finite comparison entries in the same units as ``lhs``.
    primitives
        Numeric context used to evaluate the difference and construct its scalar.

    Returns
    -------
    float
        Maximum absolute difference, or positive infinity for mismatched shapes
        or empty arrays. Incompatible observations cannot pass a finite tolerance.

    """
    if lhs.shape != rhs.shape:
        return primitives.scalar()("inf")
    if lhs.size == 0:
        return primitives.scalar()("inf")
    return primitives.scalar()(
        cast(_ComparisonNumpy, primitives.numpy()).max(
            cast(_ComparisonNumpy, primitives.numpy()).abs(lhs - rhs)
        )
    )


__all__ = [
    "TransformComparisonPrimitives",
    "as_finite_flat_array",
    "maximum_absolute_residual",
]
