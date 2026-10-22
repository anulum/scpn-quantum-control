# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Local transform composition primitives
"""Typed numerical providers for local derivative composition identities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .differentiable_finite_difference import ScalarObjective, VectorObjective

FloatArray: TypeAlias = NDArray[np.float64]
JacobianProvider: TypeAlias = Callable[[VectorObjective, ArrayLike], FloatArray]
ProductProvider: TypeAlias = Callable[[VectorObjective, ArrayLike, ArrayLike], FloatArray]


class GradientProvider(Protocol):
    """Scalar gradient provider with an explicit differentiation method."""

    def __call__(
        self, objective: ScalarObjective, values: ArrayLike, *, method: str
    ) -> FloatArray:
        """Differentiate the supplied real objective using the requested method."""
        ...


class CompositionNumpy(Protocol):
    """Real array operations needed to compare composed derivatives."""

    @property
    def float64(self) -> type[np.float64]:
        """Return the current real array dtype."""
        ...

    def vstack(self, values: Sequence[FloatArray], /) -> FloatArray:
        """Stack actual row gradients without changing their order."""
        ...

    def dot(self, left: FloatArray, right: FloatArray, /) -> np.float64:
        """Contract two real derivative vectors."""
        ...

    def array(self, values: list[float], *, dtype: type[np.float64]) -> FloatArray:
        """Construct a real array containing a contracted derivative."""
        ...


@dataclass(frozen=True)
class LocalTransformPrimitives:
    """Read numerical providers at each original expression boundary.

    Parameters
    ----------
    gradient
        Read the scalar gradient callable before evaluating its arguments.
    jacobian
        Read the Jacobian callable before evaluating its arguments, including
        an inner objective that may change the caller's numerical namespace.
    jvp
        Read the Jacobian-vector product callable.
    vjp
        Read the vector-Jacobian product callable after the JVP has executed.
    numpy
        Read the array namespace separately for each operation and dtype.
    scalar
        Read the float constructor before evaluating an inner product.

    """

    gradient: Callable[[], GradientProvider]
    jacobian: Callable[[], JacobianProvider]
    jvp: Callable[[], ProductProvider]
    vjp: Callable[[], ProductProvider]
    numpy: Callable[[], object]
    scalar: Callable[[], type[float]]


__all__ = [
    "CompositionNumpy",
    "GradientProvider",
    "JacobianProvider",
    "LocalTransformPrimitives",
    "ProductProvider",
]
