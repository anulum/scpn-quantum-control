# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase classifier numerical operations
"""Typed numerical reads used by the bounded phase classifier's host model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeAlias, overload

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


class PhaseModelNumpy(Protocol):
    """NumPy operations used to evaluate admitted phase-classifier arrays."""

    @property
    def float64(self) -> type[np.float64]:
        """Return the active double-precision scalar type."""
        ...

    def cos(self, values: FloatArray, /) -> FloatArray:
        """Evaluate the cosine of each encoded feature phase."""
        ...

    @overload
    def mean(self, values: FloatArray, *, axis: int) -> FloatArray: ...

    @overload
    def mean(self, values: FloatArray) -> np.float64: ...

    def mean(self, values: FloatArray, *, axis: int | None = None) -> FloatArray | np.float64:
        """Average feature probabilities or squared sample residuals."""
        ...

    def clip(self, values: FloatArray, lower: float, upper: float, /) -> FloatArray:
        """Bound rounded probabilities by the classifier's physical interval."""
        ...


class AnalyticPhaseModelNumpy(PhaseModelNumpy, Protocol):
    """Add analytic-derivative operations without widening the prediction protocol."""

    def sin(self, values: FloatArray, /) -> FloatArray:
        """Evaluate the sine used by the analytic phase derivative."""
        ...

    def sum(self, values: FloatArray, *, axis: int) -> FloatArray:
        """Sum sample derivatives without reducing their feature dimension."""
        ...


@dataclass(frozen=True)
class PhaseModelNumerics:
    """Read numerical names when their original expressions evaluate them.

    Parameters
    ----------
    numpy
        Read the current NumPy namespace independently for cosine, reduction,
        clipping and the final dtype lookup.
    scalar
        Read the active float constructor before evaluating the scalar loss's
        reduction argument.

    """

    numpy: Callable[[], object]
    scalar: Callable[[], type[float]]


__all__ = ["AnalyticPhaseModelNumpy", "FloatArray", "PhaseModelNumerics", "PhaseModelNumpy"]
