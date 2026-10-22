# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable API module
"""Typed live AD providers and scalar, metadata and error primitives."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import numpy as np
from numpy.typing import ArrayLike

from scpn_quantum_control.differentiable_api_contracts import UnifiedDifferentiableAPIResult
from scpn_quantum_control.differentiable_finite_difference import ScalarObjective, VectorObjective
from scpn_quantum_control.differentiable_parameter_contracts import Parameter, ParameterShiftRule
from scpn_quantum_control.differentiable_result_contracts import (
    GradientResult,
    HessianResult,
    JacobianResult,
)
from scpn_quantum_control.whole_program_ad_result import WholeProgramADResult


class ValueAndGradProvider(Protocol):
    """Read the live value_and_grad callable without copying its namespace."""

    def __call__(
        self,
        objective: Callable[[Any], Any],
        values: ArrayLike,
        *,
        parameters: Sequence[Parameter] | None = ...,
        method: str = ...,
        rule: ParameterShiftRule | None = ...,
        step: float | None = ...,
    ) -> GradientResult | WholeProgramADResult:
        """Invoke the current provider with the original public arguments."""
        ...


class ValueAndHessianProvider(Protocol):
    """Read the live value_and_hessian callable without copying its namespace."""

    def __call__(
        self,
        objective: ScalarObjective,
        values: ArrayLike,
        *,
        parameters: Sequence[Parameter] | None = ...,
        method: str = ...,
        step: float = ...,
    ) -> HessianResult:
        """Invoke the current provider with the original public arguments."""
        ...


class ValueAndJacobianProvider(Protocol):
    """Read the live value_and_jacobian callable without copying its namespace."""

    def __call__(
        self,
        objective: VectorObjective,
        values: ArrayLike,
        *,
        parameters: Sequence[Parameter] | None = ...,
        method: str = ...,
        step: float = ...,
    ) -> JacobianResult:
        """Invoke the current provider with the original public arguments."""
        ...


_Element = TypeVar("_Element")


class ScalarConversion(Protocol):
    """Convert the actual scalar result through the caller's live primitive."""

    def __call__(self, value: float, /) -> float:
        """Convert one real scalar without replacing its source value.

        Parameters
        ----------
        value
            Actual AD result value in the objective's units.

        Returns
        -------
        float
            Caller-selected real scalar conversion.

        """
        ...


class ListConversion(Protocol):
    """Copy result metadata through the caller's live generic list constructor."""

    def __call__(self, values: Iterable[_Element], /) -> list[_Element]:
        """Copy ordered metadata without losing its element type.

        Parameters
        ----------
        values
            Actual ordered parameter names or trainable flags.

        Returns
        -------
        list
            Metadata copied in its original iteration order.

        """
        ...


@dataclass(frozen=True, slots=True)
class DifferentiableValueBindings:
    """Read original globals at their original call sites, including after callbacks."""

    read_UnifiedDifferentiableAPIResult: Callable[[], type[UnifiedDifferentiableAPIResult]]
    read_value_and_grad: Callable[[], ValueAndGradProvider]
    read_value_and_hessian: Callable[[], ValueAndHessianProvider]
    read_value_and_jacobian: Callable[[], ValueAndJacobianProvider]
    read_float64: Callable[[], type[np.float64]]
    read_float: Callable[[], ScalarConversion]
    read_list: Callable[[], ListConversion]
    read_error: Callable[[], type[ValueError]]
