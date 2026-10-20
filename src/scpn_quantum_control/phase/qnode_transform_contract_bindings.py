# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Scalar QNode record dependencies
"""Typed numerical and nominal dependencies for original scalar QNode admission."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sized
from dataclasses import dataclass
from typing import Generic, Protocol, TypeAlias, TypeVar, cast, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from scpn_quantum_control.phase.transform_nesting import GradientTransformNestingPlan

FloatArray: TypeAlias = NDArray[np.float64]
ScalarObjective = Callable[[FloatArray], float]
_Result = TypeVar("_Result")
_ResultCo = TypeVar("_ResultCo", covariant=True)
_Item = TypeVar("_Item")


class _SequenceFactory(Protocol):
    """Retain the original tuple constructor for any actual record sequence."""

    def __call__(self, values: Iterable[_Item], /) -> tuple[_Item, ...]:
        """Materialise the original record sequence."""
        ...


class _NumpyTransform(Protocol):
    """Read only the original numerical operations used by scalar transforms."""

    @property
    def float64(self) -> type[np.float64]:
        """The actual copied-array scalar type."""
        ...

    @overload
    def asarray(self, values: ArrayLike, *, dtype: type[float]) -> FloatArray: ...

    @overload
    def asarray(self, values: ArrayLike) -> NDArray[np.generic]: ...

    def asarray(
        self, values: ArrayLike, *, dtype: type[float] | None = None
    ) -> NDArray[np.generic]:
        """Read original raw admission or explicit real conversion."""
        ...

    def array(self, values: ArrayLike, *, dtype: type[np.float64] | None = None) -> FloatArray:
        """Construct the original readiness parameters."""
        ...

    def isfinite(self, values: FloatArray, /) -> NDArray[np.bool_]:
        """Read the actual elementwise finite admission."""
        ...

    def all(self, values: NDArray[np.bool_], /) -> np.bool_:
        """Admit every actual numerical observation."""
        ...

    def dot(self, left: FloatArray, right: FloatArray, /) -> np.float64:
        """Compose the original directional product."""
        ...


class _ResultFactory(Protocol[_ResultCo]):
    """Construct the caller-owned scalar transform record with its original fields."""

    def __call__(
        self,
        *,
        transform: str,
        plan: GradientTransformNestingPlan,
        params: FloatArray,
        supported: bool,
        value: float | None,
        gradient: FloatArray | None,
        hessian: FloatArray | None,
        hessian_vector_product: FloatArray | None,
        jvp: float | None,
        vjp: FloatArray | None,
        jacobian: FloatArray | None,
        tangent: FloatArray | None,
        cotangent: FloatArray | None,
        parameter_shift_evaluations: int,
        failure_reason: str,
        evidence_class: str = ...,
        claim_boundary: str = ...,
        hardware_execution: bool = ...,
    ) -> _ResultCo:
        """Construct the real caller-owned record with its original fields."""
        ...


@dataclass(frozen=True)
class QNodeTransformBuiltins:
    """Read the original scalar, count, sequence and error builtins at use points."""

    scalar: Callable[[], type[float]]
    length: Callable[[], Callable[[Sized], int]]
    sequence: Callable[[], object]
    error: Callable[[], type[ValueError]]


class QNodeContractBindings(Generic[_Result]):
    """Read finite source-defined dependencies without routing namespace writes.

    Parameters
    ----------
    readers
        Literal late readers compiled in the actual defining source namespace.
    numpy
        Read the current numerical provider at each original operation.
    builtins
        Read the original cached builtin namespace at each original operation.

    """

    def __init__(
        self,
        readers: Mapping[str, Callable[[], object]],
        numpy: Callable[[], object],
        builtins: QNodeTransformBuiltins,
    ) -> None:
        """Retain the actual source readers and original primitive suppliers."""
        self._readers = readers
        self._numpy = numpy
        self.builtins = builtins

    def _read(self, name: str) -> object:
        """Read the declared source global with native builtin fallback."""
        return self._readers[name]()

    def numpy(self) -> _NumpyTransform:
        """Read the caller's current numerical namespace."""
        return cast(_NumpyTransform, self._numpy())

    def result_class(self) -> _ResultFactory[_Result]:
        """Read the current nominal result constructor at its construction point."""
        return cast(_ResultFactory[_Result], self._read("PhaseQNodeTransformResult"))

    def array_cast(self) -> Callable[[object, FloatArray], FloatArray]:
        """Read the original cast before evaluating its original arguments."""
        return cast(Callable[[object, FloatArray], FloatArray], self._read("cast"))

    def array_type(self) -> object:
        """Read the actual original array alias supplied to the cast."""
        return self._read("FloatArray")

    def sequence(self) -> _SequenceFactory:
        """Read the original generic tuple constructor."""
        return cast(_SequenceFactory, self.builtins.sequence())
