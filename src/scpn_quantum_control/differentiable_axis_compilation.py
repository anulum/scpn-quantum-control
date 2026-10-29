# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native compilation of fixed tensor-axis ordering
"""Trace fixed permutation indices without a NumPy tensor graph interruption.

Only this module's numerical bridge is registered with the SDK compiler. Eager
execution still invokes the supplied sorting operation. The traced native NumPy
case uses integer permutation arithmetic; replacement sorters remain actual
inputs and are traced normally.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from threading import RLock
from types import ModuleType
from typing import Any, TypeAlias
from weakref import WeakSet

AxisSorter: TypeAlias = Callable[[list[int]], Iterable[int]]

_REGISTRATION_LOCK = RLock()
_REGISTERED: WeakSet[ModuleType] = WeakSet()


def numpy_axis_order(sorter: AxisSorter, axes: list[int]) -> Iterable[int]:
    """Evaluate the actual supplied axis sorter at its original numerical boundary.

    Parameters
    ----------
    sorter
        Original numerical dependency, resolved by the calling adapter.
    axes
        Admitted permutation of the binary tensor dimensions.

    Returns
    -------
    Iterable[int]
        Original sorter result in eager execution. A registered compiler traces
        equivalent permutation indices; callers retain scalar conversion and
        iterator construction at their original defining scopes.

    """
    return sorter(axes)


def trace_numpy_axis_order(sorter: AxisSorter, axes: list[int]) -> Iterable[int]:
    """Lower admitted dimension permutations to traceable integer operations.

    Parameters
    ----------
    sorter
        Actual numerical dependency. NumPy's native dispatcher uses equivalent
        permutation arithmetic; every replacement callable is invoked directly.
    axes
        Admitted permutation containing each dimension index exactly once.

    Returns
    -------
    Iterable[int]
        Equivalent permutation indices, materialized by the numerical backend.
        This is a compiler lowering operation, not a general NumPy sort API.

    """
    if (
        type(sorter).__name__ == "_ArrayFunctionDispatcher"
        and getattr(sorter, "__name__", None) == "argsort"
        and getattr(sorter, "__module__", None) == "numpy"
    ):
        return tuple(axes.index(index) for index in range(len(axes)))
    return sorter(axes)


def register_torch_axis_compilation(namespace: Any) -> bool:
    """Register guarded numerical permutation tracing for an actual Torch SDK.

    Parameters
    ----------
    namespace
        Actual PyTorch module, already loaded by the original caller. Other
        namespaces and runtimes without the polyfill API remain unchanged.

    Returns
    -------
    bool
        Whether the SDK supports this registration and its handler is installed.
        Repeated registration is idempotent. Compiler signature checking and
        ordinary dependency guards stay enabled; no numerical tensor is copied
        or declared constant.

    """
    if not isinstance(namespace, ModuleType) or namespace.__name__ != "torch":
        return False
    compiler = getattr(namespace, "compiler", None)
    substitute = getattr(compiler, "substitute_in_graph", None)
    if not callable(substitute):
        return False
    with _REGISTRATION_LOCK:
        if namespace in _REGISTERED:
            return True
        substitute(numpy_axis_order)(trace_numpy_axis_order)
        _REGISTERED.add(namespace)
    return True


__all__ = [
    "AxisSorter",
    "numpy_axis_order",
    "trace_numpy_axis_order",
    "register_torch_axis_compilation",
]
