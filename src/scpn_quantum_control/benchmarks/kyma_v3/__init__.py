# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — KYMA v3 symbolic composition probe
"""KYMA v3 compositional-generalisation probe with a symbolic ground truth.

v3 removes the in-class-teacher objection to v2: the labels come from a
deterministic symbolic program over three ``Z4`` registers (:mod:`.task`), not
from oscillator dynamics. The staged gated-coupling substrate (:mod:`.substrate`)
and the non-oscillator baselines (:mod:`.baselines`) learn the three operations
from single-operation and ordered-pair training items; the held-out ordered pair
``(R0, R1)`` is evaluated on query ``a`` only. The design checks are teacher-free
and the contract (:mod:`.probe`) is frozen in the pre-registration before any
model is trained.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .task import SymbolicDataset, build_dataset, design_report

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "SymbolicDataset": ("scpn_quantum_control.benchmarks.kyma_v3.task", "SymbolicDataset"),
    "build_dataset": ("scpn_quantum_control.benchmarks.kyma_v3.task", "build_dataset"),
    "design_report": ("scpn_quantum_control.benchmarks.kyma_v3.task", "design_report"),
}


def __getattr__(name: str) -> Any:
    """Resolve and cache a public export from its original owning module.

    Parameters
    ----------
    name
        Public export requested through this package.

    Returns
    -------
    Any
        Original object, including module-valued exports.

    Raises
    ------
    AttributeError
        If the name is undeclared or the original module lacks its attribute.
    ImportError
        If the owning module cannot be imported.

    """
    target = _PUBLIC_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    origin = import_module(target[0])
    value = origin if target[1] is None else getattr(origin, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List cached and deferred names for inspection tools.

    Returns
    -------
    list[str]
        Sorted package namespace and declared lazy export names.

    """
    return sorted(set(globals()) | set(_PUBLIC_EXPORTS))


__all__ = ["SymbolicDataset", "build_dataset", "design_report"]
