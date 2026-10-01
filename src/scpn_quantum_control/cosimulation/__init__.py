# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — quantum/classical co-simulation package
"""Quantum/classical mean-field co-simulation for large K_nm networks."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .knm_partition import (
        ConservationReport,
        KnmPartition,
        partition_knm,
    )
    from .quantum_classical import CoSimulationResult, cosimulate

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "ConservationReport": (
        "scpn_quantum_control.cosimulation.knm_partition",
        "ConservationReport",
    ),
    "KnmPartition": ("scpn_quantum_control.cosimulation.knm_partition", "KnmPartition"),
    "partition_knm": ("scpn_quantum_control.cosimulation.knm_partition", "partition_knm"),
    "CoSimulationResult": (
        "scpn_quantum_control.cosimulation.quantum_classical",
        "CoSimulationResult",
    ),
    "cosimulate": ("scpn_quantum_control.cosimulation.quantum_classical", "cosimulate"),
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


__all__ = [
    "ConservationReport",
    "KnmPartition",
    "partition_knm",
    "CoSimulationResult",
    "cosimulate",
]
