# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Ψ-field Lattice Gauge Theory
"""U(1) lattice gauge simulator for the SCPN Ψ-field.

The SCPN defines a Ψ-field as a compact U(1) gauge field carrying the
infoton boson. This package implements the lattice gauge dynamics on
arbitrary graph topologies (not restricted to hypercubic lattices),
enabling simulation on the actual SCPN 15+1 layer hierarchy.

Modules:
    lattice — U(1) gauge field on arbitrary graphs with HMC update
    infoton — scalar field coupled to gauge (lattice scalar QED)
    scpn_mapping — SCPN layer hierarchy to lattice topology
    observables — Polyakov loop, topological charge, string tension
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .infoton import InfitonField, gauge_covariant_kinetic
    from .lattice import PlaquetteResult, U1LatticGauge, hmc_update
    from .observables import (
        polyakov_loop,
        string_tension_from_wilson,
        topological_charge,
    )
    from .scpn_mapping import SCPNLattice, scpn_to_lattice

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "InfitonField": ("scpn_quantum_control.psi_field.infoton", "InfitonField"),
    "gauge_covariant_kinetic": (
        "scpn_quantum_control.psi_field.infoton",
        "gauge_covariant_kinetic",
    ),
    "PlaquetteResult": ("scpn_quantum_control.psi_field.lattice", "PlaquetteResult"),
    "U1LatticGauge": ("scpn_quantum_control.psi_field.lattice", "U1LatticGauge"),
    "hmc_update": ("scpn_quantum_control.psi_field.lattice", "hmc_update"),
    "polyakov_loop": ("scpn_quantum_control.psi_field.observables", "polyakov_loop"),
    "string_tension_from_wilson": (
        "scpn_quantum_control.psi_field.observables",
        "string_tension_from_wilson",
    ),
    "topological_charge": ("scpn_quantum_control.psi_field.observables", "topological_charge"),
    "SCPNLattice": ("scpn_quantum_control.psi_field.scpn_mapping", "SCPNLattice"),
    "scpn_to_lattice": ("scpn_quantum_control.psi_field.scpn_mapping", "scpn_to_lattice"),
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
    "U1LatticGauge",
    "PlaquetteResult",
    "hmc_update",
    "InfitonField",
    "gauge_covariant_kinetic",
    "SCPNLattice",
    "scpn_to_lattice",
    "polyakov_loop",
    "topological_charge",
    "string_tension_from_wilson",
]
