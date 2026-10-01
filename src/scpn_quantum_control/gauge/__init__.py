# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — U(1) Gauge Theory Observables
"""U(1) gauge theory observables for the Kuramoto-XY quantum model."""

from __future__ import annotations

import sys as _sys
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .cft_analysis import (
        CFTResult,
        cft_analysis,
        extract_central_charge,
        find_critical_coupling,
    )
    from .confinement import ConfinementResult, confinement_analysis, confinement_vs_coupling
    from .lattice_crosscheck import GaugeLatticeCrosscheck, crosscheck_confinement_on_lattice
    from .universality import UniversalityResult, universality_analysis
    from .vortex_detector import VortexResult, measure_vortex_density, vortex_density_vs_coupling
    from .wilson_loop import WilsonLoopResult, compute_wilson_loops, wilson_loop_expectation

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "CFTResult": ("scpn_quantum_control.gauge.cft_analysis", "CFTResult"),
    "cft_analysis": ("scpn_quantum_control.gauge.cft_analysis", "cft_analysis"),
    "extract_central_charge": (
        "scpn_quantum_control.gauge.cft_analysis",
        "extract_central_charge",
    ),
    "find_critical_coupling": (
        "scpn_quantum_control.gauge.cft_analysis",
        "find_critical_coupling",
    ),
    "ConfinementResult": ("scpn_quantum_control.gauge.confinement", "ConfinementResult"),
    "confinement_analysis": ("scpn_quantum_control.gauge.confinement", "confinement_analysis"),
    "confinement_vs_coupling": (
        "scpn_quantum_control.gauge.confinement",
        "confinement_vs_coupling",
    ),
    "GaugeLatticeCrosscheck": (
        "scpn_quantum_control.gauge.lattice_crosscheck",
        "GaugeLatticeCrosscheck",
    ),
    "crosscheck_confinement_on_lattice": (
        "scpn_quantum_control.gauge.lattice_crosscheck",
        "crosscheck_confinement_on_lattice",
    ),
    "UniversalityResult": ("scpn_quantum_control.gauge.universality", "UniversalityResult"),
    "universality_analysis": ("scpn_quantum_control.gauge.universality", "universality_analysis"),
    "VortexResult": ("scpn_quantum_control.gauge.vortex_detector", "VortexResult"),
    "measure_vortex_density": (
        "scpn_quantum_control.gauge.vortex_detector",
        "measure_vortex_density",
    ),
    "vortex_density_vs_coupling": (
        "scpn_quantum_control.gauge.vortex_detector",
        "vortex_density_vs_coupling",
    ),
    "WilsonLoopResult": ("scpn_quantum_control.gauge.wilson_loop", "WilsonLoopResult"),
    "compute_wilson_loops": ("scpn_quantum_control.gauge.wilson_loop", "compute_wilson_loops"),
    "wilson_loop_expectation": (
        "scpn_quantum_control.gauge.wilson_loop",
        "wilson_loop_expectation",
    ),
}


class _ExportModule(ModuleType):
    """Keep declared object exports when Python publishes child modules."""

    def __setattr__(self, name: str, value: object) -> None:
        """Publish a module attribute while preserving same-named exports.

        Parameters
        ----------
        name
            Attribute assigned by the import system or a caller.
        value
            Value to publish in the package namespace.

        """
        target = _PUBLIC_EXPORTS.get(name)
        if (
            target is not None
            and target[1] is not None
            and isinstance(value, ModuleType)
            and value.__name__ in {target[0], f"{__name__}.{name}"}
        ):
            value = __getattr__(name)
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _ExportModule


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
    "CFTResult",
    "cft_analysis",
    "extract_central_charge",
    "find_critical_coupling",
    "ConfinementResult",
    "confinement_analysis",
    "confinement_vs_coupling",
    "GaugeLatticeCrosscheck",
    "crosscheck_confinement_on_lattice",
    "UniversalityResult",
    "universality_analysis",
    "VortexResult",
    "measure_vortex_density",
    "vortex_density_vs_coupling",
    "WilsonLoopResult",
    "compute_wilson_loops",
    "wilson_loop_expectation",
]
