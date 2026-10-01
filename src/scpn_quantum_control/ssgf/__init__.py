# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — SSGF Quantum Extensions
"""SSGF quantum extensions: gradient, cost, outer cycle."""

from __future__ import annotations

import sys as _sys
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .quantum_costs import QuantumCosts, compute_quantum_costs
    from .quantum_gradient import QuantumGradientResult, compute_quantum_gradient, quantum_cost
    from .quantum_outer_cycle import OuterCycleResult, quantum_outer_cycle
    from .quantum_spectral import SpectralBridgeResult, spectral_bridge_analysis

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "QuantumCosts": ("scpn_quantum_control.ssgf.quantum_costs", "QuantumCosts"),
    "compute_quantum_costs": ("scpn_quantum_control.ssgf.quantum_costs", "compute_quantum_costs"),
    "QuantumGradientResult": (
        "scpn_quantum_control.ssgf.quantum_gradient",
        "QuantumGradientResult",
    ),
    "compute_quantum_gradient": (
        "scpn_quantum_control.ssgf.quantum_gradient",
        "compute_quantum_gradient",
    ),
    "quantum_cost": ("scpn_quantum_control.ssgf.quantum_gradient", "quantum_cost"),
    "OuterCycleResult": ("scpn_quantum_control.ssgf.quantum_outer_cycle", "OuterCycleResult"),
    "quantum_outer_cycle": (
        "scpn_quantum_control.ssgf.quantum_outer_cycle",
        "quantum_outer_cycle",
    ),
    "SpectralBridgeResult": ("scpn_quantum_control.ssgf.quantum_spectral", "SpectralBridgeResult"),
    "spectral_bridge_analysis": (
        "scpn_quantum_control.ssgf.quantum_spectral",
        "spectral_bridge_analysis",
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
    "QuantumCosts",
    "compute_quantum_costs",
    "QuantumGradientResult",
    "compute_quantum_gradient",
    "quantum_cost",
    "OuterCycleResult",
    "quantum_outer_cycle",
    "SpectralBridgeResult",
    "spectral_bridge_analysis",
]
