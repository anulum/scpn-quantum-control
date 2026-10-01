# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — thermodynamics package exports
# scpn-quantum-control -- quantum thermodynamics
"""Quantum thermodynamics readiness tools for synchronisation transitions."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .readiness import (
        QUANTUM_THERMO_SCHEMA,
        CalibratedWorkIdentity,
        EntropyProductionRate,
        HeatDissipationRate,
        IrreversibilityResidual,
        ThermodynamicSweepConfig,
        ThermodynamicSweepResult,
        ThermodynamicSweepRow,
        calibrated_work_identity,
        entropy_production_rate,
        heat_dissipation_rate,
        irreversibility_residual,
        quantum_thermo_markdown,
        quantum_thermo_payload,
        run_k_sweep_protocol,
    )

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "QUANTUM_THERMO_SCHEMA": (
        "scpn_quantum_control.thermodynamics.readiness",
        "QUANTUM_THERMO_SCHEMA",
    ),
    "CalibratedWorkIdentity": (
        "scpn_quantum_control.thermodynamics.readiness",
        "CalibratedWorkIdentity",
    ),
    "EntropyProductionRate": (
        "scpn_quantum_control.thermodynamics.readiness",
        "EntropyProductionRate",
    ),
    "HeatDissipationRate": (
        "scpn_quantum_control.thermodynamics.readiness",
        "HeatDissipationRate",
    ),
    "IrreversibilityResidual": (
        "scpn_quantum_control.thermodynamics.readiness",
        "IrreversibilityResidual",
    ),
    "ThermodynamicSweepConfig": (
        "scpn_quantum_control.thermodynamics.readiness",
        "ThermodynamicSweepConfig",
    ),
    "ThermodynamicSweepResult": (
        "scpn_quantum_control.thermodynamics.readiness",
        "ThermodynamicSweepResult",
    ),
    "ThermodynamicSweepRow": (
        "scpn_quantum_control.thermodynamics.readiness",
        "ThermodynamicSweepRow",
    ),
    "calibrated_work_identity": (
        "scpn_quantum_control.thermodynamics.readiness",
        "calibrated_work_identity",
    ),
    "entropy_production_rate": (
        "scpn_quantum_control.thermodynamics.readiness",
        "entropy_production_rate",
    ),
    "heat_dissipation_rate": (
        "scpn_quantum_control.thermodynamics.readiness",
        "heat_dissipation_rate",
    ),
    "irreversibility_residual": (
        "scpn_quantum_control.thermodynamics.readiness",
        "irreversibility_residual",
    ),
    "quantum_thermo_markdown": (
        "scpn_quantum_control.thermodynamics.readiness",
        "quantum_thermo_markdown",
    ),
    "quantum_thermo_payload": (
        "scpn_quantum_control.thermodynamics.readiness",
        "quantum_thermo_payload",
    ),
    "run_k_sweep_protocol": (
        "scpn_quantum_control.thermodynamics.readiness",
        "run_k_sweep_protocol",
    ),
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
    "QUANTUM_THERMO_SCHEMA",
    "CalibratedWorkIdentity",
    "EntropyProductionRate",
    "HeatDissipationRate",
    "IrreversibilityResidual",
    "ThermodynamicSweepConfig",
    "ThermodynamicSweepResult",
    "ThermodynamicSweepRow",
    "calibrated_work_identity",
    "entropy_production_rate",
    "heat_dissipation_rate",
    "irreversibility_residual",
    "quantum_thermo_markdown",
    "quantum_thermo_payload",
    "run_k_sweep_protocol",
]
