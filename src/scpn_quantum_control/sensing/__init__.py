# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum sensing package
"""Quantum-sensing models, including high-field NV-centre magnetometry."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .nv_magnetometry_20T import (
        ELECTRON_GYROMAGNETIC_HZ_PER_T,
        NV_ZERO_FIELD_SPLITTING_HZ,
        NVCenter,
        NVFieldCalibration,
        calibrate_field_from_odmr,
        cw_odmr_dc_sensitivity_t_per_sqrt_hz,
        nv_energy_levels_hz,
        nv_ground_state_hamiltonian,
        odmr_resonances_hz,
        odmr_spectrum,
        simulate_odmr_measurement,
    )

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "ELECTRON_GYROMAGNETIC_HZ_PER_T": (
        "scpn_quantum_control.sensing.nv_magnetometry_20T",
        "ELECTRON_GYROMAGNETIC_HZ_PER_T",
    ),
    "NV_ZERO_FIELD_SPLITTING_HZ": (
        "scpn_quantum_control.sensing.nv_magnetometry_20T",
        "NV_ZERO_FIELD_SPLITTING_HZ",
    ),
    "NVCenter": ("scpn_quantum_control.sensing.nv_magnetometry_20T", "NVCenter"),
    "NVFieldCalibration": (
        "scpn_quantum_control.sensing.nv_magnetometry_20T",
        "NVFieldCalibration",
    ),
    "calibrate_field_from_odmr": (
        "scpn_quantum_control.sensing.nv_magnetometry_20T",
        "calibrate_field_from_odmr",
    ),
    "cw_odmr_dc_sensitivity_t_per_sqrt_hz": (
        "scpn_quantum_control.sensing.nv_magnetometry_20T",
        "cw_odmr_dc_sensitivity_t_per_sqrt_hz",
    ),
    "nv_energy_levels_hz": (
        "scpn_quantum_control.sensing.nv_magnetometry_20T",
        "nv_energy_levels_hz",
    ),
    "nv_ground_state_hamiltonian": (
        "scpn_quantum_control.sensing.nv_magnetometry_20T",
        "nv_ground_state_hamiltonian",
    ),
    "odmr_resonances_hz": (
        "scpn_quantum_control.sensing.nv_magnetometry_20T",
        "odmr_resonances_hz",
    ),
    "odmr_spectrum": ("scpn_quantum_control.sensing.nv_magnetometry_20T", "odmr_spectrum"),
    "simulate_odmr_measurement": (
        "scpn_quantum_control.sensing.nv_magnetometry_20T",
        "simulate_odmr_measurement",
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
    "ELECTRON_GYROMAGNETIC_HZ_PER_T",
    "NV_ZERO_FIELD_SPLITTING_HZ",
    "NVCenter",
    "NVFieldCalibration",
    "calibrate_field_from_odmr",
    "cw_odmr_dc_sensitivity_t_per_sqrt_hz",
    "nv_energy_levels_hz",
    "nv_ground_state_hamiltonian",
    "odmr_resonances_hz",
    "odmr_spectrum",
    "simulate_odmr_measurement",
]
