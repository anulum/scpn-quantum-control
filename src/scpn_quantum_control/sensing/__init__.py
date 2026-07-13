# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum sensing package
"""Quantum-sensing models, including high-field NV-centre magnetometry."""

from __future__ import annotations

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
