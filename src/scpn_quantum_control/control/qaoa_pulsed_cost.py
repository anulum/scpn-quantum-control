# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — FRC pulsed-shot QAOA cost function
"""Control-grade FRC pulsed-shot scheduling cost for QAOA-MPC.

This is a *control adapter* cost function, not a transport solver: the
high-fidelity field-reversed-configuration (FRC) physics is owned by
SCPN-FUSION-CORE. The closed-form expressions used here are the published
control-level evaluations of the quantities the scheduler trades off:

- **s-parameter** — number of ion gyroradii between the field null and the
  separatrix, ``s = R_s / rho_i`` (Steinhauer, *Review of field-reversed
  configurations*, Physics of Plasmas 18, 070501, 2011). Under flux-conserving
  compression the achieved ``s`` scales with the peak external field; the
  surrogate exposes the exponent so it can be matched to a FUSION run.
- **Magneto-Rayleigh-Taylor (MRTI) growth** — ``gamma = sqrt(A_T k g -
  k^2 B^2 / (mu0 (rho_h + rho_l)))`` with the magnetic-tension stabilisation
  term (Velikovich et al., Physics of Plasmas 14, 022701, 2007; Sefkow et al.,
  Physics of Plasmas 21, 072711, 2014). The perturbation amplitude grows as
  ``a0 * exp(integral gamma dt)`` over the compression window.
- **FRC n=1 tilt-mode margin** — kinetic stabilisation of the prolate-FRC tilt
  instability is governed by ``S* / E`` with ``S* = R_s / rho_i ~ s`` and
  elongation ``E`` (Belova et al., *Numerical study of tilt stability of prolate
  FRCs*, Physics of Plasmas 8, 1267, 2001).

The cost mixed by :class:`FRCQAOAObjective` is the weighted sum of four
dimensionless penalties; minimising it schedules capacitor-bank firing
(``QAOA_MPC``) towards the target ``s`` within the energy budget while keeping
MRTI growth and the tilt margin inside their limits.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_MU0 = 4.0e-7 * np.pi  # vacuum permeability [H/m]
_MAX_GROWTH = 700.0  # cap on integrated MRTI e-foldings to keep exp(growth) finite


@dataclass(frozen=True)
class FRCQAOAObjective:
    """Targets, limits, and weights for the FRC pulsed-shot cost."""

    target_s_parameter: float
    bank_energy_budget_J: float
    mrti_amplitude_max_m: float
    tilt_margin_required: float
    weight_s: float = 1.0
    weight_energy: float = 0.5
    weight_mrti: float = 2.0
    weight_tilt: float = 1.5

    def __post_init__(self) -> None:
        for name in ("target_s_parameter", "bank_energy_budget_J", "mrti_amplitude_max_m"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.tilt_margin_required) or self.tilt_margin_required <= 0.0:
            raise ValueError("tilt_margin_required must be finite and positive")
        for name in ("weight_s", "weight_energy", "weight_mrti", "weight_tilt"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class FRCPlasmaSurrogate:
    """Documented control-grade map from the field profile to FRC observables.

    Every relation is a published closed form; this is not a transport solver.
    Defaults describe a compact pulsed FRC and should be matched to a
    SCPN-FUSION-CORE run when one is available.
    """

    # Defaults are calibrated to a representative compact pulsed-FRC compression
    # (microsecond window, centimetre-scale perturbation). The physics forms are
    # exact and cited; the absolute scale is a control surrogate and should be
    # matched to a SCPN-FUSION-CORE run when one is available.
    reference_field_T: float = 3.0
    reference_s_parameter: float = 2.0
    s_compression_exponent: float = 0.5  # s ∝ B^{1/2} under flux + adiabatic compression
    atwood_number: float = 0.9
    perturbation_wavelength_m: float = 0.05
    areal_mass_kg_per_m2: float = 200.0
    plasma_mass_density_kg_per_m3: float = 2.0
    elongation: float = 5.0
    initial_perturbation_m: float = 1.0e-4
    tilt_kinetic_threshold: float = 3.5  # critical S*/E for kinetic tilt stabilisation

    def __post_init__(self) -> None:
        positives = (
            "reference_field_T",
            "reference_s_parameter",
            "perturbation_wavelength_m",
            "areal_mass_kg_per_m2",
            "plasma_mass_density_kg_per_m3",
            "elongation",
            "initial_perturbation_m",
            "tilt_kinetic_threshold",
        )
        for name in positives:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not (0.0 < self.atwood_number <= 1.0):
            raise ValueError("atwood_number must lie in (0, 1]")

    def s_parameter(self, peak_field_T: float) -> float:
        """Achieved s-parameter from flux-conserving compression of the field."""
        ratio = max(peak_field_T, 0.0) / self.reference_field_T
        return float(self.reference_s_parameter * ratio**self.s_compression_exponent)

    def mrti_amplitude(self, field_profile: NDArray[np.float64], dt_s: float) -> float:
        """Compute the final MRTI perturbation amplitude after the compression window.

        ``gamma^2 = A_T k g - k^2 B^2 / (mu0 (rho_h + rho_l))`` integrated as
        ``a = a0 * exp(sum_t max(gamma, 0) dt)``; the magnetic-tension term sets
        ``gamma = 0`` once the field stabilises the interface.
        """
        b = np.asarray(field_profile, dtype=np.float64)
        if b.size < 2:
            return float(self.initial_perturbation_m)
        k = 2.0 * np.pi / self.perturbation_wavelength_m
        growth = _mrti_growth(
            b,
            dt_s,
            k,
            self.atwood_number,
            self.areal_mass_kg_per_m2,
            self.plasma_mass_density_kg_per_m3,
        )
        return float(self.initial_perturbation_m * np.exp(growth))

    def tilt_margin(self, s_parameter: float) -> float:
        """n=1 tilt-mode stability margin from the kinetic parameter ``S*/E``.

        Margin ``= (threshold - S*/E) / threshold``; positive when the FRC is
        below the kinetic stabilisation threshold, negative when MHD-tilt-prone.
        """
        s_star_over_e = max(s_parameter, 0.0) / self.elongation
        return float((self.tilt_kinetic_threshold - s_star_over_e) / self.tilt_kinetic_threshold)


def decode_schedule_to_field(
    schedule: NDArray[np.integer] | NDArray[np.floating],
    *,
    delta_field_T: float,
    base_field_T: float = 0.0,
) -> NDArray[np.float64]:
    """Convert a per-timestep bank-firing schedule into a cumulative B(t) profile."""
    u = np.asarray(schedule, dtype=np.float64)
    if u.ndim != 1 or u.size == 0:
        raise ValueError("schedule must be a non-empty one-dimensional array")
    if delta_field_T <= 0.0:
        raise ValueError("delta_field_T must be positive")
    return base_field_T + delta_field_T * np.cumsum(u)


def frc_pulsed_shot_cost(
    schedule: NDArray[np.integer] | NDArray[np.floating],
    target_b_profile: NDArray[np.floating],
    available_capacitor_energy_J: float,
    objective: FRCQAOAObjective,
    *,
    surrogate: FRCPlasmaSurrogate | None = None,
    delta_field_T: float = 0.5,
    energy_per_bank_J: float = 1.0e5,
    dt_s: float = 1.0e-6,
    return_components: bool = False,
) -> float | tuple[float, Mapping[str, float]]:
    """Weighted FRC pulsed-shot scheduling cost (lower is better).

    The four dimensionless penalties are: relative squared deviation of the
    achieved ``s`` from target; over-budget bank energy; squared MRTI amplitude
    against its limit; and tilt-margin shortfall.
    """
    model = surrogate or FRCPlasmaSurrogate()
    target = np.asarray(target_b_profile, dtype=np.float64)
    field = decode_schedule_to_field(schedule, delta_field_T=delta_field_T)
    if target.shape != field.shape:
        raise ValueError("target_b_profile must match the schedule length")
    if available_capacitor_energy_J <= 0.0:
        raise ValueError("available_capacitor_energy_J must be positive")

    peak_field = float(np.max(field))
    s_achieved = model.s_parameter(peak_field)
    energy_used = float(np.sum(np.asarray(schedule, dtype=np.float64) > 0.5) * energy_per_bank_J)
    mrti_amplitude = model.mrti_amplitude(field, dt_s)
    tilt_margin = model.tilt_margin(s_achieved)

    s_penalty = ((s_achieved - objective.target_s_parameter) / objective.target_s_parameter) ** 2
    energy_penalty = max(0.0, energy_used - objective.bank_energy_budget_J) / (
        objective.bank_energy_budget_J
    )
    mrti_penalty = (mrti_amplitude / objective.mrti_amplitude_max_m) ** 2
    tilt_penalty = max(0.0, objective.tilt_margin_required - tilt_margin) / (
        objective.tilt_margin_required
    )

    cost = (
        objective.weight_s * s_penalty
        + objective.weight_energy * energy_penalty
        + objective.weight_mrti * mrti_penalty
        + objective.weight_tilt * tilt_penalty
    )
    cost = float(cost)
    if not return_components:
        return cost
    components: Mapping[str, float] = {
        "cost": cost,
        "s_achieved": s_achieved,
        "s_penalty": float(s_penalty),
        "energy_used_J": energy_used,
        "energy_penalty": float(energy_penalty),
        "mrti_amplitude_m": mrti_amplitude,
        "mrti_penalty": float(mrti_penalty),
        "tilt_margin": tilt_margin,
        "tilt_penalty": float(tilt_penalty),
        "peak_field_T": peak_field,
    }
    return cost, components


def _mrti_growth(
    field: NDArray[np.float64],
    dt_s: float,
    wavenumber: float,
    atwood: float,
    areal_mass: float,
    density: float,
) -> float:
    """Integrated MRTI e-foldings, via the Rust kernel when present."""
    try:
        import scpn_quantum_engine as _engine

        if hasattr(_engine, "frc_mrti_growth"):
            return float(
                _engine.frc_mrti_growth(
                    np.ascontiguousarray(field, dtype=np.float64),
                    dt_s,
                    wavenumber,
                    atwood,
                    areal_mass,
                    density,
                    _MAX_GROWTH,
                )
            )
    except (ImportError, AttributeError, ValueError):
        pass
    return _mrti_growth_numpy(field, dt_s, wavenumber, atwood, areal_mass, density)


def _mrti_growth_numpy(
    field: NDArray[np.float64],
    dt_s: float,
    wavenumber: float,
    atwood: float,
    areal_mass: float,
    density: float,
) -> float:
    # Effective acceleration from the magnetic-pressure gradient driving the
    # implosion: g = d(P_B)/dt / areal_mass, P_B = B^2 / (2 mu0).
    magnetic_pressure = field**2 / (2.0 * _MU0)
    g = np.gradient(magnetic_pressure, dt_s) / areal_mass
    tension = wavenumber**2 * field**2 / (_MU0 * 2.0 * density)
    gamma_sq = atwood * wavenumber * g - tension
    gamma = np.sqrt(np.clip(gamma_sq, 0.0, None))
    return float(np.clip(np.sum(gamma) * dt_s, 0.0, _MAX_GROWTH))


__all__ = [
    "FRCPlasmaSurrogate",
    "FRCQAOAObjective",
    "decode_schedule_to_field",
    "frc_pulsed_shot_cost",
]
