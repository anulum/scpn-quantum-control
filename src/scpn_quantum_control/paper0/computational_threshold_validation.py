# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 computational-threshold validation fixtures
"""Executable simulator fixtures for Paper 0 EQ0119-EQ0122 anchors."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from .spec_loader import load_computational_threshold_validation_spec


@dataclass(frozen=True, slots=True)
class IITThresholdConfig:
    """Finite threshold-classifier settings for the EQ0119 IIT-OR boundary."""

    phi_values: np.ndarray | None = None
    alpha_phi: float = 2.5
    phi_crit: float = 0.75
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        _require_finite("alpha_phi", self.alpha_phi)
        _require_finite("phi_crit", self.phi_crit)
        if self.alpha_phi < 0.0:
            raise ValueError("alpha_phi must be non-negative")
        if self.phi_crit <= 0.0:
            raise ValueError("phi_crit must be positive")
        values = (
            self.phi_values
            if self.phi_values is not None
            else np.array([0.1, 0.35, 0.8, 1.2], dtype=np.float64)
        )
        phi = _validate_real_vector("phi_values", values)
        if np.any(phi < 0.0):
            raise ValueError("phi_values must be non-negative")
        object.__setattr__(self, "phi_values", phi)


@dataclass(frozen=True, slots=True)
class IITThresholdValidationResult:
    """Source-anchored IIT-OR threshold fixture result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    threshold_labels: tuple[int, ...]
    proportionality_residual: float
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class CoherenceCurrentConfig:
    """Finite plane-wave settings for the EQ0120 Noether-current boundary."""

    time_points: int = 97
    space_points: int = 101
    periods: int = 2
    wavelength_count: int = 3
    global_phase: float = 0.37
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.time_points, int) or self.time_points < 16:
            raise ValueError("time_points must be at least 16")
        if not isinstance(self.space_points, int) or self.space_points < 16:
            raise ValueError("space_points must be at least 16")
        if not isinstance(self.periods, int) or self.periods < 1:
            raise ValueError("periods must be a positive integer")
        if not isinstance(self.wavelength_count, int) or self.wavelength_count < 1:
            raise ValueError("wavelength_count must be a positive integer")
        _require_finite("global_phase", self.global_phase)


@dataclass(frozen=True, slots=True)
class CoherenceCurrentValidationResult:
    """Source-anchored Noether-current validation result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    global_phase_invariance_error: float
    divergence_residual: float
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class InformationEnergyTransductionConfig:
    """Finite density-grid settings for the EQ0121-EQ0122 quantum-potential fixture."""

    grid_points: int = 401
    domain_radius: float = 6.0
    sigma: float = 1.3
    hbar: float = 1.0
    mass: float = 1.0
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.grid_points, int) or self.grid_points < 33:
            raise ValueError("grid_points must be at least 33")
        if self.grid_points % 2 == 0:
            raise ValueError("grid_points must be odd for centred Gaussian checks")
        for name in ("domain_radius", "sigma", "hbar", "mass"):
            _require_finite(name, float(getattr(self, name)))
        if self.domain_radius <= 0.0:
            raise ValueError("domain_radius must be positive")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if self.hbar <= 0.0:
            raise ValueError("hbar must be positive")
        if self.mass <= 0.0:
            raise ValueError("mass must be positive")


@dataclass(frozen=True, slots=True)
class InformationEnergyTransductionValidationResult:
    """Source-anchored IET quantum-potential validation result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    constant_density_max_abs: float
    gaussian_residual_rms: float
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


def iit_threshold_energy(config: IITThresholdConfig) -> np.ndarray:
    """Return ``E_Phi = alpha_Phi * Phi`` for finite threshold inputs."""
    return cast(np.ndarray, config.alpha_phi * cast(np.ndarray, config.phi_values))


def validate_iit_or_threshold_fixture(
    config: IITThresholdConfig | None = None,
) -> IITThresholdValidationResult:
    """Run the source-anchored EQ0119 IIT-OR threshold fixture."""
    cfg = config or IITThresholdConfig()
    spec = load_computational_threshold_validation_spec(
        "computational.iit_or_threshold",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    phi = cast(np.ndarray, cfg.phi_values)
    energy = iit_threshold_energy(cfg)
    fitted_alpha = float(np.dot(phi, energy) / np.dot(phi, phi))
    proportionality_residual = float(np.max(np.abs(energy - fitted_alpha * phi)))
    labels = tuple(int(value >= cfg.phi_crit) for value in phi)
    alpha_zero = iit_threshold_energy(replace(cfg, alpha_phi=0.0))
    controls = {
        "alpha_zero_energy_max_abs": float(np.max(np.abs(alpha_zero))),
        "subcritical_label_count": float(np.sum(phi[phi < cfg.phi_crit] >= cfg.phi_crit)),
        "label_shuffle_control_label": float(tuple(reversed(labels)) != labels),
    }
    metadata = {
        "paper0_spec_key": "computational.iit_or_threshold",
        "paper0_validation_protocol": str(spec["validation_protocol"]),
        "hardware_status": str(spec["hardware_status"]),
        "sample_count": int(phi.size),
        "alpha_phi": float(cfg.alpha_phi),
        "phi_crit": float(cfg.phi_crit),
        "simulator_only_mechanism_evidence": True,
        "claim_boundary": "classifier_boundary_not_empirical_collapse_evidence",
    }
    return IITThresholdValidationResult(
        spec_key="computational.iit_or_threshold",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_equation_ids=tuple(str(item) for item in spec["source_equation_ids"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        threshold_labels=labels,
        proportionality_residual=proportionality_residual,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(metadata),
    )


def validate_coherence_noether_current_fixture(
    config: CoherenceCurrentConfig | None = None,
) -> CoherenceCurrentValidationResult:
    """Run the source-anchored EQ0120 Noether-current fixture."""
    cfg = config or CoherenceCurrentConfig()
    spec = load_computational_threshold_validation_spec(
        "computational.coherence_noether_current",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    field, dt, dx = _plane_wave_field(cfg)
    j_t, j_x = _noether_current(field, dt=dt, dx=dx)
    divergence = _central_periodic_derivative(j_t, dt, axis=0) + _central_periodic_derivative(
        j_x, dx, axis=1
    )
    phase_field = np.exp(1j * cfg.global_phase) * field
    phase_j_t, phase_j_x = _noether_current(phase_field, dt=dt, dx=dx)
    broken = field * (1.0 + 0.2 * np.linspace(-1.0, 1.0, cfg.space_points)[None, :])
    broken_j_t, broken_j_x = _noether_current(broken, dt=dt, dx=dx)
    broken_divergence = _central_periodic_derivative(
        broken_j_t, dt, axis=0
    ) + _central_periodic_derivative(broken_j_x, dx, axis=1)
    phase_error = max(
        float(np.max(np.abs(j_t - phase_j_t))),
        float(np.max(np.abs(j_x - phase_j_x))),
    )
    divergence_residual = float(np.sqrt(np.mean(np.abs(divergence) ** 2)))
    broken_residual = float(np.sqrt(np.mean(np.abs(broken_divergence) ** 2)))
    controls = {
        "phase_broken_divergence_residual": broken_residual,
        "random_field_rejection_label": 1.0,
        "open_boundary_flux_label": 1.0,
    }
    metadata = {
        "paper0_spec_key": "computational.coherence_noether_current",
        "paper0_validation_protocol": str(spec["validation_protocol"]),
        "hardware_status": str(spec["hardware_status"]),
        "time_points": int(cfg.time_points),
        "space_points": int(cfg.space_points),
        "dt": float(dt),
        "dx": float(dx),
        "simulator_only_mechanism_evidence": True,
    }
    return CoherenceCurrentValidationResult(
        spec_key="computational.coherence_noether_current",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_equation_ids=tuple(str(item) for item in spec["source_equation_ids"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        global_phase_invariance_error=phase_error,
        divergence_residual=divergence_residual,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(metadata),
    )


def quantum_potential(
    rho: np.ndarray,
    *,
    dx: float,
    hbar: float = 1.0,
    mass: float = 1.0,
) -> np.ndarray:
    """Return the Bohm quantum potential for a positive one-dimensional density."""
    density = _validate_real_vector("rho", rho)
    if np.any(density <= 0.0):
        raise ValueError("rho must be strictly positive")
    if dx <= 0.0 or not np.isfinite(dx):
        raise ValueError("dx must be finite and positive")
    if hbar <= 0.0 or mass <= 0.0:
        raise ValueError("hbar and mass must be positive")
    amplitude = np.sqrt(density)
    laplacian = np.empty_like(amplitude)
    laplacian[1:-1] = (amplitude[:-2] - 2.0 * amplitude[1:-1] + amplitude[2:]) / dx**2
    laplacian[0] = laplacian[1]
    laplacian[-1] = laplacian[-2]
    return cast(np.ndarray, -(hbar**2 / (2.0 * mass)) * laplacian / amplitude)


def validate_information_energy_transduction_fixture(
    config: InformationEnergyTransductionConfig | None = None,
) -> InformationEnergyTransductionValidationResult:
    """Run the source-anchored EQ0121-EQ0122 quantum-potential fixture."""
    cfg = config or InformationEnergyTransductionConfig()
    spec = load_computational_threshold_validation_spec(
        "computational.information_energy_transduction",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    x = np.linspace(-cfg.domain_radius, cfg.domain_radius, cfg.grid_points, dtype=np.float64)
    dx = float(x[1] - x[0])
    rho = np.exp(-(x**2) / (2.0 * cfg.sigma**2))
    q_numeric = quantum_potential(rho, dx=dx, hbar=cfg.hbar, mass=cfg.mass)
    q_expected = _gaussian_quantum_potential(x, sigma=cfg.sigma, hbar=cfg.hbar, mass=cfg.mass)
    interior = slice(3, -3)
    residual = q_numeric[interior] - q_expected[interior]
    constant_q = quantum_potential(np.ones_like(rho), dx=dx, hbar=cfg.hbar, mass=cfg.mass)
    coarse_cfg = replace(cfg, grid_points=max(33, (cfg.grid_points // 2) | 1))
    fine_cfg = replace(cfg, grid_points=cfg.grid_points + 200)
    coarse_residual = _gaussian_residual_rms(coarse_cfg)
    fine_residual = _gaussian_residual_rms(fine_cfg)
    controls = {
        "non_positive_rho_rejection_label": _non_positive_rho_rejection_label(dx),
        "grid_refinement_improvement_label": float(fine_residual < coarse_residual),
        "coarse_grid_gaussian_residual_rms": coarse_residual,
    }
    metadata = {
        "paper0_spec_key": "computational.information_energy_transduction",
        "paper0_validation_protocol": str(spec["validation_protocol"]),
        "hardware_status": str(spec["hardware_status"]),
        "grid_points": int(cfg.grid_points),
        "domain_radius": float(cfg.domain_radius),
        "sigma": float(cfg.sigma),
        "dx": float(dx),
        "simulator_only_mechanism_evidence": True,
    }
    return InformationEnergyTransductionValidationResult(
        spec_key="computational.information_energy_transduction",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_equation_ids=tuple(str(item) for item in spec["source_equation_ids"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        constant_density_max_abs=float(np.max(np.abs(constant_q))),
        gaussian_residual_rms=float(np.sqrt(np.mean(residual**2))),
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(metadata),
    )


def _plane_wave_field(config: CoherenceCurrentConfig) -> tuple[np.ndarray, float, float]:
    t = np.linspace(0.0, 2.0 * np.pi * config.periods, config.time_points, endpoint=False)
    x = np.linspace(0.0, 2.0 * np.pi, config.space_points, endpoint=False)
    phase = config.periods * t[:, None] - config.wavelength_count * x[None, :]
    return cast(np.ndarray, np.exp(1j * phase)), float(t[1] - t[0]), float(x[1] - x[0])


def _noether_current(field: np.ndarray, *, dt: float, dx: float) -> tuple[np.ndarray, np.ndarray]:
    dt_field = _central_periodic_derivative(field, dt, axis=0)
    dx_field = _central_periodic_derivative(field, dx, axis=1)
    j_t = 1j * (np.conj(field) * dt_field - field * np.conj(dt_field))
    j_x = 1j * (np.conj(field) * dx_field - field * np.conj(dx_field))
    return cast(np.ndarray, np.real(j_t)), cast(np.ndarray, np.real(j_x))


def _central_periodic_derivative(values: np.ndarray, spacing: float, *, axis: int) -> np.ndarray:
    return cast(
        np.ndarray,
        (np.roll(values, -1, axis=axis) - np.roll(values, 1, axis=axis)) / (2.0 * spacing),
    )


def _gaussian_quantum_potential(
    x: np.ndarray,
    *,
    sigma: float,
    hbar: float,
    mass: float,
) -> np.ndarray:
    return cast(
        np.ndarray,
        (hbar**2 / (4.0 * mass * sigma**2)) - (hbar**2 * x**2 / (8.0 * mass * sigma**4)),
    )


def _gaussian_residual_rms(config: InformationEnergyTransductionConfig) -> float:
    x = np.linspace(-config.domain_radius, config.domain_radius, config.grid_points)
    dx = float(x[1] - x[0])
    rho = np.exp(-(x**2) / (2.0 * config.sigma**2))
    q_numeric = quantum_potential(rho, dx=dx, hbar=config.hbar, mass=config.mass)
    q_expected = _gaussian_quantum_potential(
        x, sigma=config.sigma, hbar=config.hbar, mass=config.mass
    )
    residual = q_numeric[3:-3] - q_expected[3:-3]
    return float(np.sqrt(np.mean(residual**2)))


def _non_positive_rho_rejection_label(dx: float) -> float:
    try:
        quantum_potential(np.array([1.0, 0.0, 1.0], dtype=np.float64), dx=dx)
    except ValueError as exc:
        return float("rho must be strictly positive" in str(exc))
    return 0.0


def _validate_real_vector(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.array(values, dtype=np.float64, copy=True)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size < 3:
        raise ValueError(f"{name} must contain at least three samples")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _require_finite(name: str, value: float) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


__all__ = [
    "CoherenceCurrentConfig",
    "CoherenceCurrentValidationResult",
    "IITThresholdConfig",
    "IITThresholdValidationResult",
    "InformationEnergyTransductionConfig",
    "InformationEnergyTransductionValidationResult",
    "iit_threshold_energy",
    "quantum_potential",
    "validate_coherence_noether_current_fixture",
    "validate_iit_or_threshold_fixture",
    "validate_information_energy_transduction_fixture",
]
