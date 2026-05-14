# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Stuart-Landau precision fixtures
"""Executable simulator fixtures for Paper 0 Stuart-Landau precision records."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from .spec_loader import load_stuart_landau_precision_validation_spec


@dataclass(frozen=True, slots=True)
class StuartLandauPrecisionConfig:
    """Finite real-coupled Stuart-Landau network parameters."""

    radius: np.ndarray | None = None
    theta: np.ndarray | None = None
    rho: np.ndarray | None = None
    omega: np.ndarray | None = None
    coupling: np.ndarray | None = None
    eta_radius: np.ndarray | None = None
    eta_theta: np.ndarray | None = None
    rho_gain_delta: float = 0.25
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        radius = _validate_positive_vector(
            "radius",
            self.radius
            if self.radius is not None
            else np.array([0.6, 1.8, 0.9], dtype=np.float64),
        )
        theta = _validate_vector(
            "theta",
            self.theta if self.theta is not None else np.array([0.1, 0.7, -0.2], dtype=np.float64),
            expected_size=radius.size,
        )
        rho = _validate_vector(
            "rho",
            self.rho if self.rho is not None else np.array([0.1, 0.4, 0.2], dtype=np.float64),
            expected_size=radius.size,
        )
        omega = _validate_vector(
            "omega",
            self.omega
            if self.omega is not None
            else np.array([0.05, -0.02, 0.01], dtype=np.float64),
            expected_size=radius.size,
        )
        coupling = _validate_coupling(
            self.coupling
            if self.coupling is not None
            else np.array([[0.0, 0.4, 0.2], [0.4, 0.0, 0.3], [0.2, 0.3, 0.0]])
        )
        eta_radius = _validate_vector(
            "eta_radius",
            self.eta_radius if self.eta_radius is not None else np.zeros(radius.size),
            expected_size=radius.size,
        )
        eta_theta = _validate_vector(
            "eta_theta",
            self.eta_theta if self.eta_theta is not None else np.zeros(radius.size),
            expected_size=radius.size,
        )
        if coupling.shape != (radius.size, radius.size):
            raise ValueError("coupling must match radius dimension")
        if not np.isfinite(self.rho_gain_delta) or self.rho_gain_delta <= 0.0:
            raise ValueError("rho_gain_delta must be finite and positive")
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "coupling", coupling)
        object.__setattr__(self, "eta_radius", eta_radius)
        object.__setattr__(self, "eta_theta", eta_theta)


@dataclass(frozen=True, slots=True)
class StuartLandauPolarRates:
    """Polar radial and phase rates for the finite Stuart-Landau network."""

    radius_dot: np.ndarray
    theta_dot: np.ndarray
    amplitude_ratio: np.ndarray
    weighted_phase_residual: np.ndarray
    radial_coupling: np.ndarray


@dataclass(frozen=True, slots=True)
class PrecisionWeightedPhaseTerms:
    """Precision-weighted phase terms and uniform-amplitude comparison."""

    weighted_phase_drive: np.ndarray
    uniform_phase_drive: np.ndarray
    max_phase_ratio_residual: float
    max_amplitude_ratio_deviation: float
    uniform_amplitude_ratio_deviation: float


@dataclass(frozen=True, slots=True)
class StuartLandauPrecisionUpgradeValidationResult:
    """Source-anchored complex-to-polar validation result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    max_complex_polar_residual: float
    phase_only_missing_amplitude_norm: float
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class StuartLandauPrecisionWeightedDynamicsValidationResult:
    """Source-anchored polar precision-weighting validation result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    max_phase_ratio_residual: float
    max_amplitude_ratio_deviation: float
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class SalienceRadialPrecisionControlValidationResult:
    """Source-anchored radial-gain precision control validation result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    rho_gain_radius_dot_delta: float
    high_incoming_over_prior_phase_drive_ratio: float
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class StuartLandauPrecisionFixtureResult:
    """Combined Paper 0 Stuart-Landau precision fixture result."""

    upgrade: StuartLandauPrecisionUpgradeValidationResult
    dynamics: StuartLandauPrecisionWeightedDynamicsValidationResult
    salience: SalienceRadialPrecisionControlValidationResult


def complex_state(config: StuartLandauPrecisionConfig) -> np.ndarray:
    """Return Z_j = R_j exp(i theta_j)."""
    radius = cast(np.ndarray, config.radius)
    theta = cast(np.ndarray, config.theta)
    return cast(np.ndarray, radius * np.exp(1j * theta))


def complex_stuart_landau_derivative(config: StuartLandauPrecisionConfig) -> np.ndarray:
    """Return dZ/dt for the real-coupled complex Stuart-Landau network."""
    z = complex_state(config)
    radius = cast(np.ndarray, config.radius)
    rho = cast(np.ndarray, config.rho)
    omega = cast(np.ndarray, config.omega)
    coupling = cast(np.ndarray, config.coupling)
    eta_radius = cast(np.ndarray, config.eta_radius)
    eta_theta = cast(np.ndarray, config.eta_theta)
    polar_noise = np.exp(1j * cast(np.ndarray, config.theta)) * (
        eta_radius + 1j * radius * eta_theta
    )
    derivative = z * (rho + 1j * omega - radius * radius) + coupling @ z + polar_noise
    return cast(np.ndarray, derivative)


def polar_stuart_landau_rates(config: StuartLandauPrecisionConfig) -> StuartLandauPolarRates:
    """Return exact polar rates implied by the real-coupled complex equation."""
    radius = cast(np.ndarray, config.radius)
    theta = cast(np.ndarray, config.theta)
    rho = cast(np.ndarray, config.rho)
    omega = cast(np.ndarray, config.omega)
    coupling = cast(np.ndarray, config.coupling)
    eta_radius = cast(np.ndarray, config.eta_radius)
    eta_theta = cast(np.ndarray, config.eta_theta)
    delta = theta[None, :] - theta[:, None]
    amplitude_ratio = radius[None, :] / radius[:, None]
    weighted_phase_residual = np.sum(coupling * amplitude_ratio * np.sin(delta), axis=1)
    radial_coupling = np.sum(coupling * radius[None, :] * np.cos(delta), axis=1)
    radius_dot = radius * (rho - radius * radius) + radial_coupling + eta_radius
    theta_dot = omega + weighted_phase_residual + eta_theta
    return StuartLandauPolarRates(
        radius_dot=cast(np.ndarray, radius_dot),
        theta_dot=cast(np.ndarray, theta_dot),
        amplitude_ratio=cast(np.ndarray, amplitude_ratio),
        weighted_phase_residual=cast(np.ndarray, weighted_phase_residual),
        radial_coupling=cast(np.ndarray, radial_coupling),
    )


def precision_weighted_phase_terms(
    config: StuartLandauPrecisionConfig,
) -> PrecisionWeightedPhaseTerms:
    """Return amplitude-weighted phase drive and uniform-amplitude comparison."""
    rates = polar_stuart_landau_rates(config)
    radius = cast(np.ndarray, config.radius)
    theta = cast(np.ndarray, config.theta)
    coupling = cast(np.ndarray, config.coupling)
    delta = theta[None, :] - theta[:, None]
    uniform_ratio = np.ones_like(rates.amplitude_ratio)
    uniform_drive = np.sum(coupling * uniform_ratio * np.sin(delta), axis=1)
    expected = np.sum(coupling * (radius[None, :] / radius[:, None]) * np.sin(delta), axis=1)
    uniform_config = replace(config, radius=np.ones_like(radius))
    uniform_rates = polar_stuart_landau_rates(uniform_config)
    return PrecisionWeightedPhaseTerms(
        weighted_phase_drive=rates.weighted_phase_residual,
        uniform_phase_drive=cast(np.ndarray, uniform_drive),
        max_phase_ratio_residual=float(np.max(np.abs(rates.weighted_phase_residual - expected))),
        max_amplitude_ratio_deviation=float(np.max(np.abs(rates.amplitude_ratio - 1.0))),
        uniform_amplitude_ratio_deviation=float(
            np.max(np.abs(uniform_rates.amplitude_ratio - 1.0))
        ),
    )


def validate_stuart_landau_precision_upgrade_fixture(
    config: StuartLandauPrecisionConfig | None = None,
) -> StuartLandauPrecisionUpgradeValidationResult:
    """Run the source-anchored complex-to-polar Stuart-Landau fixture."""
    cfg = config or StuartLandauPrecisionConfig()
    spec = load_stuart_landau_precision_validation_spec(
        "computational.stuart_landau_precision_upgrade",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    complex_derivative = complex_stuart_landau_derivative(cfg)
    rates = polar_stuart_landau_rates(cfg)
    reconstructed = np.exp(1j * cast(np.ndarray, cfg.theta)) * (
        rates.radius_dot + 1j * cast(np.ndarray, cfg.radius) * rates.theta_dot
    )
    phase_terms = precision_weighted_phase_terms(cfg)
    controls = {
        "zero_radius_rejection_label": _zero_radius_rejection_label(),
        "non_finite_complex_state_rejection_label": _non_finite_state_rejection_label(),
        "phase_only_missing_amplitude_norm": float(
            np.linalg.norm(phase_terms.weighted_phase_drive - phase_terms.uniform_phase_drive)
        ),
    }
    return StuartLandauPrecisionUpgradeValidationResult(
        spec_key="computational.stuart_landau_precision_upgrade",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        max_complex_polar_residual=float(np.max(np.abs(complex_derivative - reconstructed))),
        phase_only_missing_amplitude_norm=controls["phase_only_missing_amplitude_norm"],
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(
            {
                "simulator_only_mechanism_evidence": True,
                "claim_boundary": "finite_stuart_landau_identity_not_biological_confirmation",
            }
        ),
    )


def validate_stuart_landau_precision_weighted_dynamics_fixture(
    config: StuartLandauPrecisionConfig | None = None,
) -> StuartLandauPrecisionWeightedDynamicsValidationResult:
    """Run the source-anchored precision-weighted polar dynamics fixture."""
    cfg = config or StuartLandauPrecisionConfig()
    spec = load_stuart_landau_precision_validation_spec(
        "computational.precision_weighted_phase_amplitude_dynamics",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    terms = precision_weighted_phase_terms(cfg)
    controls = {
        "uniform_amplitude_ratio_deviation": terms.uniform_amplitude_ratio_deviation,
        "negative_amplitude_rejection_label": _negative_amplitude_rejection_label(),
        "mismatched_shape_rejection_label": _mismatched_shape_rejection_label(),
    }
    return StuartLandauPrecisionWeightedDynamicsValidationResult(
        spec_key="computational.precision_weighted_phase_amplitude_dynamics",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        max_phase_ratio_residual=terms.max_phase_ratio_residual,
        max_amplitude_ratio_deviation=terms.max_amplitude_ratio_deviation,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(
            {
                "simulator_only_mechanism_evidence": True,
                "claim_boundary": "finite_polar_precision_fixture_not_active_inference_proof",
            }
        ),
    )


def validate_salience_radial_precision_control_fixture(
    config: StuartLandauPrecisionConfig | None = None,
) -> SalienceRadialPrecisionControlValidationResult:
    """Run the source-anchored radial precision-control fixture."""
    cfg = config or StuartLandauPrecisionConfig()
    spec = load_stuart_landau_precision_validation_spec(
        "computational.salience_radial_precision_control",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    rates = polar_stuart_landau_rates(cfg)
    boosted_rho = cast(np.ndarray, cfg.rho).copy()
    boosted_rho[0] += cfg.rho_gain_delta
    boosted_rates = polar_stuart_landau_rates(replace(cfg, rho=boosted_rho))
    radius = cast(np.ndarray, cfg.radius)
    theta = cast(np.ndarray, cfg.theta)
    coupling = cast(np.ndarray, cfg.coupling)
    incoming_drive = abs(coupling[0, 1] * (radius[1] / radius[0]) * np.sin(theta[1] - theta[0]))
    prior_drive = abs(coupling[1, 0] * (radius[0] / radius[1]) * np.sin(theta[0] - theta[1]))
    denominator = max(float(prior_drive), np.finfo(np.float64).eps)
    direct_phase_rates = polar_stuart_landau_rates(replace(cfg, rho=boosted_rho)).theta_dot
    controls = {
        "direct_phase_salience_delta_abs": float(
            np.max(np.abs(direct_phase_rates - rates.theta_dot))
        ),
        "zero_incoming_amplitude_rejection_label": _zero_radius_rejection_label(),
        "non_finite_rho_rejection_label": _non_finite_rho_rejection_label(),
    }
    return SalienceRadialPrecisionControlValidationResult(
        spec_key="computational.salience_radial_precision_control",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        rho_gain_radius_dot_delta=float(boosted_rates.radius_dot[0] - rates.radius_dot[0]),
        high_incoming_over_prior_phase_drive_ratio=float(incoming_drive / denominator),
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(
            {
                "simulator_only_mechanism_evidence": True,
                "claim_boundary": "finite_radial_gain_fixture_not_salience_network_confirmation",
            }
        ),
    )


def validate_stuart_landau_precision_fixture() -> StuartLandauPrecisionFixtureResult:
    """Run all source-anchored Stuart-Landau precision fixtures."""
    return StuartLandauPrecisionFixtureResult(
        upgrade=validate_stuart_landau_precision_upgrade_fixture(),
        dynamics=validate_stuart_landau_precision_weighted_dynamics_fixture(),
        salience=validate_salience_radial_precision_control_fixture(),
    )


def _zero_radius_rejection_label() -> float:
    try:
        StuartLandauPrecisionConfig(radius=np.array([1.0, 0.0, 0.8]))
    except ValueError as exc:
        return float("strictly positive" in str(exc))
    return 0.0


def _negative_amplitude_rejection_label() -> float:
    try:
        StuartLandauPrecisionConfig(radius=np.array([1.0, -0.1, 0.8]))
    except ValueError as exc:
        return float("strictly positive" in str(exc))
    return 0.0


def _non_finite_state_rejection_label() -> float:
    try:
        StuartLandauPrecisionConfig(theta=np.array([0.0, np.nan, 0.2]))
    except ValueError as exc:
        return float("finite" in str(exc))
    return 0.0


def _mismatched_shape_rejection_label() -> float:
    try:
        StuartLandauPrecisionConfig(radius=np.array([1.0, 0.8]), theta=np.array([0.0, 0.1, 0.2]))
    except ValueError as exc:
        return float("size" in str(exc))
    return 0.0


def _non_finite_rho_rejection_label() -> float:
    try:
        StuartLandauPrecisionConfig(rho=np.array([0.1, np.inf, 0.2]))
    except ValueError as exc:
        return float("finite" in str(exc))
    return 0.0


def _validate_vector(name: str, values: np.ndarray, *, expected_size: int) -> np.ndarray:
    arr = np.array(values, dtype=np.float64, copy=True)
    if arr.ndim != 1 or arr.size != expected_size:
        raise ValueError(f"{name} must be a one-dimensional vector of size {expected_size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _validate_positive_vector(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.array(values, dtype=np.float64, copy=True)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"{name} must be a one-dimensional vector with at least two entries")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return arr


def _validate_coupling(values: np.ndarray) -> np.ndarray:
    arr = np.array(values, dtype=np.float64, copy=True)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("coupling must be square")
    if not np.all(np.isfinite(arr)):
        raise ValueError("coupling must contain only finite values")
    if not np.allclose(arr, arr.T, atol=1.0e-12, rtol=0.0):
        raise ValueError("coupling must be symmetric")
    return arr


__all__ = [
    "PrecisionWeightedPhaseTerms",
    "SalienceRadialPrecisionControlValidationResult",
    "StuartLandauPolarRates",
    "StuartLandauPrecisionConfig",
    "StuartLandauPrecisionFixtureResult",
    "StuartLandauPrecisionUpgradeValidationResult",
    "StuartLandauPrecisionWeightedDynamicsValidationResult",
    "complex_state",
    "complex_stuart_landau_derivative",
    "polar_stuart_landau_rates",
    "precision_weighted_phase_terms",
    "validate_salience_radial_precision_control_fixture",
    "validate_stuart_landau_precision_fixture",
    "validate_stuart_landau_precision_upgrade_fixture",
    "validate_stuart_landau_precision_weighted_dynamics_fixture",
]
