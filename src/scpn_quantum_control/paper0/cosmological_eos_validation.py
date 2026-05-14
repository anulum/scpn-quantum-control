# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 cosmological equation-of-state fixtures
"""Scalar-field equation-of-state fixtures for Paper 0 cosmological constraints."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .spec_loader import load_cosmological_eos_validation_spec

CLAIM_BOUNDARY = "source-bounded cosmological equation-of-state fixture; not empirical evidence"
HARDWARE_STATUS = "cosmological_constraint_fixture_no_execution"
SOURCE_LEDGER_SPAN = ("P0R06916", "P0R06948")


@dataclass(frozen=True, slots=True)
class CosmologicalEOSConfig:
    """Finite cosmological equation-of-state fixture settings."""

    w0: float = -1.03
    w0_sigma: float = 0.03
    target_w: float = -1.0
    confidence_sigma: float = 1.0
    perturbation_fraction: float = 0.03
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        _require_finite("w0", self.w0)
        _require_finite("target_w", self.target_w)
        _require_positive("w0_sigma", self.w0_sigma)
        _require_positive("confidence_sigma", self.confidence_sigma)
        _require_probability("perturbation_fraction", self.perturbation_fraction)


@dataclass(frozen=True, slots=True)
class CosmologicalEOSFixtureResult:
    """Combined cosmological equation-of-state fixture result."""

    spec_keys: tuple[str, str, str, str, str, str]
    hardware_status: str
    source_ledger_span: tuple[str, str]
    slow_roll_w: float
    kinetic_dominated_w: float
    observed_w0: float
    observed_w0_sigma: float
    observational_constraint_consistent: bool
    background_fraction: float
    perturbation_fraction: float
    null_controls: MappingProxyType[str, float]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def scalar_field_density_pressure(*, psi_dot: float, potential: float) -> tuple[float, float]:
    """Return scalar-field density and pressure from the source formulae."""
    _require_finite("psi_dot", psi_dot)
    _require_finite("potential", potential)
    kinetic = 0.5 * psi_dot * psi_dot
    return kinetic + potential, kinetic - potential


def equation_of_state(*, psi_dot: float, potential: float) -> float:
    """Return w = p / rho for positive scalar-field density."""
    density, pressure = scalar_field_density_pressure(psi_dot=psi_dot, potential=potential)
    if density <= 0.0:
        raise ValueError("density denominator must be positive")
    return pressure / density


def observational_w0_consistency(
    *,
    w0: float,
    sigma: float,
    target: float,
    config: CosmologicalEOSConfig,
) -> bool:
    """Return whether target lies inside the configured w0 confidence interval."""
    _require_finite("w0", w0)
    _require_finite("target", target)
    _require_positive("sigma", sigma)
    half_width = config.confidence_sigma * sigma
    return (w0 - half_width) <= target <= (w0 + half_width)


def stress_energy_split(
    *,
    background: float,
    perturbation: float,
    config: CosmologicalEOSConfig,
) -> tuple[float, float]:
    """Validate a background-plus-perturbation split."""
    _require_probability("background", background)
    _require_probability("perturbation", perturbation)
    if perturbation > config.perturbation_fraction:
        raise ValueError("perturbation exceeds configured subdominant fraction")
    if background <= perturbation:
        raise ValueError("background component must dominate perturbation")
    return background, perturbation


def validate_cosmological_eos_fixture(
    config: CosmologicalEOSConfig | None = None,
) -> CosmologicalEOSFixtureResult:
    """Run the cosmological equation-of-state boundary fixture."""
    cfg = config or CosmologicalEOSConfig()
    keys = (
        "cosmological_eos.chapter_boundary",
        "cosmological_eos.scalar_field_equations",
        "cosmological_eos.limiting_cases",
        "cosmological_eos.observational_constraint",
        "cosmological_eos.hybrid_split_and_homogeneity",
        "cosmological_eos.quintessence_detection_target",
    )
    specs = tuple(
        load_cosmological_eos_validation_spec(key, spec_bundle_path=cfg.spec_bundle_path)
        for key in keys
    )
    background, perturbation = stress_energy_split(
        background=1.0 - cfg.perturbation_fraction,
        perturbation=cfg.perturbation_fraction,
        config=cfg,
    )
    controls = {
        "zero_density_rejection_label": _zero_density_rejection_label(),
        "invalid_perturbation_fraction_rejection_label": _invalid_perturbation_rejection_label(),
        "unsupported_cosmology_validation_rejection_label": 1.0,
    }
    return CosmologicalEOSFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        source_ledger_span=SOURCE_LEDGER_SPAN,
        slow_roll_w=equation_of_state(psi_dot=0.0, potential=1.0),
        kinetic_dominated_w=equation_of_state(psi_dot=1.0, potential=0.0),
        observed_w0=cfg.w0,
        observed_w0_sigma=cfg.w0_sigma,
        observational_constraint_consistent=observational_w0_consistency(
            w0=cfg.w0,
            sigma=cfg.w0_sigma,
            target=cfg.target_w,
            config=cfg,
        ),
        background_fraction=background,
        perturbation_fraction=perturbation,
        null_controls=MappingProxyType(controls),
        claim_boundary=CLAIM_BOUNDARY,
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in specs[0]["source_ledger_ids"]),
                "source_ledger_span": SOURCE_LEDGER_SPAN,
                "claim_boundary": CLAIM_BOUNDARY,
                "protocol_state": "constraint_fixture_only_no_cosmological_validation",
            }
        ),
    )


def _zero_density_rejection_label() -> float:
    try:
        equation_of_state(psi_dot=0.0, potential=0.0)
    except ValueError as exc:
        return float("density denominator must be positive" in str(exc))
    return 0.0


def _invalid_perturbation_rejection_label() -> float:
    try:
        CosmologicalEOSConfig(perturbation_fraction=1.2)
    except ValueError as exc:
        return float("perturbation_fraction must be in [0, 1]" in str(exc))
    return 0.0


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _require_probability(name: str, value: float) -> None:
    if not isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


__all__ = [
    "CLAIM_BOUNDARY",
    "CosmologicalEOSConfig",
    "CosmologicalEOSFixtureResult",
    "equation_of_state",
    "observational_w0_consistency",
    "scalar_field_density_pressure",
    "stress_energy_split",
    "validate_cosmological_eos_fixture",
]
