# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 symmetry-restoration validation fixtures
"""Simulator-only MMC symmetry-restoration fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .spec_loader import load_symmetry_restoration_validation_spec

CLAIM_BOUNDARY = "source-bounded symmetry-restoration simulator contract; not empirical evidence"
SOURCE_LEDGER_SPAN = ("P0R06324", "P0R06338")
VEV_FORMULAE = (
    "lim_{t -> infinity} v(t) = 0",
    "m_A = g v; m_h = sqrt(2 lambda) v",
)


@dataclass(frozen=True, slots=True)
class SymmetryRestorationConfig:
    """Finite simulator settings for symmetry-restoration fixtures."""

    conformal_rescaling_weight: float = 0.34
    l15_preservation_weight: float = 0.33
    entropy_reset_weight: float = 0.33
    conformal_threshold: float = 0.72
    violation_threshold: float = 0.65
    restoration_threshold: float = 0.70
    vev_limit_threshold: float = 0.95
    legal_rescaling_threshold: float = 0.72
    massless_tolerance: float = 1.0e-9
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        for name in (
            "conformal_threshold",
            "violation_threshold",
            "restoration_threshold",
            "vev_limit_threshold",
            "legal_rescaling_threshold",
            "massless_tolerance",
        ):
            _require_positive(name, float(getattr(self, name)))
        for name in (
            "conformal_rescaling_weight",
            "l15_preservation_weight",
            "entropy_reset_weight",
        ):
            _require_non_negative(name, float(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class VevMeltingValidationResult:
    """VEV-melting and massless-limit result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    source_formulae: tuple[str, str]
    vev_limit_score: float
    infoton_mass_limit: float
    psi_higgs_mass_limit: float
    claim_boundary: str
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class SymmetryRestorationFixtureResult:
    """Combined symmetry-restoration fixture result."""

    spec_keys: tuple[str, str, str, str, str]
    hardware_status: str
    conformal_score: float
    violation_score: float
    restored_violation_score: float
    quadratic_coefficient_broken: float
    quadratic_coefficient_restored: float
    restoration_score: float
    vev_melting: VevMeltingValidationResult
    legal_rescaling_score: float
    config_thresholds: MappingProxyType[str, float]
    source_ledger_span: tuple[str, str]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def conformal_geometry_score(
    *,
    conformal_rescaling: float,
    l15_preservation: float,
    entropy_reset: float,
    config: SymmetryRestorationConfig,
) -> float:
    """Score source MMC conformal-geometry channels."""
    values = np.asarray([conformal_rescaling, l15_preservation, entropy_reset], dtype=np.float64)
    _require_unit_interval("conformal-geometry inputs", values)
    weights = np.asarray(
        [
            config.conformal_rescaling_weight,
            config.l15_preservation_weight,
            config.entropy_reset_weight,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("conformal-geometry weights must be finite and non-negative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("at least one conformal-geometry weight must be positive")
    return float(np.dot(weights, values) / total)


def conformal_boundary_violation_score(
    *,
    psi_mass_retained: float,
    physical_scale_retained: float,
    conformal_invariance: float,
    config: SymmetryRestorationConfig,
) -> float:
    """Score retained mass or scale as a conformal-boundary violation."""
    values = np.asarray(
        [psi_mass_retained, physical_scale_retained, conformal_invariance],
        dtype=np.float64,
    )
    _require_unit_interval("conformal-boundary inputs", values)
    invariance_defect = 1.0 - conformal_invariance
    return float(
        0.40 * psi_mass_retained + 0.35 * physical_scale_retained + 0.25 * invariance_defect
    )


def effective_quadratic_coefficient(
    *,
    mu_squared: float,
    c1_tds_squared: float,
    c2_f_r: float,
) -> float:
    """Return the source effective-potential quadratic coefficient."""
    values = np.asarray([mu_squared, c1_tds_squared, c2_f_r], dtype=np.float64)
    _require_non_negative_vector("effective-potential coefficients", values)
    return float(-mu_squared + c1_tds_squared + c2_f_r)


def symmetry_restoration_score(
    *,
    thermal_correction: float,
    geometric_correction: float,
    negative_mass_squared: float,
    config: SymmetryRestorationConfig,
) -> float:
    """Score whether thermal and geometric corrections overcome negative mass squared."""
    values = np.asarray(
        [thermal_correction, geometric_correction, negative_mass_squared],
        dtype=np.float64,
    )
    _require_non_negative_vector("symmetry-restoration inputs", values)
    if negative_mass_squared == 0.0:
        raise ValueError("negative_mass_squared must be finite and positive")
    return float((thermal_correction + geometric_correction) / negative_mass_squared)


def massless_limit_scores(
    *,
    v: float,
    gauge_coupling: float,
    lambda_coupling: float,
    config: SymmetryRestorationConfig,
) -> tuple[float, float, float]:
    """Return VEV-limit score and source mass limits."""
    values = np.asarray([v, gauge_coupling, lambda_coupling], dtype=np.float64)
    _require_non_negative_vector("massless-limit inputs", values)
    infoton_mass = float(gauge_coupling * v)
    psi_higgs_mass = float(np.sqrt(2.0 * lambda_coupling) * v)
    limit_score = float(max(0.0, 1.0 - v / config.massless_tolerance))
    return limit_score, infoton_mass, psi_higgs_mass


def legal_conformal_rescaling_score(
    *,
    scale_shedding: float,
    dimensionless_ethical_functional: float,
    metric_reset: float,
    config: SymmetryRestorationConfig,
) -> float:
    """Score legal conformal-rescaling source channels."""
    values = np.asarray(
        [scale_shedding, dimensionless_ethical_functional, metric_reset],
        dtype=np.float64,
    )
    _require_unit_interval("legal-conformal-rescaling inputs", values)
    return float(np.dot(np.asarray([0.34, 0.33, 0.33], dtype=np.float64), values))


def validate_vev_melting_fixture(
    config: SymmetryRestorationConfig | None = None,
) -> VevMeltingValidationResult:
    """Run the VEV-melting and massless-limit fixture."""
    cfg = config or SymmetryRestorationConfig()
    spec = load_symmetry_restoration_validation_spec(
        "symmetry_restoration.vev_melting_massless_limit",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    vev_score, infoton_mass, psi_higgs_mass = massless_limit_scores(
        v=0.0,
        gauge_coupling=0.71,
        lambda_coupling=0.43,
        config=cfg,
    )
    source_formulae = tuple(str(item) for item in spec["source_formulae"])
    if len(source_formulae) != 2:
        raise ValueError("VEV-melting spec must preserve exactly two source formulae")
    preserved_formulae = (source_formulae[0], source_formulae[1])
    controls = {
        "nonzero_vev_rejection_label": _nonzero_vev_rejection_label(),
        "negative_lambda_rejection_label": _negative_lambda_rejection_label(),
        "unsupported_empirical_boundary_rejection_label": 1.0,
    }
    return VevMeltingValidationResult(
        spec_key="symmetry_restoration.vev_melting_massless_limit",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        source_formulae=preserved_formulae,
        vev_limit_score=vev_score,
        infoton_mass_limit=infoton_mass,
        psi_higgs_mass_limit=psi_higgs_mass,
        claim_boundary=CLAIM_BOUNDARY,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(
            {
                "vev_limit_threshold": cfg.vev_limit_threshold,
                "massless_tolerance": cfg.massless_tolerance,
                "source_ledger_span": SOURCE_LEDGER_SPAN,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        ),
    )


def validate_symmetry_restoration_fixture(
    config: SymmetryRestorationConfig | None = None,
) -> SymmetryRestorationFixtureResult:
    """Run the combined symmetry-restoration fixture."""
    cfg = config or SymmetryRestorationConfig()
    keys = (
        "symmetry_restoration.mmc_conformal_geometry_boundary",
        "symmetry_restoration.conformal_boundary_masslessness_constraint",
        "symmetry_restoration.effective_potential_flip_boundary",
        "symmetry_restoration.vev_melting_massless_limit",
        "symmetry_restoration.legal_conformal_rescaling_boundary",
    )
    specs = tuple(
        load_symmetry_restoration_validation_spec(key, spec_bundle_path=cfg.spec_bundle_path)
        for key in keys
    )
    conformal = conformal_geometry_score(
        conformal_rescaling=0.84,
        l15_preservation=0.90,
        entropy_reset=0.76,
        config=cfg,
    )
    violation = conformal_boundary_violation_score(
        psi_mass_retained=0.92,
        physical_scale_retained=0.88,
        conformal_invariance=0.16,
        config=cfg,
    )
    restored_violation = conformal_boundary_violation_score(
        psi_mass_retained=0.08,
        physical_scale_retained=0.05,
        conformal_invariance=0.91,
        config=cfg,
    )
    broken = effective_quadratic_coefficient(
        mu_squared=0.80,
        c1_tds_squared=0.18,
        c2_f_r=0.12,
    )
    restored = effective_quadratic_coefficient(
        mu_squared=0.80,
        c1_tds_squared=0.47,
        c2_f_r=0.42,
    )
    restoration = symmetry_restoration_score(
        thermal_correction=0.47,
        geometric_correction=0.42,
        negative_mass_squared=0.80,
        config=cfg,
    )
    vev = validate_vev_melting_fixture(cfg)
    legal = legal_conformal_rescaling_score(
        scale_shedding=0.86,
        dimensionless_ethical_functional=0.88,
        metric_reset=0.81,
        config=cfg,
    )
    return SymmetryRestorationFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        conformal_score=conformal,
        violation_score=violation,
        restored_violation_score=restored_violation,
        quadratic_coefficient_broken=broken,
        quadratic_coefficient_restored=restored,
        restoration_score=restoration,
        vev_melting=vev,
        legal_rescaling_score=legal,
        config_thresholds=MappingProxyType(
            {
                "conformal_threshold": cfg.conformal_threshold,
                "violation_threshold": cfg.violation_threshold,
                "restoration_threshold": cfg.restoration_threshold,
                "vev_limit_threshold": cfg.vev_limit_threshold,
                "legal_rescaling_threshold": cfg.legal_rescaling_threshold,
            }
        ),
        source_ledger_span=SOURCE_LEDGER_SPAN,
        claim_boundary=CLAIM_BOUNDARY,
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in specs[0]["source_ledger_ids"]),
                "source_ledger_span": SOURCE_LEDGER_SPAN,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        ),
    )


def _require_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _require_non_negative(name: str, value: float) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _require_non_negative_vector(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")


def _require_unit_interval(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{name} must be in [0, 1]")


def _nonzero_vev_rejection_label() -> float:
    cfg = SymmetryRestorationConfig()
    score, infoton_mass, psi_higgs_mass = massless_limit_scores(
        v=1.0e-3,
        gauge_coupling=0.71,
        lambda_coupling=0.43,
        config=cfg,
    )
    return float(
        score < cfg.vev_limit_threshold
        and infoton_mass > cfg.massless_tolerance
        and psi_higgs_mass > cfg.massless_tolerance
    )


def _negative_lambda_rejection_label() -> float:
    try:
        massless_limit_scores(
            v=0.0,
            gauge_coupling=0.71,
            lambda_coupling=-0.43,
            config=SymmetryRestorationConfig(),
        )
    except ValueError as exc:
        return float("finite and non-negative" in str(exc))
    return 0.0


__all__ = [
    "CLAIM_BOUNDARY",
    "SymmetryRestorationConfig",
    "SymmetryRestorationFixtureResult",
    "VevMeltingValidationResult",
    "conformal_boundary_violation_score",
    "conformal_geometry_score",
    "effective_quadratic_coefficient",
    "legal_conformal_rescaling_score",
    "massless_limit_scores",
    "symmetry_restoration_score",
    "validate_symmetry_restoration_fixture",
    "validate_vev_melting_fixture",
]
