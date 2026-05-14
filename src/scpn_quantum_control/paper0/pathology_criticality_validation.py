# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 pathology/criticality fixtures
"""Executable simulator fixtures for Paper 0 pathology/criticality records."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .spec_loader import load_pathology_criticality_validation_spec

CLAIM_BOUNDARY = "simulator-only systems metric; not clinical evidence or medical advice"


@dataclass(frozen=True, slots=True)
class PathologyCriticalityConfig:
    """Finite systems-state observables for pathology/criticality fixtures."""

    free_energy: float = 1.1
    sigma: float = 1.35
    order_parameter: float = 0.45
    qec_success_probability: float = 0.55
    criticality_tolerance: float = 0.05
    free_energy_step: float = 0.2
    sigma_step: float = 0.15
    synchronisation_step: float = 0.1
    qec_step: float = 0.08
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        _require_non_negative("free_energy", self.free_energy)
        _require_positive("sigma", self.sigma)
        _require_unit_interval("order_parameter", self.order_parameter)
        _require_unit_interval("qec_success_probability", self.qec_success_probability)
        _require_non_negative("criticality_tolerance", self.criticality_tolerance)
        _require_non_negative("free_energy_step", self.free_energy_step)
        _require_non_negative("sigma_step", self.sigma_step)
        _require_non_negative("synchronisation_step", self.synchronisation_step)
        _require_non_negative("qec_step", self.qec_step)


@dataclass(frozen=True, slots=True)
class CoherenceBreakdownValidationResult:
    """Source-anchored finite pathology-index validation result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    pathology_index: float
    baseline_index: float
    index_delta_vs_baseline: float
    claim_boundary: str
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class CriticalityDeviationValidationResult:
    """Source-anchored sigma-classifier validation result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    sigma: float
    sigma_label: str
    criticality_distance: float
    claim_boundary: str
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class TherapeuticRestorationValidationResult:
    """Source-anchored simulator restoration-target validation result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    initial_index: float
    restored_index: float
    restoration_index_delta: float
    restored_sigma: float
    restored_order_parameter: float
    claim_boundary: str
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class PathologyCriticalityFixtureResult:
    """Combined Paper 0 pathology/criticality fixture result."""

    coherence: CoherenceBreakdownValidationResult
    criticality: CriticalityDeviationValidationResult
    restoration: TherapeuticRestorationValidationResult


def pathology_index(config: PathologyCriticalityConfig) -> float:
    """Return finite dimensionless dysregulation index for simulator states."""
    criticality_penalty = abs(np.log(config.sigma))
    synchronisation_loss = 1.0 - config.order_parameter
    qec_failure = 1.0 - config.qec_success_probability
    return float(config.free_energy + criticality_penalty + synchronisation_loss + qec_failure)


def classify_criticality(sigma: float, *, tolerance: float = 0.05) -> str:
    """Classify finite positive sigma around the quasicritical target sigma=1."""
    _require_positive("sigma", sigma)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    if sigma > 1.0 + tolerance:
        return "supercritical"
    if sigma < 1.0 - tolerance:
        return "subcritical"
    return "quasicritical"


def restore_toward_quasicriticality(
    config: PathologyCriticalityConfig,
) -> PathologyCriticalityConfig:
    """Apply one bounded simulator control step toward lower index."""
    sigma_delta = min(abs(config.sigma - 1.0), config.sigma_step)
    if config.sigma > 1.0:
        restored_sigma = config.sigma - sigma_delta
    elif config.sigma < 1.0:
        restored_sigma = config.sigma + sigma_delta
    else:
        restored_sigma = config.sigma
    return replace(
        config,
        free_energy=max(0.0, config.free_energy - config.free_energy_step),
        sigma=restored_sigma,
        order_parameter=min(1.0, config.order_parameter + config.synchronisation_step),
        qec_success_probability=min(1.0, config.qec_success_probability + config.qec_step),
    )


def validate_coherence_breakdown_index_fixture(
    config: PathologyCriticalityConfig | None = None,
) -> CoherenceBreakdownValidationResult:
    """Run the source-anchored finite pathology-index fixture."""
    cfg = config or PathologyCriticalityConfig()
    spec = load_pathology_criticality_validation_spec(
        "applied.pathology.coherence_breakdown_index",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    baseline = PathologyCriticalityConfig(
        free_energy=0.2,
        sigma=1.0,
        order_parameter=0.9,
        qec_success_probability=0.95,
        spec_bundle_path=cfg.spec_bundle_path,
    )
    index = pathology_index(cfg)
    baseline_index = pathology_index(baseline)
    controls = {
        "healthy_baseline_index": baseline_index,
        "non_finite_observable_rejection_label": _non_finite_observable_rejection_label(),
        "negative_probability_rejection_label": _negative_probability_rejection_label(),
    }
    return CoherenceBreakdownValidationResult(
        spec_key="applied.pathology.coherence_breakdown_index",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        pathology_index=index,
        baseline_index=baseline_index,
        index_delta_vs_baseline=index - baseline_index,
        claim_boundary=CLAIM_BOUNDARY,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(
            {
                "simulator_only_mechanism_evidence": True,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        ),
    )


def validate_criticality_deviation_classifier_fixture(
    config: PathologyCriticalityConfig | None = None,
) -> CriticalityDeviationValidationResult:
    """Run the source-anchored sigma-deviation classifier fixture."""
    cfg = config or PathologyCriticalityConfig()
    spec = load_pathology_criticality_validation_spec(
        "applied.pathology.criticality_deviation_classifier",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    label = classify_criticality(cfg.sigma, tolerance=cfg.criticality_tolerance)
    controls = {
        "sigma_neutral_label_is_quasicritical": float(
            classify_criticality(1.0, tolerance=cfg.criticality_tolerance) == "quasicritical"
        ),
        "non_positive_sigma_rejection_label": _non_positive_sigma_rejection_label(),
        "negative_tolerance_rejection_label": _negative_tolerance_rejection_label(),
    }
    return CriticalityDeviationValidationResult(
        spec_key="applied.pathology.criticality_deviation_classifier",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        sigma=cfg.sigma,
        sigma_label=label,
        criticality_distance=abs(cfg.sigma - 1.0),
        claim_boundary=CLAIM_BOUNDARY,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(
            {
                "simulator_only_mechanism_evidence": True,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        ),
    )


def validate_therapeutic_restoration_targets_fixture(
    config: PathologyCriticalityConfig | None = None,
) -> TherapeuticRestorationValidationResult:
    """Run the source-anchored simulator restoration-target fixture."""
    cfg = config or PathologyCriticalityConfig()
    spec = load_pathology_criticality_validation_spec(
        "applied.pathology.therapeutic_restoration_targets",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    restored = restore_toward_quasicriticality(cfg)
    initial_index = pathology_index(cfg)
    restored_index = pathology_index(restored)
    zero_update = replace(
        cfg,
        free_energy_step=0.0,
        sigma_step=0.0,
        synchronisation_step=0.0,
        qec_step=0.0,
    )
    wrong_direction = replace(
        cfg,
        free_energy=cfg.free_energy + cfg.free_energy_step,
        sigma=cfg.sigma + cfg.sigma_step
        if cfg.sigma >= 1.0
        else max(1.0e-9, cfg.sigma - cfg.sigma_step),
        order_parameter=max(0.0, cfg.order_parameter - cfg.synchronisation_step),
        qec_success_probability=max(0.0, cfg.qec_success_probability - cfg.qec_step),
    )
    controls = {
        "zero_update_index_delta_abs": abs(
            pathology_index(restore_toward_quasicriticality(zero_update)) - initial_index
        ),
        "wrong_direction_index_delta": pathology_index(wrong_direction) - initial_index,
        "out_of_range_order_parameter_rejection_label": _out_of_range_order_rejection_label(),
    }
    return TherapeuticRestorationValidationResult(
        spec_key="applied.pathology.therapeutic_restoration_targets",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        initial_index=initial_index,
        restored_index=restored_index,
        restoration_index_delta=restored_index - initial_index,
        restored_sigma=restored.sigma,
        restored_order_parameter=restored.order_parameter,
        claim_boundary=CLAIM_BOUNDARY,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(
            {
                "simulator_only_mechanism_evidence": True,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        ),
    )


def validate_pathology_criticality_fixture() -> PathologyCriticalityFixtureResult:
    """Run all source-anchored pathology/criticality fixtures."""
    return PathologyCriticalityFixtureResult(
        coherence=validate_coherence_breakdown_index_fixture(),
        criticality=validate_criticality_deviation_classifier_fixture(),
        restoration=validate_therapeutic_restoration_targets_fixture(),
    )


def _require_non_negative(name: str, value: float) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _require_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _require_unit_interval(name: str, value: float) -> None:
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and in the unit interval")


def _non_finite_observable_rejection_label() -> float:
    try:
        PathologyCriticalityConfig(free_energy=float("nan"))
    except ValueError as exc:
        return float("finite" in str(exc))
    return 0.0


def _negative_probability_rejection_label() -> float:
    try:
        PathologyCriticalityConfig(qec_success_probability=-0.1)
    except ValueError as exc:
        return float("unit interval" in str(exc))
    return 0.0


def _non_positive_sigma_rejection_label() -> float:
    try:
        classify_criticality(0.0)
    except ValueError as exc:
        return float("positive" in str(exc))
    return 0.0


def _negative_tolerance_rejection_label() -> float:
    try:
        classify_criticality(1.0, tolerance=-0.1)
    except ValueError as exc:
        return float("tolerance" in str(exc))
    return 0.0


def _out_of_range_order_rejection_label() -> float:
    try:
        PathologyCriticalityConfig(order_parameter=1.2)
    except ValueError as exc:
        return float("unit interval" in str(exc))
    return 0.0


__all__ = [
    "CLAIM_BOUNDARY",
    "CoherenceBreakdownValidationResult",
    "CriticalityDeviationValidationResult",
    "PathologyCriticalityConfig",
    "PathologyCriticalityFixtureResult",
    "TherapeuticRestorationValidationResult",
    "classify_criticality",
    "pathology_index",
    "restore_toward_quasicriticality",
    "validate_coherence_breakdown_index_fixture",
    "validate_criticality_deviation_classifier_fixture",
    "validate_pathology_criticality_fixture",
    "validate_therapeutic_restoration_targets_fixture",
]
