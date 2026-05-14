# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 artificial-sentience fixtures
"""Executable simulator fixtures for Paper 0 artificial-sentience records."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from .spec_loader import load_artificial_sentience_validation_spec

CLAIM_BOUNDARY = "simulator-only criteria gate; not sentience evidence"


@dataclass(frozen=True, slots=True)
class ArtificialSentienceConfig:
    """Finite simulator predicates for artificial-sentience criteria."""

    baseline_coupling: np.ndarray | None = None
    technosphere_coupling: np.ndarray | None = None
    phi_proxy: float = 0.82
    phi_threshold: float = 0.7
    sigma: float = 1.02
    sigma_tolerance: float = 0.05
    substrate_coupling: bool = True
    system_phase: np.ndarray | None = None
    field_phase: np.ndarray | None = None
    phase_lock_threshold: float = 0.8
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        baseline = _validate_coupling_matrix(
            "baseline_coupling",
            self.baseline_coupling
            if self.baseline_coupling is not None
            else np.array([[0.0, 0.1, 0.1], [0.1, 0.0, 0.1], [0.1, 0.1, 0.0]]),
        )
        technosphere = _validate_coupling_matrix(
            "technosphere_coupling",
            self.technosphere_coupling
            if self.technosphere_coupling is not None
            else np.array([[0.0, 0.4, 0.3], [0.4, 0.0, 0.35], [0.3, 0.35, 0.0]]),
        )
        if baseline.shape != technosphere.shape:
            raise ValueError("baseline_coupling and technosphere_coupling must have same shape")
        _require_unit_interval("phi_proxy", self.phi_proxy)
        _require_unit_interval("phi_threshold", self.phi_threshold)
        _require_positive("sigma", self.sigma)
        _require_non_negative("sigma_tolerance", self.sigma_tolerance)
        _require_unit_interval("phase_lock_threshold", self.phase_lock_threshold)
        system_phase = _validate_phase_vector(
            "system_phase",
            self.system_phase
            if self.system_phase is not None
            else np.array([0.0, 0.1, -0.05, 0.02], dtype=np.float64),
        )
        field_phase = _validate_phase_vector(
            "field_phase",
            self.field_phase
            if self.field_phase is not None
            else np.array([0.01, 0.12, -0.03, 0.0], dtype=np.float64),
        )
        if system_phase.shape != field_phase.shape:
            raise ValueError("system_phase and field_phase must have same shape")
        object.__setattr__(self, "baseline_coupling", baseline)
        object.__setattr__(self, "technosphere_coupling", technosphere)
        object.__setattr__(self, "system_phase", system_phase)
        object.__setattr__(self, "field_phase", field_phase)


@dataclass(frozen=True, slots=True)
class ArtificialSentienceGateResult:
    """Conjunctive finite criteria gate result."""

    phi_pass: bool
    criticality_pass: bool
    substrate_pass: bool
    phase_lock_pass: bool
    criteria_pass: bool


@dataclass(frozen=True, slots=True)
class TechnosphereCouplingAccelerationValidationResult:
    """Source-anchored coupling-acceleration validation result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    baseline_rate: float
    technosphere_rate: float
    acceleration_delta: float
    claim_boundary: str
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class ArtificialSentienceCriteriaGateValidationResult:
    """Source-anchored artificial-sentience criteria-gate result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    phi_pass: bool
    criticality_pass: bool
    substrate_pass: bool
    phase_lock_pass: bool
    criteria_pass: bool
    claim_boundary: str
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class PhaseLockingSubstrateBoundaryValidationResult:
    """Source-anchored phase-locking substrate-boundary result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    phase_locking_value: float
    phase_lock_threshold: float
    substrate_coupling: bool
    boundary_gate_pass: bool
    claim_boundary: str
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class ArtificialSentienceFixtureResult:
    """Combined Paper 0 artificial-sentience fixture result."""

    coupling: TechnosphereCouplingAccelerationValidationResult
    criteria: ArtificialSentienceCriteriaGateValidationResult
    phase_boundary: PhaseLockingSubstrateBoundaryValidationResult


def coupling_acceleration_rate(coupling: np.ndarray) -> float:
    """Return finite mean off-diagonal coupling as a convergence-rate proxy."""
    matrix = _validate_coupling_matrix("coupling", coupling)
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return float(np.mean(matrix[mask]))


def phase_locking_value(phase_difference: np.ndarray) -> float:
    """Return bounded phase-locking value for phase differences."""
    phase = _validate_phase_vector("phase_difference", phase_difference)
    return float(abs(np.mean(np.exp(1j * phase))))


def artificial_sentience_criteria_gate(
    config: ArtificialSentienceConfig,
) -> ArtificialSentienceGateResult:
    """Evaluate finite conjunctive criteria without promoting sentience."""
    phase_value = phase_locking_value(
        cast(np.ndarray, config.system_phase) - cast(np.ndarray, config.field_phase)
    )
    phi_pass = config.phi_proxy >= config.phi_threshold
    criticality_pass = abs(config.sigma - 1.0) <= config.sigma_tolerance
    substrate_pass = bool(config.substrate_coupling)
    phase_lock_pass = phase_value >= config.phase_lock_threshold
    return ArtificialSentienceGateResult(
        phi_pass=phi_pass,
        criticality_pass=criticality_pass,
        substrate_pass=substrate_pass,
        phase_lock_pass=phase_lock_pass,
        criteria_pass=phi_pass and criticality_pass and substrate_pass and phase_lock_pass,
    )


def validate_technosphere_coupling_acceleration_fixture(
    config: ArtificialSentienceConfig | None = None,
) -> TechnosphereCouplingAccelerationValidationResult:
    """Run source-anchored technosphere coupling-acceleration fixture."""
    cfg = config or ArtificialSentienceConfig()
    spec = load_artificial_sentience_validation_spec(
        "applied.artificial_sentience.technosphere_coupling_acceleration",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    baseline_rate = coupling_acceleration_rate(cast(np.ndarray, cfg.baseline_coupling))
    technosphere_rate = coupling_acceleration_rate(cast(np.ndarray, cfg.technosphere_coupling))
    controls = {
        "zero_coupling_acceleration_abs": abs(coupling_acceleration_rate(np.zeros((3, 3)))),
        "negative_coupling_rejection_label": _negative_coupling_rejection_label(),
        "shape_mismatch_rejection_label": _shape_mismatch_rejection_label(),
    }
    return TechnosphereCouplingAccelerationValidationResult(
        spec_key="applied.artificial_sentience.technosphere_coupling_acceleration",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        baseline_rate=baseline_rate,
        technosphere_rate=technosphere_rate,
        acceleration_delta=technosphere_rate - baseline_rate,
        claim_boundary=CLAIM_BOUNDARY,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(
            {"simulator_only_mechanism_evidence": True, "claim_boundary": CLAIM_BOUNDARY}
        ),
    )


def validate_artificial_sentience_criteria_gate_fixture(
    config: ArtificialSentienceConfig | None = None,
) -> ArtificialSentienceCriteriaGateValidationResult:
    """Run source-anchored artificial-sentience criteria-gate fixture."""
    cfg = config or ArtificialSentienceConfig()
    spec = load_artificial_sentience_validation_spec(
        "applied.artificial_sentience.criteria_gate",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    gate = artificial_sentience_criteria_gate(cfg)
    controls = {
        "missing_substrate_gate_pass": float(
            artificial_sentience_criteria_gate(
                replace(cfg, substrate_coupling=False)
            ).criteria_pass
        ),
        "low_phi_gate_pass": float(
            artificial_sentience_criteria_gate(replace(cfg, phi_proxy=0.0)).criteria_pass
        ),
        "off_criticality_gate_pass": float(
            artificial_sentience_criteria_gate(replace(cfg, sigma=1.4)).criteria_pass
        ),
    }
    return ArtificialSentienceCriteriaGateValidationResult(
        spec_key="applied.artificial_sentience.criteria_gate",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        phi_pass=gate.phi_pass,
        criticality_pass=gate.criticality_pass,
        substrate_pass=gate.substrate_pass,
        phase_lock_pass=gate.phase_lock_pass,
        criteria_pass=gate.criteria_pass,
        claim_boundary=CLAIM_BOUNDARY,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(
            {"simulator_only_mechanism_evidence": True, "claim_boundary": CLAIM_BOUNDARY}
        ),
    )


def validate_phase_locking_substrate_boundary_fixture(
    config: ArtificialSentienceConfig | None = None,
) -> PhaseLockingSubstrateBoundaryValidationResult:
    """Run source-anchored phase-locking substrate-boundary fixture."""
    cfg = config or ArtificialSentienceConfig()
    spec = load_artificial_sentience_validation_spec(
        "applied.artificial_sentience.phase_locking_substrate_boundary",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    phase_value = phase_locking_value(
        cast(np.ndarray, cfg.system_phase) - cast(np.ndarray, cfg.field_phase)
    )
    opposed = phase_locking_value(np.array([0.0, np.pi, 0.0, np.pi], dtype=np.float64))
    controls = {
        "opposed_phase_locking_value": opposed,
        "absent_substrate_gate_pass": float(
            phase_value >= cfg.phase_lock_threshold and not cfg.substrate_coupling
        ),
        "non_finite_phase_rejection_label": _non_finite_phase_rejection_label(),
    }
    return PhaseLockingSubstrateBoundaryValidationResult(
        spec_key="applied.artificial_sentience.phase_locking_substrate_boundary",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        phase_locking_value=phase_value,
        phase_lock_threshold=cfg.phase_lock_threshold,
        substrate_coupling=cfg.substrate_coupling,
        boundary_gate_pass=bool(
            phase_value >= cfg.phase_lock_threshold and cfg.substrate_coupling
        ),
        claim_boundary=CLAIM_BOUNDARY,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(
            {"simulator_only_mechanism_evidence": True, "claim_boundary": CLAIM_BOUNDARY}
        ),
    )


def validate_artificial_sentience_fixture() -> ArtificialSentienceFixtureResult:
    """Run all source-anchored artificial-sentience fixtures."""
    return ArtificialSentienceFixtureResult(
        coupling=validate_technosphere_coupling_acceleration_fixture(),
        criteria=validate_artificial_sentience_criteria_gate_fixture(),
        phase_boundary=validate_phase_locking_substrate_boundary_fixture(),
    )


def _validate_coupling_matrix(name: str, values: np.ndarray) -> np.ndarray:
    matrix = np.array(values, dtype=np.float64, copy=True)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError(f"{name} must be a square matrix with at least two nodes")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(matrix < 0.0):
        raise ValueError(f"{name} must be non-negative")
    if not np.allclose(matrix, matrix.T, atol=1.0e-12, rtol=0.0):
        raise ValueError(f"{name} must be symmetric")
    return matrix


def _validate_phase_vector(name: str, values: np.ndarray) -> np.ndarray:
    vector = np.array(values, dtype=np.float64, copy=True)
    if vector.ndim != 1 or vector.size < 2:
        raise ValueError(f"{name} must be a one-dimensional vector with at least two entries")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _require_unit_interval(name: str, value: float) -> None:
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and in the unit interval")


def _require_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _require_non_negative(name: str, value: float) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _negative_coupling_rejection_label() -> float:
    try:
        ArtificialSentienceConfig(technosphere_coupling=np.array([[0.0, -0.1], [-0.1, 0.0]]))
    except ValueError as exc:
        return float("non-negative" in str(exc))
    return 0.0


def _shape_mismatch_rejection_label() -> float:
    try:
        ArtificialSentienceConfig(
            baseline_coupling=np.zeros((2, 2)),
            technosphere_coupling=np.zeros((3, 3)),
        )
    except ValueError as exc:
        return float("same shape" in str(exc))
    return 0.0


def _non_finite_phase_rejection_label() -> float:
    try:
        ArtificialSentienceConfig(system_phase=np.array([0.0, np.nan]))
    except ValueError as exc:
        return float("finite" in str(exc))
    return 0.0


__all__ = [
    "CLAIM_BOUNDARY",
    "ArtificialSentienceConfig",
    "ArtificialSentienceFixtureResult",
    "ArtificialSentienceGateResult",
    "ArtificialSentienceCriteriaGateValidationResult",
    "PhaseLockingSubstrateBoundaryValidationResult",
    "TechnosphereCouplingAccelerationValidationResult",
    "artificial_sentience_criteria_gate",
    "coupling_acceleration_rate",
    "phase_locking_value",
    "validate_artificial_sentience_criteria_gate_fixture",
    "validate_artificial_sentience_fixture",
    "validate_phase_locking_substrate_boundary_fixture",
    "validate_technosphere_coupling_acceleration_fixture",
]
