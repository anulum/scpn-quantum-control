# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 system-robustness validation fixtures
"""Simulator-only robustness fixtures for Paper 0 failure-mode records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from .spec_loader import load_system_robustness_validation_spec

CLAIM_BOUNDARY = "simulator-only robustness boundary; not operational security evidence"


@dataclass(frozen=True, slots=True)
class SystemRobustnessConfig:
    """Finite simulator settings for Paper 0 system-robustness fixtures."""

    coupling_matrix: np.ndarray | None = None
    percolation_threshold: float = 0.5
    targeted_removed_nodes: tuple[int, ...] = (0,)
    sigma: float = 1.04
    reference_sigma: float = 1.4
    sigma_critical: float = 1.0
    critical_exponent: float = 1.0
    base_recovery_time: float = 1.0
    decoherence_exposure: float = 0.6
    ms_qec_redundancy: int = 4
    qec_correction_strength: float = 0.72
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        matrix = (
            self.coupling_matrix
            if self.coupling_matrix is not None
            else np.array(
                [
                    [0.0, 0.9, 0.8, 0.1],
                    [0.9, 0.0, 0.7, 0.1],
                    [0.8, 0.7, 0.0, 0.1],
                    [0.1, 0.1, 0.1, 0.0],
                ],
                dtype=np.float64,
            )
        )
        validated = _validated_coupling_matrix(matrix)
        _require_non_negative("percolation_threshold", self.percolation_threshold)
        for node in self.targeted_removed_nodes:
            if not isinstance(node, int) or node < 0 or node >= validated.shape[0]:
                raise ValueError("targeted_removed_nodes must reference valid node indices")
        _require_positive("sigma", self.sigma)
        _require_positive("reference_sigma", self.reference_sigma)
        _require_positive("sigma_critical", self.sigma_critical)
        _require_positive("critical_exponent", self.critical_exponent)
        _require_positive("base_recovery_time", self.base_recovery_time)
        _require_non_negative("decoherence_exposure", self.decoherence_exposure)
        if not isinstance(self.ms_qec_redundancy, int) or self.ms_qec_redundancy < 1:
            raise ValueError("ms_qec_redundancy must be a positive integer")
        _require_unit_interval("qec_correction_strength", self.qec_correction_strength)
        object.__setattr__(self, "coupling_matrix", validated)


@dataclass(frozen=True, slots=True)
class CascadingFailurePercolationValidationResult:
    """Weighted-graph cascade percolation validation result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    intact_largest_component_fraction: float
    attacked_largest_component_fraction: float
    largest_component_loss: float
    claim_boundary: str
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class CriticalSlowingRecoveryValidationResult:
    """Critical-slowing recovery-time validation result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    near_critical_recovery_time: float
    reference_recovery_time: float
    recovery_time_ratio: float
    claim_boundary: str
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class DecoherenceAttackMSQECBoundaryValidationResult:
    """MS-QEC/decoherence stress-boundary validation result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    ms_qec_success_probability: float
    failure_probability: float
    unprotected_failure_probability: float
    claim_boundary: str
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class SystemRobustnessFixtureResult:
    """Combined Paper 0 system-robustness fixture result."""

    cascade: CascadingFailurePercolationValidationResult
    critical_slowing: CriticalSlowingRecoveryValidationResult
    decoherence: DecoherenceAttackMSQECBoundaryValidationResult

    @property
    def hardware_status(self) -> str:
        """Return the shared simulator-only hardware status."""
        return self.cascade.hardware_status

    @property
    def spec_keys(self) -> tuple[str, str, str]:
        """Return the promoted system-robustness spec keys."""
        return (
            self.cascade.spec_key,
            self.critical_slowing.spec_key,
            self.decoherence.spec_key,
        )

    @property
    def claim_boundary(self) -> str:
        """Return the shared claim-boundary statement."""
        return CLAIM_BOUNDARY


def largest_component_fraction(
    coupling_matrix: np.ndarray | None,
    *,
    threshold: float,
    removed_nodes: tuple[int, ...] = (),
) -> float:
    """Return thresholded graph largest connected-component fraction."""
    matrix = _validated_coupling_matrix(coupling_matrix)
    _require_non_negative("threshold", threshold)
    removed = set(removed_nodes)
    for node in removed:
        if node < 0 or node >= matrix.shape[0]:
            raise ValueError("removed_nodes must reference valid node indices")
    active = [node for node in range(matrix.shape[0]) if node not in removed]
    if not active:
        return 0.0
    adjacency = matrix >= threshold
    np.fill_diagonal(adjacency, False)
    seen: set[int] = set()
    largest = 0
    for start in active:
        if start in seen:
            continue
        stack = [start]
        component: set[int] = set()
        while stack:
            node = stack.pop()
            if node in seen or node in removed:
                continue
            seen.add(node)
            component.add(node)
            neighbours = [idx for idx, connected in enumerate(adjacency[node]) if connected]
            stack.extend(neighbour for neighbour in neighbours if neighbour not in removed)
        largest = max(largest, len(component))
    return float(largest / matrix.shape[0])


def critical_recovery_time(
    sigma: float,
    *,
    sigma_critical: float = 1.0,
    critical_exponent: float = 1.0,
    base_recovery_time: float = 1.0,
) -> float:
    """Return finite critical-slowing recovery time away from sigma criticality."""
    _require_positive("sigma", sigma)
    _require_positive("sigma_critical", sigma_critical)
    _require_positive("critical_exponent", critical_exponent)
    _require_positive("base_recovery_time", base_recovery_time)
    distance = abs(sigma - sigma_critical)
    if distance <= np.finfo(np.float64).eps:
        raise ValueError("sigma must not equal sigma_critical for finite recovery time")
    return float(base_recovery_time / (distance**critical_exponent))


def ms_qec_success_probability(
    *,
    exposure: float,
    redundancy: int,
    correction_strength: float,
) -> float:
    """Return bounded MS-QEC success probability under finite decoherence exposure."""
    _require_non_negative("exposure", exposure)
    if not isinstance(redundancy, int) or redundancy < 1:
        raise ValueError("redundancy must be a positive integer")
    _require_unit_interval("correction_strength", correction_strength)
    effective_exposure = exposure * (1.0 - correction_strength) / math.sqrt(float(redundancy))
    return float(math.exp(-effective_exposure))


def validate_cascading_failure_percolation_fixture(
    config: SystemRobustnessConfig | None = None,
) -> CascadingFailurePercolationValidationResult:
    """Run the source-anchored cascade-percolation fixture."""
    cfg = config or SystemRobustnessConfig()
    spec = load_system_robustness_validation_spec(
        "applied.system_robustness.cascading_failure_percolation",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    intact = largest_component_fraction(
        cfg.coupling_matrix,
        threshold=cfg.percolation_threshold,
    )
    attacked = largest_component_fraction(
        cfg.coupling_matrix,
        threshold=cfg.percolation_threshold,
        removed_nodes=cfg.targeted_removed_nodes,
    )
    complete = np.ones((4, 4), dtype=np.float64) - np.eye(4, dtype=np.float64)
    empty = np.zeros((4, 4), dtype=np.float64)
    controls = {
        "complete_graph_largest_component_fraction": largest_component_fraction(
            complete,
            threshold=0.5,
        ),
        "empty_graph_fragmentation_label": float(
            largest_component_fraction(empty, threshold=0.5) <= 0.25
        ),
        "asymmetric_coupling_rejection_label": _asymmetric_coupling_rejection_label(),
    }
    return CascadingFailurePercolationValidationResult(
        spec_key="applied.system_robustness.cascading_failure_percolation",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        intact_largest_component_fraction=intact,
        attacked_largest_component_fraction=attacked,
        largest_component_loss=float(intact - attacked),
        claim_boundary=CLAIM_BOUNDARY,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(_metadata(spec, extra={"node_count": 4})),
    )


def validate_critical_slowing_recovery_fixture(
    config: SystemRobustnessConfig | None = None,
) -> CriticalSlowingRecoveryValidationResult:
    """Run the source-anchored critical-slowing recovery fixture."""
    cfg = config or SystemRobustnessConfig()
    spec = load_system_robustness_validation_spec(
        "applied.system_robustness.critical_slowing_recovery",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    near = critical_recovery_time(
        cfg.sigma,
        sigma_critical=cfg.sigma_critical,
        critical_exponent=cfg.critical_exponent,
        base_recovery_time=cfg.base_recovery_time,
    )
    reference = critical_recovery_time(
        cfg.reference_sigma,
        sigma_critical=cfg.sigma_critical,
        critical_exponent=cfg.critical_exponent,
        base_recovery_time=cfg.base_recovery_time,
    )
    far_control = critical_recovery_time(2.0)
    controls = {
        "far_from_transition_ratio": far_control / near,
        "critical_point_rejection_label": _critical_point_rejection_label(),
        "non_positive_sigma_rejection_label": _non_positive_sigma_rejection_label(),
    }
    return CriticalSlowingRecoveryValidationResult(
        spec_key="applied.system_robustness.critical_slowing_recovery",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        near_critical_recovery_time=near,
        reference_recovery_time=reference,
        recovery_time_ratio=float(near / reference),
        claim_boundary=CLAIM_BOUNDARY,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(
            _metadata(spec, extra={"critical_exponent": cfg.critical_exponent})
        ),
    )


def validate_decoherence_attack_ms_qec_boundary_fixture(
    config: SystemRobustnessConfig | None = None,
) -> DecoherenceAttackMSQECBoundaryValidationResult:
    """Run the source-anchored MS-QEC/decoherence stress fixture."""
    cfg = config or SystemRobustnessConfig()
    spec = load_system_robustness_validation_spec(
        "applied.system_robustness.decoherence_attack_ms_qec_boundary",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    success = ms_qec_success_probability(
        exposure=cfg.decoherence_exposure,
        redundancy=cfg.ms_qec_redundancy,
        correction_strength=cfg.qec_correction_strength,
    )
    unprotected_success = ms_qec_success_probability(
        exposure=cfg.decoherence_exposure,
        redundancy=1,
        correction_strength=0.0,
    )
    controls = {
        "zero_redundancy_rejection_label": _zero_redundancy_rejection_label(),
        "out_of_range_correction_rejection_label": _out_of_range_correction_rejection_label(),
        "unit_interval_success_label": float(0.0 <= success <= 1.0),
    }
    return DecoherenceAttackMSQECBoundaryValidationResult(
        spec_key="applied.system_robustness.decoherence_attack_ms_qec_boundary",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        ms_qec_success_probability=success,
        failure_probability=float(1.0 - success),
        unprotected_failure_probability=float(1.0 - unprotected_success),
        claim_boundary=CLAIM_BOUNDARY,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(
            _metadata(
                spec,
                extra={
                    "decoherence_exposure": cfg.decoherence_exposure,
                    "ms_qec_redundancy": cfg.ms_qec_redundancy,
                },
            )
        ),
    )


def validate_system_robustness_fixture(
    config: SystemRobustnessConfig | None = None,
) -> SystemRobustnessFixtureResult:
    """Run all Paper 0 system-robustness validation fixtures."""
    cfg = config or SystemRobustnessConfig()
    return SystemRobustnessFixtureResult(
        cascade=validate_cascading_failure_percolation_fixture(cfg),
        critical_slowing=validate_critical_slowing_recovery_fixture(cfg),
        decoherence=validate_decoherence_attack_ms_qec_boundary_fixture(cfg),
    )


def _validated_coupling_matrix(matrix: np.ndarray | None) -> np.ndarray:
    if matrix is None:
        raise ValueError("coupling_matrix must not be None")
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 2:
        raise ValueError("coupling_matrix must be a square matrix with at least two nodes")
    if not np.all(np.isfinite(arr)):
        raise ValueError("coupling_matrix must contain finite values")
    if np.any(arr < 0.0):
        raise ValueError("coupling_matrix must contain non-negative weights")
    if not np.allclose(arr, arr.T, atol=1.0e-12):
        raise ValueError("coupling_matrix must be symmetric")
    return cast(np.ndarray, arr.copy())


def _metadata(spec: dict[str, Any], *, extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper0_validation_protocol": str(spec["validation_protocol"]),
        "hardware_status": str(spec["hardware_status"]),
        "simulator_only_robustness_boundary": True,
        "claim_boundary": CLAIM_BOUNDARY,
        **extra,
    }


def _asymmetric_coupling_rejection_label() -> float:
    try:
        _validated_coupling_matrix(np.array([[0.0, 1.0], [0.2, 0.0]], dtype=np.float64))
    except ValueError as exc:
        return float("symmetric" in str(exc))
    return 0.0


def _critical_point_rejection_label() -> float:
    try:
        critical_recovery_time(1.0)
    except ValueError as exc:
        return float("sigma_critical" in str(exc))
    return 0.0


def _non_positive_sigma_rejection_label() -> float:
    try:
        critical_recovery_time(0.0)
    except ValueError as exc:
        return float("positive" in str(exc))
    return 0.0


def _zero_redundancy_rejection_label() -> float:
    try:
        ms_qec_success_probability(exposure=0.5, redundancy=0, correction_strength=0.5)
    except ValueError as exc:
        return float("redundancy" in str(exc))
    return 0.0


def _out_of_range_correction_rejection_label() -> float:
    try:
        ms_qec_success_probability(exposure=0.5, redundancy=2, correction_strength=1.2)
    except ValueError as exc:
        return float("unit interval" in str(exc))
    return 0.0


def _require_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _require_non_negative(name: str, value: float) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _require_unit_interval(name: str, value: float) -> None:
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in the unit interval")


__all__ = [
    "CLAIM_BOUNDARY",
    "CascadingFailurePercolationValidationResult",
    "CriticalSlowingRecoveryValidationResult",
    "DecoherenceAttackMSQECBoundaryValidationResult",
    "SystemRobustnessConfig",
    "SystemRobustnessFixtureResult",
    "critical_recovery_time",
    "largest_component_fraction",
    "ms_qec_success_probability",
    "validate_cascading_failure_percolation_fixture",
    "validate_critical_slowing_recovery_fixture",
    "validate_decoherence_attack_ms_qec_boundary_fixture",
    "validate_system_robustness_fixture",
]
