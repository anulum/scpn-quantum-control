# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 L11 interface validation fixtures
"""Simulator-only Noosphere-Technosphere L11 interface fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from .spec_loader import load_l11_interface_validation_spec

CLAIM_BOUNDARY = "simulator-only L11 boundary; not societal evidence"


@dataclass(frozen=True, slots=True)
class L11InterfaceConfig:
    """Finite simulator settings for Paper 0 L11 interface fixtures."""

    coherent_coupling_matrix: np.ndarray | None = None
    fragmented_coupling_matrix: np.ndarray | None = None
    technosphere_coupling_gain: float = 0.35
    effective_temperature_gain: float = 0.4
    sigma_baseline: float = 0.9
    supercritical_threshold: float = 1.0
    spin_glass_risk_threshold: float = 0.25
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        coherent = _validated_signed_matrix(
            "coherent_coupling_matrix",
            self.coherent_coupling_matrix
            if self.coherent_coupling_matrix is not None
            else np.array(
                [
                    [0.0, 0.8, 0.7, 0.6],
                    [0.8, 0.0, 0.7, 0.6],
                    [0.7, 0.7, 0.0, 0.8],
                    [0.6, 0.6, 0.8, 0.0],
                ],
                dtype=np.float64,
            ),
        )
        fragmented = _validated_signed_matrix(
            "fragmented_coupling_matrix",
            self.fragmented_coupling_matrix
            if self.fragmented_coupling_matrix is not None
            else np.array(
                [
                    [0.0, 0.8, -0.7, -0.6],
                    [0.8, 0.0, -0.7, 0.6],
                    [-0.7, -0.7, 0.0, 0.8],
                    [-0.6, 0.6, 0.8, 0.0],
                ],
                dtype=np.float64,
            ),
        )
        if coherent.shape != fragmented.shape:
            raise ValueError("coherent and fragmented coupling matrices must have same shape")
        _require_non_negative("technosphere_coupling_gain", self.technosphere_coupling_gain)
        _require_non_negative("effective_temperature_gain", self.effective_temperature_gain)
        _require_positive("sigma_baseline", self.sigma_baseline)
        _require_positive("supercritical_threshold", self.supercritical_threshold)
        _require_non_negative("spin_glass_risk_threshold", self.spin_glass_risk_threshold)
        object.__setattr__(self, "coherent_coupling_matrix", coherent)
        object.__setattr__(self, "fragmented_coupling_matrix", fragmented)


@dataclass(frozen=True, slots=True)
class HybridCollectiveCouplingValidationResult:
    """Hybrid collective coupling validation result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    baseline_coupling_sum: float
    hybrid_coupling_sum: float
    hybrid_coupling_gain: float
    claim_boundary: str
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class AcceleratedSupercriticalityBoundaryValidationResult:
    """Accelerated sigma and effective-temperature boundary result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    baseline_sigma: float
    accelerated_sigma: float
    effective_temperature_gain: float
    supercritical_label: bool
    claim_boundary: str
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class FragmentationSpinGlassRiskValidationResult:
    """Fragmentation spin-glass risk validation result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    coherent_frustration: float
    fragmented_frustration: float
    frustration_delta: float
    spin_glass_risk_label: bool
    claim_boundary: str
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class L11InterfaceFixtureResult:
    """Combined L11 interface fixture result."""

    hybrid: HybridCollectiveCouplingValidationResult
    supercriticality: AcceleratedSupercriticalityBoundaryValidationResult
    fragmentation: FragmentationSpinGlassRiskValidationResult

    @property
    def hardware_status(self) -> str:
        """Return the shared simulator-only hardware status."""
        return self.hybrid.hardware_status

    @property
    def spec_keys(self) -> tuple[str, str, str]:
        """Return the promoted L11 interface spec keys."""
        return (
            self.hybrid.spec_key,
            self.supercriticality.spec_key,
            self.fragmentation.spec_key,
        )

    @property
    def claim_boundary(self) -> str:
        """Return the shared claim-boundary statement."""
        return CLAIM_BOUNDARY


def hybrid_coupling_matrix(
    config: L11InterfaceConfig,
    *,
    include_technosphere: bool,
) -> np.ndarray:
    """Return a finite hybrid coupling matrix with optional technosphere edge gain."""
    base = cast(np.ndarray, config.coherent_coupling_matrix).copy()
    if include_technosphere and config.technosphere_coupling_gain > 0.0:
        size = base.shape[0]
        split = size // 2
        base[:split, split:] += config.technosphere_coupling_gain
        base[split:, :split] += config.technosphere_coupling_gain
        np.fill_diagonal(base, 0.0)
    return cast(np.ndarray, base)


def effective_sigma(
    sigma_baseline: float,
    *,
    coupling_gain: float,
    temperature_gain: float,
) -> float:
    """Return finite sigma after L11 coupling and effective-temperature gains."""
    _require_positive("sigma_baseline", sigma_baseline)
    _require_non_negative("coupling_gain", coupling_gain)
    _require_non_negative("temperature_gain", temperature_gain)
    return float(sigma_baseline * (1.0 + coupling_gain + temperature_gain))


def frustration_index(coupling_matrix: np.ndarray | None) -> float:
    """Return signed-triad frustration fraction for a finite symmetric graph."""
    matrix = _validated_signed_matrix("coupling_matrix", coupling_matrix)
    size = matrix.shape[0]
    frustrated = 0
    total = 0
    for i in range(size):
        for j in range(i + 1, size):
            for k in range(j + 1, size):
                edges = (matrix[i, j], matrix[i, k], matrix[j, k])
                if any(abs(edge) <= np.finfo(np.float64).eps for edge in edges):
                    continue
                total += 1
                if np.prod(np.sign(edges)) < 0.0:
                    frustrated += 1
    if total == 0:
        return 0.0
    return float(frustrated / total)


def validate_hybrid_collective_coupling_fixture(
    config: L11InterfaceConfig | None = None,
) -> HybridCollectiveCouplingValidationResult:
    """Run the source-anchored hybrid collective coupling fixture."""
    cfg = config or L11InterfaceConfig()
    spec = load_l11_interface_validation_spec(
        "applied.l11_interface.hybrid_collective_coupling",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    baseline = hybrid_coupling_matrix(cfg, include_technosphere=False)
    hybrid = hybrid_coupling_matrix(cfg, include_technosphere=True)
    zero_gain_cfg = L11InterfaceConfig(
        coherent_coupling_matrix=cfg.coherent_coupling_matrix,
        fragmented_coupling_matrix=cfg.fragmented_coupling_matrix,
        technosphere_coupling_gain=0.0,
        effective_temperature_gain=cfg.effective_temperature_gain,
        sigma_baseline=cfg.sigma_baseline,
        supercritical_threshold=cfg.supercritical_threshold,
        spin_glass_risk_threshold=cfg.spin_glass_risk_threshold,
        spec_bundle_path=cfg.spec_bundle_path,
    )
    controls = {
        "zero_gain_delta_abs": abs(
            float(
                np.sum(hybrid_coupling_matrix(zero_gain_cfg, include_technosphere=True))
                - np.sum(hybrid_coupling_matrix(zero_gain_cfg, include_technosphere=False))
            )
        ),
        "asymmetric_matrix_rejection_label": _asymmetric_matrix_rejection_label(),
        "non_finite_matrix_rejection_label": _non_finite_matrix_rejection_label(),
    }
    baseline_sum = float(np.sum(baseline))
    hybrid_sum = float(np.sum(hybrid))
    return HybridCollectiveCouplingValidationResult(
        spec_key="applied.l11_interface.hybrid_collective_coupling",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        baseline_coupling_sum=baseline_sum,
        hybrid_coupling_sum=hybrid_sum,
        hybrid_coupling_gain=hybrid_sum - baseline_sum,
        claim_boundary=CLAIM_BOUNDARY,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(_metadata(spec, extra={"node_count": hybrid.shape[0]})),
    )


def validate_accelerated_supercriticality_boundary_fixture(
    config: L11InterfaceConfig | None = None,
) -> AcceleratedSupercriticalityBoundaryValidationResult:
    """Run the source-anchored accelerated sigma boundary fixture."""
    cfg = config or L11InterfaceConfig()
    spec = load_l11_interface_validation_spec(
        "applied.l11_interface.accelerated_supercriticality_boundary",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    baseline = effective_sigma(cfg.sigma_baseline, coupling_gain=0.0, temperature_gain=0.0)
    accelerated = effective_sigma(
        cfg.sigma_baseline,
        coupling_gain=cfg.technosphere_coupling_gain,
        temperature_gain=cfg.effective_temperature_gain,
    )
    zero_gain = effective_sigma(cfg.sigma_baseline, coupling_gain=0.0, temperature_gain=0.0)
    controls = {
        "baseline_supercritical_label": float(baseline > cfg.supercritical_threshold),
        "zero_gain_sigma_delta_abs": abs(zero_gain - baseline),
        "non_positive_sigma_rejection_label": _non_positive_sigma_rejection_label(),
    }
    return AcceleratedSupercriticalityBoundaryValidationResult(
        spec_key="applied.l11_interface.accelerated_supercriticality_boundary",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        baseline_sigma=baseline,
        accelerated_sigma=accelerated,
        effective_temperature_gain=cfg.effective_temperature_gain,
        supercritical_label=accelerated > cfg.supercritical_threshold,
        claim_boundary=CLAIM_BOUNDARY,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(
            _metadata(spec, extra={"supercritical_threshold": cfg.supercritical_threshold})
        ),
    )


def validate_fragmentation_spin_glass_risk_fixture(
    config: L11InterfaceConfig | None = None,
) -> FragmentationSpinGlassRiskValidationResult:
    """Run the source-anchored fragmentation spin-glass risk fixture."""
    cfg = config or L11InterfaceConfig()
    spec = load_l11_interface_validation_spec(
        "applied.l11_interface.fragmentation_spin_glass_risk",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    coherent = frustration_index(cfg.coherent_coupling_matrix)
    fragmented = frustration_index(cfg.fragmented_coupling_matrix)
    controls = {
        "coherent_spin_glass_risk_label": float(coherent > cfg.spin_glass_risk_threshold),
        "shape_mismatch_rejection_label": _shape_mismatch_rejection_label(),
        "non_symmetric_signed_graph_rejection_label": _asymmetric_matrix_rejection_label(),
    }
    return FragmentationSpinGlassRiskValidationResult(
        spec_key="applied.l11_interface.fragmentation_spin_glass_risk",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        coherent_frustration=coherent,
        fragmented_frustration=fragmented,
        frustration_delta=fragmented - coherent,
        spin_glass_risk_label=fragmented > cfg.spin_glass_risk_threshold,
        claim_boundary=CLAIM_BOUNDARY,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(
            _metadata(spec, extra={"spin_glass_risk_threshold": cfg.spin_glass_risk_threshold})
        ),
    )


def validate_l11_interface_fixture(
    config: L11InterfaceConfig | None = None,
) -> L11InterfaceFixtureResult:
    """Run all Paper 0 L11 interface validation fixtures."""
    cfg = config or L11InterfaceConfig()
    return L11InterfaceFixtureResult(
        hybrid=validate_hybrid_collective_coupling_fixture(cfg),
        supercriticality=validate_accelerated_supercriticality_boundary_fixture(cfg),
        fragmentation=validate_fragmentation_spin_glass_risk_fixture(cfg),
    )


def _validated_signed_matrix(name: str, matrix: np.ndarray | None) -> np.ndarray:
    if matrix is None:
        raise ValueError(f"{name} must not be None")
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 2:
        raise ValueError(f"{name} must be a square matrix with at least two nodes")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    if not np.allclose(arr, arr.T, atol=1.0e-12):
        raise ValueError(f"{name} must be symmetric")
    copied = arr.copy()
    np.fill_diagonal(copied, 0.0)
    return cast(np.ndarray, copied)


def _metadata(spec: dict[str, Any], *, extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper0_validation_protocol": str(spec["validation_protocol"]),
        "hardware_status": str(spec["hardware_status"]),
        "simulator_only_l11_boundary": True,
        "claim_boundary": CLAIM_BOUNDARY,
        **extra,
    }


def _asymmetric_matrix_rejection_label() -> float:
    try:
        _validated_signed_matrix("control_matrix", np.array([[0.0, 1.0], [0.2, 0.0]]))
    except ValueError as exc:
        return float("symmetric" in str(exc))
    return 0.0


def _non_finite_matrix_rejection_label() -> float:
    try:
        _validated_signed_matrix("control_matrix", np.array([[0.0, np.nan], [np.nan, 0.0]]))
    except ValueError as exc:
        return float("finite" in str(exc))
    return 0.0


def _non_positive_sigma_rejection_label() -> float:
    try:
        effective_sigma(0.0, coupling_gain=0.0, temperature_gain=0.0)
    except ValueError as exc:
        return float("positive" in str(exc))
    return 0.0


def _shape_mismatch_rejection_label() -> float:
    try:
        L11InterfaceConfig(
            coherent_coupling_matrix=np.zeros((2, 2), dtype=np.float64),
            fragmented_coupling_matrix=np.zeros((3, 3), dtype=np.float64),
        )
    except ValueError as exc:
        return float("same shape" in str(exc))
    return 0.0


def _require_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _require_non_negative(name: str, value: float) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


__all__ = [
    "AcceleratedSupercriticalityBoundaryValidationResult",
    "CLAIM_BOUNDARY",
    "FragmentationSpinGlassRiskValidationResult",
    "HybridCollectiveCouplingValidationResult",
    "L11InterfaceConfig",
    "L11InterfaceFixtureResult",
    "effective_sigma",
    "frustration_index",
    "hybrid_coupling_matrix",
    "validate_accelerated_supercriticality_boundary_fixture",
    "validate_fragmentation_spin_glass_risk_fixture",
    "validate_hybrid_collective_coupling_fixture",
    "validate_l11_interface_fixture",
]
