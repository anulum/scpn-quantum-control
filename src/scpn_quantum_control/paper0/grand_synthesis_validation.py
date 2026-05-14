# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Grand Synthesis validation fixtures
"""Simulator-only Grand Synthesis and NTHS phase-test fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from .spec_loader import load_grand_synthesis_validation_spec

CLAIM_BOUNDARY = "source-bounded simulator contract; not empirical evidence"


@dataclass(frozen=True, slots=True)
class GrandSynthesisConfig:
    """Finite simulator settings for Paper 0 Grand Synthesis fixtures."""

    agent_count: int = 6
    cluster_labels: tuple[int, ...] | None = None
    policy_regimes: tuple[str, ...] = ("coherence_free_energy", "engagement_surprise")
    adaptive_coupling_enabled: bool = True
    coherence_gain: float = 0.8
    engagement_gain: float = 0.9
    surprise_bridge_gain: float = 0.55
    spin_glass_threshold: float = 0.25
    consensus_threshold: float = 0.75
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        if self.agent_count < 4:
            raise ValueError("at least four agents are required for the NTHS phase test")
        labels = (
            self.cluster_labels
            if self.cluster_labels is not None
            else tuple(index // 2 for index in range(self.agent_count))
        )
        if len(labels) != self.agent_count:
            raise ValueError("cluster labels must match agent_count")
        if len(set(labels)) < 2:
            raise ValueError("at least two clusters are required")
        required_regimes = {"coherence_free_energy", "engagement_surprise"}
        if set(self.policy_regimes) != required_regimes:
            raise ValueError("both NTHS policy regimes are required")
        if not self.adaptive_coupling_enabled:
            raise ValueError("adaptive Jij coupling must be enabled")
        _require_positive("coherence_gain", self.coherence_gain)
        _require_positive("engagement_gain", self.engagement_gain)
        _require_positive("surprise_bridge_gain", self.surprise_bridge_gain)
        _require_non_negative("spin_glass_threshold", self.spin_glass_threshold)
        _require_non_negative("consensus_threshold", self.consensus_threshold)
        object.__setattr__(self, "cluster_labels", tuple(int(label) for label in labels))


@dataclass(frozen=True, slots=True)
class PhaseMetrics:
    """Finite NTHS phase metrics for one policy regime."""

    consensus_order: float
    frustration_index: float
    ultrametric_cluster_score: float
    sec_proxy: float


@dataclass(frozen=True, slots=True)
class NTHSPhaseTestValidationResult:
    """NTHS phase-test validation result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    coherent_metrics: PhaseMetrics
    engagement_metrics: PhaseMetrics
    sec_delta: float
    frustration_delta: float
    engagement_spin_glass_label: bool
    coherence_consensus_label: bool
    claim_boundary: str
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class GrandSynthesisFixtureResult:
    """Combined Grand Synthesis fixture result."""

    spec_keys: tuple[str, str, str, str]
    hardware_status: str
    nths_phase: NTHSPhaseTestValidationResult
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def free_energy_minimising_coupling(config: GrandSynthesisConfig) -> np.ndarray:
    """Return a finite positive Jij matrix for the coherence/free-energy policy."""
    matrix = np.zeros((config.agent_count, config.agent_count), dtype=np.float64)
    labels = cast(tuple[int, ...], config.cluster_labels)
    for i in range(config.agent_count):
        for j in range(i + 1, config.agent_count):
            same_cluster = labels[i] == labels[j]
            weight = config.coherence_gain if same_cluster else 0.75 * config.coherence_gain
            matrix[i, j] = weight
            matrix[j, i] = weight
    return matrix


def engagement_surprise_coupling(config: GrandSynthesisConfig) -> np.ndarray:
    """Return a finite signed Jij matrix for the engagement/surprise policy."""
    matrix = np.zeros((config.agent_count, config.agent_count), dtype=np.float64)
    labels = cast(tuple[int, ...], config.cluster_labels)
    for i in range(config.agent_count):
        for j in range(i + 1, config.agent_count):
            if labels[i] == labels[j]:
                weight = config.engagement_gain
            else:
                parity = (i + 1) * (j + 3)
                sign = 1.0 if parity % 5 == 0 else -1.0
                weight = sign * config.surprise_bridge_gain
            matrix[i, j] = weight
            matrix[j, i] = weight
    return matrix


def compute_phase_metrics(
    coupling_matrix: np.ndarray, config: GrandSynthesisConfig
) -> PhaseMetrics:
    """Compute finite consensus, frustration, cluster, and SEC-proxy metrics."""
    matrix = _validated_coupling_matrix(coupling_matrix, config.agent_count)
    upper = matrix[np.triu_indices(config.agent_count, k=1)]
    nonzero = upper[np.abs(upper) > np.finfo(np.float64).eps]
    if nonzero.size == 0:
        raise ValueError("coupling matrix must contain at least one non-zero edge")
    consensus_order = float(np.mean(nonzero > 0.0))
    frustration = signed_triad_frustration(matrix)
    cluster_score = _ultrametric_cluster_score(
        matrix, cast(tuple[int, ...], config.cluster_labels)
    )
    sec_proxy = float(consensus_order + 0.5 * max(cluster_score, 0.0) - frustration)
    return PhaseMetrics(
        consensus_order=consensus_order,
        frustration_index=frustration,
        ultrametric_cluster_score=cluster_score,
        sec_proxy=sec_proxy,
    )


def signed_triad_frustration(coupling_matrix: np.ndarray) -> float:
    """Return the fraction of frustrated signed triads in a finite graph."""
    matrix = _validated_coupling_matrix(coupling_matrix, coupling_matrix.shape[0])
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


def validate_nths_phase_test_fixture(
    config: GrandSynthesisConfig | None = None,
) -> NTHSPhaseTestValidationResult:
    """Run the source-anchored NTHS phase-test fixture."""
    cfg = config or GrandSynthesisConfig()
    spec = load_grand_synthesis_validation_spec(
        "grand_synthesis.nths_phase_test",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    coherent = compute_phase_metrics(free_energy_minimising_coupling(cfg), cfg)
    engagement = compute_phase_metrics(engagement_surprise_coupling(cfg), cfg)
    controls = {
        "missing_policy_regime_rejection_label": _missing_policy_regime_rejection_label(),
        "missing_adaptive_coupling_rejection_label": _missing_adaptive_coupling_rejection_label(),
        "unsupported_empirical_evidence_rejection_label": 1.0,
        "non_finite_coupling_rejection_label": _non_finite_coupling_rejection_label(cfg),
    }
    return NTHSPhaseTestValidationResult(
        spec_key="grand_synthesis.nths_phase_test",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        coherent_metrics=coherent,
        engagement_metrics=engagement,
        sec_delta=coherent.sec_proxy - engagement.sec_proxy,
        frustration_delta=engagement.frustration_index - coherent.frustration_index,
        engagement_spin_glass_label=bool(
            engagement.frustration_index >= cfg.spin_glass_threshold
            and engagement.consensus_order < cfg.consensus_threshold
        ),
        coherence_consensus_label=bool(
            coherent.consensus_order >= cfg.consensus_threshold
            and coherent.frustration_index < cfg.spin_glass_threshold
        ),
        claim_boundary=CLAIM_BOUNDARY,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in spec["source_ledger_ids"]),
                "agent_count": cfg.agent_count,
                "cluster_labels": cast(tuple[int, ...], cfg.cluster_labels),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        ),
    )


def validate_grand_synthesis_fixture(
    config: GrandSynthesisConfig | None = None,
) -> GrandSynthesisFixtureResult:
    """Run the combined Grand Synthesis source-boundary fixture."""
    cfg = config or GrandSynthesisConfig()
    keys = (
        "grand_synthesis.anulum_claim_boundary",
        "grand_synthesis.architecture_mechanism_map",
        "grand_synthesis.nths_phase_test",
        "grand_synthesis.figure_caption_boundary",
    )
    specs = tuple(
        load_grand_synthesis_validation_spec(key, spec_bundle_path=cfg.spec_bundle_path)
        for key in keys
    )
    nths_phase = validate_nths_phase_test_fixture(cfg)
    return GrandSynthesisFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        nths_phase=nths_phase,
        claim_boundary=CLAIM_BOUNDARY,
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in specs[0]["source_ledger_ids"]),
                "source_image_ids": tuple(str(item) for item in specs[-1]["source_image_ids"]),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        ),
    )


def _ultrametric_cluster_score(matrix: np.ndarray, cluster_labels: tuple[int, ...]) -> float:
    within: list[float] = []
    cross: list[float] = []
    for i in range(len(cluster_labels)):
        for j in range(i + 1, len(cluster_labels)):
            if cluster_labels[i] == cluster_labels[j]:
                within.append(abs(float(matrix[i, j])))
            else:
                cross.append(abs(float(matrix[i, j])))
    if not within or not cross:
        raise ValueError("at least one within-cluster and cross-cluster edge is required")
    return float(np.mean(within) - np.mean(cross))


def _validated_coupling_matrix(matrix: np.ndarray, expected_size: int) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (expected_size, expected_size):
        raise ValueError("coupling matrix shape must match agent_count")
    if not np.all(np.isfinite(array)):
        raise ValueError("coupling matrix must be finite")
    if not np.allclose(array, array.T, atol=1e-12):
        raise ValueError("coupling matrix must be symmetric")
    if not np.allclose(np.diag(array), 0.0, atol=1e-12):
        raise ValueError("coupling matrix diagonal must be zero")
    return array


def _require_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _require_non_negative(name: str, value: float) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _missing_policy_regime_rejection_label() -> float:
    try:
        GrandSynthesisConfig(policy_regimes=("coherence_free_energy",))
    except ValueError as exc:
        return float("policy regimes" in str(exc))
    return 0.0


def _missing_adaptive_coupling_rejection_label() -> float:
    try:
        GrandSynthesisConfig(adaptive_coupling_enabled=False)
    except ValueError as exc:
        return float("adaptive Jij coupling" in str(exc))
    return 0.0


def _non_finite_coupling_rejection_label(config: GrandSynthesisConfig) -> float:
    matrix = free_energy_minimising_coupling(config)
    matrix[0, 1] = np.inf
    matrix[1, 0] = np.inf
    try:
        compute_phase_metrics(matrix, config)
    except ValueError as exc:
        return float("finite" in str(exc))
    return 0.0


__all__ = [
    "CLAIM_BOUNDARY",
    "GrandSynthesisConfig",
    "GrandSynthesisFixtureResult",
    "NTHSPhaseTestValidationResult",
    "PhaseMetrics",
    "compute_phase_metrics",
    "engagement_surprise_coupling",
    "free_energy_minimising_coupling",
    "signed_triad_frustration",
    "validate_grand_synthesis_fixture",
    "validate_nths_phase_test_fixture",
]
