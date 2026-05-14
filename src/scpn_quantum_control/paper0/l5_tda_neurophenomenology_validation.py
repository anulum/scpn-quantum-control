# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Layer 5 TDA/neurophenomenology fixtures
"""Simulator-only Layer 5 TDA/neurophenomenology fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .spec_loader import load_l5_tda_neurophenomenology_validation_spec

CLAIM_BOUNDARY = (
    "source-bounded Layer 5 TDA/neurophenomenology simulator contract; not empirical evidence"
)
SOURCE_LEDGER_SPAN = ("P0R06504", "P0R06518")


@dataclass(frozen=True, slots=True)
class L5TDANeurophenomenologyConfig:
    """Finite simulator settings for Layer 5 TDA/neurophenomenology fixtures."""

    correlation_threshold: float = 0.75
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.correlation_threshold) or self.correlation_threshold <= 0.0:
            raise ValueError("correlation_threshold must be finite and positive")


@dataclass(frozen=True, slots=True)
class L5TDANeurophenomenologyFixtureResult:
    """Combined Layer 5 TDA/neurophenomenology fixture result."""

    spec_keys: tuple[str, str, str, str]
    hardware_status: str
    source_ledger_span: tuple[str, str]
    geometric_qualia_score: float
    mean_persistence_lifetime: float
    correlation_score: float
    protocol_completeness: float
    null_controls: MappingProxyType[str, float]
    config_thresholds: MappingProxyType[str, float]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def persistence_lifetimes(*, persistence_pairs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return non-negative death-minus-birth lifetimes from persistence pairs."""
    pairs = np.array(persistence_pairs, dtype=np.float64, copy=True)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("persistence_pairs must have shape (n, 2)")
    if not np.all(np.isfinite(pairs)):
        raise ValueError("persistence_pairs must contain only finite values")
    lifetimes = pairs[:, 1] - pairs[:, 0]
    if np.any(lifetimes < 0.0):
        raise ValueError("death must be greater than or equal to birth")
    return cast(NDArray[np.float64], lifetimes)


def geometric_qualia_score(
    *,
    manifold_volume: float,
    betti_numbers: NDArray[np.float64],
) -> float:
    """Compute the source-bounded Vol(M) times summed Betti-number target."""
    if not np.isfinite(manifold_volume) or manifold_volume <= 0.0:
        raise ValueError("manifold_volume must be finite and positive")
    betti = _non_negative_vector("betti_numbers", betti_numbers)
    return float(manifold_volume * np.sum(betti))


def qualia_topology_correlation(
    *,
    report_scores: NDArray[np.float64],
    topology_scores: NDArray[np.float64],
) -> float:
    """Return Pearson alignment between report scores and topology scores."""
    reports = _finite_vector("report_scores", report_scores)
    topology = _finite_vector("topology_scores", topology_scores)
    if reports.shape != topology.shape:
        raise ValueError("report_scores and topology_scores must have the same shape")
    if reports.size < 2:
        raise ValueError("correlation requires at least two paired samples")
    if float(np.std(reports)) <= np.finfo(np.float64).eps:
        raise ValueError("report_scores must have non-zero variance")
    if float(np.std(topology)) <= np.finfo(np.float64).eps:
        raise ValueError("topology_scores must have non-zero variance")
    return float(np.corrcoef(reports, topology)[0, 1])


def protocol_completeness_score(
    *,
    high_density_recording: bool,
    immediate_interview: bool,
    report_scoring: bool,
    tda_features: bool,
    correlation_test: bool,
) -> float:
    """Return completeness over the five source-listed experimental protocol steps."""
    return float(
        np.mean(
            np.asarray(
                [
                    high_density_recording,
                    immediate_interview,
                    report_scoring,
                    tda_features,
                    correlation_test,
                ],
                dtype=np.float64,
            )
        )
    )


def validate_l5_tda_neurophenomenology_fixture(
    config: L5TDANeurophenomenologyConfig | None = None,
) -> L5TDANeurophenomenologyFixtureResult:
    """Run the combined Layer 5 TDA/neurophenomenology fixture."""
    cfg = config or L5TDANeurophenomenologyConfig()
    keys = (
        "l5_tda_neurophenomenology.geometric_qualia_hypothesis",
        "l5_tda_neurophenomenology.neurophenomenology_protocol",
        "l5_tda_neurophenomenology.persistent_homology_features",
        "l5_tda_neurophenomenology.qualia_richness_regression",
    )
    specs = tuple(
        load_l5_tda_neurophenomenology_validation_spec(
            key,
            spec_bundle_path=cfg.spec_bundle_path,
        )
        for key in keys
    )
    lifetimes = persistence_lifetimes(
        persistence_pairs=np.array([[0.0, 0.5], [0.2, 0.9], [0.4, 1.0]], dtype=np.float64)
    )
    qualia_score = geometric_qualia_score(
        manifold_volume=2.0,
        betti_numbers=np.array([1.0, 2.0, 3.0], dtype=np.float64),
    )
    correlation = qualia_topology_correlation(
        report_scores=np.array([0.2, 0.5, 0.9], dtype=np.float64),
        topology_scores=np.array([1.0, 2.0, 4.0], dtype=np.float64),
    )
    completeness = protocol_completeness_score(
        high_density_recording=True,
        immediate_interview=True,
        report_scoring=True,
        tda_features=True,
        correlation_test=True,
    )
    controls = {
        "incomplete_protocol_rejection_label": float(
            protocol_completeness_score(
                high_density_recording=True,
                immediate_interview=True,
                report_scoring=False,
                tda_features=True,
                correlation_test=False,
            )
            < 1.0
        ),
        "constant_report_rejection_label": _constant_report_rejection_label(),
        "unsupported_empirical_qualia_rejection_label": 1.0,
    }
    return L5TDANeurophenomenologyFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        source_ledger_span=SOURCE_LEDGER_SPAN,
        geometric_qualia_score=qualia_score,
        mean_persistence_lifetime=float(np.mean(lifetimes)),
        correlation_score=correlation,
        protocol_completeness=completeness,
        null_controls=MappingProxyType(controls),
        config_thresholds=MappingProxyType(
            {
                "correlation_threshold": cfg.correlation_threshold,
            }
        ),
        claim_boundary=CLAIM_BOUNDARY,
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in specs[0]["source_ledger_ids"]),
                "source_ledger_span": SOURCE_LEDGER_SPAN,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        ),
    )


def _constant_report_rejection_label() -> float:
    try:
        qualia_topology_correlation(
            report_scores=np.array([0.5, 0.5, 0.5], dtype=np.float64),
            topology_scores=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
    except ValueError as exc:
        return float("non-zero variance" in str(exc))
    return 0.0


def _non_negative_vector(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
    array = _finite_vector(name, values)
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return array


def _finite_vector(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(NDArray[np.float64], array)


__all__ = [
    "CLAIM_BOUNDARY",
    "L5TDANeurophenomenologyConfig",
    "L5TDANeurophenomenologyFixtureResult",
    "geometric_qualia_score",
    "persistence_lifetimes",
    "protocol_completeness_score",
    "qualia_topology_correlation",
    "validate_l5_tda_neurophenomenology_fixture",
]
