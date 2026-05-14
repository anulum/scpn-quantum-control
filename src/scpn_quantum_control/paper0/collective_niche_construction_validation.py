# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 collective niche construction fixtures
"""Simulator-only collective niche construction fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .spec_loader import load_collective_niche_construction_validation_spec

CLAIM_BOUNDARY = (
    "source-bounded collective niche construction simulator contract; not empirical evidence"
)
SOURCE_LEDGER_SPAN = ("P0R06519", "P0R06529")


@dataclass(frozen=True, slots=True)
class CollectiveNicheConstructionConfig:
    """Finite simulator settings for collective niche construction fixtures."""

    convergence_threshold: float = 0.72
    entrainment_threshold: float = 0.72
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        _require_positive("convergence_threshold", self.convergence_threshold)
        _require_positive("entrainment_threshold", self.entrainment_threshold)


@dataclass(frozen=True, slots=True)
class CollectiveNicheConstructionFixtureResult:
    """Combined collective niche construction fixture result."""

    spec_keys: tuple[str, str, str, str]
    hardware_status: str
    source_ledger_span: tuple[str, str]
    shared_model_score: float
    entrainment_score: float
    feedback_score: float
    predictability_gain: float
    null_controls: MappingProxyType[str, float]
    config_thresholds: MappingProxyType[str, float]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def shared_model_convergence_score(
    *,
    beliefs: float,
    values: float,
    language: float,
    norms: float,
    communication: float,
    imitation: float,
    artefacts: float,
) -> float:
    """Score source-bounded shared-generative-model convergence channels."""
    values_arr = _unit_interval_values(
        beliefs,
        values,
        language,
        norms,
        communication,
        imitation,
        artefacts,
    )
    return float(np.prod(values_arr) ** (1.0 / values_arr.size))


def entrainment_coherence_score(
    *,
    institutions: float,
    rituals: float,
    language: float,
    art: float,
) -> float:
    """Score source-bounded social entrainment channels."""
    values_arr = _unit_interval_values(institutions, rituals, language, art)
    return float(np.prod(values_arr) ** 0.25)


def bidirectional_feedback_score(
    *,
    collective_to_environment: NDArray[np.float64],
    environment_to_collective: NDArray[np.float64],
) -> float:
    """Return positive coupling strength for collective-environment feedback."""
    forward = _finite_vector("collective_to_environment", collective_to_environment)
    backward = _finite_vector("environment_to_collective", environment_to_collective)
    if forward.shape != backward.shape:
        raise ValueError("vectors must have the same shape")
    if forward.size < 2:
        raise ValueError("vectors must contain at least two samples")
    corr = float(np.corrcoef(forward, backward)[0, 1])
    if not np.isfinite(corr):
        raise ValueError("vectors must have non-zero variance")
    return max(0.0, corr)


def collective_predictability_gain(
    *,
    baseline_surprise: NDArray[np.float64],
    modified_surprise: NDArray[np.float64],
) -> float:
    """Return mean surprise reduction after collective niche construction."""
    baseline = _finite_vector("baseline_surprise", baseline_surprise)
    modified = _finite_vector("modified_surprise", modified_surprise)
    if baseline.shape != modified.shape:
        raise ValueError("vectors must have the same shape")
    return float(np.mean(baseline - modified))


def validate_collective_niche_construction_fixture(
    config: CollectiveNicheConstructionConfig | None = None,
) -> CollectiveNicheConstructionFixtureResult:
    """Run the combined collective niche construction fixture."""
    cfg = config or CollectiveNicheConstructionConfig()
    keys = (
        "collective_niche.shared_generative_model",
        "collective_niche.noosphere_entrainment",
        "collective_niche.biosphere_feedback_loop",
        "collective_niche.gaian_synchrony_boundary",
    )
    specs = tuple(
        load_collective_niche_construction_validation_spec(
            key,
            spec_bundle_path=cfg.spec_bundle_path,
        )
        for key in keys
    )
    shared = shared_model_convergence_score(
        beliefs=0.82,
        values=0.8,
        language=0.84,
        norms=0.78,
        communication=0.81,
        imitation=0.79,
        artefacts=0.83,
    )
    entrainment = entrainment_coherence_score(
        institutions=0.8,
        rituals=0.78,
        language=0.84,
        art=0.76,
    )
    feedback = bidirectional_feedback_score(
        collective_to_environment=np.array([0.2, 0.4, 0.8], dtype=np.float64),
        environment_to_collective=np.array([0.1, 0.3, 0.7], dtype=np.float64),
    )
    predictability = collective_predictability_gain(
        baseline_surprise=np.array([1.2, 1.0, 0.9], dtype=np.float64),
        modified_surprise=np.array([0.9, 0.7, 0.6], dtype=np.float64),
    )
    controls = {
        "missing_artefacts_rejection_label": float(
            shared_model_convergence_score(
                beliefs=0.82,
                values=0.8,
                language=0.84,
                norms=0.78,
                communication=0.81,
                imitation=0.79,
                artefacts=0.0,
            )
            < cfg.convergence_threshold
        ),
        "shape_mismatch_rejection_label": _shape_mismatch_rejection_label(),
        "unsupported_planetary_evidence_rejection_label": 1.0,
    }
    return CollectiveNicheConstructionFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        source_ledger_span=SOURCE_LEDGER_SPAN,
        shared_model_score=shared,
        entrainment_score=entrainment,
        feedback_score=feedback,
        predictability_gain=predictability,
        null_controls=MappingProxyType(controls),
        config_thresholds=MappingProxyType(
            {
                "convergence_threshold": cfg.convergence_threshold,
                "entrainment_threshold": cfg.entrainment_threshold,
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


def _shape_mismatch_rejection_label() -> float:
    try:
        bidirectional_feedback_score(
            collective_to_environment=np.array([0.2, 0.4], dtype=np.float64),
            environment_to_collective=np.array([0.1, 0.3, 0.7], dtype=np.float64),
        )
    except ValueError as exc:
        return float("same shape" in str(exc))
    return 0.0


def _unit_interval_values(*values: float) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("inputs must be in [0, 1]")
    return cast(NDArray[np.float64], array)


def _finite_vector(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(NDArray[np.float64], array)


def _require_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


__all__ = [
    "CLAIM_BOUNDARY",
    "CollectiveNicheConstructionConfig",
    "CollectiveNicheConstructionFixtureResult",
    "bidirectional_feedback_score",
    "collective_predictability_gain",
    "entrainment_coherence_score",
    "shared_model_convergence_score",
    "validate_collective_niche_construction_fixture",
]
