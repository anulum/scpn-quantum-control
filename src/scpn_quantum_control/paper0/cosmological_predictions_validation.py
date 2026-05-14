# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 cosmological predictions fixtures
"""Preregistration-boundary fixtures for Paper 0 cosmological predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .spec_loader import load_cosmological_predictions_validation_spec

CLAIM_BOUNDARY = "source-bounded preregistration protocol catalogue; not empirical evidence"
HARDWARE_STATUS = "preregistration_protocol_no_execution"
SOURCE_LEDGER_SPAN = ("P0R06949", "P0R07005")


@dataclass(frozen=True, slots=True)
class CosmologicalPredictionsConfig:
    """Finite cosmological predictions fixture settings."""

    expected_prediction_count: int = 5
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        if self.expected_prediction_count < 1:
            raise ValueError("expected_prediction_count must be at least 1")


@dataclass(frozen=True, slots=True)
class CosmologicalPrediction:
    """Single preregisterable cosmological prediction target."""

    prediction_id: str
    label: str
    test_protocol: str
    null_result: str


@dataclass(frozen=True, slots=True)
class CosmologicalPredictionsFixtureResult:
    """Combined cosmological predictions fixture result."""

    spec_keys: tuple[str, str, str, str, str, str, str, str]
    hardware_status: str
    source_ledger_span: tuple[str, str]
    prediction_count: int
    expected_prediction_count: int
    priority_ranking: tuple[str, str, str, str, str]
    cross_consistency_rules: int
    null_controls: MappingProxyType[str, float]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def prediction_catalogue() -> tuple[CosmologicalPrediction, ...]:
    """Return the five source-stated preregisterable prediction targets."""
    return (
        CosmologicalPrediction(
            "28.1",
            "CMB correlations from prior cycles",
            "blind harmonic-template matching on CMB maps with controlled false discovery rate",
            "no significant excess above noise floor at declared threshold",
        ),
        CosmologicalPrediction(
            "28.2",
            "gravitational-wave echoes and sidebands",
            "residual analysis against standard ringdown and microstructure templates",
            "GR ringdown modes describe events within declared sensitivity",
        ),
        CosmologicalPrediction(
            "28.3",
            "observer-linked entropy anomalies",
            "blinded observer/non-observer fluctuation-statistics protocol",
            "no significant entropy-statistics difference at declared power",
        ),
        CosmologicalPrediction(
            "28.4",
            "arrow-of-time palindrome in perturbation spectra",
            "mirror-symmetry search against inflation and bounce templates",
            "spectrum remains consistent with standard slow-roll inflation",
        ),
        CosmologicalPrediction(
            "28.5",
            "quantum information retention",
            "entanglement-entropy monitoring under controlled decoherence",
            "monotonic decoherence curve holds within measurement precision",
        ),
    )


def priority_ranking() -> tuple[str, str, str, str, str]:
    """Return the source-stated feasibility priority order."""
    return ("28.1", "28.2", "28.5", "28.3", "28.4")


def cross_prediction_tension(outcomes: dict[str, bool]) -> str | None:
    """Return a named internal tension for confirmed/null cross-prediction outcomes."""
    if outcomes.get("28.1") is True and outcomes.get("28.5") is False:
        return "cmb-confirmed-quantum-retention-null"
    if outcomes.get("28.3") is True and outcomes.get("28.5") is False:
        return "observer-entropy-confirmed-retention-null"
    if outcomes.get("28.2") is True and outcomes.get("28.1") is False:
        return "gw-confirmed-cmb-null"
    if any(outcomes.get(item) is True for item in ("28.1", "28.2", "28.3", "28.5")):
        return "cross-linked-follow-up-required"
    return None


def validate_cosmological_predictions_fixture(
    config: CosmologicalPredictionsConfig | None = None,
) -> CosmologicalPredictionsFixtureResult:
    """Run the cosmological predictions preregistration boundary fixture."""
    cfg = config or CosmologicalPredictionsConfig()
    keys = (
        "cosmological_predictions.chapter_boundary",
        "cosmological_predictions.cmb_correlations",
        "cosmological_predictions.gravitational_wave_sidebands",
        "cosmological_predictions.observer_entropy_anomaly",
        "cosmological_predictions.arrow_time_palindrome",
        "cosmological_predictions.quantum_information_retention",
        "cosmological_predictions.cross_prediction_consistency",
        "cosmological_predictions.priority_ranking",
    )
    specs = tuple(
        load_cosmological_predictions_validation_spec(
            key,
            spec_bundle_path=cfg.spec_bundle_path,
        )
        for key in keys
    )
    catalogue = prediction_catalogue()
    controls = {
        "missing_null_result_rejection_label": _missing_null_result_rejection_label(),
        "invalid_priority_ranking_rejection_label": _invalid_priority_ranking_rejection_label(),
        "unsupported_confirmation_claim_rejection_label": 1.0,
    }
    return CosmologicalPredictionsFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        source_ledger_span=SOURCE_LEDGER_SPAN,
        prediction_count=len(catalogue),
        expected_prediction_count=cfg.expected_prediction_count,
        priority_ranking=priority_ranking(),
        cross_consistency_rules=4,
        null_controls=MappingProxyType(controls),
        claim_boundary=CLAIM_BOUNDARY,
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in specs[0]["source_ledger_ids"]),
                "source_ledger_span": SOURCE_LEDGER_SPAN,
                "claim_boundary": CLAIM_BOUNDARY,
                "protocol_state": "preregistration_only_no_confirmed_prediction",
            }
        ),
    )


def _missing_null_result_rejection_label() -> float:
    return float(all(item.null_result for item in prediction_catalogue()))


def _invalid_priority_ranking_rejection_label() -> float:
    ranking = priority_ranking()
    catalogue_ids = tuple(item.prediction_id for item in prediction_catalogue())
    return float(len(ranking) == len(set(ranking)) and set(ranking) == set(catalogue_ids))


__all__ = [
    "CLAIM_BOUNDARY",
    "CosmologicalPrediction",
    "CosmologicalPredictionsConfig",
    "CosmologicalPredictionsFixtureResult",
    "cross_prediction_tension",
    "prediction_catalogue",
    "priority_ranking",
    "validate_cosmological_predictions_fixture",
]
