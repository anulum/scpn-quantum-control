# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Novel Falsifiable Predictions from First Principles validation
"""Source-accounting checks for Paper 0 Novel Falsifiable Predictions from First Principles records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded novel falsifiable predictions from first principles source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05162", "P0R05170")


@dataclass(frozen=True, slots=True)
class NovelFalsifiablePredictionsFromFirstPrinciplesConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 3
    next_source_boundary: str = "P0R05171"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R05171":
            raise ValueError("next_source_boundary must equal P0R05171")


@dataclass(frozen=True, slots=True)
class NovelFalsifiablePredictionsFromFirstPrinciplesFixtureResult:
    """Result for this Paper 0 source-accounting fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    components: dict[str, str]
    labels: dict[str, str]
    source_record_count: int
    component_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_novel_falsifiable_predictions_from_first_principles_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "novel_falsifiable_predictions_from_first_principles": "novel_falsifiable_predictions_from_first_principles_source_boundary",
        "prediction_i_information_geometric_deviations_in_quasicritical_systems": "prediction_i_information_geometric_deviations_in_quasicritical_systems_source_boundary",
        "theoretical_derivation": "theoretical_derivation_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown novel_falsifiable_predictions_from_first_principles component"
        ) from exc


def novel_falsifiable_predictions_from_first_principles_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Novel Falsifiable Predictions from First Principles",
        "source_span": "P0R05162-P0R05170",
        "component_count": "3",
        "next_boundary": "P0R05171",
        "component_1": "Novel Falsifiable Predictions from First Principles",
        "component_2": "Prediction I: Information-Geometric Deviations in Quasicritical Systems",
        "component_3": "Theoretical Derivation",
    }


def validate_novel_falsifiable_predictions_from_first_principles_fixture(
    config: NovelFalsifiablePredictionsFromFirstPrinciplesConfig | None = None,
) -> NovelFalsifiablePredictionsFromFirstPrinciplesFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or NovelFalsifiablePredictionsFromFirstPrinciplesConfig()
    components = (
        "novel_falsifiable_predictions_from_first_principles",
        "prediction_i_information_geometric_deviations_in_quasicritical_systems",
        "theoretical_derivation",
    )
    return NovelFalsifiablePredictionsFromFirstPrinciplesFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_novel_falsifiable_predictions_from_first_principles_component(
                component
            )
            for component in components
        },
        labels=novel_falsifiable_predictions_from_first_principles_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "novel_falsifiable_predictions_from_first_principles_is_not_empirical_validation_evidence": 1.0,
            "prediction_i_information_geometric_deviations_in_quasicritical_systems_is_not_empirical_validation_evidence": 1.0,
            "theoretical_derivation_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5162, 5171)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_novel_falsifiable_predictions_from_first_principles_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "NovelFalsifiablePredictionsFromFirstPrinciplesConfig",
    "NovelFalsifiablePredictionsFromFirstPrinciplesFixtureResult",
    "classify_novel_falsifiable_predictions_from_first_principles_component",
    "novel_falsifiable_predictions_from_first_principles_labels",
    "validate_novel_falsifiable_predictions_from_first_principles_fixture",
]
