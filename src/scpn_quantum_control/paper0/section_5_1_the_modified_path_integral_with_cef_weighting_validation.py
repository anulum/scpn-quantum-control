# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 5.1 The Modified Path Integral with CEF Weighting validation
"""Source-accounting checks for Paper 0 5.1 The Modified Path Integral with CEF Weighting records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 5 1 the modified path integral with cef weighting source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03869", "P0R03931")


@dataclass(frozen=True, slots=True)
class Section51TheModifiedPathIntegralWithCefWeightingConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 63
    expected_component_count: int = 2
    next_source_boundary: str = "P0R03932"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 63:
            raise ValueError("expected_source_record_count must equal 63")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R03932":
            raise ValueError("next_source_boundary must equal P0R03932")


@dataclass(frozen=True, slots=True)
class Section51TheModifiedPathIntegralWithCefWeightingFixtureResult:
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


def classify_section_5_1_the_modified_path_integral_with_cef_weighting_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "5_1_the_modified_path_integral_with_cef_weighting": "5_1_the_modified_path_integral_with_cef_weighting_source_boundary",
        "5_2_new_falsifiable_predictions": "5_2_new_falsifiable_predictions_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_5_1_the_modified_path_integral_with_cef_weighting component"
        ) from exc


def section_5_1_the_modified_path_integral_with_cef_weighting_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "5.1 The Modified Path Integral with CEF Weighting",
        "source_span": "P0R03869-P0R03931",
        "component_count": "2",
        "next_boundary": "P0R03932",
        "component_1": "5.1 The Modified Path Integral with CEF Weighting",
        "component_2": "5.2 New Falsifiable Predictions",
    }


def validate_section_5_1_the_modified_path_integral_with_cef_weighting_fixture(
    config: Section51TheModifiedPathIntegralWithCefWeightingConfig | None = None,
) -> Section51TheModifiedPathIntegralWithCefWeightingFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section51TheModifiedPathIntegralWithCefWeightingConfig()
    components = (
        "5_1_the_modified_path_integral_with_cef_weighting",
        "5_2_new_falsifiable_predictions",
    )
    return Section51TheModifiedPathIntegralWithCefWeightingFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_5_1_the_modified_path_integral_with_cef_weighting_component(
                component
            )
            for component in components
        },
        labels=section_5_1_the_modified_path_integral_with_cef_weighting_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "5_1_the_modified_path_integral_with_cef_weighting_is_not_empirical_validation_evidence": 1.0,
            "5_2_new_falsifiable_predictions_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3869, 3932)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_5_1_the_modified_path_integral_with_cef_weighting_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section51TheModifiedPathIntegralWithCefWeightingConfig",
    "Section51TheModifiedPathIntegralWithCefWeightingFixtureResult",
    "classify_section_5_1_the_modified_path_integral_with_cef_weighting_component",
    "section_5_1_the_modified_path_integral_with_cef_weighting_labels",
    "validate_section_5_1_the_modified_path_integral_with_cef_weighting_fixture",
]
