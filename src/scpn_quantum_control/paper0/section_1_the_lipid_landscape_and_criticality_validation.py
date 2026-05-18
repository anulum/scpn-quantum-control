# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 1. The Lipid Landscape and Criticality validation
"""Source-accounting checks for Paper 0 1. The Lipid Landscape and Criticality records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 1 the lipid landscape and criticality source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04746", "P0R04753")


@dataclass(frozen=True, slots=True)
class Section1TheLipidLandscapeAndCriticalityConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 3
    next_source_boundary: str = "P0R04754"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R04754":
            raise ValueError("next_source_boundary must equal P0R04754")


@dataclass(frozen=True, slots=True)
class Section1TheLipidLandscapeAndCriticalityFixtureResult:
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


def classify_section_1_the_lipid_landscape_and_criticality_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "1_the_lipid_landscape_and_criticality": "1_the_lipid_landscape_and_criticality_source_boundary",
        "2_the_central_role_of_cholesterol": "2_the_central_role_of_cholesterol_source_boundary",
        "3_lipid_rafts_the_organising_platforms_for_iet": "3_lipid_rafts_the_organising_platforms_for_iet_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_1_the_lipid_landscape_and_criticality component"
        ) from exc


def section_1_the_lipid_landscape_and_criticality_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "1. The Lipid Landscape and Criticality",
        "source_span": "P0R04746-P0R04753",
        "component_count": "3",
        "next_boundary": "P0R04754",
        "component_1": "1. The Lipid Landscape and Criticality",
        "component_2": "2. The Central Role of Cholesterol",
        "component_3": "3. Lipid Rafts: The Organising Platforms for IET",
    }


def validate_section_1_the_lipid_landscape_and_criticality_fixture(
    config: Section1TheLipidLandscapeAndCriticalityConfig | None = None,
) -> Section1TheLipidLandscapeAndCriticalityFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section1TheLipidLandscapeAndCriticalityConfig()
    components = (
        "1_the_lipid_landscape_and_criticality",
        "2_the_central_role_of_cholesterol",
        "3_lipid_rafts_the_organising_platforms_for_iet",
    )
    return Section1TheLipidLandscapeAndCriticalityFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_1_the_lipid_landscape_and_criticality_component(component)
            for component in components
        },
        labels=section_1_the_lipid_landscape_and_criticality_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "1_the_lipid_landscape_and_criticality_is_not_empirical_validation_evidence": 1.0,
            "2_the_central_role_of_cholesterol_is_not_empirical_validation_evidence": 1.0,
            "3_lipid_rafts_the_organising_platforms_for_iet_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4746, 4754)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_1_the_lipid_landscape_and_criticality_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section1TheLipidLandscapeAndCriticalityConfig",
    "Section1TheLipidLandscapeAndCriticalityFixtureResult",
    "classify_section_1_the_lipid_landscape_and_criticality_component",
    "section_1_the_lipid_landscape_and_criticality_labels",
    "validate_section_1_the_lipid_landscape_and_criticality_fixture",
]
