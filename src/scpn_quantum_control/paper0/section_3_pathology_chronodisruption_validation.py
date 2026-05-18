# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 3. Pathology (Chronodisruption): validation
"""Source-accounting checks for Paper 0 3. Pathology (Chronodisruption): records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 3 pathology chronodisruption source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04983", "P0R04992")


@dataclass(frozen=True, slots=True)
class Section3PathologyChronodisruptionConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 10
    expected_component_count: int = 3
    next_source_boundary: str = "P0R04993"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 10:
            raise ValueError("expected_source_record_count must equal 10")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R04993":
            raise ValueError("next_source_boundary must equal P0R04993")


@dataclass(frozen=True, slots=True)
class Section3PathologyChronodisruptionFixtureResult:
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


def classify_section_3_pathology_chronodisruption_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "3_pathology_chronodisruption": "3_pathology_chronodisruption_source_boundary",
        "iii_the_sensory_motor_loop_action_perception_and_agency": "iii_the_sensory_motor_loop_action_perception_and_agency_source_boundary",
        "1_perception_as_geometric_transduction_l1_l5": "1_perception_as_geometric_transduction_l1_l5_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown section_3_pathology_chronodisruption component") from exc


def section_3_pathology_chronodisruption_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "3. Pathology (Chronodisruption):",
        "source_span": "P0R04983-P0R04992",
        "component_count": "3",
        "next_boundary": "P0R04993",
        "component_1": "3. Pathology (Chronodisruption):",
        "component_2": "III. The Sensory-Motor Loop: Action, Perception, and Agency",
        "component_3": "1. Perception as Geometric Transduction (L1-L5)",
    }


def validate_section_3_pathology_chronodisruption_fixture(
    config: Section3PathologyChronodisruptionConfig | None = None,
) -> Section3PathologyChronodisruptionFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section3PathologyChronodisruptionConfig()
    components = (
        "3_pathology_chronodisruption",
        "iii_the_sensory_motor_loop_action_perception_and_agency",
        "1_perception_as_geometric_transduction_l1_l5",
    )
    return Section3PathologyChronodisruptionFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_3_pathology_chronodisruption_component(component)
            for component in components
        },
        labels=section_3_pathology_chronodisruption_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "3_pathology_chronodisruption_is_not_empirical_validation_evidence": 1.0,
            "iii_the_sensory_motor_loop_action_perception_and_agency_is_not_empirical_validation_evidence": 1.0,
            "1_perception_as_geometric_transduction_l1_l5_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4983, 4993)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_3_pathology_chronodisruption_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section3PathologyChronodisruptionConfig",
    "Section3PathologyChronodisruptionFixtureResult",
    "classify_section_3_pathology_chronodisruption_component",
    "section_3_pathology_chronodisruption_labels",
    "validate_section_3_pathology_chronodisruption_fixture",
]
