# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Locus of the Interaction: validation
"""Source-accounting checks for Paper 0 The Locus of the Interaction: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = (
    "source-bounded the locus of the interaction source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R02551", "P0R02565")


@dataclass(frozen=True, slots=True)
class TheLocusOfTheInteractionConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 15
    expected_component_count: int = 5
    next_source_boundary: str = "P0R02566"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 15:
            raise ValueError("expected_source_record_count must equal 15")
        if self.expected_component_count != 5:
            raise ValueError("expected_component_count must equal 5")
        if self.next_source_boundary != "P0R02566":
            raise ValueError("next_source_boundary must equal P0R02566")


@dataclass(frozen=True, slots=True)
class TheLocusOfTheInteractionFixtureResult:
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


def classify_the_locus_of_the_interaction_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_locus_of_the_interaction": "the_locus_of_the_interaction_source_boundary",
        "holonomy_as_memory_of_coupling": "holonomy_as_memory_of_coupling_source_boundary",
        "the_dynamic_visualisation_the_scpn_torus": "the_dynamic_visualisation_the_scpn_torus_source_boundary",
        "conceptual_specification_of_the_scpn_torus": "conceptual_specification_of_the_scpn_torus_source_boundary",
        "1_the_geometry_and_flow": "1_the_geometry_and_flow_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown the_locus_of_the_interaction component") from exc


def the_locus_of_the_interaction_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Locus of the Interaction:",
        "source_span": "P0R02551-P0R02565",
        "component_count": "5",
        "next_boundary": "P0R02566",
        "component_1": "The Locus of the Interaction:",
        "component_2": "Holonomy as Memory of Coupling:",
        "component_3": "The Dynamic Visualisation: The SCPN Torus",
        "component_4": "Conceptual Specification of the SCPN Torus:",
        "component_5": "1. The Geometry and Flow:",
    }


def validate_the_locus_of_the_interaction_fixture(
    config: TheLocusOfTheInteractionConfig | None = None,
) -> TheLocusOfTheInteractionFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TheLocusOfTheInteractionConfig()
    components = (
        "the_locus_of_the_interaction",
        "holonomy_as_memory_of_coupling",
        "the_dynamic_visualisation_the_scpn_torus",
        "conceptual_specification_of_the_scpn_torus",
        "1_the_geometry_and_flow",
    )
    return TheLocusOfTheInteractionFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_locus_of_the_interaction_component(component)
            for component in components
        },
        labels=the_locus_of_the_interaction_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_locus_of_the_interaction_is_not_empirical_validation_evidence": 1.0,
            "holonomy_as_memory_of_coupling_is_not_empirical_validation_evidence": 1.0,
            "the_dynamic_visualisation_the_scpn_torus_is_not_empirical_validation_evidence": 1.0,
            "conceptual_specification_of_the_scpn_torus_is_not_empirical_validation_evidence": 1.0,
            "1_the_geometry_and_flow_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(2551, 2566)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_locus_of_the_interaction_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TheLocusOfTheInteractionConfig",
    "TheLocusOfTheInteractionFixtureResult",
    "classify_the_locus_of_the_interaction_component",
    "the_locus_of_the_interaction_labels",
    "validate_the_locus_of_the_interaction_fixture",
]
