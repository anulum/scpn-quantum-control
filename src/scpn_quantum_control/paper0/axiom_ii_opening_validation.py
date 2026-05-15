# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom II opening validation
"""Source-accounting checks for Paper 0 Axiom II opening records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded Axiom II opening map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00761", "P0R00769")


@dataclass(frozen=True, slots=True)
class AxiomIIOpeningConfig:
    """Configuration for the Axiom II opening fixture."""

    expected_heading_record_count: int = 4
    expected_axiom_statement_count: int = 1
    next_source_boundary: str = "P0R00770"

    def __post_init__(self) -> None:
        if self.expected_heading_record_count != 4:
            raise ValueError("expected_heading_record_count must equal 4")
        if self.expected_axiom_statement_count != 1:
            raise ValueError("expected_axiom_statement_count must equal 1")
        if self.next_source_boundary != "P0R00770":
            raise ValueError("next_source_boundary must equal P0R00770")


@dataclass(frozen=True, slots=True)
class AxiomIIOpeningFixtureResult:
    """Result for the Paper 0 Axiom II opening fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    components: dict[str, str]
    labels: dict[str, str]
    source_record_count: int
    heading_record_count: int
    axiom_statement_count: int
    falsifiability_boundary_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_axiom_ii_component(component: str) -> str:
    """Classify source-defined Axiom II opening components."""
    mapping = {
        "section_headings": "axiom_ii_navigation_headings",
        "source_material": "interactions_as_information_geometry",
        "ontology_to_dynamics": "psi_field_substance_to_interaction_language",
        "interaction_axiom": "informational_geometric_falsifiable_hypothesis",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown Axiom II opening component") from exc


def axiom_ii_opening_labels() -> dict[str, str]:
    """Return source-bounded labels for the Axiom II opening slice."""
    return {
        "section": "Axiom II: The Language of Information Geometry",
        "axiom": "Axiom II: The Axiom of Interaction (Information Geometry)",
        "next_boundary": 'The Central Problem: The Geometry of the "Infoton"',
    }


def validate_axiom_ii_opening_fixture(
    config: AxiomIIOpeningConfig | None = None,
) -> AxiomIIOpeningFixtureResult:
    """Validate source accounting for the Axiom II opening slice."""
    cfg = config or AxiomIIOpeningConfig()
    components = (
        "section_headings",
        "source_material",
        "ontology_to_dynamics",
        "interaction_axiom",
    )

    return AxiomIIOpeningFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={component: classify_axiom_ii_component(component) for component in components},
        labels=axiom_ii_opening_labels(),
        source_record_count=9,
        heading_record_count=cfg.expected_heading_record_count,
        axiom_statement_count=cfg.expected_axiom_statement_count,
        falsifiability_boundary_count=1,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "axiom_statement_is_not_empirical_validation": 1.0,
            "falsifiability_boundary_requires_downstream_protocols": 1.0,
            "information_geometry_claim_is_source_hypothesis_only": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(761, 770)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_axiom_ii_opening_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "AxiomIIOpeningConfig",
    "AxiomIIOpeningFixtureResult",
    "axiom_ii_opening_labels",
    "classify_axiom_ii_component",
    "validate_axiom_ii_opening_fixture",
]
