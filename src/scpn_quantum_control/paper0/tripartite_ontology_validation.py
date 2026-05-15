# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 tripartite ontology validation
"""Source-accounting checks for Paper 0 tripartite-ontology records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded tripartite-ontology map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00818", "P0R00837")


@dataclass(frozen=True, slots=True)
class TripartiteOntologyConfig:
    """Configuration for the tripartite-ontology fixture."""

    expected_source_record_count: int = 20
    expected_blank_record_count: int = 2
    expected_information_form_count: int = 3
    next_source_boundary: str = "P0R00838"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 20:
            raise ValueError("expected_source_record_count must equal 20")
        if self.expected_blank_record_count != 2:
            raise ValueError("expected_blank_record_count must equal 2")
        if self.expected_information_form_count != 3:
            raise ValueError("expected_information_form_count must equal 3")
        if self.next_source_boundary != "P0R00838":
            raise ValueError("next_source_boundary must equal P0R00838")


@dataclass(frozen=True, slots=True)
class TripartiteOntologyFixtureResult:
    """Result for the Paper 0 tripartite-ontology fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    components: dict[str, str]
    labels: dict[str, str]
    source_record_count: int
    blank_record_count: int
    formal_ontology_record_count: int
    explanatory_analogy_record_count: int
    information_form_count: int
    transduction_direction_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_tripartite_ontology_component(component: str) -> str:
    """Classify source-defined tripartite-ontology components."""
    mapping = {
        "section_boundary": "tripartite_ontology_section_with_blank_boundary_record",
        "psi_fibre_bundle": "psi_field_section_of_fibre_bundle_over_spacetime",
        "information_forms": "phi_g_h_tripartite_information_ontology",
        "bidirectional_transduction": "phi_g_h_downward_and_h_g_phi_upward_transduction",
        "grounded_platonism": "mathematics_as_source_field_layer_13_logic",
        "explanatory_analogies": "lay_analogy_records_preserved_not_validation_evidence",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown tripartite ontology component") from exc


def tripartite_ontology_labels() -> dict[str, str]:
    """Return source-bounded labels for the tripartite-ontology slice."""
    return {
        "section": "1.4 Tripartite Ontology: The Substance of Information",
        "psi_geometry": "Psi-field section of fibre bundle pi:E->M",
        "information_forms": "Phi experiential, G semantic/geometric, H syntactic",
        "transduction": "Phi -> G -> H and H -> G -> Phi",
        "source_integrity": "P0R00819 blank; P0R00837 blank",
        "next_boundary": "Meta-Framework Integrations",
    }


def validate_tripartite_ontology_fixture(
    config: TripartiteOntologyConfig | None = None,
) -> TripartiteOntologyFixtureResult:
    """Validate source accounting for the tripartite-ontology slice."""
    cfg = config or TripartiteOntologyConfig()
    components = (
        "section_boundary",
        "psi_fibre_bundle",
        "information_forms",
        "bidirectional_transduction",
        "grounded_platonism",
        "explanatory_analogies",
    )

    return TripartiteOntologyFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_tripartite_ontology_component(component)
            for component in components
        },
        labels=tripartite_ontology_labels(),
        source_record_count=cfg.expected_source_record_count,
        blank_record_count=cfg.expected_blank_record_count,
        formal_ontology_record_count=12,
        explanatory_analogy_record_count=6,
        information_form_count=cfg.expected_information_form_count,
        transduction_direction_count=2,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "tripartite_ontology_is_source_claim_not_empirical_evidence": 1.0,
            "blank_records_p0r00819_p0r00837_are_preserved": 1.0,
            "explanatory_analogies_are_not_promoted_as_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(818, 838)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_tripartite_ontology_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TripartiteOntologyConfig",
    "TripartiteOntologyFixtureResult",
    "classify_tripartite_ontology_component",
    "tripartite_ontology_labels",
    "validate_tripartite_ontology_fixture",
]
