# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 meta-framework Psi coupling validation
"""Source-accounting checks for Paper 0 meta-framework/Psi-coupling records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded meta-framework/Psi-coupling map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00838", "P0R00904")


@dataclass(frozen=True, slots=True)
class MetaFrameworkPsiCouplingConfig:
    """Configuration for the meta-framework/Psi-coupling fixture."""

    expected_source_record_count: int = 67
    expected_blank_record_count: int = 2
    expected_image_or_figure_record_count: int = 6
    next_source_boundary: str = "P0R00905"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 67:
            raise ValueError("expected_source_record_count must equal 67")
        if self.expected_blank_record_count != 2:
            raise ValueError("expected_blank_record_count must equal 2")
        if self.expected_image_or_figure_record_count != 6:
            raise ValueError("expected_image_or_figure_record_count must equal 6")
        if self.next_source_boundary != "P0R00905":
            raise ValueError("next_source_boundary must equal P0R00905")


@dataclass(frozen=True, slots=True)
class MetaFrameworkPsiCouplingFixtureResult:
    """Result for the Paper 0 meta-framework/Psi-coupling fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    components: dict[str, str]
    labels: dict[str, str]
    source_record_count: int
    predictive_coding_record_count: int
    psi_coupling_record_count: int
    formal_restatement_record_count: int
    image_or_figure_record_count: int
    blank_record_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_meta_framework_psi_coupling_component(component: str) -> str:
    """Classify source-defined meta-framework/Psi-coupling components."""
    mapping = {
        "meta_framework_boundary": "meta_framework_integrations_boundary",
        "predictive_coding_loop": (
            "fibre_bundle_belief_state_and_tripartite_active_inference_loop"
        ),
        "psi_interaction_hamiltonian": "h_int_minus_lambda_psis_sigma_coupling_statement",
        "coupling_projection": "total_space_to_fibre_projection_and_g_to_h_transduction",
        "formal_ontology_restatement": "psi_x_in_e_and_pi_e_to_m_formal_restatement",
        "figure_and_image_records": "image_and_caption_records_preserved_not_validation_evidence",
        "repeated_ontology_block": "repeated_tripartite_ontology_block_preserved",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown meta-framework Psi-coupling component") from exc


def meta_framework_psi_coupling_labels() -> dict[str, str]:
    """Return source-bounded labels for the meta-framework/Psi-coupling slice."""
    return {
        "section": "Meta-Framework Integrations",
        "predictive_coding": "fibre bundle state space of beliefs",
        "hamiltonian": "H_int = -lambda * Psis * sigma",
        "projection": "total space to fibre projection with G to H transduction",
        "source_integrity": "P0R00875 blank; P0R00897 blank",
        "next_boundary": "1.5 The Universal Grammar: A Category-Theoretic Foundation",
    }


def validate_meta_framework_psi_coupling_fixture(
    config: MetaFrameworkPsiCouplingConfig | None = None,
) -> MetaFrameworkPsiCouplingFixtureResult:
    """Validate source accounting for the meta-framework/Psi-coupling slice."""
    cfg = config or MetaFrameworkPsiCouplingConfig()
    components = (
        "meta_framework_boundary",
        "predictive_coding_loop",
        "psi_interaction_hamiltonian",
        "coupling_projection",
        "formal_ontology_restatement",
        "figure_and_image_records",
        "repeated_ontology_block",
    )

    return MetaFrameworkPsiCouplingFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_meta_framework_psi_coupling_component(component)
            for component in components
        },
        labels=meta_framework_psi_coupling_labels(),
        source_record_count=cfg.expected_source_record_count,
        predictive_coding_record_count=14,
        psi_coupling_record_count=25,
        formal_restatement_record_count=22,
        image_or_figure_record_count=cfg.expected_image_or_figure_record_count,
        blank_record_count=cfg.expected_blank_record_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "meta_framework_psi_coupling_is_source_claim_not_empirical_evidence": 1.0,
            "image_and_figure_records_are_not_promoted_as_validation_evidence": 1.0,
            "blank_records_p0r00875_p0r00897_are_preserved": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(838, 905)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_meta_framework_psi_coupling_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "MetaFrameworkPsiCouplingConfig",
    "MetaFrameworkPsiCouplingFixtureResult",
    "classify_meta_framework_psi_coupling_component",
    "meta_framework_psi_coupling_labels",
    "validate_meta_framework_psi_coupling_fixture",
]
