# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Anulum Collection mandate validation
"""Executable source-accounting checks for the Paper 0 collection mandate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded Anulum Collection mandate; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00401", "P0R00435")
COUPLING_EQUATION = "H_int = -lambda * Psi_s * sigma"


@dataclass(frozen=True, slots=True)
class AnulumCollectionMandateConfig:
    """Configuration for the Anulum Collection mandate fixture."""

    expected_blank_separator_count: int = 2
    next_source_boundary: str = "P0R00436"

    def __post_init__(self) -> None:
        if self.expected_blank_separator_count != 2:
            raise ValueError("expected_blank_separator_count must equal 2")
        if self.next_source_boundary != "P0R00436":
            raise ValueError("next_source_boundary must equal P0R00436")


@dataclass(frozen=True, slots=True)
class AnulumCollectionMandateFixtureResult:
    """Result for the Paper 0 Anulum Collection mandate fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    book_roles: dict[str, str]
    meta_framework_integrations: dict[str, str]
    master_publications: dict[str, str]
    book_count: int
    meta_framework_count: int
    blank_separator_count: int
    validation_suite_range: tuple[str, str]
    coupling_equation: str
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_book_role(book: str) -> str:
    """Classify a source book label into its programme role."""
    mapping = {
        "book_i": "foundational_physics",
        "book_ii": "engineering_and_architecture",
        "book_iii": "philosophical_interpretation",
        "book_iv": "boundary_probe",
        "book_v": "practitioner_tooling",
    }
    try:
        return mapping[book]
    except KeyError as exc:
        raise ValueError("unknown Anulum Collection book") from exc


def classify_meta_framework_integration(integration: str) -> str:
    """Classify mandate meta-framework entries into their operational role."""
    mapping = {
        "predictive_coding": "research_process_hpc",
        "paper0_deep_priors": "slow_prior_source",
        "papers_1_16_generative_cascade": "layer_hypothesis_cascade",
        "part_iii_prediction_error": "validation_error_minimisation",
        "psi_sigma_coupling": "layer_sigma_lambda_measurement_plan",
    }
    try:
        return mapping[integration]
    except KeyError as exc:
        raise ValueError("unknown meta-framework integration") from exc


def master_publication_entries() -> dict[str, str]:
    """Return the source-bounded master-publication map."""
    return {
        "book_i": "The Anulum Framework",
        "book_ii": "The Sentient-Consciousness Projection Network",
        "book_iii": "Metatron's Coda",
        "book_iv": "The Godelian Koans",
        "book_v": "VIBRANA",
        "paper0_location": "Paper 0: The Foundational Framework - You are Here",
    }


def validate_anulum_collection_mandate_fixture(
    config: AnulumCollectionMandateConfig | None = None,
) -> AnulumCollectionMandateFixtureResult:
    """Validate source accounting for the Anulum Collection mandate run."""
    cfg = config or AnulumCollectionMandateConfig()
    books = tuple(f"book_{label}" for label in ("i", "ii", "iii", "iv", "v"))
    integrations = (
        "predictive_coding",
        "paper0_deep_priors",
        "papers_1_16_generative_cascade",
        "part_iii_prediction_error",
        "psi_sigma_coupling",
    )
    book_roles = {book: classify_book_role(book) for book in books}
    meta_frameworks = {
        integration: classify_meta_framework_integration(integration)
        for integration in integrations
    }

    return AnulumCollectionMandateFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        book_roles=book_roles,
        meta_framework_integrations=meta_frameworks,
        master_publications=master_publication_entries(),
        book_count=len(book_roles),
        meta_framework_count=len(meta_frameworks),
        blank_separator_count=cfg.expected_blank_separator_count,
        validation_suite_range=("Papers 17", "Papers 20"),
        coupling_equation=COUPLING_EQUATION,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "programme_map_is_not_empirical_validation": 1.0,
            "unmeasured_lambda_rejection_label": 1.0,
            "unisolated_sigma_rejection_label": 1.0,
            "curriculum_analogy_not_ontology_label": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(401, 436)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_mandate_mapping_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "COUPLING_EQUATION",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "AnulumCollectionMandateConfig",
    "AnulumCollectionMandateFixtureResult",
    "classify_book_role",
    "classify_meta_framework_integration",
    "master_publication_entries",
    "validate_anulum_collection_mandate_fixture",
]
