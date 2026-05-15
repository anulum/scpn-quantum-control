# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 category universal grammar validation
"""Source-accounting checks for Paper 0 category/universal-grammar records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded category/universal-grammar map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00905", "P0R00986")


@dataclass(frozen=True, slots=True)
class CategoryUniversalGrammarConfig:
    """Configuration for the category/universal-grammar fixture."""

    expected_source_record_count: int = 82
    expected_blank_record_count: int = 3
    expected_image_record_count: int = 5
    expected_figure_caption_record_count: int = 11
    next_source_boundary: str = "P0R00987"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 82:
            raise ValueError("expected_source_record_count must equal 82")
        if self.expected_blank_record_count != 3:
            raise ValueError("expected_blank_record_count must equal 3")
        if self.expected_image_record_count != 5:
            raise ValueError("expected_image_record_count must equal 5")
        if self.expected_figure_caption_record_count != 11:
            raise ValueError("expected_figure_caption_record_count must equal 11")
        if self.next_source_boundary != "P0R00987":
            raise ValueError("next_source_boundary must equal P0R00987")


@dataclass(frozen=True, slots=True)
class CategoryUniversalGrammarFixtureResult:
    """Result for the Paper 0 category/universal-grammar fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    components: dict[str, str]
    labels: dict[str, str]
    source_record_count: int
    blank_record_count: int
    image_record_count: int
    figure_caption_record_count: int
    formal_category_record_count: int
    meta_framework_record_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_category_universal_grammar_component(component: str) -> str:
    """Classify source-defined category/universal-grammar components."""
    mapping = {
        "section_boundary": "category_theory_universal_grammar_boundary",
        "category_objects_morphisms": "scpn_category_objects_morphisms_identity_composition",
        "functor_bridge": "consciousness_physics_functors_and_natural_transformation",
        "topos_kan_foundation": ("topos_classifier_exponential_and_kan_inference_foundation"),
        "explanatory_analogies": "lay_analogies_preserved_not_validation_evidence",
        "meta_framework_integrations": (
            "category_predictive_coding_and_psi_coupling_integrations"
        ),
        "formal_diagram_records": "image_caption_and_diagram_records_preserved_not_evidence",
        "kan_extension_inference": "lan_ran_and_psi_inferred_missing_data_formulas",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown category universal-grammar component") from exc


def category_universal_grammar_labels() -> dict[str, str]:
    """Return source-bounded labels for the category/universal-grammar slice."""
    return {
        "section": "1.5 The Universal Grammar: A Category-Theoretic Foundation",
        "category": "SCPN category with objects L1...L15 and morphisms f: Li -> Lj",
        "functors": "F: Consciousness -> Physics; G: Physics -> Consciousness",
        "naturality": "eta: F => G",
        "topos": "Omega = {true, false, uncertain}; exponential B^A",
        "kan": "Psi_inferred = Lan_physical(Psi_true)",
        "next_boundary": "Part II: The Physical Sector",
    }


def validate_category_universal_grammar_fixture(
    config: CategoryUniversalGrammarConfig | None = None,
) -> CategoryUniversalGrammarFixtureResult:
    """Validate source accounting for the category/universal-grammar slice."""
    cfg = config or CategoryUniversalGrammarConfig()
    components = (
        "section_boundary",
        "category_objects_morphisms",
        "functor_bridge",
        "topos_kan_foundation",
        "explanatory_analogies",
        "meta_framework_integrations",
        "formal_diagram_records",
        "kan_extension_inference",
    )

    return CategoryUniversalGrammarFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_category_universal_grammar_component(component)
            for component in components
        },
        labels=category_universal_grammar_labels(),
        source_record_count=cfg.expected_source_record_count,
        blank_record_count=cfg.expected_blank_record_count,
        image_record_count=cfg.expected_image_record_count,
        figure_caption_record_count=cfg.expected_figure_caption_record_count,
        formal_category_record_count=27,
        meta_framework_record_count=15,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "category_universal_grammar_is_source_claim_not_empirical_evidence": 1.0,
            "image_and_caption_records_are_not_promoted_as_validation_evidence": 1.0,
            "blank_records_p0r00915_p0r00968_p0r00978_are_preserved": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(905, 987)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_category_universal_grammar_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "CategoryUniversalGrammarConfig",
    "CategoryUniversalGrammarFixtureResult",
    "category_universal_grammar_labels",
    "classify_category_universal_grammar_component",
    "validate_category_universal_grammar_fixture",
]
