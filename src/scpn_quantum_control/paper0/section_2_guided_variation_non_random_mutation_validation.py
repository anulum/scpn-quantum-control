# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. Guided Variation (Non-Random Mutation): validation
"""Source-accounting checks for Paper 0 2. Guided Variation (Non-Random Mutation): records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 2 guided variation non random mutation source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R06031", "P0R06038")


@dataclass(frozen=True, slots=True)
class Section2GuidedVariationNonRandomMutationConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 3
    next_source_boundary: str = "P0R06039"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R06039":
            raise ValueError("next_source_boundary must equal P0R06039")


@dataclass(frozen=True, slots=True)
class Section2GuidedVariationNonRandomMutationFixtureResult:
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


def classify_section_2_guided_variation_non_random_mutation_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "2_guided_variation_non_random_mutation": "2_guided_variation_non_random_mutation_source_boundary",
        "3_teleological_dynamics_rg_flow": "3_teleological_dynamics_rg_flow_source_boundary",
        "4_the_co_evolutionary_spiral": "4_the_co_evolutionary_spiral_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_2_guided_variation_non_random_mutation component"
        ) from exc


def section_2_guided_variation_non_random_mutation_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "2. Guided Variation (Non-Random Mutation):",
        "source_span": "P0R06031-P0R06038",
        "component_count": "3",
        "next_boundary": "P0R06039",
        "component_1": "2. Guided Variation (Non-Random Mutation):",
        "component_2": "3. Teleological Dynamics (RG Flow):",
        "component_3": "4. The Co-Evolutionary Spiral:",
    }


def validate_section_2_guided_variation_non_random_mutation_fixture(
    config: Section2GuidedVariationNonRandomMutationConfig | None = None,
) -> Section2GuidedVariationNonRandomMutationFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section2GuidedVariationNonRandomMutationConfig()
    components = (
        "2_guided_variation_non_random_mutation",
        "3_teleological_dynamics_rg_flow",
        "4_the_co_evolutionary_spiral",
    )
    return Section2GuidedVariationNonRandomMutationFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_2_guided_variation_non_random_mutation_component(component)
            for component in components
        },
        labels=section_2_guided_variation_non_random_mutation_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "2_guided_variation_non_random_mutation_is_not_empirical_validation_evidence": 1.0,
            "3_teleological_dynamics_rg_flow_is_not_empirical_validation_evidence": 1.0,
            "4_the_co_evolutionary_spiral_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(6031, 6039)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_2_guided_variation_non_random_mutation_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section2GuidedVariationNonRandomMutationConfig",
    "Section2GuidedVariationNonRandomMutationFixtureResult",
    "classify_section_2_guided_variation_non_random_mutation_component",
    "section_2_guided_variation_non_random_mutation_labels",
    "validate_section_2_guided_variation_non_random_mutation_fixture",
]
