# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Section 4: Justification for a Teleological Least-Action Principle validation
"""Source-accounting checks for Paper 0 Section 4: Justification for a Teleological Least-Action Principle records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 4 justification for a teleological least action principle source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03638", "P0R03652")


@dataclass(frozen=True, slots=True)
class Section4JustificationForATeleologicalLeastActionPrincipleConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 15
    expected_component_count: int = 3
    next_source_boundary: str = "P0R03653"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 15:
            raise ValueError("expected_source_record_count must equal 15")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R03653":
            raise ValueError("next_source_boundary must equal P0R03653")


@dataclass(frozen=True, slots=True)
class Section4JustificationForATeleologicalLeastActionPrincipleFixtureResult:
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


def classify_section_4_justification_for_a_teleological_least_action_principle_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "section_4_justification_for_a_teleological_least_action_principle": "section_4_justification_for_a_teleological_least_action_principle_source_boundary",
        "4_1_the_problem_of_teleology_in_physics": "4_1_the_problem_of_teleology_in_physics_source_boundary",
        "4_2_causal_entropic_forces_cef_as_the_underlying_mechanism": "4_2_causal_entropic_forces_cef_as_the_underlying_mechanism_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_4_justification_for_a_teleological_least_action_principle component"
        ) from exc


def section_4_justification_for_a_teleological_least_action_principle_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Section 4: Justification for a Teleological Least-Action Principle",
        "source_span": "P0R03638-P0R03652",
        "component_count": "3",
        "next_boundary": "P0R03653",
        "component_1": "Section 4: Justification for a Teleological Least-Action Principle",
        "component_2": "4.1. The Problem of Teleology in Physics",
        "component_3": "4.2. Causal Entropic Forces (CEF) as the Underlying Mechanism",
    }


def validate_section_4_justification_for_a_teleological_least_action_principle_fixture(
    config: Section4JustificationForATeleologicalLeastActionPrincipleConfig | None = None,
) -> Section4JustificationForATeleologicalLeastActionPrincipleFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section4JustificationForATeleologicalLeastActionPrincipleConfig()
    components = (
        "section_4_justification_for_a_teleological_least_action_principle",
        "4_1_the_problem_of_teleology_in_physics",
        "4_2_causal_entropic_forces_cef_as_the_underlying_mechanism",
    )
    return Section4JustificationForATeleologicalLeastActionPrincipleFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_4_justification_for_a_teleological_least_action_principle_component(
                component
            )
            for component in components
        },
        labels=section_4_justification_for_a_teleological_least_action_principle_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "section_4_justification_for_a_teleological_least_action_principle_is_not_empirical_validation_evidence": 1.0,
            "4_1_the_problem_of_teleology_in_physics_is_not_empirical_validation_evidence": 1.0,
            "4_2_causal_entropic_forces_cef_as_the_underlying_mechanism_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3638, 3653)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_4_justification_for_a_teleological_least_action_principle_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section4JustificationForATeleologicalLeastActionPrincipleConfig",
    "Section4JustificationForATeleologicalLeastActionPrincipleFixtureResult",
    "classify_section_4_justification_for_a_teleological_least_action_principle_component",
    "section_4_justification_for_a_teleological_least_action_principle_labels",
    "validate_section_4_justification_for_a_teleological_least_action_principle_fixture",
]
