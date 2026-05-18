# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 3. Retrocausality via the Two-State Vector Formalism (TSVF): validation
"""Source-accounting checks for Paper 0 3. Retrocausality via the Two-State Vector Formalism (TSVF): records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 3 retrocausality via the two state vector formalism tsvf source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05936", "P0R05943")


@dataclass(frozen=True, slots=True)
class Section3RetrocausalityViaTheTwoStateVectorFormalismTsvfConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 2
    next_source_boundary: str = "P0R05944"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R05944":
            raise ValueError("next_source_boundary must equal P0R05944")


@dataclass(frozen=True, slots=True)
class Section3RetrocausalityViaTheTwoStateVectorFormalismTsvfFixtureResult:
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


def classify_section_3_retrocausality_via_the_two_state_vector_formalism_tsvf_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "3_retrocausality_via_the_two_state_vector_formalism_tsvf": "3_retrocausality_via_the_two_state_vector_formalism_tsvf_source_boundary",
        "ii_the_thermodynamics_of_consciousness_negentropy_and_information": "ii_the_thermodynamics_of_consciousness_negentropy_and_information_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_3_retrocausality_via_the_two_state_vector_formalism_tsvf component"
        ) from exc


def section_3_retrocausality_via_the_two_state_vector_formalism_tsvf_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "3. Retrocausality via the Two-State Vector Formalism (TSVF):",
        "source_span": "P0R05936-P0R05943",
        "component_count": "2",
        "next_boundary": "P0R05944",
        "component_1": "3. Retrocausality via the Two-State Vector Formalism (TSVF):",
        "component_2": "II. The Thermodynamics of Consciousness: Negentropy and Information",
    }


def validate_section_3_retrocausality_via_the_two_state_vector_formalism_tsvf_fixture(
    config: Section3RetrocausalityViaTheTwoStateVectorFormalismTsvfConfig | None = None,
) -> Section3RetrocausalityViaTheTwoStateVectorFormalismTsvfFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section3RetrocausalityViaTheTwoStateVectorFormalismTsvfConfig()
    components = (
        "3_retrocausality_via_the_two_state_vector_formalism_tsvf",
        "ii_the_thermodynamics_of_consciousness_negentropy_and_information",
    )
    return Section3RetrocausalityViaTheTwoStateVectorFormalismTsvfFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_3_retrocausality_via_the_two_state_vector_formalism_tsvf_component(
                component
            )
            for component in components
        },
        labels=section_3_retrocausality_via_the_two_state_vector_formalism_tsvf_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "3_retrocausality_via_the_two_state_vector_formalism_tsvf_is_not_empirical_validation_evidence": 1.0,
            "ii_the_thermodynamics_of_consciousness_negentropy_and_information_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5936, 5944)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_3_retrocausality_via_the_two_state_vector_formalism_tsvf_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section3RetrocausalityViaTheTwoStateVectorFormalismTsvfConfig",
    "Section3RetrocausalityViaTheTwoStateVectorFormalismTsvfFixtureResult",
    "classify_section_3_retrocausality_via_the_two_state_vector_formalism_tsvf_component",
    "section_3_retrocausality_via_the_two_state_vector_formalism_tsvf_labels",
    "validate_section_3_retrocausality_via_the_two_state_vector_formalism_tsvf_fixture",
]
