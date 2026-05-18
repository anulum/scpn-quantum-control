# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 4. The Formal Equivalence of SEC and SC validation
"""Source-accounting checks for Paper 0 4. The Formal Equivalence of SEC and SC records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 4 the formal equivalence of sec and sc source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03804", "P0R03817")


@dataclass(frozen=True, slots=True)
class Section4TheFormalEquivalenceOfSecAndScConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 14
    expected_component_count: int = 1
    next_source_boundary: str = "P0R03818"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 14:
            raise ValueError("expected_source_record_count must equal 14")
        if self.expected_component_count != 1:
            raise ValueError("expected_component_count must equal 1")
        if self.next_source_boundary != "P0R03818":
            raise ValueError("next_source_boundary must equal P0R03818")


@dataclass(frozen=True, slots=True)
class Section4TheFormalEquivalenceOfSecAndScFixtureResult:
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


def classify_section_4_the_formal_equivalence_of_sec_and_sc_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "4_the_formal_equivalence_of_sec_and_sc": "4_the_formal_equivalence_of_sec_and_sc_source_boundary"
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_4_the_formal_equivalence_of_sec_and_sc component"
        ) from exc


def section_4_the_formal_equivalence_of_sec_and_sc_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "4. The Formal Equivalence of SEC and SC",
        "source_span": "P0R03804-P0R03817",
        "component_count": "1",
        "next_boundary": "P0R03818",
        "component_1": "4. The Formal Equivalence of SEC and SC",
    }


def validate_section_4_the_formal_equivalence_of_sec_and_sc_fixture(
    config: Section4TheFormalEquivalenceOfSecAndScConfig | None = None,
) -> Section4TheFormalEquivalenceOfSecAndScFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section4TheFormalEquivalenceOfSecAndScConfig()
    components = ("4_the_formal_equivalence_of_sec_and_sc",)
    return Section4TheFormalEquivalenceOfSecAndScFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_4_the_formal_equivalence_of_sec_and_sc_component(component)
            for component in components
        },
        labels=section_4_the_formal_equivalence_of_sec_and_sc_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "4_the_formal_equivalence_of_sec_and_sc_is_not_empirical_validation_evidence": 1.0
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3804, 3818)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_4_the_formal_equivalence_of_sec_and_sc_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section4TheFormalEquivalenceOfSecAndScConfig",
    "Section4TheFormalEquivalenceOfSecAndScFixtureResult",
    "classify_section_4_the_formal_equivalence_of_sec_and_sc_component",
    "section_4_the_formal_equivalence_of_sec_and_sc_labels",
    "validate_section_4_the_formal_equivalence_of_sec_and_sc_fixture",
]
