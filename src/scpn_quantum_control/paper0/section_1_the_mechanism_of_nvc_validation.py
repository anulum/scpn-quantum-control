# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 1. The Mechanism of NVC: validation
"""Source-accounting checks for Paper 0 1. The Mechanism of NVC: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 1 the mechanism of nvc source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04879", "P0R04893")


@dataclass(frozen=True, slots=True)
class Section1TheMechanismOfNvcConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 15
    expected_component_count: int = 2
    next_source_boundary: str = "P0R04894"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 15:
            raise ValueError("expected_source_record_count must equal 15")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R04894":
            raise ValueError("next_source_boundary must equal P0R04894")


@dataclass(frozen=True, slots=True)
class Section1TheMechanismOfNvcFixtureResult:
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


def classify_section_1_the_mechanism_of_nvc_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "1_the_mechanism_of_nvc": "1_the_mechanism_of_nvc_source_boundary",
        "2_hemodynamics_and_the_upde": "2_hemodynamics_and_the_upde_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown section_1_the_mechanism_of_nvc component") from exc


def section_1_the_mechanism_of_nvc_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "1. The Mechanism of NVC:",
        "source_span": "P0R04879-P0R04893",
        "component_count": "2",
        "next_boundary": "P0R04894",
        "component_1": "1. The Mechanism of NVC:",
        "component_2": "2. Hemodynamics and the UPDE:",
    }


def validate_section_1_the_mechanism_of_nvc_fixture(
    config: Section1TheMechanismOfNvcConfig | None = None,
) -> Section1TheMechanismOfNvcFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section1TheMechanismOfNvcConfig()
    components = ("1_the_mechanism_of_nvc", "2_hemodynamics_and_the_upde")
    return Section1TheMechanismOfNvcFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_1_the_mechanism_of_nvc_component(component)
            for component in components
        },
        labels=section_1_the_mechanism_of_nvc_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "1_the_mechanism_of_nvc_is_not_empirical_validation_evidence": 1.0,
            "2_hemodynamics_and_the_upde_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4879, 4894)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_1_the_mechanism_of_nvc_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section1TheMechanismOfNvcConfig",
    "Section1TheMechanismOfNvcFixtureResult",
    "classify_section_1_the_mechanism_of_nvc_component",
    "section_1_the_mechanism_of_nvc_labels",
    "validate_section_1_the_mechanism_of_nvc_fixture",
]
