# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 1. The Generalised Second Law (GSL): validation
"""Source-accounting checks for Paper 0 1. The Generalised Second Law (GSL): records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 1 the generalised second law gsl source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05944", "P0R05952")


@dataclass(frozen=True, slots=True)
class Section1TheGeneralisedSecondLawGslConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 3
    next_source_boundary: str = "P0R05953"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R05953":
            raise ValueError("next_source_boundary must equal P0R05953")


@dataclass(frozen=True, slots=True)
class Section1TheGeneralisedSecondLawGslFixtureResult:
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


def classify_section_1_the_generalised_second_law_gsl_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "1_the_generalised_second_law_gsl": "1_the_generalised_second_law_gsl_source_boundary",
        "2_the_psi_field_as_a_negentropy_source_information_thermodynamics": "2_the_psi_field_as_a_negentropy_source_information_thermodynamics_source_boundary",
        "the_rate_of_negentropy_injection_npsi_is_proportional_to_the_mutual_info": "the_rate_of_negentropy_injection_npsi_is_proportional_to_the_mutual_info_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown section_1_the_generalised_second_law_gsl component") from exc


def section_1_the_generalised_second_law_gsl_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "1. The Generalised Second Law (GSL):",
        "source_span": "P0R05944-P0R05952",
        "component_count": "3",
        "next_boundary": "P0R05953",
        "component_1": "1. The Generalised Second Law (GSL):",
        "component_2": "2. The Psi-Field as a Negentropy Source (Information Thermodynamics):",
        "component_3": "The rate of negentropy injection (NPsi) is proportional to the mutual information (I) between the Psi-field and the biological substrate (B):",
    }


def validate_section_1_the_generalised_second_law_gsl_fixture(
    config: Section1TheGeneralisedSecondLawGslConfig | None = None,
) -> Section1TheGeneralisedSecondLawGslFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section1TheGeneralisedSecondLawGslConfig()
    components = (
        "1_the_generalised_second_law_gsl",
        "2_the_psi_field_as_a_negentropy_source_information_thermodynamics",
        "the_rate_of_negentropy_injection_npsi_is_proportional_to_the_mutual_info",
    )
    return Section1TheGeneralisedSecondLawGslFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_1_the_generalised_second_law_gsl_component(component)
            for component in components
        },
        labels=section_1_the_generalised_second_law_gsl_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "1_the_generalised_second_law_gsl_is_not_empirical_validation_evidence": 1.0,
            "2_the_psi_field_as_a_negentropy_source_information_thermodynamics_is_not_empirical_validation_evidence": 1.0,
            "the_rate_of_negentropy_injection_npsi_is_proportional_to_the_mutual_info_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5944, 5953)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_1_the_generalised_second_law_gsl_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section1TheGeneralisedSecondLawGslConfig",
    "Section1TheGeneralisedSecondLawGslFixtureResult",
    "classify_section_1_the_generalised_second_law_gsl_component",
    "section_1_the_generalised_second_law_gsl_labels",
    "validate_section_1_the_generalised_second_law_gsl_fixture",
]
