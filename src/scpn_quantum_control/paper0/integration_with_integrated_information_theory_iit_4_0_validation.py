# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Integration with Integrated Information Theory (IIT) 4.0 validation
"""Source-accounting checks for Paper 0 Integration with Integrated Information Theory (IIT) 4.0 records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded integration with integrated information theory iit 4 0 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03521", "P0R03529")


@dataclass(frozen=True, slots=True)
class IntegrationWithIntegratedInformationTheoryIit40Config:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 1
    next_source_boundary: str = "P0R03530"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 1:
            raise ValueError("expected_component_count must equal 1")
        if self.next_source_boundary != "P0R03530":
            raise ValueError("next_source_boundary must equal P0R03530")


@dataclass(frozen=True, slots=True)
class IntegrationWithIntegratedInformationTheoryIit40FixtureResult:
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


def classify_integration_with_integrated_information_theory_iit_4_0_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "integration_with_integrated_information_theory_iit_4_0": "integration_with_integrated_information_theory_iit_4_0_source_boundary"
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown integration_with_integrated_information_theory_iit_4_0 component"
        ) from exc


def integration_with_integrated_information_theory_iit_4_0_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Integration with Integrated Information Theory (IIT) 4.0",
        "source_span": "P0R03521-P0R03529",
        "component_count": "1",
        "next_boundary": "P0R03530",
        "component_1": "Integration with Integrated Information Theory (IIT) 4.0",
    }


def validate_integration_with_integrated_information_theory_iit_4_0_fixture(
    config: IntegrationWithIntegratedInformationTheoryIit40Config | None = None,
) -> IntegrationWithIntegratedInformationTheoryIit40FixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or IntegrationWithIntegratedInformationTheoryIit40Config()
    components = ("integration_with_integrated_information_theory_iit_4_0",)
    return IntegrationWithIntegratedInformationTheoryIit40FixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_integration_with_integrated_information_theory_iit_4_0_component(
                component
            )
            for component in components
        },
        labels=integration_with_integrated_information_theory_iit_4_0_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "integration_with_integrated_information_theory_iit_4_0_is_not_empirical_validation_evidence": 1.0
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3521, 3530)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_integration_with_integrated_information_theory_iit_4_0_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "IntegrationWithIntegratedInformationTheoryIit40Config",
    "IntegrationWithIntegratedInformationTheoryIit40FixtureResult",
    "classify_integration_with_integrated_information_theory_iit_4_0_component",
    "integration_with_integrated_information_theory_iit_4_0_labels",
    "validate_integration_with_integrated_information_theory_iit_4_0_fixture",
]
