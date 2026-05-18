# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Predictive Coding Integration validation
"""Source-accounting checks for Paper 0 Predictive Coding Integration records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded predictive coding integration p0r02839 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R02839", "P0R02847")


@dataclass(frozen=True, slots=True)
class PredictiveCodingIntegrationP0r02839Config:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 4
    next_source_boundary: str = "P0R02848"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R02848":
            raise ValueError("next_source_boundary must equal P0R02848")


@dataclass(frozen=True, slots=True)
class PredictiveCodingIntegrationP0r02839FixtureResult:
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


def classify_predictive_coding_integration_p0r02839_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "predictive_coding_integration": "predictive_coding_integration_source_boundary",
        "maximising_inferential_capacity": "maximising_inferential_capacity_source_boundary",
        "soc_as_precision_tuning": "soc_as_precision_tuning_source_boundary",
        "psis_field_coupling_integration": "psis_field_coupling_integration_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown predictive_coding_integration_p0r02839 component") from exc


def predictive_coding_integration_p0r02839_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Predictive Coding Integration",
        "source_span": "P0R02839-P0R02847",
        "component_count": "4",
        "next_boundary": "P0R02848",
        "component_1": "Predictive Coding Integration",
        "component_2": "Maximising Inferential Capacity:",
        "component_3": "SOC as Precision Tuning:",
        "component_4": "Psis Field Coupling Integration",
    }


def validate_predictive_coding_integration_p0r02839_fixture(
    config: PredictiveCodingIntegrationP0r02839Config | None = None,
) -> PredictiveCodingIntegrationP0r02839FixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or PredictiveCodingIntegrationP0r02839Config()
    components = (
        "predictive_coding_integration",
        "maximising_inferential_capacity",
        "soc_as_precision_tuning",
        "psis_field_coupling_integration",
    )
    return PredictiveCodingIntegrationP0r02839FixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_predictive_coding_integration_p0r02839_component(component)
            for component in components
        },
        labels=predictive_coding_integration_p0r02839_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "predictive_coding_integration_is_not_empirical_validation_evidence": 1.0,
            "maximising_inferential_capacity_is_not_empirical_validation_evidence": 1.0,
            "soc_as_precision_tuning_is_not_empirical_validation_evidence": 1.0,
            "psis_field_coupling_integration_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(2839, 2848)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_predictive_coding_integration_p0r02839_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "PredictiveCodingIntegrationP0r02839Config",
    "PredictiveCodingIntegrationP0r02839FixtureResult",
    "classify_predictive_coding_integration_p0r02839_component",
    "predictive_coding_integration_p0r02839_labels",
    "validate_predictive_coding_integration_p0r02839_fixture",
]
