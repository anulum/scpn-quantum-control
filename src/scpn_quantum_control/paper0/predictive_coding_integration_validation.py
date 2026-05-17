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

CLAIM_BOUNDARY = "source-bounded predictive coding integration source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R01763", "P0R01770")


@dataclass(frozen=True, slots=True)
class PredictiveCodingIntegrationConfig:
    """Configuration for the Predictive Coding Integration fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 4
    next_source_boundary: str = "P0R01771"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R01771":
            raise ValueError("next_source_boundary must equal P0R01771")


@dataclass(frozen=True, slots=True)
class PredictiveCodingIntegrationFixtureResult:
    """Result for the Paper 0 Predictive Coding Integration fixture."""

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


def classify_predictive_coding_integration_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "predictive_coding_integration": "predictive_coding_integration_source_boundary",
        "the_potential_v_psi_as_the_landscape_of_priors": "the_potential_v_psi_as_the_landscape_of_priors_source_boundary",
        "the_sextic_term_as_a_sanity_check": "the_sextic_term_as_a_sanity_check_source_boundary",
        "psis_field_coupling_integration": "psis_field_coupling_integration_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown predictive_coding_integration component") from exc


def predictive_coding_integration_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Predictive Coding Integration",
        "source_span": "P0R01763-P0R01770",
        "component_count": "4",
        "next_boundary": "P0R01771",
        "component_1": "Predictive Coding Integration",
        "component_2": "The Potential V(|Psi|) as the Landscape of Priors:",
        "component_3": 'The Sextic Term as a "Sanity Check":',
        "component_4": "Psis Field Coupling Integration",
    }


def validate_predictive_coding_integration_fixture(
    config: PredictiveCodingIntegrationConfig | None = None,
) -> PredictiveCodingIntegrationFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or PredictiveCodingIntegrationConfig()
    components = (
        "predictive_coding_integration",
        "the_potential_v_psi_as_the_landscape_of_priors",
        "the_sextic_term_as_a_sanity_check",
        "psis_field_coupling_integration",
    )
    return PredictiveCodingIntegrationFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_predictive_coding_integration_component(component)
            for component in components
        },
        labels=predictive_coding_integration_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "predictive_coding_integration_is_not_empirical_validation_evidence": 1.0,
            "the_potential_v_psi_as_the_landscape_of_priors_is_not_empirical_validation_evidence": 1.0,
            "the_sextic_term_as_a_sanity_check_is_not_empirical_validation_evidence": 1.0,
            "psis_field_coupling_integration_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(1763, 1771)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_predictive_coding_integration_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "PredictiveCodingIntegrationConfig",
    "PredictiveCodingIntegrationFixtureResult",
    "classify_predictive_coding_integration_component",
    "predictive_coding_integration_labels",
    "validate_predictive_coding_integration_fixture",
]
