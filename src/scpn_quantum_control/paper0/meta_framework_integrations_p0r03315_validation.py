# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Meta-Framework Integrations validation
"""Source-accounting checks for Paper 0 Meta-Framework Integrations records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded meta framework integrations p0r03315 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03315", "P0R03323")


@dataclass(frozen=True, slots=True)
class MetaFrameworkIntegrationsP0r03315Config:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 5
    next_source_boundary: str = "P0R03324"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 5:
            raise ValueError("expected_component_count must equal 5")
        if self.next_source_boundary != "P0R03324":
            raise ValueError("next_source_boundary must equal P0R03324")


@dataclass(frozen=True, slots=True)
class MetaFrameworkIntegrationsP0r03315FixtureResult:
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


def classify_meta_framework_integrations_p0r03315_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "meta_framework_integrations": "meta_framework_integrations_source_boundary",
        "predictive_coding_integration": "predictive_coding_integration_source_boundary",
        "guided_einselection_is_prediction": "guided_einselection_is_prediction_source_boundary",
        "qsr_is_error_signal_amplification": "qsr_is_error_signal_amplification_source_boundary",
        "psis_field_coupling_integration": "psis_field_coupling_integration_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown meta_framework_integrations_p0r03315 component") from exc


def meta_framework_integrations_p0r03315_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Meta-Framework Integrations",
        "source_span": "P0R03315-P0R03323",
        "component_count": "5",
        "next_boundary": "P0R03324",
        "component_1": "Meta-Framework Integrations",
        "component_2": "Predictive Coding Integration",
        "component_3": "Guided Einselection is Prediction:",
        "component_4": "QSR is Error-Signal Amplification:",
        "component_5": "Psis Field Coupling Integration",
    }


def validate_meta_framework_integrations_p0r03315_fixture(
    config: MetaFrameworkIntegrationsP0r03315Config | None = None,
) -> MetaFrameworkIntegrationsP0r03315FixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or MetaFrameworkIntegrationsP0r03315Config()
    components = (
        "meta_framework_integrations",
        "predictive_coding_integration",
        "guided_einselection_is_prediction",
        "qsr_is_error_signal_amplification",
        "psis_field_coupling_integration",
    )
    return MetaFrameworkIntegrationsP0r03315FixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_meta_framework_integrations_p0r03315_component(component)
            for component in components
        },
        labels=meta_framework_integrations_p0r03315_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "meta_framework_integrations_is_not_empirical_validation_evidence": 1.0,
            "predictive_coding_integration_is_not_empirical_validation_evidence": 1.0,
            "guided_einselection_is_prediction_is_not_empirical_validation_evidence": 1.0,
            "qsr_is_error_signal_amplification_is_not_empirical_validation_evidence": 1.0,
            "psis_field_coupling_integration_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3315, 3324)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_meta_framework_integrations_p0r03315_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "MetaFrameworkIntegrationsP0r03315Config",
    "MetaFrameworkIntegrationsP0r03315FixtureResult",
    "classify_meta_framework_integrations_p0r03315_component",
    "meta_framework_integrations_p0r03315_labels",
    "validate_meta_framework_integrations_p0r03315_fixture",
]
