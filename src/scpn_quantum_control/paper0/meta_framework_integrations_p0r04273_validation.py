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

CLAIM_BOUNDARY = "source-bounded meta framework integrations p0r04273 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04273", "P0R04282")


@dataclass(frozen=True, slots=True)
class MetaFrameworkIntegrationsP0r04273Config:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 10
    expected_component_count: int = 5
    next_source_boundary: str = "P0R04283"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 10:
            raise ValueError("expected_source_record_count must equal 10")
        if self.expected_component_count != 5:
            raise ValueError("expected_component_count must equal 5")
        if self.next_source_boundary != "P0R04283":
            raise ValueError("next_source_boundary must equal P0R04283")


@dataclass(frozen=True, slots=True)
class MetaFrameworkIntegrationsP0r04273FixtureResult:
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


def classify_meta_framework_integrations_p0r04273_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "meta_framework_integrations": "meta_framework_integrations_source_boundary",
        "predictive_coding_integration": "predictive_coding_integration_source_boundary",
        "downward_causation_psi_em_is_the_write_operation": "downward_causation_psi_em_is_the_write_operation_source_boundary",
        "upward_causation_em_psi_is_the_read_operation": "upward_causation_em_psi_is_the_read_operation_source_boundary",
        "psis_field_coupling_integration": "psis_field_coupling_integration_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown meta_framework_integrations_p0r04273 component") from exc


def meta_framework_integrations_p0r04273_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Meta-Framework Integrations",
        "source_span": "P0R04273-P0R04282",
        "component_count": "5",
        "next_boundary": "P0R04283",
        "component_1": "Meta-Framework Integrations",
        "component_2": "Predictive Coding Integration",
        "component_3": 'Downward Causation (Psi->EM) is the "Write" operation:',
        "component_4": 'Upward Causation (EM->Psi) is the "Read" operation:',
        "component_5": "Psis Field Coupling Integration",
    }


def validate_meta_framework_integrations_p0r04273_fixture(
    config: MetaFrameworkIntegrationsP0r04273Config | None = None,
) -> MetaFrameworkIntegrationsP0r04273FixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or MetaFrameworkIntegrationsP0r04273Config()
    components = (
        "meta_framework_integrations",
        "predictive_coding_integration",
        "downward_causation_psi_em_is_the_write_operation",
        "upward_causation_em_psi_is_the_read_operation",
        "psis_field_coupling_integration",
    )
    return MetaFrameworkIntegrationsP0r04273FixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_meta_framework_integrations_p0r04273_component(component)
            for component in components
        },
        labels=meta_framework_integrations_p0r04273_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "meta_framework_integrations_is_not_empirical_validation_evidence": 1.0,
            "predictive_coding_integration_is_not_empirical_validation_evidence": 1.0,
            "downward_causation_psi_em_is_the_write_operation_is_not_empirical_validation_evidence": 1.0,
            "upward_causation_em_psi_is_the_read_operation_is_not_empirical_validation_evidence": 1.0,
            "psis_field_coupling_integration_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4273, 4283)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_meta_framework_integrations_p0r04273_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "MetaFrameworkIntegrationsP0r04273Config",
    "MetaFrameworkIntegrationsP0r04273FixtureResult",
    "classify_meta_framework_integrations_p0r04273_component",
    "meta_framework_integrations_p0r04273_labels",
    "validate_meta_framework_integrations_p0r04273_fixture",
]
