# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Psis Field Coupling Integration validation
"""Source-accounting checks for Paper 0 Psis Field Coupling Integration records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded psis field coupling integration source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R02315", "P0R02366")


@dataclass(frozen=True, slots=True)
class PsisFieldCouplingIntegrationConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 52
    expected_component_count: int = 3
    next_source_boundary: str = "P0R02367"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 52:
            raise ValueError("expected_source_record_count must equal 52")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R02367":
            raise ValueError("next_source_boundary must equal P0R02367")


@dataclass(frozen=True, slots=True)
class PsisFieldCouplingIntegrationFixtureResult:
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


def classify_psis_field_coupling_integration_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "psis_field_coupling_integration": "psis_field_coupling_integration_source_boundary",
        "the_collective_state_variable_sigma": "the_collective_state_variable_sigma_source_boundary",
        "the_coupling_mechanism": "the_coupling_mechanism_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown psis_field_coupling_integration component") from exc


def psis_field_coupling_integration_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Psis Field Coupling Integration",
        "source_span": "P0R02315-P0R02366",
        "component_count": "3",
        "next_boundary": "P0R02367",
        "component_1": "Psis Field Coupling Integration",
        "component_2": "The Collective State Variable (sigma):",
        "component_3": "The Coupling Mechanism:",
    }


def validate_psis_field_coupling_integration_fixture(
    config: PsisFieldCouplingIntegrationConfig | None = None,
) -> PsisFieldCouplingIntegrationFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or PsisFieldCouplingIntegrationConfig()
    components = (
        "psis_field_coupling_integration",
        "the_collective_state_variable_sigma",
        "the_coupling_mechanism",
    )
    return PsisFieldCouplingIntegrationFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_psis_field_coupling_integration_component(component)
            for component in components
        },
        labels=psis_field_coupling_integration_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "psis_field_coupling_integration_is_not_empirical_validation_evidence": 1.0,
            "the_collective_state_variable_sigma_is_not_empirical_validation_evidence": 1.0,
            "the_coupling_mechanism_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(2315, 2367)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_psis_field_coupling_integration_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "PsisFieldCouplingIntegrationConfig",
    "PsisFieldCouplingIntegrationFixtureResult",
    "classify_psis_field_coupling_integration_component",
    "psis_field_coupling_integration_labels",
    "validate_psis_field_coupling_integration_fixture",
]
