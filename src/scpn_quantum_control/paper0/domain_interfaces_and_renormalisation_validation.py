# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Domain Interfaces and Renormalisation validation
"""Source-accounting checks for Paper 0 Domain Interfaces and Renormalisation records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded domain interfaces and renormalisation source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05633", "P0R05640")


@dataclass(frozen=True, slots=True)
class DomainInterfacesAndRenormalisationConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 2
    next_source_boundary: str = "P0R05641"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R05641":
            raise ValueError("next_source_boundary must equal P0R05641")


@dataclass(frozen=True, slots=True)
class DomainInterfacesAndRenormalisationFixtureResult:
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


def classify_domain_interfaces_and_renormalisation_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "domain_interfaces_and_renormalisation": "domain_interfaces_and_renormalisation_source_boundary",
        "1_renormalisation_group_rg_flow_across_domains": "1_renormalisation_group_rg_flow_across_domains_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown domain_interfaces_and_renormalisation component") from exc


def domain_interfaces_and_renormalisation_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Domain Interfaces and Renormalisation",
        "source_span": "P0R05633-P0R05640",
        "component_count": "2",
        "next_boundary": "P0R05641",
        "component_1": "Domain Interfaces and Renormalisation",
        "component_2": "1. Renormalisation Group (RG) Flow Across Domains:",
    }


def validate_domain_interfaces_and_renormalisation_fixture(
    config: DomainInterfacesAndRenormalisationConfig | None = None,
) -> DomainInterfacesAndRenormalisationFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or DomainInterfacesAndRenormalisationConfig()
    components = (
        "domain_interfaces_and_renormalisation",
        "1_renormalisation_group_rg_flow_across_domains",
    )
    return DomainInterfacesAndRenormalisationFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_domain_interfaces_and_renormalisation_component(component)
            for component in components
        },
        labels=domain_interfaces_and_renormalisation_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "domain_interfaces_and_renormalisation_is_not_empirical_validation_evidence": 1.0,
            "1_renormalisation_group_rg_flow_across_domains_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5633, 5641)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_domain_interfaces_and_renormalisation_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "DomainInterfacesAndRenormalisationConfig",
    "DomainInterfacesAndRenormalisationFixtureResult",
    "classify_domain_interfaces_and_renormalisation_component",
    "domain_interfaces_and_renormalisation_labels",
    "validate_domain_interfaces_and_renormalisation_fixture",
]
