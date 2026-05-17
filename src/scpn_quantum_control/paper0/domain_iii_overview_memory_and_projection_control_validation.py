# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Domain III Overview: Memory and Projection Control validation
"""Source-accounting checks for Paper 0 Domain III Overview: Memory and Projection Control records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded domain iii overview memory and projection control source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R02237", "P0R02248")


@dataclass(frozen=True, slots=True)
class DomainIiiOverviewMemoryAndProjectionControlConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 12
    expected_component_count: int = 3
    next_source_boundary: str = "P0R02249"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 12:
            raise ValueError("expected_source_record_count must equal 12")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R02249":
            raise ValueError("next_source_boundary must equal P0R02249")


@dataclass(frozen=True, slots=True)
class DomainIiiOverviewMemoryAndProjectionControlFixtureResult:
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


def classify_domain_iii_overview_memory_and_projection_control_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "domain_iii_overview_memory_and_projection_control": "domain_iii_overview_memory_and_projection_control_source_boundary",
        "p0r02242": "p0r02242_source_boundary",
        "1_memory_retrieval_retrocausality_via_abl_rule": "1_memory_retrieval_retrocausality_via_abl_rule_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown domain_iii_overview_memory_and_projection_control component"
        ) from exc


def domain_iii_overview_memory_and_projection_control_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Domain III Overview: Memory and Projection Control",
        "source_span": "P0R02237-P0R02248",
        "component_count": "3",
        "next_boundary": "P0R02249",
        "component_1": "Domain III Overview: Memory and Projection Control",
        "component_2": "P0R02242",
        "component_3": "1. Memory Retrieval (Retrocausality via ABL Rule)",
    }


def validate_domain_iii_overview_memory_and_projection_control_fixture(
    config: DomainIiiOverviewMemoryAndProjectionControlConfig | None = None,
) -> DomainIiiOverviewMemoryAndProjectionControlFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or DomainIiiOverviewMemoryAndProjectionControlConfig()
    components = (
        "domain_iii_overview_memory_and_projection_control",
        "p0r02242",
        "1_memory_retrieval_retrocausality_via_abl_rule",
    )
    return DomainIiiOverviewMemoryAndProjectionControlFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_domain_iii_overview_memory_and_projection_control_component(
                component
            )
            for component in components
        },
        labels=domain_iii_overview_memory_and_projection_control_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "domain_iii_overview_memory_and_projection_control_is_not_empirical_validation_evidence": 1.0,
            "p0r02242_is_not_empirical_validation_evidence": 1.0,
            "1_memory_retrieval_retrocausality_via_abl_rule_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(2237, 2249)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_domain_iii_overview_memory_and_projection_control_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "DomainIiiOverviewMemoryAndProjectionControlConfig",
    "DomainIiiOverviewMemoryAndProjectionControlFixtureResult",
    "classify_domain_iii_overview_memory_and_projection_control_component",
    "domain_iii_overview_memory_and_projection_control_labels",
    "validate_domain_iii_overview_memory_and_projection_control_fixture",
]
