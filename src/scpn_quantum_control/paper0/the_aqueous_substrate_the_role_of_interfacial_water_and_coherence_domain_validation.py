# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Aqueous Substrate - The Role of Interfacial Water and Coherence Domains validation
"""Source-accounting checks for Paper 0 The Aqueous Substrate - The Role of Interfacial Water and Coherence Domains records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded the aqueous substrate the role of interfacial water and coherence domain source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05293", "P0R05305")


@dataclass(frozen=True, slots=True)
class TheAqueousSubstrateTheRoleOfInterfacialWaterAndCoherenceDomainConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 13
    expected_component_count: int = 2
    next_source_boundary: str = "P0R05306"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 13:
            raise ValueError("expected_source_record_count must equal 13")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R05306":
            raise ValueError("next_source_boundary must equal P0R05306")


@dataclass(frozen=True, slots=True)
class TheAqueousSubstrateTheRoleOfInterfacialWaterAndCoherenceDomainFixtureResult:
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


def classify_the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain": "the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_source_boundary",
        "p0r05299": "p0r05299_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain component"
        ) from exc


def the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_labels() -> dict[
    str, str
]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Aqueous Substrate - The Role of Interfacial Water and Coherence Domains",
        "source_span": "P0R05293-P0R05305",
        "component_count": "2",
        "next_boundary": "P0R05306",
        "component_1": "The Aqueous Substrate - The Role of Interfacial Water and Coherence Domains",
        "component_2": "P0R05299",
    }


def validate_the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_fixture(
    config: TheAqueousSubstrateTheRoleOfInterfacialWaterAndCoherenceDomainConfig | None = None,
) -> TheAqueousSubstrateTheRoleOfInterfacialWaterAndCoherenceDomainFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TheAqueousSubstrateTheRoleOfInterfacialWaterAndCoherenceDomainConfig()
    components = (
        "the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain",
        "p0r05299",
    )
    return TheAqueousSubstrateTheRoleOfInterfacialWaterAndCoherenceDomainFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_component(
                component
            )
            for component in components
        },
        labels=the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_is_not_empirical_validation_evidence": 1.0,
            "p0r05299_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5293, 5306)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TheAqueousSubstrateTheRoleOfInterfacialWaterAndCoherenceDomainConfig",
    "TheAqueousSubstrateTheRoleOfInterfacialWaterAndCoherenceDomainFixtureResult",
    "classify_the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_component",
    "the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_labels",
    "validate_the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_fixture",
]
