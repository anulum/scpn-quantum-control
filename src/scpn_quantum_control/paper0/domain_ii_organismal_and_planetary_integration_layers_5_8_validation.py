# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Domain II: Organismal and Planetary Integration (Layers 5-8) validation
"""Source-accounting checks for Paper 0 Domain II: Organismal and Planetary Integration (Layers 5-8) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded domain ii organismal and planetary integration layers 5 8 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05537", "P0R05550")


@dataclass(frozen=True, slots=True)
class DomainIiOrganismalAndPlanetaryIntegrationLayers58Config:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 14
    expected_component_count: int = 2
    next_source_boundary: str = "P0R05551"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 14:
            raise ValueError("expected_source_record_count must equal 14")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R05551":
            raise ValueError("next_source_boundary must equal P0R05551")


@dataclass(frozen=True, slots=True)
class DomainIiOrganismalAndPlanetaryIntegrationLayers58FixtureResult:
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


def classify_domain_ii_organismal_and_planetary_integration_layers_5_8_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "domain_ii_organismal_and_planetary_integration_layers_5_8": "domain_ii_organismal_and_planetary_integration_layers_5_8_source_boundary",
        "citations": "citations_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown domain_ii_organismal_and_planetary_integration_layers_5_8 component"
        ) from exc


def domain_ii_organismal_and_planetary_integration_layers_5_8_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Domain II: Organismal and Planetary Integration (Layers 5-8)",
        "source_span": "P0R05537-P0R05550",
        "component_count": "2",
        "next_boundary": "P0R05551",
        "component_1": "Domain II: Organismal and Planetary Integration (Layers 5-8)",
        "component_2": "Citations:",
    }


def validate_domain_ii_organismal_and_planetary_integration_layers_5_8_fixture(
    config: DomainIiOrganismalAndPlanetaryIntegrationLayers58Config | None = None,
) -> DomainIiOrganismalAndPlanetaryIntegrationLayers58FixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or DomainIiOrganismalAndPlanetaryIntegrationLayers58Config()
    components = ("domain_ii_organismal_and_planetary_integration_layers_5_8", "citations")
    return DomainIiOrganismalAndPlanetaryIntegrationLayers58FixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_domain_ii_organismal_and_planetary_integration_layers_5_8_component(
                component
            )
            for component in components
        },
        labels=domain_ii_organismal_and_planetary_integration_layers_5_8_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "domain_ii_organismal_and_planetary_integration_layers_5_8_is_not_empirical_validation_evidence": 1.0,
            "citations_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5537, 5551)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_domain_ii_organismal_and_planetary_integration_layers_5_8_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "DomainIiOrganismalAndPlanetaryIntegrationLayers58Config",
    "DomainIiOrganismalAndPlanetaryIntegrationLayers58FixtureResult",
    "classify_domain_ii_organismal_and_planetary_integration_layers_5_8_component",
    "domain_ii_organismal_and_planetary_integration_layers_5_8_labels",
    "validate_domain_ii_organismal_and_planetary_integration_layers_5_8_fixture",
]
