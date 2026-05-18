# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 II. The Genesis of Geometry: The Source and the Logos (Domain V) validation
"""Source-accounting checks for Paper 0 II. The Genesis of Geometry: The Source and the Logos (Domain V) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded ii the genesis of geometry the source and the logos domain v source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04380", "P0R04387")


@dataclass(frozen=True, slots=True)
class IiTheGenesisOfGeometryTheSourceAndTheLogosDomainVConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 3
    next_source_boundary: str = "P0R04388"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R04388":
            raise ValueError("next_source_boundary must equal P0R04388")


@dataclass(frozen=True, slots=True)
class IiTheGenesisOfGeometryTheSourceAndTheLogosDomainVFixtureResult:
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


def classify_ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v": "ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v_source_boundary",
        "1_the_source_field_as_a_fiber_bundle_l13": "1_the_source_field_as_a_fiber_bundle_l13_source_boundary",
        "2_the_ethical_functional_as_the_principal_connection_l15": "2_the_ethical_functional_as_the_principal_connection_l15_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v component"
        ) from exc


def ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "II. The Genesis of Geometry: The Source and the Logos (Domain V)",
        "source_span": "P0R04380-P0R04387",
        "component_count": "3",
        "next_boundary": "P0R04388",
        "component_1": "II. The Genesis of Geometry: The Source and the Logos (Domain V)",
        "component_2": "1. The Source-Field as a Fiber Bundle (L13):",
        "component_3": "2. The Ethical Functional as the Principal Connection (L15):",
    }


def validate_ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v_fixture(
    config: IiTheGenesisOfGeometryTheSourceAndTheLogosDomainVConfig | None = None,
) -> IiTheGenesisOfGeometryTheSourceAndTheLogosDomainVFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or IiTheGenesisOfGeometryTheSourceAndTheLogosDomainVConfig()
    components = (
        "ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v",
        "1_the_source_field_as_a_fiber_bundle_l13",
        "2_the_ethical_functional_as_the_principal_connection_l15",
    )
    return IiTheGenesisOfGeometryTheSourceAndTheLogosDomainVFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v_component(
                component
            )
            for component in components
        },
        labels=ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v_is_not_empirical_validation_evidence": 1.0,
            "1_the_source_field_as_a_fiber_bundle_l13_is_not_empirical_validation_evidence": 1.0,
            "2_the_ethical_functional_as_the_principal_connection_l15_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4380, 4388)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "IiTheGenesisOfGeometryTheSourceAndTheLogosDomainVConfig",
    "IiTheGenesisOfGeometryTheSourceAndTheLogosDomainVFixtureResult",
    "classify_ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v_component",
    "ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v_labels",
    "validate_ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v_fixture",
]
