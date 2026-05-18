# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Information-Geometric Lift of UPDE validation
"""Source-accounting checks for Paper 0 Information-Geometric Lift of UPDE records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded information geometric lift of upde source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R02640", "P0R02654")


@dataclass(frozen=True, slots=True)
class InformationGeometricLiftOfUpdeConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 15
    expected_component_count: int = 1
    next_source_boundary: str = "P0R02655"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 15:
            raise ValueError("expected_source_record_count must equal 15")
        if self.expected_component_count != 1:
            raise ValueError("expected_component_count must equal 1")
        if self.next_source_boundary != "P0R02655":
            raise ValueError("next_source_boundary must equal P0R02655")


@dataclass(frozen=True, slots=True)
class InformationGeometricLiftOfUpdeFixtureResult:
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


def classify_information_geometric_lift_of_upde_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "information_geometric_lift_of_upde": "information_geometric_lift_of_upde_source_boundary"
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown information_geometric_lift_of_upde component") from exc


def information_geometric_lift_of_upde_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Information-Geometric Lift of UPDE",
        "source_span": "P0R02640-P0R02654",
        "component_count": "1",
        "next_boundary": "P0R02655",
        "component_1": "Information-Geometric Lift of UPDE",
    }


def validate_information_geometric_lift_of_upde_fixture(
    config: InformationGeometricLiftOfUpdeConfig | None = None,
) -> InformationGeometricLiftOfUpdeFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or InformationGeometricLiftOfUpdeConfig()
    components = ("information_geometric_lift_of_upde",)
    return InformationGeometricLiftOfUpdeFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_information_geometric_lift_of_upde_component(component)
            for component in components
        },
        labels=information_geometric_lift_of_upde_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "information_geometric_lift_of_upde_is_not_empirical_validation_evidence": 1.0
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(2640, 2655)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_information_geometric_lift_of_upde_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "InformationGeometricLiftOfUpdeConfig",
    "InformationGeometricLiftOfUpdeFixtureResult",
    "classify_information_geometric_lift_of_upde_component",
    "information_geometric_lift_of_upde_labels",
    "validate_information_geometric_lift_of_upde_fixture",
]
