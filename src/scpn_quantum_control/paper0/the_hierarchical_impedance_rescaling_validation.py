# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Hierarchical Impedance Rescaling validation
"""Source-accounting checks for Paper 0 The Hierarchical Impedance Rescaling records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded the hierarchical impedance rescaling source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R02655", "P0R02681")


@dataclass(frozen=True, slots=True)
class TheHierarchicalImpedanceRescalingConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 27
    expected_component_count: int = 3
    next_source_boundary: str = "P0R02682"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 27:
            raise ValueError("expected_source_record_count must equal 27")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R02682":
            raise ValueError("next_source_boundary must equal P0R02682")


@dataclass(frozen=True, slots=True)
class TheHierarchicalImpedanceRescalingFixtureResult:
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


def classify_the_hierarchical_impedance_rescaling_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_hierarchical_impedance_rescaling": "the_hierarchical_impedance_rescaling_source_boundary",
        "p0r02661": "p0r02661_source_boundary",
        "formal_integration_of_the_enhanced_boundary_set_ebs_into_the_upde": "formal_integration_of_the_enhanced_boundary_set_ebs_into_the_upde_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown the_hierarchical_impedance_rescaling component") from exc


def the_hierarchical_impedance_rescaling_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Hierarchical Impedance Rescaling",
        "source_span": "P0R02655-P0R02681",
        "component_count": "3",
        "next_boundary": "P0R02682",
        "component_1": "The Hierarchical Impedance Rescaling",
        "component_2": "P0R02661",
        "component_3": "Formal Integration of the Enhanced Boundary Set (EBS) into the UPDE",
    }


def validate_the_hierarchical_impedance_rescaling_fixture(
    config: TheHierarchicalImpedanceRescalingConfig | None = None,
) -> TheHierarchicalImpedanceRescalingFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TheHierarchicalImpedanceRescalingConfig()
    components = (
        "the_hierarchical_impedance_rescaling",
        "p0r02661",
        "formal_integration_of_the_enhanced_boundary_set_ebs_into_the_upde",
    )
    return TheHierarchicalImpedanceRescalingFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_hierarchical_impedance_rescaling_component(component)
            for component in components
        },
        labels=the_hierarchical_impedance_rescaling_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_hierarchical_impedance_rescaling_is_not_empirical_validation_evidence": 1.0,
            "p0r02661_is_not_empirical_validation_evidence": 1.0,
            "formal_integration_of_the_enhanced_boundary_set_ebs_into_the_upde_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(2655, 2682)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_hierarchical_impedance_rescaling_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TheHierarchicalImpedanceRescalingConfig",
    "TheHierarchicalImpedanceRescalingFixtureResult",
    "classify_the_hierarchical_impedance_rescaling_component",
    "the_hierarchical_impedance_rescaling_labels",
    "validate_the_hierarchical_impedance_rescaling_fixture",
]
