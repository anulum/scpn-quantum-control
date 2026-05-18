# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. Impedance Matching: validation
"""Source-accounting checks for Paper 0 2. Impedance Matching: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = (
    "source-bounded section 2 impedance matching source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05641", "P0R05649")


@dataclass(frozen=True, slots=True)
class Section2ImpedanceMatchingConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 3
    next_source_boundary: str = "P0R05650"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R05650":
            raise ValueError("next_source_boundary must equal P0R05650")


@dataclass(frozen=True, slots=True)
class Section2ImpedanceMatchingFixtureResult:
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


def classify_section_2_impedance_matching_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "2_impedance_matching": "2_impedance_matching_source_boundary",
        "citations_cross_domain_anchors": "citations_cross_domain_anchors_source_boundary",
        "cross_domain_anchors_core_dynamics": "cross_domain_anchors_core_dynamics_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown section_2_impedance_matching component") from exc


def section_2_impedance_matching_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "2. Impedance Matching:",
        "source_span": "P0R05641-P0R05649",
        "component_count": "3",
        "next_boundary": "P0R05650",
        "component_1": "2. Impedance Matching:",
        "component_2": "Citations: Cross-Domain Anchors",
        "component_3": "Cross-Domain Anchors (Core Dynamics)",
    }


def validate_section_2_impedance_matching_fixture(
    config: Section2ImpedanceMatchingConfig | None = None,
) -> Section2ImpedanceMatchingFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section2ImpedanceMatchingConfig()
    components = (
        "2_impedance_matching",
        "citations_cross_domain_anchors",
        "cross_domain_anchors_core_dynamics",
    )
    return Section2ImpedanceMatchingFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_2_impedance_matching_component(component)
            for component in components
        },
        labels=section_2_impedance_matching_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "2_impedance_matching_is_not_empirical_validation_evidence": 1.0,
            "citations_cross_domain_anchors_is_not_empirical_validation_evidence": 1.0,
            "cross_domain_anchors_core_dynamics_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5641, 5650)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_2_impedance_matching_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section2ImpedanceMatchingConfig",
    "Section2ImpedanceMatchingFixtureResult",
    "classify_section_2_impedance_matching_component",
    "section_2_impedance_matching_labels",
    "validate_section_2_impedance_matching_fixture",
]
