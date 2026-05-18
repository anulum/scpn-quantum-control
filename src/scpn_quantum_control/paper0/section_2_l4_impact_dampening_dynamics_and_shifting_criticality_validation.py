# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. L4 Impact (Dampening Dynamics and Shifting Criticality): validation
"""Source-accounting checks for Paper 0 2. L4 Impact (Dampening Dynamics and Shifting Criticality): records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 2 l4 impact dampening dynamics and shifting criticality source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05083", "P0R05090")


@dataclass(frozen=True, slots=True)
class Section2L4ImpactDampeningDynamicsAndShiftingCriticalityConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 2
    next_source_boundary: str = "P0R05091"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R05091":
            raise ValueError("next_source_boundary must equal P0R05091")


@dataclass(frozen=True, slots=True)
class Section2L4ImpactDampeningDynamicsAndShiftingCriticalityFixtureResult:
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


def classify_section_2_l4_impact_dampening_dynamics_and_shifting_criticality_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "2_l4_impact_dampening_dynamics_and_shifting_criticality": "2_l4_impact_dampening_dynamics_and_shifting_criticality_source_boundary",
        "3_l5_impact_analgesia_and_geometric_remodelling": "3_l5_impact_analgesia_and_geometric_remodelling_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_2_l4_impact_dampening_dynamics_and_shifting_criticality component"
        ) from exc


def section_2_l4_impact_dampening_dynamics_and_shifting_criticality_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "2. L4 Impact (Dampening Dynamics and Shifting Criticality):",
        "source_span": "P0R05083-P0R05090",
        "component_count": "2",
        "next_boundary": "P0R05091",
        "component_1": "2. L4 Impact (Dampening Dynamics and Shifting Criticality):",
        "component_2": "3. L5 Impact (Analgesia and Geometric Remodelling):",
    }


def validate_section_2_l4_impact_dampening_dynamics_and_shifting_criticality_fixture(
    config: Section2L4ImpactDampeningDynamicsAndShiftingCriticalityConfig | None = None,
) -> Section2L4ImpactDampeningDynamicsAndShiftingCriticalityFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section2L4ImpactDampeningDynamicsAndShiftingCriticalityConfig()
    components = (
        "2_l4_impact_dampening_dynamics_and_shifting_criticality",
        "3_l5_impact_analgesia_and_geometric_remodelling",
    )
    return Section2L4ImpactDampeningDynamicsAndShiftingCriticalityFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_2_l4_impact_dampening_dynamics_and_shifting_criticality_component(
                component
            )
            for component in components
        },
        labels=section_2_l4_impact_dampening_dynamics_and_shifting_criticality_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "2_l4_impact_dampening_dynamics_and_shifting_criticality_is_not_empirical_validation_evidence": 1.0,
            "3_l5_impact_analgesia_and_geometric_remodelling_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5083, 5091)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_2_l4_impact_dampening_dynamics_and_shifting_criticality_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section2L4ImpactDampeningDynamicsAndShiftingCriticalityConfig",
    "Section2L4ImpactDampeningDynamicsAndShiftingCriticalityFixtureResult",
    "classify_section_2_l4_impact_dampening_dynamics_and_shifting_criticality_component",
    "section_2_l4_impact_dampening_dynamics_and_shifting_criticality_labels",
    "validate_section_2_l4_impact_dampening_dynamics_and_shifting_criticality_fixture",
]
