# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Maximizing  as the Goal of Coupling: validation
"""Source-accounting checks for Paper 0 Maximizing  as the Goal of Coupling: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded maximizing as the goal of coupling source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03539", "P0R03554")


@dataclass(frozen=True, slots=True)
class MaximizingAsTheGoalOfCouplingConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 16
    expected_component_count: int = 4
    next_source_boundary: str = "P0R03555"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 16:
            raise ValueError("expected_source_record_count must equal 16")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R03555":
            raise ValueError("next_source_boundary must equal P0R03555")


@dataclass(frozen=True, slots=True)
class MaximizingAsTheGoalOfCouplingFixtureResult:
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


def classify_maximizing_as_the_goal_of_coupling_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "maximizing_as_the_goal_of_coupling": "maximizing_as_the_goal_of_coupling_source_boundary",
        "integration_with_integrated_information_theory_4_0": "integration_with_integrated_information_theory_4_0_source_boundary",
        "iit_4_0_integration": "iit_4_0_integration_source_boundary",
        "bridging_scpn_with_iit_4_0_s_mathematical_framework": "bridging_scpn_with_iit_4_0_s_mathematical_framework_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown maximizing_as_the_goal_of_coupling component") from exc


def maximizing_as_the_goal_of_coupling_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Maximizing  as the Goal of Coupling:",
        "source_span": "P0R03539-P0R03554",
        "component_count": "4",
        "next_boundary": "P0R03555",
        "component_1": "Maximizing as the Goal of Coupling:",
        "component_2": "Integration with Integrated Information Theory 4.0",
        "component_3": "IIT 4.0 Integration",
        "component_4": "Bridging SCPN with IIT 4.0's Mathematical Framework",
    }


def validate_maximizing_as_the_goal_of_coupling_fixture(
    config: MaximizingAsTheGoalOfCouplingConfig | None = None,
) -> MaximizingAsTheGoalOfCouplingFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or MaximizingAsTheGoalOfCouplingConfig()
    components = (
        "maximizing_as_the_goal_of_coupling",
        "integration_with_integrated_information_theory_4_0",
        "iit_4_0_integration",
        "bridging_scpn_with_iit_4_0_s_mathematical_framework",
    )
    return MaximizingAsTheGoalOfCouplingFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_maximizing_as_the_goal_of_coupling_component(component)
            for component in components
        },
        labels=maximizing_as_the_goal_of_coupling_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "maximizing_as_the_goal_of_coupling_is_not_empirical_validation_evidence": 1.0,
            "integration_with_integrated_information_theory_4_0_is_not_empirical_validation_evidence": 1.0,
            "iit_4_0_integration_is_not_empirical_validation_evidence": 1.0,
            "bridging_scpn_with_iit_4_0_s_mathematical_framework_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3539, 3555)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_maximizing_as_the_goal_of_coupling_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "MaximizingAsTheGoalOfCouplingConfig",
    "MaximizingAsTheGoalOfCouplingFixtureResult",
    "classify_maximizing_as_the_goal_of_coupling_component",
    "maximizing_as_the_goal_of_coupling_labels",
    "validate_maximizing_as_the_goal_of_coupling_fixture",
]
