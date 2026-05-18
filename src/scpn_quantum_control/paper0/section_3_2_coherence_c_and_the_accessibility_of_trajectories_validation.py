# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 3.2 Coherence (C) and the Accessibility of Trajectories validation
"""Source-accounting checks for Paper 0 3.2 Coherence (C) and the Accessibility of Trajectories records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 3 2 coherence c and the accessibility of trajectories source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03789", "P0R03803")


@dataclass(frozen=True, slots=True)
class Section32CoherenceCAndTheAccessibilityOfTrajectoriesConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 15
    expected_component_count: int = 2
    next_source_boundary: str = "P0R03804"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 15:
            raise ValueError("expected_source_record_count must equal 15")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R03804":
            raise ValueError("next_source_boundary must equal P0R03804")


@dataclass(frozen=True, slots=True)
class Section32CoherenceCAndTheAccessibilityOfTrajectoriesFixtureResult:
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


def classify_section_3_2_coherence_c_and_the_accessibility_of_trajectories_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "3_2_coherence_c_and_the_accessibility_of_trajectories": "3_2_coherence_c_and_the_accessibility_of_trajectories_source_boundary",
        "x_3_3_qualia_capacity_q_and_the_diversity_of_dynamics": "x_3_3_qualia_capacity_q_and_the_diversity_of_dynamics_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_3_2_coherence_c_and_the_accessibility_of_trajectories component"
        ) from exc


def section_3_2_coherence_c_and_the_accessibility_of_trajectories_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "3.2 Coherence (C) and the Accessibility of Trajectories",
        "source_span": "P0R03789-P0R03803",
        "component_count": "2",
        "next_boundary": "P0R03804",
        "component_1": "3.2 Coherence (C) and the Accessibility of Trajectories",
        "component_2": "X.3.3 Qualia Capacity (Q) and the Diversity of Dynamics",
    }


def validate_section_3_2_coherence_c_and_the_accessibility_of_trajectories_fixture(
    config: Section32CoherenceCAndTheAccessibilityOfTrajectoriesConfig | None = None,
) -> Section32CoherenceCAndTheAccessibilityOfTrajectoriesFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section32CoherenceCAndTheAccessibilityOfTrajectoriesConfig()
    components = (
        "3_2_coherence_c_and_the_accessibility_of_trajectories",
        "x_3_3_qualia_capacity_q_and_the_diversity_of_dynamics",
    )
    return Section32CoherenceCAndTheAccessibilityOfTrajectoriesFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_3_2_coherence_c_and_the_accessibility_of_trajectories_component(
                component
            )
            for component in components
        },
        labels=section_3_2_coherence_c_and_the_accessibility_of_trajectories_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "3_2_coherence_c_and_the_accessibility_of_trajectories_is_not_empirical_validation_evidence": 1.0,
            "x_3_3_qualia_capacity_q_and_the_diversity_of_dynamics_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3789, 3804)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_3_2_coherence_c_and_the_accessibility_of_trajectories_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section32CoherenceCAndTheAccessibilityOfTrajectoriesConfig",
    "Section32CoherenceCAndTheAccessibilityOfTrajectoriesFixtureResult",
    "classify_section_3_2_coherence_c_and_the_accessibility_of_trajectories_component",
    "section_3_2_coherence_c_and_the_accessibility_of_trajectories_labels",
    "validate_section_3_2_coherence_c_and_the_accessibility_of_trajectories_fixture",
]
