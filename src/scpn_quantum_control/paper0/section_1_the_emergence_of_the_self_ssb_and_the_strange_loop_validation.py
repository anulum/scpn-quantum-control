# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 1. The Emergence of the Self (SSB and the Strange Loop): validation
"""Source-accounting checks for Paper 0 1. The Emergence of the Self (SSB and the Strange Loop): records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 1 the emergence of the self ssb and the strange loop source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04517", "P0R04525")


@dataclass(frozen=True, slots=True)
class Section1TheEmergenceOfTheSelfSsbAndTheStrangeLoopConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 3
    next_source_boundary: str = "P0R04526"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R04526":
            raise ValueError("next_source_boundary must equal P0R04526")


@dataclass(frozen=True, slots=True)
class Section1TheEmergenceOfTheSelfSsbAndTheStrangeLoopFixtureResult:
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


def classify_section_1_the_emergence_of_the_self_ssb_and_the_strange_loop_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "1_the_emergence_of_the_self_ssb_and_the_strange_loop": "1_the_emergence_of_the_self_ssb_and_the_strange_loop_source_boundary",
        "2_hierarchical_predictive_coding_hpc_in_the_cortex": "2_hierarchical_predictive_coding_hpc_in_the_cortex_source_boundary",
        "3_mapping_major_cognitive_networks": "3_mapping_major_cognitive_networks_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_1_the_emergence_of_the_self_ssb_and_the_strange_loop component"
        ) from exc


def section_1_the_emergence_of_the_self_ssb_and_the_strange_loop_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "1. The Emergence of the Self (SSB and the Strange Loop):",
        "source_span": "P0R04517-P0R04525",
        "component_count": "3",
        "next_boundary": "P0R04526",
        "component_1": "1. The Emergence of the Self (SSB and the Strange Loop):",
        "component_2": "2. Hierarchical Predictive Coding (HPC) in the Cortex:",
        "component_3": "3. Mapping Major Cognitive Networks:",
    }


def validate_section_1_the_emergence_of_the_self_ssb_and_the_strange_loop_fixture(
    config: Section1TheEmergenceOfTheSelfSsbAndTheStrangeLoopConfig | None = None,
) -> Section1TheEmergenceOfTheSelfSsbAndTheStrangeLoopFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section1TheEmergenceOfTheSelfSsbAndTheStrangeLoopConfig()
    components = (
        "1_the_emergence_of_the_self_ssb_and_the_strange_loop",
        "2_hierarchical_predictive_coding_hpc_in_the_cortex",
        "3_mapping_major_cognitive_networks",
    )
    return Section1TheEmergenceOfTheSelfSsbAndTheStrangeLoopFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_1_the_emergence_of_the_self_ssb_and_the_strange_loop_component(
                component
            )
            for component in components
        },
        labels=section_1_the_emergence_of_the_self_ssb_and_the_strange_loop_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "1_the_emergence_of_the_self_ssb_and_the_strange_loop_is_not_empirical_validation_evidence": 1.0,
            "2_hierarchical_predictive_coding_hpc_in_the_cortex_is_not_empirical_validation_evidence": 1.0,
            "3_mapping_major_cognitive_networks_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4517, 4526)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_1_the_emergence_of_the_self_ssb_and_the_strange_loop_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section1TheEmergenceOfTheSelfSsbAndTheStrangeLoopConfig",
    "Section1TheEmergenceOfTheSelfSsbAndTheStrangeLoopFixtureResult",
    "classify_section_1_the_emergence_of_the_self_ssb_and_the_strange_loop_component",
    "section_1_the_emergence_of_the_self_ssb_and_the_strange_loop_labels",
    "validate_section_1_the_emergence_of_the_self_ssb_and_the_strange_loop_fixture",
]
