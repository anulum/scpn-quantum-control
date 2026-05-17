# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Case Study: The Layer 5 (Organismal Self) Action-Perception Cycle validation
"""Source-accounting checks for Paper 0 Case Study: The Layer 5 (Organismal Self) Action-Perception Cycle records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded case study the layer 5 organismal self action perception cycle source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R02177", "P0R02188")


@dataclass(frozen=True, slots=True)
class CaseStudyTheLayer5OrganismalSelfActionPerceptionCycleConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 12
    expected_component_count: int = 1
    next_source_boundary: str = "P0R02189"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 12:
            raise ValueError("expected_source_record_count must equal 12")
        if self.expected_component_count != 1:
            raise ValueError("expected_component_count must equal 1")
        if self.next_source_boundary != "P0R02189":
            raise ValueError("next_source_boundary must equal P0R02189")


@dataclass(frozen=True, slots=True)
class CaseStudyTheLayer5OrganismalSelfActionPerceptionCycleFixtureResult:
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


def classify_case_study_the_layer_5_organismal_self_action_perception_cycle_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "case_study_the_layer_5_organismal_self_action_perception_cycle": "case_study_the_layer_5_organismal_self_action_perception_cycle_source_boundary"
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown case_study_the_layer_5_organismal_self_action_perception_cycle component"
        ) from exc


def case_study_the_layer_5_organismal_self_action_perception_cycle_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Case Study: The Layer 5 (Organismal Self) Action-Perception Cycle",
        "source_span": "P0R02177-P0R02188",
        "component_count": "1",
        "next_boundary": "P0R02189",
        "component_1": "Case Study: The Layer 5 (Organismal Self) Action-Perception Cycle",
    }


def validate_case_study_the_layer_5_organismal_self_action_perception_cycle_fixture(
    config: CaseStudyTheLayer5OrganismalSelfActionPerceptionCycleConfig | None = None,
) -> CaseStudyTheLayer5OrganismalSelfActionPerceptionCycleFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or CaseStudyTheLayer5OrganismalSelfActionPerceptionCycleConfig()
    components = ("case_study_the_layer_5_organismal_self_action_perception_cycle",)
    return CaseStudyTheLayer5OrganismalSelfActionPerceptionCycleFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_case_study_the_layer_5_organismal_self_action_perception_cycle_component(
                component
            )
            for component in components
        },
        labels=case_study_the_layer_5_organismal_self_action_perception_cycle_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "case_study_the_layer_5_organismal_self_action_perception_cycle_is_not_empirical_validation_evidence": 1.0
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(2177, 2189)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_case_study_the_layer_5_organismal_self_action_perception_cycle_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "CaseStudyTheLayer5OrganismalSelfActionPerceptionCycleConfig",
    "CaseStudyTheLayer5OrganismalSelfActionPerceptionCycleFixtureResult",
    "classify_case_study_the_layer_5_organismal_self_action_perception_cycle_component",
    "case_study_the_layer_5_organismal_self_action_perception_cycle_labels",
    "validate_case_study_the_layer_5_organismal_self_action_perception_cycle_fixture",
]
