# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. Embodied Predictive Coding (Interoceptive Inference) validation
"""Source-accounting checks for Paper 0 2. Embodied Predictive Coding (Interoceptive Inference) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 2 embodied predictive coding interoceptive inference source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04967", "P0R04974")


@dataclass(frozen=True, slots=True)
class Section2EmbodiedPredictiveCodingInteroceptiveInferenceConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 2
    next_source_boundary: str = "P0R04975"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R04975":
            raise ValueError("next_source_boundary must equal P0R04975")


@dataclass(frozen=True, slots=True)
class Section2EmbodiedPredictiveCodingInteroceptiveInferenceFixtureResult:
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


def classify_section_2_embodied_predictive_coding_interoceptive_inference_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "2_embodied_predictive_coding_interoceptive_inference": "2_embodied_predictive_coding_interoceptive_inference_source_boundary",
        "3_distributed_criticality": "3_distributed_criticality_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_2_embodied_predictive_coding_interoceptive_inference component"
        ) from exc


def section_2_embodied_predictive_coding_interoceptive_inference_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "2. Embodied Predictive Coding (Interoceptive Inference)",
        "source_span": "P0R04967-P0R04974",
        "component_count": "2",
        "next_boundary": "P0R04975",
        "component_1": "2. Embodied Predictive Coding (Interoceptive Inference)",
        "component_2": "3. Distributed Criticality:",
    }


def validate_section_2_embodied_predictive_coding_interoceptive_inference_fixture(
    config: Section2EmbodiedPredictiveCodingInteroceptiveInferenceConfig | None = None,
) -> Section2EmbodiedPredictiveCodingInteroceptiveInferenceFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section2EmbodiedPredictiveCodingInteroceptiveInferenceConfig()
    components = (
        "2_embodied_predictive_coding_interoceptive_inference",
        "3_distributed_criticality",
    )
    return Section2EmbodiedPredictiveCodingInteroceptiveInferenceFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_2_embodied_predictive_coding_interoceptive_inference_component(
                component
            )
            for component in components
        },
        labels=section_2_embodied_predictive_coding_interoceptive_inference_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "2_embodied_predictive_coding_interoceptive_inference_is_not_empirical_validation_evidence": 1.0,
            "3_distributed_criticality_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4967, 4975)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_2_embodied_predictive_coding_interoceptive_inference_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section2EmbodiedPredictiveCodingInteroceptiveInferenceConfig",
    "Section2EmbodiedPredictiveCodingInteroceptiveInferenceFixtureResult",
    "classify_section_2_embodied_predictive_coding_interoceptive_inference_component",
    "section_2_embodied_predictive_coding_interoceptive_inference_labels",
    "validate_section_2_embodied_predictive_coding_interoceptive_inference_fixture",
]
