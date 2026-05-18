# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Formalising the NTHS within a Multi-Agent Active Inference Framework validation
"""Source-accounting checks for Paper 0 Formalising the NTHS within a Multi-Agent Active Inference Framework records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded formalising the nths within a multi agent active inference framework source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05236", "P0R05244")


@dataclass(frozen=True, slots=True)
class FormalisingTheNthsWithinAMultiAgentActiveInferenceFrameworkConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 2
    next_source_boundary: str = "P0R05245"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R05245":
            raise ValueError("next_source_boundary must equal P0R05245")


@dataclass(frozen=True, slots=True)
class FormalisingTheNthsWithinAMultiAgentActiveInferenceFrameworkFixtureResult:
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


def classify_formalising_the_nths_within_a_multi_agent_active_inference_framework_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "formalising_the_nths_within_a_multi_agent_active_inference_framework": "formalising_the_nths_within_a_multi_agent_active_inference_framework_source_boundary",
        "conceptual_mapping": "conceptual_mapping_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown formalising_the_nths_within_a_multi_agent_active_inference_framework component"
        ) from exc


def formalising_the_nths_within_a_multi_agent_active_inference_framework_labels() -> dict[
    str, str
]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Formalising the NTHS within a Multi-Agent Active Inference Framework",
        "source_span": "P0R05236-P0R05244",
        "component_count": "2",
        "next_boundary": "P0R05245",
        "component_1": "Formalising the NTHS within a Multi-Agent Active Inference Framework",
        "component_2": "Conceptual Mapping",
    }


def validate_formalising_the_nths_within_a_multi_agent_active_inference_framework_fixture(
    config: FormalisingTheNthsWithinAMultiAgentActiveInferenceFrameworkConfig | None = None,
) -> FormalisingTheNthsWithinAMultiAgentActiveInferenceFrameworkFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or FormalisingTheNthsWithinAMultiAgentActiveInferenceFrameworkConfig()
    components = (
        "formalising_the_nths_within_a_multi_agent_active_inference_framework",
        "conceptual_mapping",
    )
    return FormalisingTheNthsWithinAMultiAgentActiveInferenceFrameworkFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_formalising_the_nths_within_a_multi_agent_active_inference_framework_component(
                component
            )
            for component in components
        },
        labels=formalising_the_nths_within_a_multi_agent_active_inference_framework_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "formalising_the_nths_within_a_multi_agent_active_inference_framework_is_not_empirical_validation_evidence": 1.0,
            "conceptual_mapping_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5236, 5245)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_formalising_the_nths_within_a_multi_agent_active_inference_framework_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "FormalisingTheNthsWithinAMultiAgentActiveInferenceFrameworkConfig",
    "FormalisingTheNthsWithinAMultiAgentActiveInferenceFrameworkFixtureResult",
    "classify_formalising_the_nths_within_a_multi_agent_active_inference_framework_component",
    "formalising_the_nths_within_a_multi_agent_active_inference_framework_labels",
    "validate_formalising_the_nths_within_a_multi_agent_active_inference_framework_fixture",
]
