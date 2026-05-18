# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 4.3 The Origin of Purpose: Causal Entropic Forces & Negative Entropy validation
"""Source-accounting checks for Paper 0 4.3 The Origin of Purpose: Causal Entropic Forces & Negative Entropy records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 4 3 the origin of purpose causal entropic forces negative entropy source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03653", "P0R03663")


@dataclass(frozen=True, slots=True)
class Section43TheOriginOfPurposeCausalEntropicForcesNegativeEntropyConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 11
    expected_component_count: int = 2
    next_source_boundary: str = "P0R03664"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 11:
            raise ValueError("expected_source_record_count must equal 11")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R03664":
            raise ValueError("next_source_boundary must equal P0R03664")


@dataclass(frozen=True, slots=True)
class Section43TheOriginOfPurposeCausalEntropicForcesNegativeEntropyFixtureResult:
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


def classify_section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy": "4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_source_boundary",
        "causal_entropic_forces_cef": "causal_entropic_forces_cef_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy component"
        ) from exc


def section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_labels() -> dict[
    str, str
]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "4.3 The Origin of Purpose: Causal Entropic Forces & Negative Entropy",
        "source_span": "P0R03653-P0R03663",
        "component_count": "2",
        "next_boundary": "P0R03664",
        "component_1": "4.3 The Origin of Purpose: Causal Entropic Forces & Negative Entropy",
        "component_2": "Causal Entropic Forces (CEF)",
    }


def validate_section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_fixture(
    config: Section43TheOriginOfPurposeCausalEntropicForcesNegativeEntropyConfig | None = None,
) -> Section43TheOriginOfPurposeCausalEntropicForcesNegativeEntropyFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section43TheOriginOfPurposeCausalEntropicForcesNegativeEntropyConfig()
    components = (
        "4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy",
        "causal_entropic_forces_cef",
    )
    return Section43TheOriginOfPurposeCausalEntropicForcesNegativeEntropyFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_component(
                component
            )
            for component in components
        },
        labels=section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_is_not_empirical_validation_evidence": 1.0,
            "causal_entropic_forces_cef_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3653, 3664)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section43TheOriginOfPurposeCausalEntropicForcesNegativeEntropyConfig",
    "Section43TheOriginOfPurposeCausalEntropicForcesNegativeEntropyFixtureResult",
    "classify_section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_component",
    "section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_labels",
    "validate_section_4_3_the_origin_of_purpose_causal_entropic_forces_negative_entropy_fixture",
]
