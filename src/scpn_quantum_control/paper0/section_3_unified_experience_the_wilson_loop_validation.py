# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 3. Unified Experience (The Wilson Loop): validation
"""Source-accounting checks for Paper 0 3. Unified Experience (The Wilson Loop): records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 3 unified experience the wilson loop source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03216", "P0R03231")


@dataclass(frozen=True, slots=True)
class Section3UnifiedExperienceTheWilsonLoopConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 16
    expected_component_count: int = 3
    next_source_boundary: str = "P0R03232"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 16:
            raise ValueError("expected_source_record_count must equal 16")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R03232":
            raise ValueError("next_source_boundary must equal P0R03232")


@dataclass(frozen=True, slots=True)
class Section3UnifiedExperienceTheWilsonLoopFixtureResult:
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


def classify_section_3_unified_experience_the_wilson_loop_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "3_unified_experience_the_wilson_loop": "3_unified_experience_the_wilson_loop_source_boundary",
        "iii_information_flow_dynamics_ilit": "iii_information_flow_dynamics_ilit_source_boundary",
        "1_quantifying_causality_via_transfer_entropy_te": "1_quantifying_causality_via_transfer_entropy_te_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown section_3_unified_experience_the_wilson_loop component") from exc


def section_3_unified_experience_the_wilson_loop_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "3. Unified Experience (The Wilson Loop):",
        "source_span": "P0R03216-P0R03231",
        "component_count": "3",
        "next_boundary": "P0R03232",
        "component_1": "3. Unified Experience (The Wilson Loop):",
        "component_2": "III. Information Flow Dynamics (ILIT)",
        "component_3": "1. Quantifying Causality via Transfer Entropy (TE):",
    }


def validate_section_3_unified_experience_the_wilson_loop_fixture(
    config: Section3UnifiedExperienceTheWilsonLoopConfig | None = None,
) -> Section3UnifiedExperienceTheWilsonLoopFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section3UnifiedExperienceTheWilsonLoopConfig()
    components = (
        "3_unified_experience_the_wilson_loop",
        "iii_information_flow_dynamics_ilit",
        "1_quantifying_causality_via_transfer_entropy_te",
    )
    return Section3UnifiedExperienceTheWilsonLoopFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_3_unified_experience_the_wilson_loop_component(component)
            for component in components
        },
        labels=section_3_unified_experience_the_wilson_loop_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "3_unified_experience_the_wilson_loop_is_not_empirical_validation_evidence": 1.0,
            "iii_information_flow_dynamics_ilit_is_not_empirical_validation_evidence": 1.0,
            "1_quantifying_causality_via_transfer_entropy_te_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3216, 3232)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_3_unified_experience_the_wilson_loop_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section3UnifiedExperienceTheWilsonLoopConfig",
    "Section3UnifiedExperienceTheWilsonLoopFixtureResult",
    "classify_section_3_unified_experience_the_wilson_loop_component",
    "section_3_unified_experience_the_wilson_loop_labels",
    "validate_section_3_unified_experience_the_wilson_loop_fixture",
]
