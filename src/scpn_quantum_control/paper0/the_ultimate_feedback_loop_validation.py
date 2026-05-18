# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Ultimate Feedback Loop: validation
"""Source-accounting checks for Paper 0 The Ultimate Feedback Loop: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = (
    "source-bounded the ultimate feedback loop source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03067", "P0R03075")


@dataclass(frozen=True, slots=True)
class TheUltimateFeedbackLoopConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 2
    next_source_boundary: str = "P0R03076"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R03076":
            raise ValueError("next_source_boundary must equal P0R03076")


@dataclass(frozen=True, slots=True)
class TheUltimateFeedbackLoopFixtureResult:
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


def classify_the_ultimate_feedback_loop_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_ultimate_feedback_loop": "the_ultimate_feedback_loop_source_boundary",
        "the_quantum_error_correction_qec_imperative_and_the_role_of_the_psi_fiel": "the_quantum_error_correction_qec_imperative_and_the_role_of_the_psi_fiel_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown the_ultimate_feedback_loop component") from exc


def the_ultimate_feedback_loop_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Ultimate Feedback Loop:",
        "source_span": "P0R03067-P0R03075",
        "component_count": "2",
        "next_boundary": "P0R03076",
        "component_1": "The Ultimate Feedback Loop:",
        "component_2": "The Quantum Error Correction (QEC) Imperative and the Role of the Psi-Field",
    }


def validate_the_ultimate_feedback_loop_fixture(
    config: TheUltimateFeedbackLoopConfig | None = None,
) -> TheUltimateFeedbackLoopFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TheUltimateFeedbackLoopConfig()
    components = (
        "the_ultimate_feedback_loop",
        "the_quantum_error_correction_qec_imperative_and_the_role_of_the_psi_fiel",
    )
    return TheUltimateFeedbackLoopFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_ultimate_feedback_loop_component(component)
            for component in components
        },
        labels=the_ultimate_feedback_loop_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_ultimate_feedback_loop_is_not_empirical_validation_evidence": 1.0,
            "the_quantum_error_correction_qec_imperative_and_the_role_of_the_psi_fiel_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3067, 3076)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_ultimate_feedback_loop_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TheUltimateFeedbackLoopConfig",
    "TheUltimateFeedbackLoopFixtureResult",
    "classify_the_ultimate_feedback_loop_component",
    "the_ultimate_feedback_loop_labels",
    "validate_the_ultimate_feedback_loop_fixture",
]
