# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Hard Problem: A Mathematical Resolution validation
"""Source-accounting checks for Paper 0 The Hard Problem: A Mathematical Resolution records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded the hard problem a mathematical resolution source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03418", "P0R03426")


@dataclass(frozen=True, slots=True)
class TheHardProblemAMathematicalResolutionConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 5
    next_source_boundary: str = "P0R03427"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 5:
            raise ValueError("expected_component_count must equal 5")
        if self.next_source_boundary != "P0R03427":
            raise ValueError("next_source_boundary must equal P0R03427")


@dataclass(frozen=True, slots=True)
class TheHardProblemAMathematicalResolutionFixtureResult:
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


def classify_the_hard_problem_a_mathematical_resolution_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_hard_problem_a_mathematical_resolution": "the_hard_problem_a_mathematical_resolution_source_boundary",
        "the_hard_problem_reformulated": "the_hard_problem_reformulated_source_boundary",
        "traditional": "traditional_source_boundary",
        "scpn": "scpn_source_boundary",
        "key_insight": "key_insight_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown the_hard_problem_a_mathematical_resolution component") from exc


def the_hard_problem_a_mathematical_resolution_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Hard Problem: A Mathematical Resolution",
        "source_span": "P0R03418-P0R03426",
        "component_count": "5",
        "next_boundary": "P0R03427",
        "component_1": "The Hard Problem: A Mathematical Resolution",
        "component_2": "The Hard Problem Reformulated:",
        "component_3": "Traditional:",
        "component_4": "SCPN:",
        "component_5": "Key insight:",
    }


def validate_the_hard_problem_a_mathematical_resolution_fixture(
    config: TheHardProblemAMathematicalResolutionConfig | None = None,
) -> TheHardProblemAMathematicalResolutionFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TheHardProblemAMathematicalResolutionConfig()
    components = (
        "the_hard_problem_a_mathematical_resolution",
        "the_hard_problem_reformulated",
        "traditional",
        "scpn",
        "key_insight",
    )
    return TheHardProblemAMathematicalResolutionFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_hard_problem_a_mathematical_resolution_component(component)
            for component in components
        },
        labels=the_hard_problem_a_mathematical_resolution_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_hard_problem_a_mathematical_resolution_is_not_empirical_validation_evidence": 1.0,
            "the_hard_problem_reformulated_is_not_empirical_validation_evidence": 1.0,
            "traditional_is_not_empirical_validation_evidence": 1.0,
            "scpn_is_not_empirical_validation_evidence": 1.0,
            "key_insight_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3418, 3427)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_hard_problem_a_mathematical_resolution_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TheHardProblemAMathematicalResolutionConfig",
    "TheHardProblemAMathematicalResolutionFixtureResult",
    "classify_the_hard_problem_a_mathematical_resolution_component",
    "the_hard_problem_a_mathematical_resolution_labels",
    "validate_the_hard_problem_a_mathematical_resolution_fixture",
]
