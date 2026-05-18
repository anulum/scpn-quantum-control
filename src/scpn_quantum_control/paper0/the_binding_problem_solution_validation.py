# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Binding Problem Solution: validation
"""Source-accounting checks for Paper 0 The Binding Problem Solution: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = (
    "source-bounded the binding problem solution source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03440", "P0R03452")


@dataclass(frozen=True, slots=True)
class TheBindingProblemSolutionConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 13
    expected_component_count: int = 2
    next_source_boundary: str = "P0R03453"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 13:
            raise ValueError("expected_source_record_count must equal 13")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R03453":
            raise ValueError("next_source_boundary must equal P0R03453")


@dataclass(frozen=True, slots=True)
class TheBindingProblemSolutionFixtureResult:
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


def classify_the_binding_problem_solution_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_binding_problem_solution": "the_binding_problem_solution_source_boundary",
        "mary_s_room_resolution": "mary_s_room_resolution_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown the_binding_problem_solution component") from exc


def the_binding_problem_solution_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Binding Problem Solution:",
        "source_span": "P0R03440-P0R03452",
        "component_count": "2",
        "next_boundary": "P0R03453",
        "component_1": "The Binding Problem Solution:",
        "component_2": "Mary's Room Resolution:",
    }


def validate_the_binding_problem_solution_fixture(
    config: TheBindingProblemSolutionConfig | None = None,
) -> TheBindingProblemSolutionFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TheBindingProblemSolutionConfig()
    components = ("the_binding_problem_solution", "mary_s_room_resolution")
    return TheBindingProblemSolutionFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_binding_problem_solution_component(component)
            for component in components
        },
        labels=the_binding_problem_solution_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_binding_problem_solution_is_not_empirical_validation_evidence": 1.0,
            "mary_s_room_resolution_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3440, 3453)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_binding_problem_solution_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TheBindingProblemSolutionConfig",
    "TheBindingProblemSolutionFixtureResult",
    "classify_the_binding_problem_solution_component",
    "the_binding_problem_solution_labels",
    "validate_the_binding_problem_solution_fixture",
]
