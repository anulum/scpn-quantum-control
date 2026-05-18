# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Time, Retrocausality, and Two-State Vector validation
"""Source-accounting checks for Paper 0  Time, Retrocausality, and Two-State Vector records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded time retrocausality and two state vector source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05902", "P0R05909")


@dataclass(frozen=True, slots=True)
class TimeRetrocausalityAndTwoStateVectorConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 2
    next_source_boundary: str = "P0R05910"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R05910":
            raise ValueError("next_source_boundary must equal P0R05910")


@dataclass(frozen=True, slots=True)
class TimeRetrocausalityAndTwoStateVectorFixtureResult:
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


def classify_time_retrocausality_and_two_state_vector_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "time_retrocausality_and_two_state_vector": "time_retrocausality_and_two_state_vector_source_boundary",
        "memory_holography": "memory_holography_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown time_retrocausality_and_two_state_vector component") from exc


def time_retrocausality_and_two_state_vector_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": " Time, Retrocausality, and Two-State Vector",
        "source_span": "P0R05902-P0R05909",
        "component_count": "2",
        "next_boundary": "P0R05910",
        "component_1": "Time, Retrocausality, and Two-State Vector",
        "component_2": "Memory & Holography",
    }


def validate_time_retrocausality_and_two_state_vector_fixture(
    config: TimeRetrocausalityAndTwoStateVectorConfig | None = None,
) -> TimeRetrocausalityAndTwoStateVectorFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TimeRetrocausalityAndTwoStateVectorConfig()
    components = ("time_retrocausality_and_two_state_vector", "memory_holography")
    return TimeRetrocausalityAndTwoStateVectorFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_time_retrocausality_and_two_state_vector_component(component)
            for component in components
        },
        labels=time_retrocausality_and_two_state_vector_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "time_retrocausality_and_two_state_vector_is_not_empirical_validation_evidence": 1.0,
            "memory_holography_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5902, 5910)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_time_retrocausality_and_two_state_vector_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TimeRetrocausalityAndTwoStateVectorConfig",
    "TimeRetrocausalityAndTwoStateVectorFixtureResult",
    "classify_time_retrocausality_and_two_state_vector_component",
    "time_retrocausality_and_two_state_vector_labels",
    "validate_time_retrocausality_and_two_state_vector_fixture",
]
