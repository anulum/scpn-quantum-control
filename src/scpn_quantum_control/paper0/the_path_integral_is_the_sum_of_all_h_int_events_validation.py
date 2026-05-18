# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Path Integral is the Sum of all H_int events: validation
"""Source-accounting checks for Paper 0 The Path Integral is the Sum of all H_int events: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded the path integral is the sum of all h int events source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03673", "P0R03703")


@dataclass(frozen=True, slots=True)
class ThePathIntegralIsTheSumOfAllHIntEventsConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 31
    expected_component_count: int = 3
    next_source_boundary: str = "P0R03704"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 31:
            raise ValueError("expected_source_record_count must equal 31")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R03704":
            raise ValueError("next_source_boundary must equal P0R03704")


@dataclass(frozen=True, slots=True)
class ThePathIntegralIsTheSumOfAllHIntEventsFixtureResult:
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


def classify_the_path_integral_is_the_sum_of_all_h_int_events_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_path_integral_is_the_sum_of_all_h_int_events": "the_path_integral_is_the_sum_of_all_h_int_events_source_boundary",
        "cef_biases_the_coupling": "cef_biases_the_coupling_source_boundary",
        "ethics_as_causal_entropic_forces_cef": "ethics_as_causal_entropic_forces_cef_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown the_path_integral_is_the_sum_of_all_h_int_events component"
        ) from exc


def the_path_integral_is_the_sum_of_all_h_int_events_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Path Integral is the Sum of all H_int events:",
        "source_span": "P0R03673-P0R03703",
        "component_count": "3",
        "next_boundary": "P0R03704",
        "component_1": "The Path Integral is the Sum of all H_int events:",
        "component_2": "CEF Biases the Coupling:",
        "component_3": "Ethics as Causal Entropic Forces (CEF)",
    }


def validate_the_path_integral_is_the_sum_of_all_h_int_events_fixture(
    config: ThePathIntegralIsTheSumOfAllHIntEventsConfig | None = None,
) -> ThePathIntegralIsTheSumOfAllHIntEventsFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or ThePathIntegralIsTheSumOfAllHIntEventsConfig()
    components = (
        "the_path_integral_is_the_sum_of_all_h_int_events",
        "cef_biases_the_coupling",
        "ethics_as_causal_entropic_forces_cef",
    )
    return ThePathIntegralIsTheSumOfAllHIntEventsFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_path_integral_is_the_sum_of_all_h_int_events_component(
                component
            )
            for component in components
        },
        labels=the_path_integral_is_the_sum_of_all_h_int_events_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_path_integral_is_the_sum_of_all_h_int_events_is_not_empirical_validation_evidence": 1.0,
            "cef_biases_the_coupling_is_not_empirical_validation_evidence": 1.0,
            "ethics_as_causal_entropic_forces_cef_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3673, 3704)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_path_integral_is_the_sum_of_all_h_int_events_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "ThePathIntegralIsTheSumOfAllHIntEventsConfig",
    "ThePathIntegralIsTheSumOfAllHIntEventsFixtureResult",
    "classify_the_path_integral_is_the_sum_of_all_h_int_events_component",
    "the_path_integral_is_the_sum_of_all_h_int_events_labels",
    "validate_the_path_integral_is_the_sum_of_all_h_int_events_fixture",
]
