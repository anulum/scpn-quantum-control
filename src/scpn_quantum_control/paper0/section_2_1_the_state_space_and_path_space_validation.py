# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2.1 The State Space and Path Space validation
"""Source-accounting checks for Paper 0 2.1 The State Space and Path Space records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 2 1 the state space and path space source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03753", "P0R03761")


@dataclass(frozen=True, slots=True)
class Section21TheStateSpaceAndPathSpaceConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 2
    next_source_boundary: str = "P0R03762"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R03762":
            raise ValueError("next_source_boundary must equal P0R03762")


@dataclass(frozen=True, slots=True)
class Section21TheStateSpaceAndPathSpaceFixtureResult:
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


def classify_section_2_1_the_state_space_and_path_space_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "2_1_the_state_space_and_path_space": "2_1_the_state_space_and_path_space_source_boundary",
        "2_2_the_path_integral_measure": "2_2_the_path_integral_measure_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown section_2_1_the_state_space_and_path_space component") from exc


def section_2_1_the_state_space_and_path_space_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "2.1 The State Space and Path Space",
        "source_span": "P0R03753-P0R03761",
        "component_count": "2",
        "next_boundary": "P0R03762",
        "component_1": "2.1 The State Space and Path Space",
        "component_2": "2.2 The Path Integral Measure",
    }


def validate_section_2_1_the_state_space_and_path_space_fixture(
    config: Section21TheStateSpaceAndPathSpaceConfig | None = None,
) -> Section21TheStateSpaceAndPathSpaceFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section21TheStateSpaceAndPathSpaceConfig()
    components = ("2_1_the_state_space_and_path_space", "2_2_the_path_integral_measure")
    return Section21TheStateSpaceAndPathSpaceFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_2_1_the_state_space_and_path_space_component(component)
            for component in components
        },
        labels=section_2_1_the_state_space_and_path_space_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "2_1_the_state_space_and_path_space_is_not_empirical_validation_evidence": 1.0,
            "2_2_the_path_integral_measure_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3753, 3762)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_2_1_the_state_space_and_path_space_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section21TheStateSpaceAndPathSpaceConfig",
    "Section21TheStateSpaceAndPathSpaceFixtureResult",
    "classify_section_2_1_the_state_space_and_path_space_component",
    "section_2_1_the_state_space_and_path_space_labels",
    "validate_section_2_1_the_state_space_and_path_space_fixture",
]
