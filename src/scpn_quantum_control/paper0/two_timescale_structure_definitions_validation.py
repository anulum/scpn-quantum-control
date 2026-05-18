# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Two-timescale structure (definitions). validation
"""Source-accounting checks for Paper 0 Two-timescale structure (definitions). records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded two timescale structure definitions source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R02958", "P0R02966")


@dataclass(frozen=True, slots=True)
class TwoTimescaleStructureDefinitionsConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 2
    next_source_boundary: str = "P0R02967"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R02967":
            raise ValueError("next_source_boundary must equal P0R02967")


@dataclass(frozen=True, slots=True)
class TwoTimescaleStructureDefinitionsFixtureResult:
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


def classify_two_timescale_structure_definitions_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "two_timescale_structure_definitions": "two_timescale_structure_definitions_source_boundary",
        "gain_scheduling_via_the_affective_field": "gain_scheduling_via_the_affective_field_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown two_timescale_structure_definitions component") from exc


def two_timescale_structure_definitions_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Two-timescale structure (definitions).",
        "source_span": "P0R02958-P0R02966",
        "component_count": "2",
        "next_boundary": "P0R02967",
        "component_1": "Two-timescale structure (definitions).",
        "component_2": "Gain scheduling via the Affective Field.",
    }


def validate_two_timescale_structure_definitions_fixture(
    config: TwoTimescaleStructureDefinitionsConfig | None = None,
) -> TwoTimescaleStructureDefinitionsFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TwoTimescaleStructureDefinitionsConfig()
    components = ("two_timescale_structure_definitions", "gain_scheduling_via_the_affective_field")
    return TwoTimescaleStructureDefinitionsFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_two_timescale_structure_definitions_component(component)
            for component in components
        },
        labels=two_timescale_structure_definitions_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "two_timescale_structure_definitions_is_not_empirical_validation_evidence": 1.0,
            "gain_scheduling_via_the_affective_field_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(2958, 2967)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_two_timescale_structure_definitions_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TwoTimescaleStructureDefinitionsConfig",
    "TwoTimescaleStructureDefinitionsFixtureResult",
    "classify_two_timescale_structure_definitions_component",
    "two_timescale_structure_definitions_labels",
    "validate_two_timescale_structure_definitions_fixture",
]
