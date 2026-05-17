# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 3. Memory Capacity (Bekenstein-Hawking Bound) validation
"""Source-accounting checks for Paper 0 3. Memory Capacity (Bekenstein-Hawking Bound) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 3 memory capacity bekenstein hawking bound source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R02257", "P0R02277")


@dataclass(frozen=True, slots=True)
class Section3MemoryCapacityBekensteinHawkingBoundConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 21
    expected_component_count: int = 3
    next_source_boundary: str = "P0R02278"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 21:
            raise ValueError("expected_source_record_count must equal 21")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R02278":
            raise ValueError("next_source_boundary must equal P0R02278")


@dataclass(frozen=True, slots=True)
class Section3MemoryCapacityBekensteinHawkingBoundFixtureResult:
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


def classify_section_3_memory_capacity_bekenstein_hawking_bound_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "3_memory_capacity_bekenstein_hawking_bound": "3_memory_capacity_bekenstein_hawking_bound_source_boundary",
        "p0r02263": "p0r02263_source_boundary",
        "4_emergent_spacetime_ryu_takayanagi_formula": "4_emergent_spacetime_ryu_takayanagi_formula_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_3_memory_capacity_bekenstein_hawking_bound component"
        ) from exc


def section_3_memory_capacity_bekenstein_hawking_bound_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "3. Memory Capacity (Bekenstein-Hawking Bound)",
        "source_span": "P0R02257-P0R02277",
        "component_count": "3",
        "next_boundary": "P0R02278",
        "component_1": "3. Memory Capacity (Bekenstein-Hawking Bound)",
        "component_2": "P0R02263",
        "component_3": "4. Emergent Spacetime (Ryu-Takayanagi Formula)",
    }


def validate_section_3_memory_capacity_bekenstein_hawking_bound_fixture(
    config: Section3MemoryCapacityBekensteinHawkingBoundConfig | None = None,
) -> Section3MemoryCapacityBekensteinHawkingBoundFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section3MemoryCapacityBekensteinHawkingBoundConfig()
    components = (
        "3_memory_capacity_bekenstein_hawking_bound",
        "p0r02263",
        "4_emergent_spacetime_ryu_takayanagi_formula",
    )
    return Section3MemoryCapacityBekensteinHawkingBoundFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_3_memory_capacity_bekenstein_hawking_bound_component(
                component
            )
            for component in components
        },
        labels=section_3_memory_capacity_bekenstein_hawking_bound_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "3_memory_capacity_bekenstein_hawking_bound_is_not_empirical_validation_evidence": 1.0,
            "p0r02263_is_not_empirical_validation_evidence": 1.0,
            "4_emergent_spacetime_ryu_takayanagi_formula_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(2257, 2278)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_3_memory_capacity_bekenstein_hawking_bound_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section3MemoryCapacityBekensteinHawkingBoundConfig",
    "Section3MemoryCapacityBekensteinHawkingBoundFixtureResult",
    "classify_section_3_memory_capacity_bekenstein_hawking_bound_component",
    "section_3_memory_capacity_bekenstein_hawking_bound_labels",
    "validate_section_3_memory_capacity_bekenstein_hawking_bound_fixture",
]
