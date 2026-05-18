# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. The Strange Loop (L5): The Geometry of Self-Reference validation
"""Source-accounting checks for Paper 0 2. The Strange Loop (L5): The Geometry of Self-Reference records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 2 the strange loop l5 the geometry of self reference source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04433", "P0R04440")


@dataclass(frozen=True, slots=True)
class Section2TheStrangeLoopL5TheGeometryOfSelfReferenceConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 3
    next_source_boundary: str = "P0R04441"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R04441":
            raise ValueError("next_source_boundary must equal P0R04441")


@dataclass(frozen=True, slots=True)
class Section2TheStrangeLoopL5TheGeometryOfSelfReferenceFixtureResult:
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


def classify_section_2_the_strange_loop_l5_the_geometry_of_self_reference_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "2_the_strange_loop_l5_the_geometry_of_self_reference": "2_the_strange_loop_l5_the_geometry_of_self_reference_source_boundary",
        "3_symbols_as_geometric_operators_l7": "3_symbols_as_geometric_operators_l7_source_boundary",
        "vi_the_geometry_of_memory_and_spacetime_domain_iii_l9_l10": "vi_the_geometry_of_memory_and_spacetime_domain_iii_l9_l10_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_2_the_strange_loop_l5_the_geometry_of_self_reference component"
        ) from exc


def section_2_the_strange_loop_l5_the_geometry_of_self_reference_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "2. The Strange Loop (L5): The Geometry of Self-Reference",
        "source_span": "P0R04433-P0R04440",
        "component_count": "3",
        "next_boundary": "P0R04441",
        "component_1": "2. The Strange Loop (L5): The Geometry of Self-Reference",
        "component_2": "3. Symbols as Geometric Operators (L7):",
        "component_3": "VI. The Geometry of Memory and Spacetime (Domain III: L9/L10)",
    }


def validate_section_2_the_strange_loop_l5_the_geometry_of_self_reference_fixture(
    config: Section2TheStrangeLoopL5TheGeometryOfSelfReferenceConfig | None = None,
) -> Section2TheStrangeLoopL5TheGeometryOfSelfReferenceFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section2TheStrangeLoopL5TheGeometryOfSelfReferenceConfig()
    components = (
        "2_the_strange_loop_l5_the_geometry_of_self_reference",
        "3_symbols_as_geometric_operators_l7",
        "vi_the_geometry_of_memory_and_spacetime_domain_iii_l9_l10",
    )
    return Section2TheStrangeLoopL5TheGeometryOfSelfReferenceFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_2_the_strange_loop_l5_the_geometry_of_self_reference_component(
                component
            )
            for component in components
        },
        labels=section_2_the_strange_loop_l5_the_geometry_of_self_reference_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "2_the_strange_loop_l5_the_geometry_of_self_reference_is_not_empirical_validation_evidence": 1.0,
            "3_symbols_as_geometric_operators_l7_is_not_empirical_validation_evidence": 1.0,
            "vi_the_geometry_of_memory_and_spacetime_domain_iii_l9_l10_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4433, 4441)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_2_the_strange_loop_l5_the_geometry_of_self_reference_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section2TheStrangeLoopL5TheGeometryOfSelfReferenceConfig",
    "Section2TheStrangeLoopL5TheGeometryOfSelfReferenceFixtureResult",
    "classify_section_2_the_strange_loop_l5_the_geometry_of_self_reference_component",
    "section_2_the_strange_loop_l5_the_geometry_of_self_reference_labels",
    "validate_section_2_the_strange_loop_l5_the_geometry_of_self_reference_fixture",
]
