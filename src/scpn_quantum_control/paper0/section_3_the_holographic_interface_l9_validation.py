# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 3. The Holographic Interface (L9): validation
"""Source-accounting checks for Paper 0 3. The Holographic Interface (L9): records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 3 the holographic interface l9 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05009", "P0R05016")


@dataclass(frozen=True, slots=True)
class Section3TheHolographicInterfaceL9Config:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 4
    next_source_boundary: str = "P0R05017"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R05017":
            raise ValueError("next_source_boundary must equal P0R05017")


@dataclass(frozen=True, slots=True)
class Section3TheHolographicInterfaceL9FixtureResult:
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


def classify_section_3_the_holographic_interface_l9_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "3_the_holographic_interface_l9": "3_the_holographic_interface_l9_source_boundary",
        "v_dynamics_across_the_lifespan_development_ageing_and_sleep": "v_dynamics_across_the_lifespan_development_ageing_and_sleep_source_boundary",
        "1_development_the_ascent_to_criticality": "1_development_the_ascent_to_criticality_source_boundary",
        "2_ageing_the_descent_from_criticality_and_decoherence": "2_ageing_the_descent_from_criticality_and_decoherence_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown section_3_the_holographic_interface_l9 component") from exc


def section_3_the_holographic_interface_l9_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "3. The Holographic Interface (L9):",
        "source_span": "P0R05009-P0R05016",
        "component_count": "4",
        "next_boundary": "P0R05017",
        "component_1": "3. The Holographic Interface (L9):",
        "component_2": "V. Dynamics Across the Lifespan: Development, Ageing, and Sleep",
        "component_3": "1. Development (The Ascent to Criticality):",
        "component_4": "2. Ageing (The Descent from Criticality and Decoherence):",
    }


def validate_section_3_the_holographic_interface_l9_fixture(
    config: Section3TheHolographicInterfaceL9Config | None = None,
) -> Section3TheHolographicInterfaceL9FixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section3TheHolographicInterfaceL9Config()
    components = (
        "3_the_holographic_interface_l9",
        "v_dynamics_across_the_lifespan_development_ageing_and_sleep",
        "1_development_the_ascent_to_criticality",
        "2_ageing_the_descent_from_criticality_and_decoherence",
    )
    return Section3TheHolographicInterfaceL9FixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_3_the_holographic_interface_l9_component(component)
            for component in components
        },
        labels=section_3_the_holographic_interface_l9_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "3_the_holographic_interface_l9_is_not_empirical_validation_evidence": 1.0,
            "v_dynamics_across_the_lifespan_development_ageing_and_sleep_is_not_empirical_validation_evidence": 1.0,
            "1_development_the_ascent_to_criticality_is_not_empirical_validation_evidence": 1.0,
            "2_ageing_the_descent_from_criticality_and_decoherence_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5009, 5017)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_3_the_holographic_interface_l9_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section3TheHolographicInterfaceL9Config",
    "Section3TheHolographicInterfaceL9FixtureResult",
    "classify_section_3_the_holographic_interface_l9_component",
    "section_3_the_holographic_interface_l9_labels",
    "validate_section_3_the_holographic_interface_l9_fixture",
]
