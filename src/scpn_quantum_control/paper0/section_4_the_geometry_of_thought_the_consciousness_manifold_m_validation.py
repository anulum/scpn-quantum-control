# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 4. The Geometry of Thought (The Consciousness Manifold M): validation
"""Source-accounting checks for Paper 0 4. The Geometry of Thought (The Consciousness Manifold M): records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 4 the geometry of thought the consciousness manifold m source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04526", "P0R04533")


@dataclass(frozen=True, slots=True)
class Section4TheGeometryOfThoughtTheConsciousnessManifoldMConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 3
    next_source_boundary: str = "P0R04534"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R04534":
            raise ValueError("next_source_boundary must equal P0R04534")


@dataclass(frozen=True, slots=True)
class Section4TheGeometryOfThoughtTheConsciousnessManifoldMFixtureResult:
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


def classify_section_4_the_geometry_of_thought_the_consciousness_manifold_m_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "4_the_geometry_of_thought_the_consciousness_manifold_m": "4_the_geometry_of_thought_the_consciousness_manifold_m_source_boundary",
        "5_the_neural_correlates_of_consciousness_ncc_an_scpn_synthesis": "5_the_neural_correlates_of_consciousness_ncc_an_scpn_synthesis_source_boundary",
        "vi_the_extended_brain_neuro_visceral_and_neuro_immune_axes": "vi_the_extended_brain_neuro_visceral_and_neuro_immune_axes_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_4_the_geometry_of_thought_the_consciousness_manifold_m component"
        ) from exc


def section_4_the_geometry_of_thought_the_consciousness_manifold_m_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "4. The Geometry of Thought (The Consciousness Manifold M):",
        "source_span": "P0R04526-P0R04533",
        "component_count": "3",
        "next_boundary": "P0R04534",
        "component_1": "4. The Geometry of Thought (The Consciousness Manifold M):",
        "component_2": "5. The Neural Correlates of Consciousness (NCC) - An SCPN Synthesis:",
        "component_3": "VI. The Extended Brain: Neuro-Visceral and Neuro-Immune Axes",
    }


def validate_section_4_the_geometry_of_thought_the_consciousness_manifold_m_fixture(
    config: Section4TheGeometryOfThoughtTheConsciousnessManifoldMConfig | None = None,
) -> Section4TheGeometryOfThoughtTheConsciousnessManifoldMFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section4TheGeometryOfThoughtTheConsciousnessManifoldMConfig()
    components = (
        "4_the_geometry_of_thought_the_consciousness_manifold_m",
        "5_the_neural_correlates_of_consciousness_ncc_an_scpn_synthesis",
        "vi_the_extended_brain_neuro_visceral_and_neuro_immune_axes",
    )
    return Section4TheGeometryOfThoughtTheConsciousnessManifoldMFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_4_the_geometry_of_thought_the_consciousness_manifold_m_component(
                component
            )
            for component in components
        },
        labels=section_4_the_geometry_of_thought_the_consciousness_manifold_m_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "4_the_geometry_of_thought_the_consciousness_manifold_m_is_not_empirical_validation_evidence": 1.0,
            "5_the_neural_correlates_of_consciousness_ncc_an_scpn_synthesis_is_not_empirical_validation_evidence": 1.0,
            "vi_the_extended_brain_neuro_visceral_and_neuro_immune_axes_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4526, 4534)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_4_the_geometry_of_thought_the_consciousness_manifold_m_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section4TheGeometryOfThoughtTheConsciousnessManifoldMConfig",
    "Section4TheGeometryOfThoughtTheConsciousnessManifoldMFixtureResult",
    "classify_section_4_the_geometry_of_thought_the_consciousness_manifold_m_component",
    "section_4_the_geometry_of_thought_the_consciousness_manifold_m_labels",
    "validate_section_4_the_geometry_of_thought_the_consciousness_manifold_m_fixture",
]
