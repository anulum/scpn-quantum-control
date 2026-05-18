# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 3. Structural Geometry and Morphogenesis (L3): validation
"""Source-accounting checks for Paper 0 3. Structural Geometry and Morphogenesis (L3): records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 3 structural geometry and morphogenesis l3 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04396", "P0R04403")


@dataclass(frozen=True, slots=True)
class Section3StructuralGeometryAndMorphogenesisL3Config:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 3
    next_source_boundary: str = "P0R04404"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R04404":
            raise ValueError("next_source_boundary must equal P0R04404")


@dataclass(frozen=True, slots=True)
class Section3StructuralGeometryAndMorphogenesisL3FixtureResult:
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


def classify_section_3_structural_geometry_and_morphogenesis_l3_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "3_structural_geometry_and_morphogenesis_l3": "3_structural_geometry_and_morphogenesis_l3_source_boundary",
        "4_the_infoton_ciss_bridge_coupling_the_psi_field_to_spin_dynamics": "4_the_infoton_ciss_bridge_coupling_the_psi_field_to_spin_dynamics_source_boundary",
        "the_interaction_lagrangian": "the_interaction_lagrangian_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_3_structural_geometry_and_morphogenesis_l3 component"
        ) from exc


def section_3_structural_geometry_and_morphogenesis_l3_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "3. Structural Geometry and Morphogenesis (L3):",
        "source_span": "P0R04396-P0R04403",
        "component_count": "3",
        "next_boundary": "P0R04404",
        "component_1": "3. Structural Geometry and Morphogenesis (L3):",
        "component_2": "4. The Infoton-CISS Bridge: Coupling the Psi-Field to Spin Dynamics",
        "component_3": "The Interaction Lagrangian",
    }


def validate_section_3_structural_geometry_and_morphogenesis_l3_fixture(
    config: Section3StructuralGeometryAndMorphogenesisL3Config | None = None,
) -> Section3StructuralGeometryAndMorphogenesisL3FixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section3StructuralGeometryAndMorphogenesisL3Config()
    components = (
        "3_structural_geometry_and_morphogenesis_l3",
        "4_the_infoton_ciss_bridge_coupling_the_psi_field_to_spin_dynamics",
        "the_interaction_lagrangian",
    )
    return Section3StructuralGeometryAndMorphogenesisL3FixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_3_structural_geometry_and_morphogenesis_l3_component(
                component
            )
            for component in components
        },
        labels=section_3_structural_geometry_and_morphogenesis_l3_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "3_structural_geometry_and_morphogenesis_l3_is_not_empirical_validation_evidence": 1.0,
            "4_the_infoton_ciss_bridge_coupling_the_psi_field_to_spin_dynamics_is_not_empirical_validation_evidence": 1.0,
            "the_interaction_lagrangian_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4396, 4404)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_3_structural_geometry_and_morphogenesis_l3_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section3StructuralGeometryAndMorphogenesisL3Config",
    "Section3StructuralGeometryAndMorphogenesisL3FixtureResult",
    "classify_section_3_structural_geometry_and_morphogenesis_l3_component",
    "section_3_structural_geometry_and_morphogenesis_l3_labels",
    "validate_section_3_structural_geometry_and_morphogenesis_l3_fixture",
]
