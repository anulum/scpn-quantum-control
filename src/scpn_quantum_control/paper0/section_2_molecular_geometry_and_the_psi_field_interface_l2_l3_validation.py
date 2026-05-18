# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. Molecular Geometry and the Psi-Field Interface (L2/L3): validation
"""Source-accounting checks for Paper 0 2. Molecular Geometry and the Psi-Field Interface (L2/L3): records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 2 molecular geometry and the psi field interface l2 l3 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04824", "P0R04831")


@dataclass(frozen=True, slots=True)
class Section2MolecularGeometryAndThePsiFieldInterfaceL2L3Config:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 4
    next_source_boundary: str = "P0R04832"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R04832":
            raise ValueError("next_source_boundary must equal P0R04832")


@dataclass(frozen=True, slots=True)
class Section2MolecularGeometryAndThePsiFieldInterfaceL2L3FixtureResult:
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


def classify_section_2_molecular_geometry_and_the_psi_field_interface_l2_l3_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "2_molecular_geometry_and_the_psi_field_interface_l2_l3": "2_molecular_geometry_and_the_psi_field_interface_l2_l3_source_boundary",
        "3_cellular_geometry_tensegrity_and_fractals_l3": "3_cellular_geometry_tensegrity_and_fractals_l3_source_boundary",
        "iii_meso_scale_geometry_circuits_columns_and_fields_l3_l4": "iii_meso_scale_geometry_circuits_columns_and_fields_l3_l4_source_boundary",
        "1_the_geometry_of_the_bioelectric_field_l3": "1_the_geometry_of_the_bioelectric_field_l3_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_2_molecular_geometry_and_the_psi_field_interface_l2_l3 component"
        ) from exc


def section_2_molecular_geometry_and_the_psi_field_interface_l2_l3_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "2. Molecular Geometry and the Psi-Field Interface (L2/L3):",
        "source_span": "P0R04824-P0R04831",
        "component_count": "4",
        "next_boundary": "P0R04832",
        "component_1": "2. Molecular Geometry and the Psi-Field Interface (L2/L3):",
        "component_2": "3. Cellular Geometry: Tensegrity and Fractals (L3):",
        "component_3": "III. Meso-Scale Geometry: Circuits, Columns, and Fields (L3-L4)",
        "component_4": "1. The Geometry of the Bioelectric Field (L3):",
    }


def validate_section_2_molecular_geometry_and_the_psi_field_interface_l2_l3_fixture(
    config: Section2MolecularGeometryAndThePsiFieldInterfaceL2L3Config | None = None,
) -> Section2MolecularGeometryAndThePsiFieldInterfaceL2L3FixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section2MolecularGeometryAndThePsiFieldInterfaceL2L3Config()
    components = (
        "2_molecular_geometry_and_the_psi_field_interface_l2_l3",
        "3_cellular_geometry_tensegrity_and_fractals_l3",
        "iii_meso_scale_geometry_circuits_columns_and_fields_l3_l4",
        "1_the_geometry_of_the_bioelectric_field_l3",
    )
    return Section2MolecularGeometryAndThePsiFieldInterfaceL2L3FixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_2_molecular_geometry_and_the_psi_field_interface_l2_l3_component(
                component
            )
            for component in components
        },
        labels=section_2_molecular_geometry_and_the_psi_field_interface_l2_l3_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "2_molecular_geometry_and_the_psi_field_interface_l2_l3_is_not_empirical_validation_evidence": 1.0,
            "3_cellular_geometry_tensegrity_and_fractals_l3_is_not_empirical_validation_evidence": 1.0,
            "iii_meso_scale_geometry_circuits_columns_and_fields_l3_l4_is_not_empirical_validation_evidence": 1.0,
            "1_the_geometry_of_the_bioelectric_field_l3_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4824, 4832)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_2_molecular_geometry_and_the_psi_field_interface_l2_l3_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section2MolecularGeometryAndThePsiFieldInterfaceL2L3Config",
    "Section2MolecularGeometryAndThePsiFieldInterfaceL2L3FixtureResult",
    "classify_section_2_molecular_geometry_and_the_psi_field_interface_l2_l3_component",
    "section_2_molecular_geometry_and_the_psi_field_interface_l2_l3_labels",
    "validate_section_2_molecular_geometry_and_the_psi_field_interface_l2_l3_fixture",
]
