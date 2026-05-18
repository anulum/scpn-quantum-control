# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 3. Neurotransmitters as Tuners of the Psi-Field Interface (L2): validation
"""Source-accounting checks for Paper 0 3. Neurotransmitters as Tuners of the Psi-Field Interface (L2): records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 3 neurotransmitters as tuners of the psi field interface l2 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04649", "P0R04656")


@dataclass(frozen=True, slots=True)
class Section3NeurotransmittersAsTunersOfThePsiFieldInterfaceL2Config:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 3
    next_source_boundary: str = "P0R04657"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R04657":
            raise ValueError("next_source_boundary must equal P0R04657")


@dataclass(frozen=True, slots=True)
class Section3NeurotransmittersAsTunersOfThePsiFieldInterfaceL2FixtureResult:
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


def classify_section_3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2": "3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2_source_boundary",
        "4_the_coherent_milieu_csf_and_the_glymphatic_system": "4_the_coherent_milieu_csf_and_the_glymphatic_system_source_boundary",
        "ii_the_architecture_of_structure_and_plasticity_domain_i_l3": "ii_the_architecture_of_structure_and_plasticity_domain_i_l3_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2 component"
        ) from exc


def section_3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "3. Neurotransmitters as Tuners of the Psi-Field Interface (L2):",
        "source_span": "P0R04649-P0R04656",
        "component_count": "3",
        "next_boundary": "P0R04657",
        "component_1": "3. Neurotransmitters as Tuners of the Psi-Field Interface (L2):",
        "component_2": "4. The Coherent Milieu: CSF and the Glymphatic System:",
        "component_3": "II. The Architecture of Structure and Plasticity (Domain I: L3)",
    }


def validate_section_3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2_fixture(
    config: Section3NeurotransmittersAsTunersOfThePsiFieldInterfaceL2Config | None = None,
) -> Section3NeurotransmittersAsTunersOfThePsiFieldInterfaceL2FixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section3NeurotransmittersAsTunersOfThePsiFieldInterfaceL2Config()
    components = (
        "3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2",
        "4_the_coherent_milieu_csf_and_the_glymphatic_system",
        "ii_the_architecture_of_structure_and_plasticity_domain_i_l3",
    )
    return Section3NeurotransmittersAsTunersOfThePsiFieldInterfaceL2FixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2_component(
                component
            )
            for component in components
        },
        labels=section_3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2_is_not_empirical_validation_evidence": 1.0,
            "4_the_coherent_milieu_csf_and_the_glymphatic_system_is_not_empirical_validation_evidence": 1.0,
            "ii_the_architecture_of_structure_and_plasticity_domain_i_l3_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4649, 4657)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section3NeurotransmittersAsTunersOfThePsiFieldInterfaceL2Config",
    "Section3NeurotransmittersAsTunersOfThePsiFieldInterfaceL2FixtureResult",
    "classify_section_3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2_component",
    "section_3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2_labels",
    "validate_section_3_neurotransmitters_as_tuners_of_the_psi_field_interface_l2_fixture",
]
