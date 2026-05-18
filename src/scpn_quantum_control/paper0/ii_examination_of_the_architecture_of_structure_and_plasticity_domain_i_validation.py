# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 II. Examination of The Architecture of Structure and Plasticity (Domain I: L3) validation
"""Source-accounting checks for Paper 0 II. Examination of The Architecture of Structure and Plasticity (Domain I: L3) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded ii examination of the architecture of structure and plasticity domain i source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04560", "P0R04571")


@dataclass(frozen=True, slots=True)
class IiExaminationOfTheArchitectureOfStructureAndPlasticityDomainIConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 12
    expected_component_count: int = 3
    next_source_boundary: str = "P0R04572"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 12:
            raise ValueError("expected_source_record_count must equal 12")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R04572":
            raise ValueError("next_source_boundary must equal P0R04572")


@dataclass(frozen=True, slots=True)
class IiExaminationOfTheArchitectureOfStructureAndPlasticityDomainIFixtureResult:
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


def classify_ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i": "ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_source_boundary",
        "the_optimised_connectome_the_geometric_scaffold_of_thought": "the_optimised_connectome_the_geometric_scaffold_of_thought_source_boundary",
        "the_active_role_of_glia_the_slow_control_network": "the_active_role_of_glia_the_slow_control_network_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i component"
        ) from exc


def ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_labels() -> dict[
    str, str
]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "II. Examination of The Architecture of Structure and Plasticity (Domain I: L3)",
        "source_span": "P0R04560-P0R04571",
        "component_count": "3",
        "next_boundary": "P0R04572",
        "component_1": "II. Examination of The Architecture of Structure and Plasticity (Domain I: L3)",
        "component_2": "The Optimised Connectome: The Geometric Scaffold of Thought",
        "component_3": "The Active Role of Glia: The Slow Control Network",
    }


def validate_ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_fixture(
    config: IiExaminationOfTheArchitectureOfStructureAndPlasticityDomainIConfig | None = None,
) -> IiExaminationOfTheArchitectureOfStructureAndPlasticityDomainIFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or IiExaminationOfTheArchitectureOfStructureAndPlasticityDomainIConfig()
    components = (
        "ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i",
        "the_optimised_connectome_the_geometric_scaffold_of_thought",
        "the_active_role_of_glia_the_slow_control_network",
    )
    return IiExaminationOfTheArchitectureOfStructureAndPlasticityDomainIFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_component(
                component
            )
            for component in components
        },
        labels=ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_is_not_empirical_validation_evidence": 1.0,
            "the_optimised_connectome_the_geometric_scaffold_of_thought_is_not_empirical_validation_evidence": 1.0,
            "the_active_role_of_glia_the_slow_control_network_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4560, 4572)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "IiExaminationOfTheArchitectureOfStructureAndPlasticityDomainIConfig",
    "IiExaminationOfTheArchitectureOfStructureAndPlasticityDomainIFixtureResult",
    "classify_ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_component",
    "ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_labels",
    "validate_ii_examination_of_the_architecture_of_structure_and_plasticity_domain_i_fixture",
]
