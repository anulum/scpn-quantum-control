# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. Dendritic Spines: The Loci of Plasticity and IET validation
"""Source-accounting checks for Paper 0 2. Dendritic Spines: The Loci of Plasticity and IET records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 2 dendritic spines the loci of plasticity and iet source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04703", "P0R04710")


@dataclass(frozen=True, slots=True)
class Section2DendriticSpinesTheLociOfPlasticityAndIetConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 3
    next_source_boundary: str = "P0R04711"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R04711":
            raise ValueError("next_source_boundary must equal P0R04711")


@dataclass(frozen=True, slots=True)
class Section2DendriticSpinesTheLociOfPlasticityAndIetFixtureResult:
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


def classify_section_2_dendritic_spines_the_loci_of_plasticity_and_iet_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "2_dendritic_spines_the_loci_of_plasticity_and_iet": "2_dendritic_spines_the_loci_of_plasticity_and_iet_source_boundary",
        "3_the_axon_and_the_axon_initial_segment_ais_the_decision_point": "3_the_axon_and_the_axon_initial_segment_ais_the_decision_point_source_boundary",
        "ii_the_chemical_milieu_and_the_primacy_of_water_l1_l2": "ii_the_chemical_milieu_and_the_primacy_of_water_l1_l2_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_2_dendritic_spines_the_loci_of_plasticity_and_iet component"
        ) from exc


def section_2_dendritic_spines_the_loci_of_plasticity_and_iet_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "2. Dendritic Spines: The Loci of Plasticity and IET",
        "source_span": "P0R04703-P0R04710",
        "component_count": "3",
        "next_boundary": "P0R04711",
        "component_1": "2. Dendritic Spines: The Loci of Plasticity and IET",
        "component_2": "3. The Axon and the Axon Initial Segment (AIS): The Decision Point",
        "component_3": "II. The Chemical Milieu and the Primacy of Water (L1/L2)",
    }


def validate_section_2_dendritic_spines_the_loci_of_plasticity_and_iet_fixture(
    config: Section2DendriticSpinesTheLociOfPlasticityAndIetConfig | None = None,
) -> Section2DendriticSpinesTheLociOfPlasticityAndIetFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section2DendriticSpinesTheLociOfPlasticityAndIetConfig()
    components = (
        "2_dendritic_spines_the_loci_of_plasticity_and_iet",
        "3_the_axon_and_the_axon_initial_segment_ais_the_decision_point",
        "ii_the_chemical_milieu_and_the_primacy_of_water_l1_l2",
    )
    return Section2DendriticSpinesTheLociOfPlasticityAndIetFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_2_dendritic_spines_the_loci_of_plasticity_and_iet_component(
                component
            )
            for component in components
        },
        labels=section_2_dendritic_spines_the_loci_of_plasticity_and_iet_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "2_dendritic_spines_the_loci_of_plasticity_and_iet_is_not_empirical_validation_evidence": 1.0,
            "3_the_axon_and_the_axon_initial_segment_ais_the_decision_point_is_not_empirical_validation_evidence": 1.0,
            "ii_the_chemical_milieu_and_the_primacy_of_water_l1_l2_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4703, 4711)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_2_dendritic_spines_the_loci_of_plasticity_and_iet_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section2DendriticSpinesTheLociOfPlasticityAndIetConfig",
    "Section2DendriticSpinesTheLociOfPlasticityAndIetFixtureResult",
    "classify_section_2_dendritic_spines_the_loci_of_plasticity_and_iet_component",
    "section_2_dendritic_spines_the_loci_of_plasticity_and_iet_labels",
    "validate_section_2_dendritic_spines_the_loci_of_plasticity_and_iet_fixture",
]
