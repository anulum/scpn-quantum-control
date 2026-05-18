# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 1. The Cytoskeleton-Water Interface and QEC: validation
"""Source-accounting checks for Paper 0 1. The Cytoskeleton-Water Interface and QEC: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 1 the cytoskeleton water interface and qec source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04794", "P0R04801")


@dataclass(frozen=True, slots=True)
class Section1TheCytoskeletonWaterInterfaceAndQecConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 3
    next_source_boundary: str = "P0R04802"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R04802":
            raise ValueError("next_source_boundary must equal P0R04802")


@dataclass(frozen=True, slots=True)
class Section1TheCytoskeletonWaterInterfaceAndQecFixtureResult:
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


def classify_section_1_the_cytoskeleton_water_interface_and_qec_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "1_the_cytoskeleton_water_interface_and_qec": "1_the_cytoskeleton_water_interface_and_qec_source_boundary",
        "2_endogenous_em_fields_and_biophotons": "2_endogenous_em_fields_and_biophotons_source_boundary",
        "3_spin_chemistry_and_the_radical_pair_mechanism_rpm": "3_spin_chemistry_and_the_radical_pair_mechanism_rpm_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_1_the_cytoskeleton_water_interface_and_qec component"
        ) from exc


def section_1_the_cytoskeleton_water_interface_and_qec_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "1. The Cytoskeleton-Water Interface and QEC:",
        "source_span": "P0R04794-P0R04801",
        "component_count": "3",
        "next_boundary": "P0R04802",
        "component_1": "1. The Cytoskeleton-Water Interface and QEC:",
        "component_2": "2. Endogenous EM Fields and Biophotons:",
        "component_3": "3. Spin Chemistry and the Radical Pair Mechanism (RPM):",
    }


def validate_section_1_the_cytoskeleton_water_interface_and_qec_fixture(
    config: Section1TheCytoskeletonWaterInterfaceAndQecConfig | None = None,
) -> Section1TheCytoskeletonWaterInterfaceAndQecFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section1TheCytoskeletonWaterInterfaceAndQecConfig()
    components = (
        "1_the_cytoskeleton_water_interface_and_qec",
        "2_endogenous_em_fields_and_biophotons",
        "3_spin_chemistry_and_the_radical_pair_mechanism_rpm",
    )
    return Section1TheCytoskeletonWaterInterfaceAndQecFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_1_the_cytoskeleton_water_interface_and_qec_component(
                component
            )
            for component in components
        },
        labels=section_1_the_cytoskeleton_water_interface_and_qec_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "1_the_cytoskeleton_water_interface_and_qec_is_not_empirical_validation_evidence": 1.0,
            "2_endogenous_em_fields_and_biophotons_is_not_empirical_validation_evidence": 1.0,
            "3_spin_chemistry_and_the_radical_pair_mechanism_rpm_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4794, 4802)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_1_the_cytoskeleton_water_interface_and_qec_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section1TheCytoskeletonWaterInterfaceAndQecConfig",
    "Section1TheCytoskeletonWaterInterfaceAndQecFixtureResult",
    "classify_section_1_the_cytoskeleton_water_interface_and_qec_component",
    "section_1_the_cytoskeleton_water_interface_and_qec_labels",
    "validate_section_1_the_cytoskeleton_water_interface_and_qec_fixture",
]
