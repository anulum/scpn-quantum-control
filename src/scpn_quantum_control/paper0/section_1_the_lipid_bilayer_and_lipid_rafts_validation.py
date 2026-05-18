# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 1. The Lipid Bilayer and Lipid Rafts: validation
"""Source-accounting checks for Paper 0 1. The Lipid Bilayer and Lipid Rafts: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 1 the lipid bilayer and lipid rafts source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04720", "P0R04727")


@dataclass(frozen=True, slots=True)
class Section1TheLipidBilayerAndLipidRaftsConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 3
    next_source_boundary: str = "P0R04728"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R04728":
            raise ValueError("next_source_boundary must equal P0R04728")


@dataclass(frozen=True, slots=True)
class Section1TheLipidBilayerAndLipidRaftsFixtureResult:
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


def classify_section_1_the_lipid_bilayer_and_lipid_rafts_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "1_the_lipid_bilayer_and_lipid_rafts": "1_the_lipid_bilayer_and_lipid_rafts_source_boundary",
        "2_ion_channels_the_molecular_transistors_l1_l2_interface": "2_ion_channels_the_molecular_transistors_l1_l2_interface_source_boundary",
        "iv_the_internal_architecture_cytoskeleton_and_organelles_l1_l3": "iv_the_internal_architecture_cytoskeleton_and_organelles_l1_l3_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown section_1_the_lipid_bilayer_and_lipid_rafts component") from exc


def section_1_the_lipid_bilayer_and_lipid_rafts_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "1. The Lipid Bilayer and Lipid Rafts:",
        "source_span": "P0R04720-P0R04727",
        "component_count": "3",
        "next_boundary": "P0R04728",
        "component_1": "1. The Lipid Bilayer and Lipid Rafts:",
        "component_2": "2. Ion Channels: The Molecular Transistors (L1/L2 Interface)",
        "component_3": "IV. The Internal Architecture: Cytoskeleton and Organelles (L1/L3)",
    }


def validate_section_1_the_lipid_bilayer_and_lipid_rafts_fixture(
    config: Section1TheLipidBilayerAndLipidRaftsConfig | None = None,
) -> Section1TheLipidBilayerAndLipidRaftsFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section1TheLipidBilayerAndLipidRaftsConfig()
    components = (
        "1_the_lipid_bilayer_and_lipid_rafts",
        "2_ion_channels_the_molecular_transistors_l1_l2_interface",
        "iv_the_internal_architecture_cytoskeleton_and_organelles_l1_l3",
    )
    return Section1TheLipidBilayerAndLipidRaftsFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_1_the_lipid_bilayer_and_lipid_rafts_component(component)
            for component in components
        },
        labels=section_1_the_lipid_bilayer_and_lipid_rafts_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "1_the_lipid_bilayer_and_lipid_rafts_is_not_empirical_validation_evidence": 1.0,
            "2_ion_channels_the_molecular_transistors_l1_l2_interface_is_not_empirical_validation_evidence": 1.0,
            "iv_the_internal_architecture_cytoskeleton_and_organelles_l1_l3_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4720, 4728)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_1_the_lipid_bilayer_and_lipid_rafts_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section1TheLipidBilayerAndLipidRaftsConfig",
    "Section1TheLipidBilayerAndLipidRaftsFixtureResult",
    "classify_section_1_the_lipid_bilayer_and_lipid_rafts_component",
    "section_1_the_lipid_bilayer_and_lipid_rafts_labels",
    "validate_section_1_the_lipid_bilayer_and_lipid_rafts_fixture",
]
