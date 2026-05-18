# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 1. Ion Gradients: The Electrochemical Battery validation
"""Source-accounting checks for Paper 0 1. Ion Gradients: The Electrochemical Battery records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 1 ion gradients the electrochemical battery source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04711", "P0R04719")


@dataclass(frozen=True, slots=True)
class Section1IonGradientsTheElectrochemicalBatteryConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 3
    next_source_boundary: str = "P0R04720"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R04720":
            raise ValueError("next_source_boundary must equal P0R04720")


@dataclass(frozen=True, slots=True)
class Section1IonGradientsTheElectrochemicalBatteryFixtureResult:
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


def classify_section_1_ion_gradients_the_electrochemical_battery_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "1_ion_gradients_the_electrochemical_battery": "1_ion_gradients_the_electrochemical_battery_source_boundary",
        "2_the_aqueous_substrate_qed_water_and_coherence_domains_l1": "2_the_aqueous_substrate_qed_water_and_coherence_domains_l1_source_boundary",
        "iii_the_neuronal_membrane_the_critical_interface_l2_l3": "iii_the_neuronal_membrane_the_critical_interface_l2_l3_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_1_ion_gradients_the_electrochemical_battery component"
        ) from exc


def section_1_ion_gradients_the_electrochemical_battery_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "1. Ion Gradients: The Electrochemical Battery",
        "source_span": "P0R04711-P0R04719",
        "component_count": "3",
        "next_boundary": "P0R04720",
        "component_1": "1. Ion Gradients: The Electrochemical Battery",
        "component_2": "2. The Aqueous Substrate: QED Water and Coherence Domains (L1)",
        "component_3": "III. The Neuronal Membrane: The Critical Interface (L2/L3)",
    }


def validate_section_1_ion_gradients_the_electrochemical_battery_fixture(
    config: Section1IonGradientsTheElectrochemicalBatteryConfig | None = None,
) -> Section1IonGradientsTheElectrochemicalBatteryFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section1IonGradientsTheElectrochemicalBatteryConfig()
    components = (
        "1_ion_gradients_the_electrochemical_battery",
        "2_the_aqueous_substrate_qed_water_and_coherence_domains_l1",
        "iii_the_neuronal_membrane_the_critical_interface_l2_l3",
    )
    return Section1IonGradientsTheElectrochemicalBatteryFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_1_ion_gradients_the_electrochemical_battery_component(
                component
            )
            for component in components
        },
        labels=section_1_ion_gradients_the_electrochemical_battery_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "1_ion_gradients_the_electrochemical_battery_is_not_empirical_validation_evidence": 1.0,
            "2_the_aqueous_substrate_qed_water_and_coherence_domains_l1_is_not_empirical_validation_evidence": 1.0,
            "iii_the_neuronal_membrane_the_critical_interface_l2_l3_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4711, 4720)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_1_ion_gradients_the_electrochemical_battery_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section1IonGradientsTheElectrochemicalBatteryConfig",
    "Section1IonGradientsTheElectrochemicalBatteryFixtureResult",
    "classify_section_1_ion_gradients_the_electrochemical_battery_component",
    "section_1_ion_gradients_the_electrochemical_battery_labels",
    "validate_section_1_ion_gradients_the_electrochemical_battery_fixture",
]
