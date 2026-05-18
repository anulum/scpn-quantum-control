# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 IV. The Integrated Scenario: Pharmacological Modulation of the Traumatised Brain validation
"""Source-accounting checks for Paper 0 IV. The Integrated Scenario: Pharmacological Modulation of the Traumatised Brain records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded iv the integrated scenario pharmacological modulation of the traumatised source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05091", "P0R05101")


@dataclass(frozen=True, slots=True)
class IvTheIntegratedScenarioPharmacologicalModulationOfTheTraumatisedConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 11
    expected_component_count: int = 4
    next_source_boundary: str = "P0R05102"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 11:
            raise ValueError("expected_source_record_count must equal 11")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R05102":
            raise ValueError("next_source_boundary must equal P0R05102")


@dataclass(frozen=True, slots=True)
class IvTheIntegratedScenarioPharmacologicalModulationOfTheTraumatisedFixtureResult:
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


def classify_iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised": "iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised_source_boundary",
        "1_the_therapeutic_effects_stabilisation_and_f_minimisation": "1_the_therapeutic_effects_stabilisation_and_f_minimisation_source_boundary",
        "2_the_synergistic_risks_the_dangers_of_dyscritia": "2_the_synergistic_risks_the_dangers_of_dyscritia_source_boundary",
        "v_intervention_adjunct_agents": "v_intervention_adjunct_agents_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised component"
        ) from exc


def iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised_labels() -> dict[
    str, str
]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "IV. The Integrated Scenario: Pharmacological Modulation of the Traumatised Brain",
        "source_span": "P0R05091-P0R05101",
        "component_count": "4",
        "next_boundary": "P0R05102",
        "component_1": "IV. The Integrated Scenario: Pharmacological Modulation of the Traumatised Brain",
        "component_2": "1. The Therapeutic Effects (Stabilisation and F Minimisation)",
        "component_3": "2. The Synergistic Risks (The Dangers of Dyscritia)",
        "component_4": "V. Intervention: Adjunct Agents",
    }


def validate_iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised_fixture(
    config: IvTheIntegratedScenarioPharmacologicalModulationOfTheTraumatisedConfig | None = None,
) -> IvTheIntegratedScenarioPharmacologicalModulationOfTheTraumatisedFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or IvTheIntegratedScenarioPharmacologicalModulationOfTheTraumatisedConfig()
    components = (
        "iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised",
        "1_the_therapeutic_effects_stabilisation_and_f_minimisation",
        "2_the_synergistic_risks_the_dangers_of_dyscritia",
        "v_intervention_adjunct_agents",
    )
    return IvTheIntegratedScenarioPharmacologicalModulationOfTheTraumatisedFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised_component(
                component
            )
            for component in components
        },
        labels=iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised_is_not_empirical_validation_evidence": 1.0,
            "1_the_therapeutic_effects_stabilisation_and_f_minimisation_is_not_empirical_validation_evidence": 1.0,
            "2_the_synergistic_risks_the_dangers_of_dyscritia_is_not_empirical_validation_evidence": 1.0,
            "v_intervention_adjunct_agents_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5091, 5102)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "IvTheIntegratedScenarioPharmacologicalModulationOfTheTraumatisedConfig",
    "IvTheIntegratedScenarioPharmacologicalModulationOfTheTraumatisedFixtureResult",
    "classify_iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised_component",
    "iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised_labels",
    "validate_iv_the_integrated_scenario_pharmacological_modulation_of_the_traumatised_fixture",
]
