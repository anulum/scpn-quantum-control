# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Clinical Scenario Analysis: Traumatic Brain Injury (TBI) and Pharmacological Intervention validation
"""Source-accounting checks for Paper 0 Clinical Scenario Analysis: Traumatic Brain Injury (TBI) and Pharmacological Intervention records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded clinical scenario analysis traumatic brain injury tbi and pharmacologica source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05050", "P0R05057")


@dataclass(frozen=True, slots=True)
class ClinicalScenarioAnalysisTraumaticBrainInjuryTbiAndPharmacologicaConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 3
    next_source_boundary: str = "P0R05058"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R05058":
            raise ValueError("next_source_boundary must equal P0R05058")


@dataclass(frozen=True, slots=True)
class ClinicalScenarioAnalysisTraumaticBrainInjuryTbiAndPharmacologicaFixtureResult:
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


def classify_clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica": "clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_source_boundary",
        "i_the_traumatised_brain_catastrophic_disruption_of_the_scpn_architecture": "i_the_traumatised_brain_catastrophic_disruption_of_the_scpn_architecture_source_boundary",
        "1_l1_disruption_the_decoherence_cascade": "1_l1_disruption_the_decoherence_cascade_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica component"
        ) from exc


def clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_labels() -> dict[
    str, str
]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Clinical Scenario Analysis: Traumatic Brain Injury (TBI) and Pharmacological Intervention",
        "source_span": "P0R05050-P0R05057",
        "component_count": "3",
        "next_boundary": "P0R05058",
        "component_1": "Clinical Scenario Analysis: Traumatic Brain Injury (TBI) and Pharmacological Intervention",
        "component_2": "I. The Traumatised Brain: Catastrophic Disruption of the SCPN Architecture",
        "component_3": "1. L1 Disruption (The Decoherence Cascade):",
    }


def validate_clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_fixture(
    config: ClinicalScenarioAnalysisTraumaticBrainInjuryTbiAndPharmacologicaConfig | None = None,
) -> ClinicalScenarioAnalysisTraumaticBrainInjuryTbiAndPharmacologicaFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or ClinicalScenarioAnalysisTraumaticBrainInjuryTbiAndPharmacologicaConfig()
    components = (
        "clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica",
        "i_the_traumatised_brain_catastrophic_disruption_of_the_scpn_architecture",
        "1_l1_disruption_the_decoherence_cascade",
    )
    return ClinicalScenarioAnalysisTraumaticBrainInjuryTbiAndPharmacologicaFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_component(
                component
            )
            for component in components
        },
        labels=clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_is_not_empirical_validation_evidence": 1.0,
            "i_the_traumatised_brain_catastrophic_disruption_of_the_scpn_architecture_is_not_empirical_validation_evidence": 1.0,
            "1_l1_disruption_the_decoherence_cascade_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5050, 5058)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "ClinicalScenarioAnalysisTraumaticBrainInjuryTbiAndPharmacologicaConfig",
    "ClinicalScenarioAnalysisTraumaticBrainInjuryTbiAndPharmacologicaFixtureResult",
    "classify_clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_component",
    "clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_labels",
    "validate_clinical_scenario_analysis_traumatic_brain_injury_tbi_and_pharmacologica_fixture",
]
