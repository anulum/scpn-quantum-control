# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Applied SCPN: Pathology, Technology, and Anomalies validation
"""Source-accounting checks for Paper 0 Applied SCPN: Pathology, Technology, and Anomalies records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded applied scpn pathology technology and anomalies source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R06197", "P0R06205")


@dataclass(frozen=True, slots=True)
class AppliedScpnPathologyTechnologyAndAnomaliesConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 6
    next_source_boundary: str = "P0R06206"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 6:
            raise ValueError("expected_component_count must equal 6")
        if self.next_source_boundary != "P0R06206":
            raise ValueError("next_source_boundary must equal P0R06206")


@dataclass(frozen=True, slots=True)
class AppliedScpnPathologyTechnologyAndAnomaliesFixtureResult:
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


def classify_applied_scpn_pathology_technology_and_anomalies_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "applied_scpn_pathology_technology_and_anomalies": "applied_scpn_pathology_technology_and_anomalies_source_boundary",
        "i_pathology_and_therapeutics": "i_pathology_and_therapeutics_source_boundary",
        "aetiology_of_disorder": "aetiology_of_disorder_source_boundary",
        "dissonance_free_energy_accumulation_sustained_accumulation_of_prediction": "dissonance_free_energy_accumulation_sustained_accumulation_of_prediction_source_boundary",
        "deviation_from_criticality_shifts_away_from_sigma_1_supercriticality_sig": "deviation_from_criticality_shifts_away_from_sigma_1_supercriticality_sig_source_boundary",
        "fragmentation_failure_of_integration_e_g_l5_self_fragmentation_in_dissoc": "fragmentation_failure_of_integration_e_g_l5_self_fragmentation_in_dissoc_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown applied_scpn_pathology_technology_and_anomalies component"
        ) from exc


def applied_scpn_pathology_technology_and_anomalies_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Applied SCPN: Pathology, Technology, and Anomalies",
        "source_span": "P0R06197-P0R06205",
        "component_count": "6",
        "next_boundary": "P0R06206",
        "component_1": "Applied SCPN: Pathology, Technology, and Anomalies",
        "component_2": "I. Pathology and Therapeutics",
        "component_3": "Aetiology of Disorder:",
        "component_4": "Dissonance (Free Energy Accumulation): Sustained accumulation of Prediction Errors (EL) in the HPC architecture. PathologyIndexFGlobal.",
        "component_5": "Deviation from Criticality: Shifts away from sigma=1. Supercriticality (sigma > 1, e.g., mania, seizures); Subcriticality (sigma < 1, e.g., depression, coma).",
        "component_6": "Fragmentation: Failure of integration (e.g., L5 Self fragmentation in dissociation; L11 societal polarisation).",
    }


def validate_applied_scpn_pathology_technology_and_anomalies_fixture(
    config: AppliedScpnPathologyTechnologyAndAnomaliesConfig | None = None,
) -> AppliedScpnPathologyTechnologyAndAnomaliesFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or AppliedScpnPathologyTechnologyAndAnomaliesConfig()
    components = (
        "applied_scpn_pathology_technology_and_anomalies",
        "i_pathology_and_therapeutics",
        "aetiology_of_disorder",
        "dissonance_free_energy_accumulation_sustained_accumulation_of_prediction",
        "deviation_from_criticality_shifts_away_from_sigma_1_supercriticality_sig",
        "fragmentation_failure_of_integration_e_g_l5_self_fragmentation_in_dissoc",
    )
    return AppliedScpnPathologyTechnologyAndAnomaliesFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_applied_scpn_pathology_technology_and_anomalies_component(
                component
            )
            for component in components
        },
        labels=applied_scpn_pathology_technology_and_anomalies_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "applied_scpn_pathology_technology_and_anomalies_is_not_empirical_validation_evidence": 1.0,
            "i_pathology_and_therapeutics_is_not_empirical_validation_evidence": 1.0,
            "aetiology_of_disorder_is_not_empirical_validation_evidence": 1.0,
            "dissonance_free_energy_accumulation_sustained_accumulation_of_prediction_is_not_empirical_validation_evidence": 1.0,
            "deviation_from_criticality_shifts_away_from_sigma_1_supercriticality_sig_is_not_empirical_validation_evidence": 1.0,
            "fragmentation_failure_of_integration_e_g_l5_self_fragmentation_in_dissoc_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(6197, 6206)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_applied_scpn_pathology_technology_and_anomalies_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "AppliedScpnPathologyTechnologyAndAnomaliesConfig",
    "AppliedScpnPathologyTechnologyAndAnomaliesFixtureResult",
    "classify_applied_scpn_pathology_technology_and_anomalies_component",
    "applied_scpn_pathology_technology_and_anomalies_labels",
    "validate_applied_scpn_pathology_technology_and_anomalies_fixture",
]
