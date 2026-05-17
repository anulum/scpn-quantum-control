# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 SSB Psi-field validation
"""Source-accounting checks for Paper 0 SSB Psi-field records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded SSB Psi-field mechanism claims; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R01272", "P0R01332")


@dataclass(frozen=True, slots=True)
class SSBPsiFieldConfig:
    """Configuration for the SSB Psi-field fixture."""

    expected_source_record_count: int = 61
    expected_component_count: int = 8
    next_source_boundary: str = "P0R01333"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 61:
            raise ValueError("expected_source_record_count must equal 61")
        if self.expected_component_count != 8:
            raise ValueError("expected_component_count must equal 8")
        if self.next_source_boundary != "P0R01333":
            raise ValueError("next_source_boundary must equal P0R01333")


@dataclass(frozen=True, slots=True)
class SSBPsiFieldFixtureResult:
    """Result for the Paper 0 SSB Psi-field fixture."""

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


def classify_ssb_psi_field_component(component: str) -> str:
    """Classify source-defined SSB Psi-field components."""
    mapping = {
        "section_overview_and_three_implications": "ssb_psi_field_section_overview_boundary",
        "popular_context_short_range_particle_self": "popular_context_not_validation_boundary",
        "predictive_coding_core_belief": "predictive_coding_mapping_source_boundary",
        "psi_s_coupling_integration": "h_int_psi_s_sigma_coupling_integration_boundary",
        "mexican_hat_vacuum_selection": "mexican_hat_potential_and_vacuum_selection_boundary",
        "eft_sextic_stability_and_mass": "eft_sextic_stability_and_radial_mass_boundary",
        "global_goldstone_boundary": "global_u1_goldstone_counterfactual_boundary",
        "local_higgs_architecture_implications": "local_u1_higgs_infoton_architecture_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown SSB Psi-field component") from exc


def ssb_psi_field_labels() -> dict[str, str]:
    """Return source-bounded labels for the SSB Psi-field slice."""
    return {
        "section": "The Physics of Form: Spontaneous Symmetry Breaking and the Psi-Field",
        "potential": "V(|Psi|) = -mu^2 |Psi|^2 + lambda |Psi|^4",
        "eft_potential": "V(|Psi|) = -mu^2 |Psi|^2 + lambda |Psi|^4 + (gamma/Lambda^2)|Psi|^6",
        "higgs_mass": "m_A = sqrt(2) g v",
        "next_boundary": "The Phenomenological Formulation: An Evolutionary Starting Point",
    }


def validate_ssb_psi_field_fixture(
    config: SSBPsiFieldConfig | None = None,
) -> SSBPsiFieldFixtureResult:
    """Validate source accounting for the SSB Psi-field slice."""
    cfg = config or SSBPsiFieldConfig()
    components = (
        "section_overview_and_three_implications",
        "popular_context_short_range_particle_self",
        "predictive_coding_core_belief",
        "psi_s_coupling_integration",
        "mexican_hat_vacuum_selection",
        "eft_sextic_stability_and_mass",
        "global_goldstone_boundary",
        "local_higgs_architecture_implications",
    )

    return SSBPsiFieldFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_ssb_psi_field_component(component) for component in components
        },
        labels=ssb_psi_field_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "psi_higgs_prediction_is_not_particle_detection": 1.0,
            "global_goldstone_case_must_not_be_mixed_with_local_higgs_case": 1.0,
            "quartic_only_potential_rejected_for_eft_stability_boundary": 1.0,
            "popular_context_is_not_empirical_validation": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(1272, 1333)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_ssb_psi_field_mechanism_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "SSBPsiFieldConfig",
    "SSBPsiFieldFixtureResult",
    "classify_ssb_psi_field_component",
    "ssb_psi_field_labels",
    "validate_ssb_psi_field_fixture",
]
