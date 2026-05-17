# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 derived interaction opening validation
"""Source-accounting checks for Paper 0 derived interaction opening records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded derived interaction opening; not experimental validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R01384", "P0R01421")


@dataclass(frozen=True, slots=True)
class DerivedInteractionOpeningConfig:
    """Configuration for the derived interaction opening fixture."""

    expected_source_record_count: int = 38
    expected_component_count: int = 5
    next_source_boundary: str = "P0R01422"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 38:
            raise ValueError("expected_source_record_count must equal 38")
        if self.expected_component_count != 5:
            raise ValueError("expected_component_count must equal 5")
        if self.next_source_boundary != "P0R01422":
            raise ValueError("next_source_boundary must equal P0R01422")


@dataclass(frozen=True, slots=True)
class DerivedInteractionOpeningFixtureResult:
    """Result for the Paper 0 derived interaction opening fixture."""

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


def classify_derived_interaction_opening_component(component: str) -> str:
    """Classify source-defined derived interaction opening components."""
    mapping = {
        "gauge_theory_grounding": "complex_scalar_u1_infoton_grounding_boundary",
        "predictive_coding_mapping": "psi_charge_belief_infoton_prediction_error_mapping_boundary",
        "h_int_gauge_identification": "h_int_to_u1_gauge_interaction_identification_boundary",
        "intrinsic_properties_quantum_numbers": "spin_zero_psi_charge_infoton_fim_quantum_number_boundary",
        "gauge_principle_nonabelian_boundary": "local_u1_su_n_hypothesis_fim_dynamics_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown derived interaction opening component") from exc


def derived_interaction_opening_labels() -> dict[str, str]:
    """Return source-bounded labels for the derived interaction opening slice."""
    return {
        "section": "Deriving the Master Interaction Lagrangian",
        "field": "Psi is a complex scalar field with spin-0 quanta and Psi-charge",
        "interaction": "L_interaction = i g A_mu J_mu",
        "current": "J_mu = Psi* partial_mu Psi - Psi partial_mu Psi*",
        "next_boundary": "The Master Interaction Lagrangian (Derived from First Principles)",
    }


def validate_derived_interaction_opening_fixture(
    config: DerivedInteractionOpeningConfig | None = None,
) -> DerivedInteractionOpeningFixtureResult:
    """Validate source accounting for the derived interaction opening slice."""
    cfg = config or DerivedInteractionOpeningConfig()
    components = (
        "gauge_theory_grounding",
        "predictive_coding_mapping",
        "h_int_gauge_identification",
        "intrinsic_properties_quantum_numbers",
        "gauge_principle_nonabelian_boundary",
    )

    return DerivedInteractionOpeningFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_derived_interaction_opening_component(component)
            for component in components
        },
        labels=derived_interaction_opening_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "phenomenological_black_box_h_int_rejected_for_derived_boundary": 1.0,
            "predictive_coding_mapping_is_not_observed_infoton_signal": 1.0,
            "su_n_qualia_confinement_remains_hypothesis_not_established_gauge_group": 1.0,
            "diagram_caption_is_not_particle_detection_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(1384, 1422)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_derived_interaction_opening_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "DerivedInteractionOpeningConfig",
    "DerivedInteractionOpeningFixtureResult",
    "classify_derived_interaction_opening_component",
    "derived_interaction_opening_labels",
    "validate_derived_interaction_opening_fixture",
]
