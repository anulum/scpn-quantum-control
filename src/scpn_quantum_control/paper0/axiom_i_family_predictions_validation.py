# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom I family predictions validation
"""Source-accounting checks for Paper 0 Axiom I family-prediction records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded Axiom I family-predictions map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00747", "P0R00756")


@dataclass(frozen=True, slots=True)
class AxiomIFamilyPredictionsConfig:
    """Configuration for the Axiom I family-predictions fixture."""

    expected_conditional_prediction_count: int = 3
    expected_rejected_model_class_count: int = 5
    next_source_boundary: str = "P0R00757"

    def __post_init__(self) -> None:
        if self.expected_conditional_prediction_count != 3:
            raise ValueError("expected_conditional_prediction_count must equal 3")
        if self.expected_rejected_model_class_count != 5:
            raise ValueError("expected_rejected_model_class_count must equal 5")
        if self.next_source_boundary != "P0R00757":
            raise ValueError("next_source_boundary must equal P0R00757")


@dataclass(frozen=True, slots=True)
class AxiomIFamilyPredictionsFixtureResult:
    """Result for the Paper 0 Axiom I family-predictions fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    rejected_families: dict[str, str]
    conditional_predictions: dict[str, str]
    labels: dict[str, str]
    conditional_prediction_count: int
    rejected_model_class_count: int
    blank_separator_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_rejected_family(family: str) -> str:
    """Classify source-defined rejected model families."""
    mapping = {
        "real_scalar": "lacks_phase_charge_and_solitons",
        "global_u1": "long_range_goldstone_no_locality",
        "vector_tensor": "unnecessary_lorentz_structure",
        "spinor": "spin_representation_not_universal_fibre",
        "non_abelian_minimal": "deferred_su_n_field_content",
    }
    try:
        return mapping[family]
    except KeyError as exc:
        raise ValueError("unknown rejected model family") from exc


def classify_conditional_prediction(prediction: str) -> str:
    """Classify source-defined conditional predictions."""
    mapping = {
        "psi_charge": "conserved_noether_current_q_psi",
        "massive_infoton": "ssb_mass_m_a_equals_g_v",
        "psi_higgs": "massive_radial_spin0_excitation",
    }
    try:
        return mapping[prediction]
    except KeyError as exc:
        raise ValueError("unknown conditional prediction") from exc


def axiom_i_family_predictions_labels() -> dict[str, str]:
    """Return source-bounded labels for the Axiom I family-predictions slice."""
    return {
        "section": "Why this family satisfies (i)-(iii)",
        "decision_rule": "model-class escalation or replacement after contrary evidence",
        "next_boundary": "Extension to SU(N) Qualia Confinement",
    }


def validate_axiom_i_family_predictions_fixture(
    config: AxiomIFamilyPredictionsConfig | None = None,
) -> AxiomIFamilyPredictionsFixtureResult:
    """Validate source accounting for the Axiom I family-predictions slice."""
    cfg = config or AxiomIFamilyPredictionsConfig()
    rejected_families = (
        "real_scalar",
        "global_u1",
        "vector_tensor",
        "spinor",
        "non_abelian_minimal",
    )
    conditional_predictions = ("psi_charge", "massive_infoton", "psi_higgs")

    return AxiomIFamilyPredictionsFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        rejected_families={
            family: classify_rejected_family(family) for family in rejected_families
        },
        conditional_predictions={
            prediction: classify_conditional_prediction(prediction)
            for prediction in conditional_predictions
        },
        labels=axiom_i_family_predictions_labels(),
        conditional_prediction_count=cfg.expected_conditional_prediction_count,
        rejected_model_class_count=cfg.expected_rejected_model_class_count,
        blank_separator_count=1,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "conditional_predictions_are_not_observed_results": 1.0,
            "rejected_model_classes_remain_source_boundary_claims": 1.0,
            "su_n_extension_header_is_not_promoted_model_selection": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(747, 757)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_family_predictions_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "AxiomIFamilyPredictionsConfig",
    "AxiomIFamilyPredictionsFixtureResult",
    "axiom_i_family_predictions_labels",
    "classify_conditional_prediction",
    "classify_rejected_family",
    "validate_axiom_i_family_predictions_fixture",
]
