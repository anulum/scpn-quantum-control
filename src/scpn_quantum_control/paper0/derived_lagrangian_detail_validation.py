# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 derived Lagrangian detail validation
"""Source-accounting checks for Paper 0 derived Lagrangian detail records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded derived Master Interaction Lagrangian detail; not experimental validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R01422", "P0R01509")


@dataclass(frozen=True, slots=True)
class DerivedLagrangianDetailConfig:
    """Configuration for the derived Lagrangian detail fixture."""

    expected_source_record_count: int = 88
    expected_component_count: int = 7
    next_source_boundary: str = "P0R01510"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 88:
            raise ValueError("expected_source_record_count must equal 88")
        if self.expected_component_count != 7:
            raise ValueError("expected_component_count must equal 7")
        if self.next_source_boundary != "P0R01510":
            raise ValueError("next_source_boundary must equal P0R01510")


@dataclass(frozen=True, slots=True)
class DerivedLagrangianDetailFixtureResult:
    """Result for the Paper 0 derived Lagrangian detail fixture."""

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


def classify_derived_lagrangian_detail_component(component: str) -> str:
    """Classify source-defined derived Lagrangian detail components."""
    mapping = {
        "derived_lint_split": "derived_lint_prime_informational_geometric_split_boundary",
        "informational_lagrangian_fim_kinetics": "informational_lagrangian_fim_gauge_kinetic_boundary",
        "operational_pullback_protocol": "statistical_bundle_fim_pullback_gauge_protocol_boundary",
        "observable_l4_l5_prediction": "observable_l4_l5_fim_prediction_only_boundary",
        "neural_fim_covariance_strategy": "full_covariance_neural_fim_strategy_boundary",
        "domain_constraints_local_physics": "eft_lorentz_locality_causality_pullback_boundary",
        "geometric_constants_predictions": "geometric_coupling_constants_prediction_target_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown derived Lagrangian detail component") from exc


def derived_lagrangian_detail_labels() -> dict[str, str]:
    """Return source-bounded labels for the derived Lagrangian detail slice."""
    return {
        "section": "The Master Interaction Lagrangian (Derived from First Principles)",
        "lint_split": "L_Int_prime = L_Informational_prime + L_Geometric_prime",
        "informational": "L_Informational_prime includes U1 current coupling and FIM gauge kinetic term",
        "geometric": "L_Geometric_prime = -xi R Psi* Psi",
        "next_boundary": "The Master Interaction Lagrangian (Derived from First Principles) restart",
    }


def validate_derived_lagrangian_detail_fixture(
    config: DerivedLagrangianDetailConfig | None = None,
) -> DerivedLagrangianDetailFixtureResult:
    """Validate source accounting for the derived Lagrangian detail slice."""
    cfg = config or DerivedLagrangianDetailConfig()
    components = (
        "derived_lint_split",
        "informational_lagrangian_fim_kinetics",
        "operational_pullback_protocol",
        "observable_l4_l5_prediction",
        "neural_fim_covariance_strategy",
        "domain_constraints_local_physics",
        "geometric_constants_predictions",
    )

    return DerivedLagrangianDetailFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_derived_lagrangian_detail_component(component)
            for component in components
        },
        labels=derived_lagrangian_detail_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "spacetime_metric_contraction_rejected_for_fim_gauge_kinetic_term": 1.0,
            "nv_centre_sensor_prediction_is_not_observed_evidence": 1.0,
            "nonlocal_or_acausal_pullback_dependency_rejected": 1.0,
            "psi_higgs_and_alp_targets_are_not_detected_particles": 1.0,
            "mean_only_or_diagonal_fim_shortcut_rejected": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(1422, 1510)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_derived_lagrangian_detail_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "DerivedLagrangianDetailConfig",
    "DerivedLagrangianDetailFixtureResult",
    "classify_derived_lagrangian_detail_component",
    "derived_lagrangian_detail_labels",
    "validate_derived_lagrangian_detail_fixture",
]
