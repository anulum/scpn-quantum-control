# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 final LInt SM interface validation
"""Source-accounting checks for Paper 0 final LInt and SM-interface records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = (
    "source-bounded final LInt and Standard Model interface; not experimental validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R01510", "P0R01581")


@dataclass(frozen=True, slots=True)
class FinalLIntSMInterfaceConfig:
    """Configuration for the final LInt and SM-interface fixture."""

    expected_source_record_count: int = 72
    expected_component_count: int = 6
    next_source_boundary: str = "P0R01582"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 72:
            raise ValueError("expected_source_record_count must equal 72")
        if self.expected_component_count != 6:
            raise ValueError("expected_component_count must equal 6")
        if self.next_source_boundary != "P0R01582":
            raise ValueError("next_source_boundary must equal P0R01582")


@dataclass(frozen=True, slots=True)
class FinalLIntSMInterfaceFixtureResult:
    """Result for the Paper 0 final LInt and SM-interface fixture."""

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


def classify_final_lint_sm_interface_component(component: str) -> str:
    """Classify source-defined final LInt and SM-interface components."""
    mapping = {
        "final_lint_dual_clause": "final_lint_geometric_informational_dual_clause_boundary",
        "free_energy_and_h_int_mapping": "free_energy_h_int_mapping_source_boundary",
        "foundational_physics_equations": "compact_lint_foundational_physics_equation_boundary",
        "standard_model_indirect_coupling": "standard_model_indirect_coupling_no_direct_force_boundary",
        "predictive_interface_mapping": "predictive_downward_causation_interface_mapping_boundary",
        "downstream_sm_manifestations": "downstream_sm_manifestation_exploratory_hypothesis_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown final LInt SM-interface component") from exc


def final_lint_sm_interface_labels() -> dict[str, str]:
    """Return source-bounded labels for the final LInt SM-interface slice."""
    return {
        "section": "The Master Interaction Lagrangian (Derived from First Principles)",
        "lint": "L_Int = L_Geometric + L_Informational",
        "geometric": "L_Geometric = -xi R Psi*Psi",
        "heuristic": "H_int = -lambda * Psi_s * sigma",
        "next_boundary": "How Reality Gets Its Structure: A Cascade of Broken Symmetries",
    }


def validate_final_lint_sm_interface_fixture(
    config: FinalLIntSMInterfaceConfig | None = None,
) -> FinalLIntSMInterfaceFixtureResult:
    """Validate source accounting for the final LInt and SM-interface slice."""
    cfg = config or FinalLIntSMInterfaceConfig()
    components = (
        "final_lint_dual_clause",
        "free_energy_and_h_int_mapping",
        "foundational_physics_equations",
        "standard_model_indirect_coupling",
        "predictive_interface_mapping",
        "downstream_sm_manifestations",
    )

    return FinalLIntSMInterfaceFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_final_lint_sm_interface_component(component)
            for component in components
        },
        labels=final_lint_sm_interface_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "direct_standard_model_force_carrier_claim_rejected": 1.0,
            "weak_force_and_alp_extensions_remain_exploratory_hypotheses": 1.0,
            "free_energy_mapping_is_not_measured_cost_function": 1.0,
            "prediction_mapping_is_not_observed_probability_bias": 1.0,
            "diagrammatic_sm_interface_is_not_experimental_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(1510, 1582)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_final_lint_sm_interface_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "FinalLIntSMInterfaceConfig",
    "FinalLIntSMInterfaceFixtureResult",
    "classify_final_lint_sm_interface_component",
    "final_lint_sm_interface_labels",
    "validate_final_lint_sm_interface_fixture",
]
