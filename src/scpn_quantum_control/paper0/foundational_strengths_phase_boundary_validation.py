# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 foundational strengths and phase boundary validation
"""Source-accounting checks for Paper 0 foundational-strengths and phase-boundary records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = (
    "source-bounded foundational-strength and phase-regime claims; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R01189", "P0R01241")


@dataclass(frozen=True, slots=True)
class FoundationalStrengthsPhaseBoundaryConfig:
    """Configuration for the foundational-strengths and phase-boundary fixture."""

    expected_source_record_count: int = 53
    expected_component_count: int = 6
    next_source_boundary: str = "P0R01242"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 53:
            raise ValueError("expected_source_record_count must equal 53")
        if self.expected_component_count != 6:
            raise ValueError("expected_component_count must equal 6")
        if self.next_source_boundary != "P0R01242":
            raise ValueError("next_source_boundary must equal P0R01242")


@dataclass(frozen=True, slots=True)
class FoundationalStrengthsPhaseBoundaryFixtureResult:
    """Result for the Paper 0 foundational-strengths and phase-boundary fixture."""

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


def classify_foundational_strengths_phase_boundary_component(component: str) -> str:
    """Classify source-defined foundational-strengths and phase-boundary components."""
    mapping = {
        "foundational_strengths": "predictive_constrained_explanatory_falsifiable_source_claim_boundary",
        "architecture_integration": "fifteen_layer_architecture_integration_source_claim_boundary",
        "future_research_and_parameter_constraints": "future_research_parameter_constraint_queue_boundary",
        "modulus_phase_decomposition": "single_psi_modulus_phase_regime_decomposition",
        "axion_analogy_and_em_interface": "axion_analogy_phase_sensitive_em_interface_boundary",
        "gauge_choice_and_kinetic_phase_boundary": "fixed_phase_gravity_vs_released_phase_quantum_regime_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown foundational-strengths phase-boundary component") from exc


def foundational_strengths_phase_boundary_labels() -> dict[str, str]:
    """Return source-bounded labels for the foundational-strengths phase-boundary slice."""
    return {
        "section": "Foundational Strengths of the SCPN Lagrangian",
        "parameter_boundary": "g and v remain unconstrained source parameters",
        "phase_split": "Psi = (v + h) exp(i theta)",
        "alp_interface": "L_a_gamma_gamma = g_a_gamma_gamma a F F_tilde",
        "next_boundary": "2.3 The Physics of Form: Spontaneous Symmetry Breaking",
    }


def validate_foundational_strengths_phase_boundary_fixture(
    config: FoundationalStrengthsPhaseBoundaryConfig | None = None,
) -> FoundationalStrengthsPhaseBoundaryFixtureResult:
    """Validate source accounting for the foundational-strengths phase-boundary slice."""
    cfg = config or FoundationalStrengthsPhaseBoundaryConfig()
    components = (
        "foundational_strengths",
        "architecture_integration",
        "future_research_and_parameter_constraints",
        "modulus_phase_decomposition",
        "axion_analogy_and_em_interface",
        "gauge_choice_and_kinetic_phase_boundary",
    )

    return FoundationalStrengthsPhaseBoundaryFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_foundational_strengths_phase_boundary_component(component)
            for component in components
        },
        labels=foundational_strengths_phase_boundary_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "foundational_strengths_are_source_claims_not_validation_evidence": 1.0,
            "parameter_values_g_and_v_remain_unconstrained_without_external_bounds": 1.0,
            "fixed_phase_gravity_and_phase_varying_quantum_regimes_must_not_be_mixed": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(1189, 1242)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_foundational_strengths_phase_boundary_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "FoundationalStrengthsPhaseBoundaryConfig",
    "FoundationalStrengthsPhaseBoundaryFixtureResult",
    "classify_foundational_strengths_phase_boundary_component",
    "foundational_strengths_phase_boundary_labels",
    "validate_foundational_strengths_phase_boundary_fixture",
]
