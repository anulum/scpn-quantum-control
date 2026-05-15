# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom I minimal Lagrangian validation
"""Source-accounting checks for Paper 0 Axiom I minimal Lagrangian records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded Axiom I minimal Lagrangian map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00733", "P0R00746")


@dataclass(frozen=True, slots=True)
class AxiomIMinimalLagrangianConfig:
    """Configuration for the Axiom I minimal Lagrangian fixture."""

    expected_minimal_criterion_count: int = 3
    expected_equation_record_count: int = 4
    next_source_boundary: str = "P0R00747"

    def __post_init__(self) -> None:
        if self.expected_minimal_criterion_count != 3:
            raise ValueError("expected_minimal_criterion_count must equal 3")
        if self.expected_equation_record_count != 4:
            raise ValueError("expected_equation_record_count must equal 4")
        if self.next_source_boundary != "P0R00747":
            raise ValueError("next_source_boundary must equal P0R00747")


@dataclass(frozen=True, slots=True)
class AxiomIMinimalLagrangianFixtureResult:
    """Result for the Paper 0 Axiom I minimal Lagrangian fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    minimal_requirements: dict[str, str]
    lagrangian_operators: dict[str, str]
    labels: dict[str, str]
    minimal_criterion_count: int
    lagrangian_operator_count: int
    equation_record_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_minimal_lagrangian_requirement(requirement: str) -> str:
    """Classify source-defined minimal Lagrangian requirements."""
    mapping = {
        "spin0": "single_irreducible_spin0_dof",
        "phase": "intentional_phase_variable",
        "soliton": "stable_finite_energy_organismal_soliton",
    }
    try:
        return mapping[requirement]
    except KeyError as exc:
        raise ValueError("unknown minimal Lagrangian requirement") from exc


def classify_lagrangian_operator(operator: str) -> str:
    """Classify source-defined minimal Lagrangian operator roles."""
    mapping = {
        "kinetic": "covariant_psi_kinetic_term",
        "potential": "quartic_ssb_potential",
        "curvature": "nonminimal_curvature_coupling",
        "infoton": "pulled_back_information_metric_dynamics",
    }
    try:
        return mapping[operator]
    except KeyError as exc:
        raise ValueError("unknown Lagrangian operator") from exc


def axiom_i_minimal_lagrangian_labels() -> dict[str, str]:
    """Return source-bounded labels for the Axiom I minimal Lagrangian slice."""
    return {
        "section": "Model-Class Justification: From Axiom 1 to a Minimal Psi-Field Lagrangian",
        "lagrangian": "L_min = |D_mu Psi|^2 - V(|Psi|) - 1/4 g_F F F - xi R |Psi|^2",
        "next_boundary": "Why this family satisfies (i)-(iii)",
    }


def validate_axiom_i_minimal_lagrangian_fixture(
    config: AxiomIMinimalLagrangianConfig | None = None,
) -> AxiomIMinimalLagrangianFixtureResult:
    """Validate source accounting for the Axiom I minimal Lagrangian slice."""
    cfg = config or AxiomIMinimalLagrangianConfig()
    minimal_requirements = ("spin0", "phase", "soliton")
    lagrangian_operators = ("kinetic", "potential", "curvature", "infoton")

    return AxiomIMinimalLagrangianFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        minimal_requirements={
            requirement: classify_minimal_lagrangian_requirement(requirement)
            for requirement in minimal_requirements
        },
        lagrangian_operators={
            operator: classify_lagrangian_operator(operator) for operator in lagrangian_operators
        },
        labels=axiom_i_minimal_lagrangian_labels(),
        minimal_criterion_count=cfg.expected_minimal_criterion_count,
        lagrangian_operator_count=len(lagrangian_operators),
        equation_record_count=cfg.expected_equation_record_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "minimal_lagrangian_is_source_formula_not_empirical_fit": 1.0,
            "curvature_and_information_metric_terms_require_downstream_tests": 1.0,
            "ssb_boundedness_claim_requires_model_validation": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(733, 747)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_minimal_lagrangian_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "AxiomIMinimalLagrangianConfig",
    "AxiomIMinimalLagrangianFixtureResult",
    "axiom_i_minimal_lagrangian_labels",
    "classify_lagrangian_operator",
    "classify_minimal_lagrangian_requirement",
    "validate_axiom_i_minimal_lagrangian_fixture",
]
