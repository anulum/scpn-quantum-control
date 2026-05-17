# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 geometric coupling consistency validation
"""Source-accounting checks for Paper 0 geometric-coupling consistency records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = (
    "source-bounded geometric-coupling consistency derivation; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R01135", "P0R01188")


@dataclass(frozen=True, slots=True)
class GeometricCouplingConsistencyConfig:
    """Configuration for the geometric-coupling consistency fixture."""

    expected_source_record_count: int = 54
    expected_component_count: int = 6
    expected_math_id_count: int = 3
    next_source_boundary: str = "P0R01189"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 54:
            raise ValueError("expected_source_record_count must equal 54")
        if self.expected_component_count != 6:
            raise ValueError("expected_component_count must equal 6")
        if self.expected_math_id_count != 3:
            raise ValueError("expected_math_id_count must equal 3")
        if self.next_source_boundary != "P0R01189":
            raise ValueError("next_source_boundary must equal P0R01189")


@dataclass(frozen=True, slots=True)
class GeometricCouplingConsistencyFixtureResult:
    """Result for the Paper 0 geometric-coupling consistency fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    components: dict[str, str]
    labels: dict[str, str]
    source_record_count: int
    component_count: int
    math_ids: tuple[str, ...]
    image_ids: tuple[str, ...]
    table_ids: tuple[str, ...]
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_geometric_coupling_consistency_component(component: str) -> str:
    """Classify source-defined geometric-coupling consistency components."""
    mapping = {
        "coupling_problem_boundary": "internal_gauge_symmetry_not_spacetime_curvature_coupling",
        "minimal_curved_spacetime_coupling": "minimal_scalar_curved_spacetime_coupling_and_limitation",
        "non_minimal_consistency_condition": "conformal_and_renormalizability_non_minimal_coupling_argument",
        "derived_geometric_lagrangian": "derived_scalar_curvature_interaction_source_equation",
        "complete_covariant_action": "generally_covariant_gauge_invariant_total_action_boundary",
        "interpretation_prediction_comments": "interpretation_infoton_prediction_and_derivation_comment_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown geometric-coupling consistency component") from exc


def geometric_coupling_consistency_labels() -> dict[str, str]:
    """Return source-bounded labels for the geometric-coupling consistency slice."""
    return {
        "section": "Consistency Conditions and the Origin of Geometric Coupling",
        "minimal_coupling": "L_Psi_curved uses g_mu_nu and covariant derivatives",
        "non_minimal_coupling": "L_non_minimal = - xi R Psi^* Psi",
        "derived_geometric_term": "L_Geometric_prime = - g_PsiG R Psi^* Psi",
        "next_boundary": "Foundational Strengths of the SCPN Lagrangian",
    }


def validate_geometric_coupling_consistency_fixture(
    config: GeometricCouplingConsistencyConfig | None = None,
) -> GeometricCouplingConsistencyFixtureResult:
    """Validate source accounting for the geometric-coupling consistency slice."""
    cfg = config or GeometricCouplingConsistencyConfig()
    components = (
        "coupling_problem_boundary",
        "minimal_curved_spacetime_coupling",
        "non_minimal_consistency_condition",
        "derived_geometric_lagrangian",
        "complete_covariant_action",
        "interpretation_prediction_comments",
    )

    return GeometricCouplingConsistencyFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_geometric_coupling_consistency_component(component)
            for component in components
        },
        labels=geometric_coupling_consistency_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        math_ids=("EQ0010", "EQ0011", "EQ0012"),
        image_ids=("IMG0020",),
        table_ids=("TBL002",),
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "geometric_coupling_consistency_is_source_derivation_not_empirical_evidence": 1.0,
            "minimal_coupling_alone_does_not_satisfy_direct_curvature_coupling": 1.0,
            "infoton_prediction_is_not_detector_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(1135, 1189)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_geometric_coupling_consistency_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "GeometricCouplingConsistencyConfig",
    "GeometricCouplingConsistencyFixtureResult",
    "classify_geometric_coupling_consistency_component",
    "geometric_coupling_consistency_labels",
    "validate_geometric_coupling_consistency_fixture",
]
