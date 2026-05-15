# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom I meta-coupling validation
"""Source-accounting checks for Paper 0 Axiom I meta-coupling records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded Axiom I meta-coupling map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00717", "P0R00732")


@dataclass(frozen=True, slots=True)
class AxiomIMetaCouplingConfig:
    """Configuration for the Axiom I meta-coupling fixture."""

    expected_interaction_component_count: int = 3
    next_source_boundary: str = "P0R00733"

    def __post_init__(self) -> None:
        if self.expected_interaction_component_count != 3:
            raise ValueError("expected_interaction_component_count must equal 3")
        if self.next_source_boundary != "P0R00733":
            raise ValueError("next_source_boundary must equal P0R00733")


@dataclass(frozen=True, slots=True)
class AxiomIMetaCouplingFixtureResult:
    """Result for the Paper 0 Axiom I meta-coupling fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    predictive_hardware_roles: dict[str, str]
    coupling_requirements: dict[str, str]
    labels: dict[str, str]
    predictive_hardware_role_count: int
    interaction_component_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_predictive_hardware_role(role: str) -> str:
    """Classify source-defined predictive-hardware roles."""
    mapping = {
        "spin0": "minimal_processor_components",
        "phase": "belief_prior_carrier_updated_by_u1",
        "soliton": "persistent_deep_prior_self",
    }
    try:
        return mapping[role]
    except KeyError as exc:
        raise ValueError("unknown predictive-hardware role") from exc


def classify_coupling_requirement(requirement: str) -> str:
    """Classify source-defined Psi-s interaction requirements."""
    mapping = {
        "psi_s": "complex_scalar_stability_intentionality",
        "gauge": "local_well_behaved_infoton_mediation",
        "sigma": "stable_charge_supported_q_ball_self",
    }
    try:
        return mapping[requirement]
    except KeyError as exc:
        raise ValueError("unknown coupling requirement") from exc


def axiom_i_meta_coupling_labels() -> dict[str, str]:
    """Return source-bounded labels for the Axiom I meta-coupling slice."""
    return {
        "section": "Meta-Framework Integrations and Psi-s Coupling",
        "h_int": "H_int = -lambda * Psi_s * sigma",
        "next_boundary": (
            "Model-Class Justification: From Axiom 1 to a Minimal Psi-Field Lagrangian"
        ),
    }


def validate_axiom_i_meta_coupling_fixture(
    config: AxiomIMetaCouplingConfig | None = None,
) -> AxiomIMetaCouplingFixtureResult:
    """Validate source accounting for the Axiom I meta-coupling slice."""
    cfg = config or AxiomIMetaCouplingConfig()
    predictive_hardware_roles = ("spin0", "phase", "soliton")
    coupling_requirements = ("psi_s", "gauge", "sigma")

    return AxiomIMetaCouplingFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        predictive_hardware_roles={
            role: classify_predictive_hardware_role(role) for role in predictive_hardware_roles
        },
        coupling_requirements={
            requirement: classify_coupling_requirement(requirement)
            for requirement in coupling_requirements
        },
        labels=axiom_i_meta_coupling_labels(),
        predictive_hardware_role_count=len(predictive_hardware_roles),
        interaction_component_count=cfg.expected_interaction_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "predictive_coding_hardware_is_not_empirical_validation": 1.0,
            "hint_component_justification_requires_lagrangian_tests": 1.0,
            "necessity_language_requires_downstream_falsification": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(717, 733)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_meta_coupling_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "AxiomIMetaCouplingConfig",
    "AxiomIMetaCouplingFixtureResult",
    "axiom_i_meta_coupling_labels",
    "classify_coupling_requirement",
    "classify_predictive_hardware_role",
    "validate_axiom_i_meta_coupling_fixture",
]
