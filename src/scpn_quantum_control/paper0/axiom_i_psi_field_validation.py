# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom I Psi-field validation
"""Source-accounting checks for Paper 0 Axiom I Psi-field records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded Axiom I Psi-field map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00670", "P0R00702")


@dataclass(frozen=True, slots=True)
class AxiomIPsiFieldConfig:
    """Configuration for the Axiom I Psi-field fixture."""

    expected_blank_separator_count: int = 2
    next_source_boundary: str = "P0R00703"

    def __post_init__(self) -> None:
        if self.expected_blank_separator_count != 2:
            raise ValueError("expected_blank_separator_count must equal 2")
        if self.next_source_boundary != "P0R00703":
            raise ValueError("next_source_boundary must equal P0R00703")


@dataclass(frozen=True, slots=True)
class AxiomIPsiFieldFixtureResult:
    """Result for the Paper 0 Axiom I Psi-field fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    axiom_i_roles: dict[str, str]
    psi_field_claims: dict[str, str]
    labels: dict[str, str]
    axiom_i_role_count: int
    psi_field_claim_count: int
    blank_separator_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_axiom_i_role(role: str) -> str:
    """Classify one source-defined role in the Axiom I slice."""
    mapping = {
        "ontological_primitive": "metaphysical_generative_postulate",
        "psi_field": "universal_complex_scalar_field_formalisation",
        "generative_model": "cosmic_priors_physical_substrate",
        "hint_ground": "psi_s_ontological_ground_not_peer_field",
    }
    try:
        return mapping[role]
    except KeyError as exc:
        raise ValueError("unknown Axiom I role") from exc


def classify_psi_field_claim(claim: str) -> str:
    """Classify source-bounded Psi-field claim types."""
    mapping = {
        "not_emergent": "matter_emerges_from_psi_not_reverse",
        "complex_scalar": "universal_complex_scalar_field",
        "hierarchical_definition": "ontological_physical_experiential_layers",
    }
    try:
        return mapping[claim]
    except KeyError as exc:
        raise ValueError("unknown Psi-field claim") from exc


def axiom_i_psi_field_labels() -> dict[str, str]:
    """Return source-bounded labels for the Axiom I Psi-field slice."""
    return {
        "section": "Axiom I: The Primacy of Consciousness (Psi)",
        "h_int": "H_int = -lambda * Psi_s * sigma",
        "next_boundary": "Model-Class Justification: From Axiom to Lagrangian",
    }


def validate_axiom_i_psi_field_fixture(
    config: AxiomIPsiFieldConfig | None = None,
) -> AxiomIPsiFieldFixtureResult:
    """Validate source accounting for the Axiom I Psi-field slice."""
    cfg = config or AxiomIPsiFieldConfig()
    axiom_i_roles = (
        "ontological_primitive",
        "psi_field",
        "generative_model",
        "hint_ground",
    )
    psi_field_claims = ("not_emergent", "complex_scalar", "hierarchical_definition")

    return AxiomIPsiFieldFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        axiom_i_roles={role: classify_axiom_i_role(role) for role in axiom_i_roles},
        psi_field_claims={claim: classify_psi_field_claim(claim) for claim in psi_field_claims},
        labels=axiom_i_psi_field_labels(),
        axiom_i_role_count=len(axiom_i_roles),
        psi_field_claim_count=len(psi_field_claims),
        blank_separator_count=cfg.expected_blank_separator_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "axiom_i_is_not_empirical_result": 1.0,
            "psi_field_formalisation_requires_downstream_model_tests": 1.0,
            "image_boundary_is_not_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(670, 703)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_axiom_i_map_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "AxiomIPsiFieldConfig",
    "AxiomIPsiFieldFixtureResult",
    "axiom_i_psi_field_labels",
    "classify_axiom_i_role",
    "classify_psi_field_claim",
    "validate_axiom_i_psi_field_fixture",
]
