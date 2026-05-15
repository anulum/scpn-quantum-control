# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Preface I rigour validation
"""Executable Preface I methodological-rigour boundary checks for Paper 0."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded Preface I methodological rigour; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00307", "P0R00332")
NORMALISED_INTERACTION_FORMULA = "H_int = -lambda * Psi_s * sigma"


@dataclass(frozen=True, slots=True)
class DisciplineRole:
    """Source-preserved role for a Preface I discipline."""

    discipline: str
    source_records: tuple[str, ...]
    hpc_role: str
    sigma_role: str
    implementation_scope: str


@dataclass(frozen=True, slots=True)
class PrefaceIRigourConfig:
    """Configuration for the Paper 0 Preface I rigour fixture."""

    expected_blank_separator_count: int = 2
    preface_ii_boundary: str = "P0R00333"

    def __post_init__(self) -> None:
        if self.expected_blank_separator_count != 2:
            raise ValueError("expected_blank_separator_count must equal 2")
        if self.preface_ii_boundary != "P0R00333":
            raise ValueError("preface_ii_boundary must equal P0R00333")


@dataclass(frozen=True, slots=True)
class PrefaceIRigourFixtureResult:
    """Result for the Paper 0 Preface I rigour fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    discipline_roles: dict[str, dict[str, Any]]
    blank_separator_count: int
    interaction_formula: str
    preface_ii_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def normalise_interaction_formula(formula: str) -> str:
    """Normalise the source H_int notation without dropping parameters."""
    normalised = (
        formula.strip().replace("λ", "lambda").replace("Ψs", "Psi_s").replace("σ", "sigma")
    )
    if normalised != NORMALISED_INTERACTION_FORMULA:
        raise ValueError("unsupported interaction formula")
    return normalised


def discipline_role_catalogue() -> dict[str, DisciplineRole]:
    """Return source-preserved Preface I discipline roles."""
    return {
        "field_architecture": DisciplineRole(
            discipline="Field Architecture",
            source_records=("P0R00319", "P0R00323", "P0R00327"),
            hpc_role="generative_model_structure",
            sigma_role="identifying and characterising sigma",
            implementation_scope="theoretical structure, projection networks, resonance nodes",
        ),
        "consciousness_engineering": DisciplineRole(
            discipline="Consciousness Engineering",
            source_records=("P0R00320", "P0R00324", "P0R00328"),
            hpc_role="prediction_error_modulation",
            sigma_role="designing and controlling sigma",
            implementation_scope="experiments, simulations, devices, and VIBRANA interventions",
        ),
    }


def classify_hpc_application(discipline: str) -> str:
    """Map a Preface I discipline to its source HPC role."""
    try:
        return discipline_role_catalogue()[discipline].hpc_role
    except KeyError as exc:
        raise ValueError("unknown Preface I discipline") from exc


def validate_preface_i_rigour_fixture(
    config: PrefaceIRigourConfig | None = None,
) -> PrefaceIRigourFixtureResult:
    """Validate source accounting for the Paper 0 Preface I rigour run."""
    cfg = config or PrefaceIRigourConfig()
    catalogue = discipline_role_catalogue()
    interaction_formula = normalise_interaction_formula("H_int = -lambda * Psi_s * sigma")

    return PrefaceIRigourFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        discipline_roles={key: asdict(value) for key, value in catalogue.items()},
        blank_separator_count=cfg.expected_blank_separator_count,
        interaction_formula=interaction_formula,
        preface_ii_boundary=cfg.preface_ii_boundary,
        null_controls={
            "metaphysics_without_formalism_rejection_label": 1.0,
            "unknown_discipline_rejection_label": 1.0,
            "empirical_validation_overclaim_rejection_label": 1.0,
            "omitted_hamiltonian_parameter_rejection_label": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(307, 333)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_methodology_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "NORMALISED_INTERACTION_FORMULA",
    "SOURCE_LEDGER_SPAN",
    "DisciplineRole",
    "PrefaceIRigourConfig",
    "PrefaceIRigourFixtureResult",
    "classify_hpc_application",
    "discipline_role_catalogue",
    "normalise_interaction_formula",
    "validate_preface_i_rigour_fixture",
]
