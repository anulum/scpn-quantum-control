# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Preface II visionary validation
"""Executable Preface II visionary-register boundary checks for Paper 0."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded Preface II visionary register; not validation evidence"
HARDWARE_STATUS = "source_visionary_register_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00333", "P0R00357")
NORMALISED_VISIONARY_INTERACTION_FORMULA = "H_int = -lambda * Psi_s * sigma"


@dataclass(frozen=True, slots=True)
class VisionaryOperator:
    """Source-preserved visionary operator role."""

    operator: str
    source_records: tuple[str, ...]
    active_inference_role: str
    sigma_role: str
    claim_boundary: str


@dataclass(frozen=True, slots=True)
class PrefaceIIVisionaryConfig:
    """Configuration for the Paper 0 Preface II visionary fixture."""

    expected_blank_separator_count: int = 1
    status_method_boundary: str = "P0R00358"

    def __post_init__(self) -> None:
        if self.expected_blank_separator_count != 1:
            raise ValueError("expected_blank_separator_count must equal 1")
        if self.status_method_boundary != "P0R00358":
            raise ValueError("status_method_boundary must equal P0R00358")


@dataclass(frozen=True, slots=True)
class PrefaceIIVisionaryFixtureResult:
    """Result for the Paper 0 Preface II visionary fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    visionary_operators: dict[str, dict[str, Any]]
    blank_separator_count: int
    interaction_formula: str
    status_method_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def normalise_visionary_interaction_formula(formula: str) -> str:
    """Normalise the visionary-register H_int notation without dropping parameters."""
    normalised = (
        formula.strip().replace("λ", "lambda").replace("Ψs", "Psi_s").replace("σ", "sigma")
    )
    if normalised != NORMALISED_VISIONARY_INTERACTION_FORMULA:
        raise ValueError("unsupported visionary interaction formula")
    return normalised


def visionary_operator_catalogue() -> dict[str, VisionaryOperator]:
    """Return source-preserved Preface II operator roles."""
    return {
        "projection_lattices": VisionaryOperator(
            operator="projection lattices",
            source_records=("P0R00345", "P0R00353"),
            active_inference_role="prior_pathway",
            sigma_role="pathway for high-level priors and field design language",
            claim_boundary="source operator vocabulary; not measured topology",
        ),
        "resonance_hubs": VisionaryOperator(
            operator="resonance hubs",
            source_records=("P0R00349", "P0R00353"),
            active_inference_role="coherent_coupling_site",
            sigma_role="coherent sigma coupling site",
            claim_boundary="source coupling-site vocabulary; not measured resonance",
        ),
        "vibrational_codes": VisionaryOperator(
            operator="vibrational codes",
            source_records=("P0R00346", "P0R00354", "P0R00356"),
            active_inference_role="generative_model_intervention",
            sigma_role="designed sigma organisation",
            claim_boundary="source intervention vocabulary; not validated control protocol",
        ),
    }


def classify_visionary_active_inference_role(operator: str) -> str:
    """Map a Preface II operator to its source active-inference role."""
    try:
        return visionary_operator_catalogue()[operator].active_inference_role
    except KeyError as exc:
        raise ValueError("unknown Preface II operator") from exc


def validate_preface_ii_visionary_fixture(
    config: PrefaceIIVisionaryConfig | None = None,
) -> PrefaceIIVisionaryFixtureResult:
    """Validate source accounting for the Paper 0 Preface II visionary run."""
    cfg = config or PrefaceIIVisionaryConfig()
    catalogue = visionary_operator_catalogue()
    interaction_formula = normalise_visionary_interaction_formula(
        "H_int = -lambda * Psi_s * sigma"
    )

    return PrefaceIIVisionaryFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        visionary_operators={key: asdict(value) for key, value in catalogue.items()},
        blank_separator_count=cfg.expected_blank_separator_count,
        interaction_formula=interaction_formula,
        status_method_boundary=cfg.status_method_boundary,
        null_controls={
            "manifesto_as_empirical_evidence_rejection_label": 1.0,
            "unknown_operator_rejection_label": 1.0,
            "mastery_as_validation_rejection_label": 1.0,
            "sigma_design_without_testability_rejection_label": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(333, 358)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_visionary_register_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "NORMALISED_VISIONARY_INTERACTION_FORMULA",
    "SOURCE_LEDGER_SPAN",
    "PrefaceIIVisionaryConfig",
    "PrefaceIIVisionaryFixtureResult",
    "VisionaryOperator",
    "classify_visionary_active_inference_role",
    "normalise_visionary_interaction_formula",
    "validate_preface_ii_visionary_fixture",
    "visionary_operator_catalogue",
]
