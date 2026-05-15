# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Status and Method validation
"""Executable Status and Method boundary checks for Paper 0."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded Status and Method protocol; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00358", "P0R00390")
NORMALISED_STATUS_METHOD_INTERACTION_FORMULA = "H_int = -lambda * Psi_s * sigma"


@dataclass(frozen=True, slots=True)
class MethodologyCommitment:
    """Source-preserved operational commitment."""

    key: str
    source_record: str
    operational_role: str
    rejection_control: str


@dataclass(frozen=True, slots=True)
class StatusMethodConfig:
    """Configuration for the Paper 0 Status and Method fixture."""

    expected_blank_separator_count: int = 2
    next_boundary: str = "P0R00391"

    def __post_init__(self) -> None:
        if self.expected_blank_separator_count != 2:
            raise ValueError("expected_blank_separator_count must equal 2")
        if self.next_boundary != "P0R00391":
            raise ValueError("next_boundary must equal P0R00391")


@dataclass(frozen=True, slots=True)
class StatusMethodFixtureResult:
    """Result for the Paper 0 Status and Method fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    methodology_commitments: dict[str, dict[str, Any]]
    scientific_inference_roles: dict[str, str]
    blank_separator_count: int
    interaction_formula: str
    next_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def normalise_status_method_interaction_formula(formula: str) -> str:
    """Normalise the Status and Method H_int notation without dropping parameters."""
    normalised = (
        formula.strip().replace("λ", "lambda").replace("Ψs", "Psi_s").replace("σ", "sigma")
    )
    if normalised != NORMALISED_STATUS_METHOD_INTERACTION_FORMULA:
        raise ValueError("unsupported Status and Method interaction formula")
    return normalised


def methodology_commitment_catalogue() -> dict[str, MethodologyCommitment]:
    """Return source-preserved methodology commitments."""
    return {
        "falsifiability_first": MethodologyCommitment(
            key="falsifiability_first",
            source_record="P0R00360",
            operational_role="admission_gate",
            rejection_control="untestable_sigma_rejection_label",
        ),
        "hypothesis_registry": MethodologyCommitment(
            key="hypothesis_registry",
            source_record="P0R00360",
            operational_role="prediction_inventory",
            rejection_control="unregistered_prediction_rejection_label",
        ),
        "tiered_status": MethodologyCommitment(
            key="tiered_status",
            source_record="P0R00360",
            operational_role="claim_status_tracking",
            rejection_control="status_collapse_rejection_label",
        ),
        "versioning_and_correction": MethodologyCommitment(
            key="versioning_and_correction",
            source_record="P0R00360",
            operational_role="model_update",
            rejection_control="unversioned_update_rejection_label",
        ),
    }


def classify_scientific_inference_step(step: str) -> str:
    """Map Status and Method scientific steps to FEP roles."""
    mapping = {
        "theory": "generative_model",
        "experiment": "sensory_evidence",
        "falsification": "prediction_error",
        "revision": "model_update",
    }
    try:
        return mapping[step]
    except KeyError as exc:
        raise ValueError("unknown scientific inference step") from exc


def validate_status_method_fixture(
    config: StatusMethodConfig | None = None,
) -> StatusMethodFixtureResult:
    """Validate source accounting for the Paper 0 Status and Method run."""
    cfg = config or StatusMethodConfig()
    commitments = methodology_commitment_catalogue()
    inference_roles = {
        step: classify_scientific_inference_step(step)
        for step in ("theory", "experiment", "falsification", "revision")
    }
    interaction_formula = normalise_status_method_interaction_formula(
        "H_int = -lambda * Psi_s * sigma"
    )

    return StatusMethodFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        methodology_commitments={key: asdict(value) for key, value in commitments.items()},
        scientific_inference_roles=inference_roles,
        blank_separator_count=cfg.expected_blank_separator_count,
        interaction_formula=interaction_formula,
        next_boundary=cfg.next_boundary,
        null_controls={
            "doctrine_promotion_rejection_label": 1.0,
            "untestable_sigma_rejection_label": 1.0,
            "analogy_without_empirical_handle_rejection_label": 1.0,
            "unknown_inference_step_rejection_label": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(358, 391)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_methodology_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "NORMALISED_STATUS_METHOD_INTERACTION_FORMULA",
    "SOURCE_LEDGER_SPAN",
    "MethodologyCommitment",
    "StatusMethodConfig",
    "StatusMethodFixtureResult",
    "classify_scientific_inference_step",
    "methodology_commitment_catalogue",
    "normalise_status_method_interaction_formula",
    "validate_status_method_fixture",
]
