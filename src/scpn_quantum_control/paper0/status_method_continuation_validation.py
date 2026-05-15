# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Status and Method continuation validation
"""Executable Status and Method continuation boundary checks for Paper 0."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded Status and Method continuation; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00391", "P0R00400")


@dataclass(frozen=True, slots=True)
class StatusMethodContinuationConfig:
    """Configuration for the Status and Method continuation fixture."""

    expected_blank_separator_count: int = 1
    scp_mandate_boundary: str = "P0R00401"

    def __post_init__(self) -> None:
        if self.expected_blank_separator_count != 1:
            raise ValueError("expected_blank_separator_count must equal 1")
        if self.scp_mandate_boundary != "P0R00401":
            raise ValueError("scp_mandate_boundary must equal P0R00401")


@dataclass(frozen=True, slots=True)
class StatusMethodContinuationFixtureResult:
    """Result for the Paper 0 Status and Method continuation fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    not_boundaries: dict[str, str]
    disagreement_moves: dict[str, str]
    operational_commitments: tuple[str, ...]
    operational_commitment_count: int
    blank_separator_count: int
    scp_mandate_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_not_boundary(boundary: str) -> str:
    """Classify a source not-boundary into its rejection role."""
    mapping = {
        "absolute_truths": "reject_doctrine_status",
        "metaphor_literalisation": "reject_ontological_load",
        "empirical_bypass": "reject_method_bypass",
    }
    try:
        return mapping[boundary]
    except KeyError as exc:
        raise ValueError("unknown Status and Method boundary") from exc


def classify_disagreement_move(move: str) -> str:
    """Classify productive-disagreement moves into required actions."""
    mapping = {
        "prediction_baseline": "run_comparison",
        "analogy_handle": "supply_or_refute_empirical_handle",
        "replacement_model": "same_slot_stricter_fit",
    }
    try:
        return mapping[move]
    except KeyError as exc:
        raise ValueError("unknown disagreement move") from exc


def operational_commitment_labels() -> tuple[str, ...]:
    """Return the four source operational commitment labels."""
    return (
        "falsifiability_first",
        "hypothesis_registry",
        "tiered_status",
        "versioning_and_correction",
    )


def validate_status_method_continuation_fixture(
    config: StatusMethodContinuationConfig | None = None,
) -> StatusMethodContinuationFixtureResult:
    """Validate source accounting for the Status and Method continuation run."""
    cfg = config or StatusMethodContinuationConfig()
    not_boundaries = {
        key: classify_not_boundary(key)
        for key in ("absolute_truths", "metaphor_literalisation", "empirical_bypass")
    }
    disagreement_moves = {
        key: classify_disagreement_move(key)
        for key in ("prediction_baseline", "analogy_handle", "replacement_model")
    }
    commitments = operational_commitment_labels()

    return StatusMethodContinuationFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        not_boundaries=not_boundaries,
        disagreement_moves=disagreement_moves,
        operational_commitments=commitments,
        operational_commitment_count=len(commitments),
        blank_separator_count=cfg.expected_blank_separator_count,
        scp_mandate_boundary=cfg.scp_mandate_boundary,
        null_controls={
            "literalised_metaphor_rejection_label": 1.0,
            "analogy_without_handle_rejection_label": 1.0,
            "doctrine_status_rejection_label": 1.0,
            "capstone_finality_rejection_label": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(391, 401)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_methodology_continuation_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "StatusMethodContinuationConfig",
    "StatusMethodContinuationFixtureResult",
    "classify_disagreement_move",
    "classify_not_boundary",
    "operational_commitment_labels",
    "validate_status_method_continuation_fixture",
]
