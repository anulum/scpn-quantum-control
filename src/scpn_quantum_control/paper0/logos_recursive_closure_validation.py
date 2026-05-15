# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Logos recursive closure validation
"""Source-accounting checks for Logos recursive-closure records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded Logos recursive closure; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00545", "P0R00577")


@dataclass(frozen=True, slots=True)
class LogosRecursiveClosureConfig:
    """Configuration for the Logos recursive-closure fixture."""

    expected_blank_separator_count: int = 3
    next_source_boundary: str = "P0R00578"

    def __post_init__(self) -> None:
        if self.expected_blank_separator_count != 3:
            raise ValueError("expected_blank_separator_count must equal 3")
        if self.next_source_boundary != "P0R00578":
            raise ValueError("next_source_boundary must equal P0R00578")


@dataclass(frozen=True, slots=True)
class LogosRecursiveClosureFixtureResult:
    """Result for the Paper 0 Logos recursive-closure fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    axiom_statuses: dict[str, str]
    hint_axiom_roles: dict[str, str]
    recursive_closure: dict[str, str]
    axiom_count: int
    hint_role_count: int
    blank_separator_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_axiom_status(axiom: str) -> str:
    """Classify Logos axiom source status."""
    mapping = {
        "axiom_1": "metaphysical_postulate",
        "axiom_2": "falsifiable_physical_hypothesis",
        "axiom_3": "normative_teleological_postulate",
    }
    try:
        return mapping[axiom]
    except KeyError as exc:
        raise ValueError("unknown Logos axiom") from exc


def classify_hint_axiom_role(axiom: str) -> str:
    """Classify each axiom's source role in the interaction Hamiltonian."""
    mapping = {
        "axiom_1": "defines_psi_s_ground",
        "axiom_2": "defines_lambda_sigma_information_geometry",
        "axiom_3": "defines_sec_directional_bias",
    }
    try:
        return mapping[axiom]
    except KeyError as exc:
        raise ValueError("unknown H_int axiom role") from exc


def recursive_closure_labels() -> dict[str, str]:
    """Return source-bounded recursive-closure labels."""
    return {
        "hierarchy": "15-layer hierarchy",
        "closure": "recursive closure",
        "figure": "SCPN Hierarchy & Recursive Closure",
    }


def validate_logos_recursive_closure_fixture(
    config: LogosRecursiveClosureConfig | None = None,
) -> LogosRecursiveClosureFixtureResult:
    """Validate source accounting for the Logos recursive-closure run."""
    cfg = config or LogosRecursiveClosureConfig()
    axioms = ("axiom_1", "axiom_2", "axiom_3")
    axiom_statuses = {axiom: classify_axiom_status(axiom) for axiom in axioms}
    hint_roles = {axiom: classify_hint_axiom_role(axiom) for axiom in axioms}

    return LogosRecursiveClosureFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        axiom_statuses=axiom_statuses,
        hint_axiom_roles=hint_roles,
        recursive_closure=recursive_closure_labels(),
        axiom_count=len(axiom_statuses),
        hint_role_count=len(hint_roles),
        blank_separator_count=cfg.expected_blank_separator_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "axioms_are_not_established_truths": 1.0,
            "figure_caption_is_not_validation_evidence": 1.0,
            "recursive_closure_is_architectural_boundary": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(545, 578)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_axiom_status_map_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "LogosRecursiveClosureConfig",
    "LogosRecursiveClosureFixtureResult",
    "classify_axiom_status",
    "classify_hint_axiom_role",
    "recursive_closure_labels",
    "validate_logos_recursive_closure_fixture",
]
