# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Positioning Preface context validation
"""Executable Positioning Preface context boundary checks for Paper 0."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded Positioning Preface context; not validation evidence"
HARDWARE_STATUS = "source_context_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00249", "P0R00267")


@dataclass(frozen=True, slots=True)
class PositioningPrefaceContextConfig:
    """Configuration for the Paper 0 Positioning Preface context fixture."""

    expected_blank_separator_count: int = 2
    expected_image_marker_count: int = 1
    part_i_boundary: str = "P0R00268"

    def __post_init__(self) -> None:
        if self.expected_blank_separator_count < 0:
            raise ValueError("expected_blank_separator_count must be non-negative")
        if self.expected_image_marker_count < 0:
            raise ValueError("expected_image_marker_count must be non-negative")
        if self.part_i_boundary != "P0R00268":
            raise ValueError("part_i_boundary must equal P0R00268")


@dataclass(frozen=True, slots=True)
class PositioningPrefaceContextFixtureResult:
    """Result for the Paper 0 Positioning Preface context fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    positioning_targets: tuple[str, ...]
    blank_separator_count: int
    image_marker_count: int
    part_i_boundary: str
    context_classification: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def positioning_preface_targets() -> tuple[str, ...]:
    """Return source-positioning targets preserved by this preface slice."""
    return (
        "Field Architecture as formal study of consciousness-field structures",
        "Consciousness Engineering as applied modulation and implementation branch",
        "Noetic Field Theory with rigour, explicit equations, and testable couplings",
        "VIBRANA and symbolic operators as later architecture-manual targets",
        "Author's Note two frequencies: academic register and visionary register",
    )


def classify_positioning_preface_context(
    *,
    blank_separator_count: int,
    image_marker_count: int,
    has_dual_register_author_note: bool,
    requested_status: str,
) -> str:
    """Classify Positioning Preface records without allowing evidence promotion."""
    if blank_separator_count < 0:
        raise ValueError("blank_separator_count cannot be negative")
    if image_marker_count < 0:
        raise ValueError("image_marker_count cannot be negative")
    if requested_status == "validation_evidence":
        raise ValueError("positioning preface cannot be validation evidence")
    if has_dual_register_author_note:
        return "dual_register_preface_context"
    if image_marker_count:
        return "preface_with_image_context"
    return "positioning_preface_context"


def validate_positioning_preface_context_fixture(
    config: PositioningPrefaceContextConfig | None = None,
) -> PositioningPrefaceContextFixtureResult:
    """Validate source accounting for the Paper 0 Positioning Preface context run."""
    cfg = config or PositioningPrefaceContextConfig()
    classification = classify_positioning_preface_context(
        blank_separator_count=cfg.expected_blank_separator_count,
        image_marker_count=cfg.expected_image_marker_count,
        has_dual_register_author_note=True,
        requested_status="context_boundary",
    )

    return PositioningPrefaceContextFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        positioning_targets=positioning_preface_targets(),
        blank_separator_count=cfg.expected_blank_separator_count,
        image_marker_count=cfg.expected_image_marker_count,
        part_i_boundary=cfg.part_i_boundary,
        context_classification=classification,
        null_controls={
            "preface_as_validation_evidence_rejection_label": 1.0,
            "image_marker_as_data_rejection_label": 1.0,
            "missing_part_i_boundary_rejection_label": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(249, 268)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_context_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "PositioningPrefaceContextConfig",
    "PositioningPrefaceContextFixtureResult",
    "classify_positioning_preface_context",
    "positioning_preface_targets",
    "validate_positioning_preface_context_fixture",
]
