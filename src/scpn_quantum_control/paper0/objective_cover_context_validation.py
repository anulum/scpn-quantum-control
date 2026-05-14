# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 objective cover context validation
"""Executable objective-cover context boundary checks for Paper 0."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded objective and cover context; not validation evidence"
HARDWARE_STATUS = "source_context_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00218", "P0R00248")


@dataclass(frozen=True, slots=True)
class ObjectiveCoverContextConfig:
    """Configuration for the Paper 0 objective cover context fixture."""

    expected_image_marker_count: int = 3
    expected_collection_book_count: int = 5
    positioning_preface_boundary: str = "P0R00249"

    def __post_init__(self) -> None:
        if self.expected_image_marker_count < 0:
            raise ValueError("expected_image_marker_count must be non-negative")
        if self.expected_collection_book_count != 5:
            raise ValueError("expected_collection_book_count must equal 5")
        if self.positioning_preface_boundary != "P0R00249":
            raise ValueError("positioning_preface_boundary must equal P0R00249")


@dataclass(frozen=True, slots=True)
class ObjectiveCoverContextFixtureResult:
    """Result for the Paper 0 objective cover context fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    objective_targets: tuple[str, ...]
    image_marker_count: int
    collection_book_count: int
    positioning_preface_boundary: str
    context_classification: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def objective_cover_targets() -> tuple[str, ...]:
    """Return source-positioning targets preserved by this cover slice."""
    return (
        "Multi-Scale Interaction of Consciousness and Biology",
        "self-organising, self-optimising, and self-correcting universe",
        "SCPN architecture as a cosmic-scale active inference engine",
        "Meta Metatron Cycle cyclic-operator positioning",
        "Book II - The Sentient-Consciousness Projection Network",
    )


def classify_objective_cover_context(
    *,
    image_marker_count: int,
    has_cyclic_operator_positioning: bool,
    requested_status: str,
) -> str:
    """Classify objective and cover records without allowing evidence promotion."""
    if image_marker_count < 0:
        raise ValueError("image_marker_count cannot be negative")
    if requested_status == "validation_evidence":
        raise ValueError("objective and cover context cannot be validation evidence")
    if has_cyclic_operator_positioning:
        return "cover_positioning_context"
    if image_marker_count:
        return "image_cover_context"
    return "objective_context"


def validate_objective_cover_context_fixture(
    config: ObjectiveCoverContextConfig | None = None,
) -> ObjectiveCoverContextFixtureResult:
    """Validate source accounting for the Paper 0 objective cover context run."""
    cfg = config or ObjectiveCoverContextConfig()
    classification = classify_objective_cover_context(
        image_marker_count=cfg.expected_image_marker_count,
        has_cyclic_operator_positioning=True,
        requested_status="context_boundary",
    )

    return ObjectiveCoverContextFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        objective_targets=objective_cover_targets(),
        image_marker_count=cfg.expected_image_marker_count,
        collection_book_count=cfg.expected_collection_book_count,
        positioning_preface_boundary=cfg.positioning_preface_boundary,
        context_classification=classification,
        null_controls={
            "cover_as_validation_evidence_rejection_label": 1.0,
            "image_marker_as_data_rejection_label": 1.0,
            "missing_positioning_preface_boundary_rejection_label": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(218, 249)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_context_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "ObjectiveCoverContextConfig",
    "ObjectiveCoverContextFixtureResult",
    "classify_objective_cover_context",
    "objective_cover_targets",
    "validate_objective_cover_context_fixture",
]
