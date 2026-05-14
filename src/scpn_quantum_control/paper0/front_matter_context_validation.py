# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 front matter context validation
"""Executable front matter context boundary checks for Paper 0."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded front matter context; not validation evidence"
HARDWARE_STATUS = "source_context_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00018", "P0R00104")


@dataclass(frozen=True, slots=True)
class FrontMatterContextConfig:
    """Configuration for the Paper 0 front matter context fixture."""

    expected_collection_book_count: int = 5
    expected_layer_monograph_count: int = 16
    expected_validation_suite_paper_count: int = 4
    expected_blank_placeholder_count: int = 45

    def __post_init__(self) -> None:
        if self.expected_collection_book_count != 5:
            raise ValueError("expected_collection_book_count must equal 5")
        if self.expected_layer_monograph_count != 16:
            raise ValueError("expected_layer_monograph_count must equal 16")
        if self.expected_validation_suite_paper_count != 4:
            raise ValueError("expected_validation_suite_paper_count must equal 4")
        if self.expected_blank_placeholder_count < 0:
            raise ValueError("expected_blank_placeholder_count must be non-negative")


@dataclass(frozen=True, slots=True)
class FrontMatterContextFixtureResult:
    """Result for the Paper 0 front matter context fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    collection_books: tuple[str, ...]
    collection_book_count: int
    layer_monograph_count: int
    validation_suite_paper_count: int
    blank_placeholder_count: int
    fragmented_toc_warning_present: bool
    context_classification: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def collection_books() -> tuple[str, ...]:
    """Return the five source-listed collection books in ledger order."""
    return (
        "Book I - The Anulum Framework",
        "Book II - The Sentient-Consciousness Projection Network",
        "Book III - Metatron's Coda",
        "Book IV - The Godelian Koans",
        "Book V - VIBRANA",
    )


def classify_toc_context(
    *,
    blank_placeholder_count: int,
    fragmented_warning_present: bool,
    requested_status: str,
) -> str:
    """Classify ToC records without allowing empirical-evidence promotion."""
    if blank_placeholder_count < 0:
        raise ValueError("blank_placeholder_count cannot be negative")
    if requested_status == "empirical_evidence":
        raise ValueError("front matter cannot be promoted as empirical evidence")
    if fragmented_warning_present:
        return "fragmented_context_boundary"
    if blank_placeholder_count:
        return "placeholder_context_boundary"
    return "context_boundary"


def validate_front_matter_context_fixture(
    config: FrontMatterContextConfig | None = None,
) -> FrontMatterContextFixtureResult:
    """Validate source accounting for the Paper 0 front matter context run."""
    cfg = config or FrontMatterContextConfig()
    books = collection_books()
    classification = classify_toc_context(
        blank_placeholder_count=cfg.expected_blank_placeholder_count,
        fragmented_warning_present=True,
        requested_status="context_boundary",
    )
    if len(books) != cfg.expected_collection_book_count:
        raise ValueError("collection book count does not match Paper 0 front matter")

    return FrontMatterContextFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        collection_books=books,
        collection_book_count=len(books),
        layer_monograph_count=cfg.expected_layer_monograph_count,
        validation_suite_paper_count=cfg.expected_validation_suite_paper_count,
        blank_placeholder_count=cfg.expected_blank_placeholder_count,
        fragmented_toc_warning_present=True,
        context_classification=classification,
        null_controls={
            "toc_as_empirical_evidence_rejection_label": 1.0,
            "blank_record_skip_rejection_label": 1.0,
            "missing_fragment_warning_rejection_label": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(18, 105)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_context_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "FrontMatterContextConfig",
    "FrontMatterContextFixtureResult",
    "classify_toc_context",
    "collection_books",
    "validate_front_matter_context_fixture",
]
