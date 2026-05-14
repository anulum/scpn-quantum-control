# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 chapter roadmap context validation
"""Executable chapter-roadmap context boundary checks for Paper 0."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded chapter roadmap context; not validation evidence"
HARDWARE_STATUS = "source_context_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R00105", "P0R00217")


@dataclass(frozen=True, slots=True)
class ChapterRoadmapContextConfig:
    """Configuration for the Paper 0 chapter roadmap context fixture."""

    expected_part_count: int = 5
    expected_chapter_count: int = 18
    expected_blank_marker_count: int = 5

    def __post_init__(self) -> None:
        if self.expected_part_count != 5:
            raise ValueError("expected_part_count must equal 5")
        if self.expected_chapter_count != 18:
            raise ValueError("expected_chapter_count must equal 18")
        if self.expected_blank_marker_count < 0:
            raise ValueError("expected_blank_marker_count must be non-negative")


@dataclass(frozen=True, slots=True)
class ChapterRoadmapContextFixtureResult:
    """Result for the Paper 0 chapter roadmap context fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    chapter_roadmap: tuple[str, ...]
    part_count: int
    chapter_count: int
    blank_marker_count: int
    numbering_inconsistency_present: bool
    context_classification: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def chapter_roadmap_entries() -> tuple[str, ...]:
    """Return the 18 source-listed chapter roadmap entries in ledger order."""
    return (
        "Chapter 1: The Invitation - Field Architecture and the Anulum Framework",
        "Chapter 2: The Logos - The Three Foundational Axioms of Reality",
        "Chapter 3: The Universal Grammar of Interaction",
        "Chapter 4: The Substance of Being - A Tripartite Ontology of Information",
        "Chapter 5: The Gauge Principle - Deriving the Master Interaction Lagrangian",
        "Chapter 6: The Geometry of Interaction - The Fisher Information Metric",
        "Chapter 7: The Psi-Field-Electromagnetic Interface",
        "Chapter 8: The Physics of Form - Spontaneous Symmetry Breaking and the Solitonic Self",
        "Chapter 9: The Master Diagram - The 15 Layers and 6 Domains",
        "Chapter 10: The Spine of the Network - The Unified Phase Dynamics Equation",
        "Chapter 11: The Universal Dynamic Regime - Quasicriticality and Self-Organisation",
        "Chapter 12: The Coherence Backbone - Multi-Scale Quantum Error Correction",
        "Chapter 13: The Cosmic Algorithm - Hierarchical Predictive Coding and Free Energy",
        "Chapter 14: The Shape of Feeling - The Geometric Qualia Hypothesis",
        "Chapter 15: The Origin of Purpose - Causal Entropic Forces",
        "Chapter 16: The Cosmic Compass - The Ethical Functional and the Consilium",
        "Chapter 17: The Strange Loop of Closure - Recursive Optimisation Hamiltonian",
        "Chapter 18: Falsifiable Predictions and Empirical Trajectories",
    )


def classify_chapter_roadmap_context(
    *,
    blank_marker_count: int,
    numbering_inconsistency_present: bool,
    requested_status: str,
) -> str:
    """Classify chapter roadmap records without allowing evidence promotion."""
    if blank_marker_count < 0:
        raise ValueError("blank_marker_count cannot be negative")
    if requested_status == "validation_evidence":
        raise ValueError("chapter roadmap cannot be promoted as validation evidence")
    if numbering_inconsistency_present:
        return "numbering_inconsistent_roadmap_context"
    if blank_marker_count:
        return "placeholder_roadmap_context"
    return "roadmap_context"


def validate_chapter_roadmap_context_fixture(
    config: ChapterRoadmapContextConfig | None = None,
) -> ChapterRoadmapContextFixtureResult:
    """Validate source accounting for the Paper 0 chapter roadmap context run."""
    cfg = config or ChapterRoadmapContextConfig()
    roadmap = chapter_roadmap_entries()
    classification = classify_chapter_roadmap_context(
        blank_marker_count=cfg.expected_blank_marker_count,
        numbering_inconsistency_present=True,
        requested_status="context_boundary",
    )
    if len(roadmap) != cfg.expected_chapter_count:
        raise ValueError("chapter roadmap count does not match Paper 0 ToC")

    return ChapterRoadmapContextFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        chapter_roadmap=roadmap,
        part_count=cfg.expected_part_count,
        chapter_count=len(roadmap),
        blank_marker_count=cfg.expected_blank_marker_count,
        numbering_inconsistency_present=True,
        context_classification=classification,
        null_controls={
            "roadmap_as_validation_evidence_rejection_label": 1.0,
            "toc_heading_as_equation_rejection_label": 1.0,
            "missing_numbering_inconsistency_rejection_label": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(105, 218)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_context_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "ChapterRoadmapContextConfig",
    "ChapterRoadmapContextFixtureResult",
    "chapter_roadmap_entries",
    "classify_chapter_roadmap_context",
    "validate_chapter_roadmap_context_fixture",
]
