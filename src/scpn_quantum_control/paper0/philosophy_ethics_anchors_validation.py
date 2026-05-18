# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Philosophy & Ethics Anchors validation
"""Source-accounting checks for Paper 0  Philosophy & Ethics Anchors records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = (
    "source-bounded philosophy ethics anchors source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05770", "P0R05777")


@dataclass(frozen=True, slots=True)
class PhilosophyEthicsAnchorsConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 2
    next_source_boundary: str = "P0R05778"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R05778":
            raise ValueError("next_source_boundary must equal P0R05778")


@dataclass(frozen=True, slots=True)
class PhilosophyEthicsAnchorsFixtureResult:
    """Result for this Paper 0 source-accounting fixture."""

    source_ledger_span: tuple[str, str]
    hardware_status: str
    claim_boundary: str
    components: dict[str, str]
    labels: dict[str, str]
    source_record_count: int
    component_count: int
    next_source_boundary: str
    null_controls: dict[str, float]
    problem_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result dictionary."""
        return asdict(self)


def classify_philosophy_ethics_anchors_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "philosophy_ethics_anchors": "philosophy_ethics_anchors_source_boundary",
        "philosophy_of_science_methodology": "philosophy_of_science_methodology_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown philosophy_ethics_anchors component") from exc


def philosophy_ethics_anchors_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": " Philosophy & Ethics Anchors",
        "source_span": "P0R05770-P0R05777",
        "component_count": "2",
        "next_boundary": "P0R05778",
        "component_1": "Philosophy & Ethics Anchors",
        "component_2": "Philosophy of Science & Methodology",
    }


def validate_philosophy_ethics_anchors_fixture(
    config: PhilosophyEthicsAnchorsConfig | None = None,
) -> PhilosophyEthicsAnchorsFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or PhilosophyEthicsAnchorsConfig()
    components = ("philosophy_ethics_anchors", "philosophy_of_science_methodology")
    return PhilosophyEthicsAnchorsFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_philosophy_ethics_anchors_component(component)
            for component in components
        },
        labels=philosophy_ethics_anchors_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "philosophy_ethics_anchors_is_not_empirical_validation_evidence": 1.0,
            "philosophy_of_science_methodology_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5770, 5778)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_philosophy_ethics_anchors_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "PhilosophyEthicsAnchorsConfig",
    "PhilosophyEthicsAnchorsFixtureResult",
    "classify_philosophy_ethics_anchors_component",
    "philosophy_ethics_anchors_labels",
    "validate_philosophy_ethics_anchors_fixture",
]
