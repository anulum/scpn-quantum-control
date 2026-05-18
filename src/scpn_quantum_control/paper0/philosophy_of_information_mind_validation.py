# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Philosophy of Information / Mind validation
"""Source-accounting checks for Paper 0  Philosophy of Information / Mind records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded philosophy of information mind source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05778", "P0R05785")


@dataclass(frozen=True, slots=True)
class PhilosophyOfInformationMindConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 2
    next_source_boundary: str = "P0R05786"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R05786":
            raise ValueError("next_source_boundary must equal P0R05786")


@dataclass(frozen=True, slots=True)
class PhilosophyOfInformationMindFixtureResult:
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


def classify_philosophy_of_information_mind_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "philosophy_of_information_mind": "philosophy_of_information_mind_source_boundary",
        "philosophy_of_science_method": "philosophy_of_science_method_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown philosophy_of_information_mind component") from exc


def philosophy_of_information_mind_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": " Philosophy of Information / Mind",
        "source_span": "P0R05778-P0R05785",
        "component_count": "2",
        "next_boundary": "P0R05786",
        "component_1": "Philosophy of Information / Mind",
        "component_2": "Philosophy of Science & Method",
    }


def validate_philosophy_of_information_mind_fixture(
    config: PhilosophyOfInformationMindConfig | None = None,
) -> PhilosophyOfInformationMindFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or PhilosophyOfInformationMindConfig()
    components = ("philosophy_of_information_mind", "philosophy_of_science_method")
    return PhilosophyOfInformationMindFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_philosophy_of_information_mind_component(component)
            for component in components
        },
        labels=philosophy_of_information_mind_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "philosophy_of_information_mind_is_not_empirical_validation_evidence": 1.0,
            "philosophy_of_science_method_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5778, 5786)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_philosophy_of_information_mind_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "PhilosophyOfInformationMindConfig",
    "PhilosophyOfInformationMindFixtureResult",
    "classify_philosophy_of_information_mind_component",
    "philosophy_of_information_mind_labels",
    "validate_philosophy_of_information_mind_fixture",
]
