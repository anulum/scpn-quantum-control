# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 A Cascade of Directed Couplings: validation
"""Source-accounting checks for Paper 0 A Cascade of Directed Couplings: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded a cascade of directed couplings source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03954", "P0R03967")


@dataclass(frozen=True, slots=True)
class ACascadeOfDirectedCouplingsConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 14
    expected_component_count: int = 2
    next_source_boundary: str = "P0R03968"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 14:
            raise ValueError("expected_source_record_count must equal 14")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R03968":
            raise ValueError("next_source_boundary must equal P0R03968")


@dataclass(frozen=True, slots=True)
class ACascadeOfDirectedCouplingsFixtureResult:
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


def classify_a_cascade_of_directed_couplings_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "a_cascade_of_directed_couplings": "a_cascade_of_directed_couplings_source_boundary",
        "the_physics_of_teleology_and_the_origin_of_ethics": "the_physics_of_teleology_and_the_origin_of_ethics_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown a_cascade_of_directed_couplings component") from exc


def a_cascade_of_directed_couplings_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "A Cascade of Directed Couplings:",
        "source_span": "P0R03954-P0R03967",
        "component_count": "2",
        "next_boundary": "P0R03968",
        "component_1": "A Cascade of Directed Couplings:",
        "component_2": "The Physics of Teleology and the Origin of Ethics",
    }


def validate_a_cascade_of_directed_couplings_fixture(
    config: ACascadeOfDirectedCouplingsConfig | None = None,
) -> ACascadeOfDirectedCouplingsFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or ACascadeOfDirectedCouplingsConfig()
    components = (
        "a_cascade_of_directed_couplings",
        "the_physics_of_teleology_and_the_origin_of_ethics",
    )
    return ACascadeOfDirectedCouplingsFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_a_cascade_of_directed_couplings_component(component)
            for component in components
        },
        labels=a_cascade_of_directed_couplings_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "a_cascade_of_directed_couplings_is_not_empirical_validation_evidence": 1.0,
            "the_physics_of_teleology_and_the_origin_of_ethics_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3954, 3968)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_a_cascade_of_directed_couplings_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "ACascadeOfDirectedCouplingsConfig",
    "ACascadeOfDirectedCouplingsFixtureResult",
    "classify_a_cascade_of_directed_couplings_component",
    "a_cascade_of_directed_couplings_labels",
    "validate_a_cascade_of_directed_couplings_fixture",
]
