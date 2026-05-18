# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Functional Implications: validation
"""Source-accounting checks for Paper 0 Functional Implications: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = (
    "source-bounded functional implications source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R02859", "P0R02868")


@dataclass(frozen=True, slots=True)
class FunctionalImplicationsConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 10
    expected_component_count: int = 3
    next_source_boundary: str = "P0R02869"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 10:
            raise ValueError("expected_source_record_count must equal 10")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R02869":
            raise ValueError("next_source_boundary must equal P0R02869")


@dataclass(frozen=True, slots=True)
class FunctionalImplicationsFixtureResult:
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


def classify_functional_implications_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "functional_implications": "functional_implications_source_boundary",
        "maximised_information_capacity": "maximised_information_capacity_source_boundary",
        "efficient_communication": "efficient_communication_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown functional_implications component") from exc


def functional_implications_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Functional Implications:",
        "source_span": "P0R02859-P0R02868",
        "component_count": "3",
        "next_boundary": "P0R02869",
        "component_1": "Functional Implications:",
        "component_2": "Maximised Information Capacity:",
        "component_3": "Efficient Communication:",
    }


def validate_functional_implications_fixture(
    config: FunctionalImplicationsConfig | None = None,
) -> FunctionalImplicationsFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or FunctionalImplicationsConfig()
    components = (
        "functional_implications",
        "maximised_information_capacity",
        "efficient_communication",
    )
    return FunctionalImplicationsFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_functional_implications_component(component)
            for component in components
        },
        labels=functional_implications_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "functional_implications_is_not_empirical_validation_evidence": 1.0,
            "maximised_information_capacity_is_not_empirical_validation_evidence": 1.0,
            "efficient_communication_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(2859, 2869)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_functional_implications_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "FunctionalImplicationsConfig",
    "FunctionalImplicationsFixtureResult",
    "classify_functional_implications_component",
    "functional_implications_labels",
    "validate_functional_implications_fixture",
]
