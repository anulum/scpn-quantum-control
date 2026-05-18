# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Ethics & Teleology validation
"""Source-accounting checks for Paper 0  Ethics & Teleology records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = (
    "source-bounded ethics teleology source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05762", "P0R05769")


@dataclass(frozen=True, slots=True)
class EthicsTeleologyConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 2
    next_source_boundary: str = "P0R05770"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R05770":
            raise ValueError("next_source_boundary must equal P0R05770")


@dataclass(frozen=True, slots=True)
class EthicsTeleologyFixtureResult:
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


def classify_ethics_teleology_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "ethics_teleology": "ethics_teleology_source_boundary",
        "ethics_evolution_systems": "ethics_evolution_systems_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown ethics_teleology component") from exc


def ethics_teleology_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": " Ethics & Teleology",
        "source_span": "P0R05762-P0R05769",
        "component_count": "2",
        "next_boundary": "P0R05770",
        "component_1": "Ethics & Teleology",
        "component_2": "Ethics, Evolution & Systems",
    }


def validate_ethics_teleology_fixture(
    config: EthicsTeleologyConfig | None = None,
) -> EthicsTeleologyFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or EthicsTeleologyConfig()
    components = ("ethics_teleology", "ethics_evolution_systems")
    return EthicsTeleologyFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_ethics_teleology_component(component) for component in components
        },
        labels=ethics_teleology_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "ethics_teleology_is_not_empirical_validation_evidence": 1.0,
            "ethics_evolution_systems_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5762, 5770)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_ethics_teleology_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "EthicsTeleologyConfig",
    "EthicsTeleologyFixtureResult",
    "classify_ethics_teleology_component",
    "ethics_teleology_labels",
    "validate_ethics_teleology_fixture",
]
