# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Complexity Economics & Social Physics validation
"""Source-accounting checks for Paper 0  Complexity Economics & Social Physics records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded complexity economics social physics source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05886", "P0R05893")


@dataclass(frozen=True, slots=True)
class ComplexityEconomicsSocialPhysicsConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 2
    next_source_boundary: str = "P0R05894"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R05894":
            raise ValueError("next_source_boundary must equal P0R05894")


@dataclass(frozen=True, slots=True)
class ComplexityEconomicsSocialPhysicsFixtureResult:
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


def classify_complexity_economics_social_physics_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "complexity_economics_social_physics": "complexity_economics_social_physics_source_boundary",
        "biosemiotics_meaning": "biosemiotics_meaning_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown complexity_economics_social_physics component") from exc


def complexity_economics_social_physics_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": " Complexity Economics & Social Physics",
        "source_span": "P0R05886-P0R05893",
        "component_count": "2",
        "next_boundary": "P0R05894",
        "component_1": "Complexity Economics & Social Physics",
        "component_2": "Biosemiotics & Meaning",
    }


def validate_complexity_economics_social_physics_fixture(
    config: ComplexityEconomicsSocialPhysicsConfig | None = None,
) -> ComplexityEconomicsSocialPhysicsFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or ComplexityEconomicsSocialPhysicsConfig()
    components = ("complexity_economics_social_physics", "biosemiotics_meaning")
    return ComplexityEconomicsSocialPhysicsFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_complexity_economics_social_physics_component(component)
            for component in components
        },
        labels=complexity_economics_social_physics_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "complexity_economics_social_physics_is_not_empirical_validation_evidence": 1.0,
            "biosemiotics_meaning_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5886, 5894)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_complexity_economics_social_physics_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "ComplexityEconomicsSocialPhysicsConfig",
    "ComplexityEconomicsSocialPhysicsFixtureResult",
    "classify_complexity_economics_social_physics_component",
    "complexity_economics_social_physics_labels",
    "validate_complexity_economics_social_physics_fixture",
]
