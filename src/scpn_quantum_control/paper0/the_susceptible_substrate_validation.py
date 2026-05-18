# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Susceptible Substrate: validation
"""Source-accounting checks for Paper 0 The Susceptible Substrate: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = (
    "source-bounded the susceptible substrate source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R02848", "P0R02858")


@dataclass(frozen=True, slots=True)
class TheSusceptibleSubstrateConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 11
    expected_component_count: int = 4
    next_source_boundary: str = "P0R02859"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 11:
            raise ValueError("expected_source_record_count must equal 11")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R02859":
            raise ValueError("next_source_boundary must equal P0R02859")


@dataclass(frozen=True, slots=True)
class TheSusceptibleSubstrateFixtureResult:
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


def classify_the_susceptible_substrate_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_susceptible_substrate": "the_susceptible_substrate_source_boundary",
        "the_branching_parameter_as_sigma": "the_branching_parameter_as_sigma_source_boundary",
        "overarching_dynamic_principles": "overarching_dynamic_principles_source_boundary",
        "a_the_universal_dynamic_regime_quasicriticality": "a_the_universal_dynamic_regime_quasicriticality_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown the_susceptible_substrate component") from exc


def the_susceptible_substrate_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Susceptible Substrate:",
        "source_span": "P0R02848-P0R02858",
        "component_count": "4",
        "next_boundary": "P0R02859",
        "component_1": "The Susceptible Substrate:",
        "component_2": "The Branching Parameter as sigma:",
        "component_3": "Overarching Dynamic Principles",
        "component_4": "A The Universal Dynamic Regime: Quasicriticality",
    }


def validate_the_susceptible_substrate_fixture(
    config: TheSusceptibleSubstrateConfig | None = None,
) -> TheSusceptibleSubstrateFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TheSusceptibleSubstrateConfig()
    components = (
        "the_susceptible_substrate",
        "the_branching_parameter_as_sigma",
        "overarching_dynamic_principles",
        "a_the_universal_dynamic_regime_quasicriticality",
    )
    return TheSusceptibleSubstrateFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_susceptible_substrate_component(component)
            for component in components
        },
        labels=the_susceptible_substrate_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_susceptible_substrate_is_not_empirical_validation_evidence": 1.0,
            "the_branching_parameter_as_sigma_is_not_empirical_validation_evidence": 1.0,
            "overarching_dynamic_principles_is_not_empirical_validation_evidence": 1.0,
            "a_the_universal_dynamic_regime_quasicriticality_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(2848, 2859)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_susceptible_substrate_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TheSusceptibleSubstrateConfig",
    "TheSusceptibleSubstrateFixtureResult",
    "classify_the_susceptible_substrate_component",
    "the_susceptible_substrate_labels",
    "validate_the_susceptible_substrate_fixture",
]
