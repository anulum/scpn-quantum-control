# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Universe's Built-in Moral Compass validation
"""Source-accounting checks for Paper 0 The Universe's Built-in Moral Compass records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded the universe s built in moral compass source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03715", "P0R03722")


@dataclass(frozen=True, slots=True)
class TheUniverseSBuiltInMoralCompassConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 3
    next_source_boundary: str = "P0R03723"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R03723":
            raise ValueError("next_source_boundary must equal P0R03723")


@dataclass(frozen=True, slots=True)
class TheUniverseSBuiltInMoralCompassFixtureResult:
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


def classify_the_universe_s_built_in_moral_compass_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_universe_s_built_in_moral_compass": "the_universe_s_built_in_moral_compass_source_boundary",
        "the_pull_of_the_future_how_purpose_guides_the_present": "the_pull_of_the_future_how_purpose_guides_the_present_source_boundary",
        "formalisation_of_the_causal_entropic_principle": "formalisation_of_the_causal_entropic_principle_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown the_universe_s_built_in_moral_compass component") from exc


def the_universe_s_built_in_moral_compass_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Universe's Built-in Moral Compass",
        "source_span": "P0R03715-P0R03722",
        "component_count": "3",
        "next_boundary": "P0R03723",
        "component_1": "The Universe's Built-in Moral Compass",
        "component_2": "The Pull of the Future: How Purpose Guides the Present",
        "component_3": "Formalisation of the Causal Entropic Principle:",
    }


def validate_the_universe_s_built_in_moral_compass_fixture(
    config: TheUniverseSBuiltInMoralCompassConfig | None = None,
) -> TheUniverseSBuiltInMoralCompassFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TheUniverseSBuiltInMoralCompassConfig()
    components = (
        "the_universe_s_built_in_moral_compass",
        "the_pull_of_the_future_how_purpose_guides_the_present",
        "formalisation_of_the_causal_entropic_principle",
    )
    return TheUniverseSBuiltInMoralCompassFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_universe_s_built_in_moral_compass_component(component)
            for component in components
        },
        labels=the_universe_s_built_in_moral_compass_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_universe_s_built_in_moral_compass_is_not_empirical_validation_evidence": 1.0,
            "the_pull_of_the_future_how_purpose_guides_the_present_is_not_empirical_validation_evidence": 1.0,
            "formalisation_of_the_causal_entropic_principle_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3715, 3723)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_universe_s_built_in_moral_compass_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TheUniverseSBuiltInMoralCompassConfig",
    "TheUniverseSBuiltInMoralCompassFixtureResult",
    "classify_the_universe_s_built_in_moral_compass_component",
    "the_universe_s_built_in_moral_compass_labels",
    "validate_the_universe_s_built_in_moral_compass_fixture",
]
