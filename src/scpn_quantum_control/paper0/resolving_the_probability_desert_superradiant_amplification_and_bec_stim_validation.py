# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Resolving the Probability Desert: Superradiant Amplification and BEC Stimulated Emission validation
"""Source-accounting checks for Paper 0 Resolving the Probability Desert: Superradiant Amplification and BEC Stimulated Emission records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded resolving the probability desert superradiant amplification and bec stim source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04257", "P0R04272")


@dataclass(frozen=True, slots=True)
class ResolvingTheProbabilityDesertSuperradiantAmplificationAndBecStimConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 16
    expected_component_count: int = 1
    next_source_boundary: str = "P0R04273"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 16:
            raise ValueError("expected_source_record_count must equal 16")
        if self.expected_component_count != 1:
            raise ValueError("expected_component_count must equal 1")
        if self.next_source_boundary != "P0R04273":
            raise ValueError("next_source_boundary must equal P0R04273")


@dataclass(frozen=True, slots=True)
class ResolvingTheProbabilityDesertSuperradiantAmplificationAndBecStimFixtureResult:
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


def classify_resolving_the_probability_desert_superradiant_amplification_and_bec_stim_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "resolving_the_probability_desert_superradiant_amplification_and_bec_stim": "resolving_the_probability_desert_superradiant_amplification_and_bec_stim_source_boundary"
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown resolving_the_probability_desert_superradiant_amplification_and_bec_stim component"
        ) from exc


def resolving_the_probability_desert_superradiant_amplification_and_bec_stim_labels() -> dict[
    str, str
]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Resolving the Probability Desert: Superradiant Amplification and BEC Stimulated Emission",
        "source_span": "P0R04257-P0R04272",
        "component_count": "1",
        "next_boundary": "P0R04273",
        "component_1": "Resolving the Probability Desert: Superradiant Amplification and BEC Stimulated Emission",
    }


def validate_resolving_the_probability_desert_superradiant_amplification_and_bec_stim_fixture(
    config: ResolvingTheProbabilityDesertSuperradiantAmplificationAndBecStimConfig | None = None,
) -> ResolvingTheProbabilityDesertSuperradiantAmplificationAndBecStimFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or ResolvingTheProbabilityDesertSuperradiantAmplificationAndBecStimConfig()
    components = ("resolving_the_probability_desert_superradiant_amplification_and_bec_stim",)
    return ResolvingTheProbabilityDesertSuperradiantAmplificationAndBecStimFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_resolving_the_probability_desert_superradiant_amplification_and_bec_stim_component(
                component
            )
            for component in components
        },
        labels=resolving_the_probability_desert_superradiant_amplification_and_bec_stim_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "resolving_the_probability_desert_superradiant_amplification_and_bec_stim_is_not_empirical_validation_evidence": 1.0
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4257, 4273)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_resolving_the_probability_desert_superradiant_amplification_and_bec_stim_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "ResolvingTheProbabilityDesertSuperradiantAmplificationAndBecStimConfig",
    "ResolvingTheProbabilityDesertSuperradiantAmplificationAndBecStimFixtureResult",
    "classify_resolving_the_probability_desert_superradiant_amplification_and_bec_stim_component",
    "resolving_the_probability_desert_superradiant_amplification_and_bec_stim_labels",
    "validate_resolving_the_probability_desert_superradiant_amplification_and_bec_stim_fixture",
]
