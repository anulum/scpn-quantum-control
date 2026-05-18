# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Universe's Path of Least Resistance validation
"""Source-accounting checks for Paper 0 The Universe's Path of Least Resistance records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded the universe s path of least resistance source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04098", "P0R04114")


@dataclass(frozen=True, slots=True)
class TheUniverseSPathOfLeastResistanceConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 17
    expected_component_count: int = 2
    next_source_boundary: str = "P0R04115"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 17:
            raise ValueError("expected_source_record_count must equal 17")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R04115":
            raise ValueError("next_source_boundary must equal P0R04115")


@dataclass(frozen=True, slots=True)
class TheUniverseSPathOfLeastResistanceFixtureResult:
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


def classify_the_universe_s_path_of_least_resistance_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_universe_s_path_of_least_resistance": "the_universe_s_path_of_least_resistance_source_boundary",
        "seeing_the_bigger_picture_how_all_scales_align": "seeing_the_bigger_picture_how_all_scales_align_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown the_universe_s_path_of_least_resistance component") from exc


def the_universe_s_path_of_least_resistance_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Universe's Path of Least Resistance",
        "source_span": "P0R04098-P0R04114",
        "component_count": "2",
        "next_boundary": "P0R04115",
        "component_1": "The Universe's Path of Least Resistance",
        "component_2": "Seeing the Bigger Picture: How All Scales Align",
    }


def validate_the_universe_s_path_of_least_resistance_fixture(
    config: TheUniverseSPathOfLeastResistanceConfig | None = None,
) -> TheUniverseSPathOfLeastResistanceFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TheUniverseSPathOfLeastResistanceConfig()
    components = (
        "the_universe_s_path_of_least_resistance",
        "seeing_the_bigger_picture_how_all_scales_align",
    )
    return TheUniverseSPathOfLeastResistanceFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_universe_s_path_of_least_resistance_component(component)
            for component in components
        },
        labels=the_universe_s_path_of_least_resistance_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_universe_s_path_of_least_resistance_is_not_empirical_validation_evidence": 1.0,
            "seeing_the_bigger_picture_how_all_scales_align_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4098, 4115)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_universe_s_path_of_least_resistance_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TheUniverseSPathOfLeastResistanceConfig",
    "TheUniverseSPathOfLeastResistanceFixtureResult",
    "classify_the_universe_s_path_of_least_resistance_component",
    "the_universe_s_path_of_least_resistance_labels",
    "validate_the_universe_s_path_of_least_resistance_fixture",
]
