# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 sigma is the Q-ball Soliton: validation
"""Source-accounting checks for Paper 0 sigma is the Q-ball Soliton: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = (
    "source-bounded sigma is the q ball soliton source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R01812", "P0R01819")


@dataclass(frozen=True, slots=True)
class SigmaIsTheQBallSolitonConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 4
    next_source_boundary: str = "P0R01820"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R01820":
            raise ValueError("next_source_boundary must equal P0R01820")


@dataclass(frozen=True, slots=True)
class SigmaIsTheQBallSolitonFixtureResult:
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


def classify_sigma_is_the_q_ball_soliton_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "sigma_is_the_q_ball_soliton": "sigma_is_the_q_ball_soliton_source_boundary",
        "why_the_q_ball_is_the_perfect_sigma": "why_the_q_ball_is_the_perfect_sigma_source_boundary",
        "collective": "collective_source_boundary",
        "stable": "stable_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown sigma_is_the_q_ball_soliton component") from exc


def sigma_is_the_q_ball_soliton_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "sigma is the Q-ball Soliton:",
        "source_span": "P0R01812-P0R01819",
        "component_count": "4",
        "next_boundary": "P0R01820",
        "component_1": "sigma is the Q-ball Soliton:",
        "component_2": "Why the Q-ball is the Perfect sigma:",
        "component_3": "Collective:",
        "component_4": "Stable:",
    }


def validate_sigma_is_the_q_ball_soliton_fixture(
    config: SigmaIsTheQBallSolitonConfig | None = None,
) -> SigmaIsTheQBallSolitonFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or SigmaIsTheQBallSolitonConfig()
    components = (
        "sigma_is_the_q_ball_soliton",
        "why_the_q_ball_is_the_perfect_sigma",
        "collective",
        "stable",
    )
    return SigmaIsTheQBallSolitonFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_sigma_is_the_q_ball_soliton_component(component)
            for component in components
        },
        labels=sigma_is_the_q_ball_soliton_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "sigma_is_the_q_ball_soliton_is_not_empirical_validation_evidence": 1.0,
            "why_the_q_ball_is_the_perfect_sigma_is_not_empirical_validation_evidence": 1.0,
            "collective_is_not_empirical_validation_evidence": 1.0,
            "stable_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(1812, 1820)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_sigma_is_the_q_ball_soliton_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "SigmaIsTheQBallSolitonConfig",
    "SigmaIsTheQBallSolitonFixtureResult",
    "classify_sigma_is_the_q_ball_soliton_component",
    "sigma_is_the_q_ball_soliton_labels",
    "validate_sigma_is_the_q_ball_soliton_fixture",
]
