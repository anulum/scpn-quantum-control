# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Mathematical Bridge: validation
"""Source-accounting checks for Paper 0 Mathematical Bridge: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = (
    "source-bounded mathematical bridge source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03427", "P0R03439")


@dataclass(frozen=True, slots=True)
class MathematicalBridgeConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 13
    expected_component_count: int = 2
    next_source_boundary: str = "P0R03440"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 13:
            raise ValueError("expected_source_record_count must equal 13")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R03440":
            raise ValueError("next_source_boundary must equal P0R03440")


@dataclass(frozen=True, slots=True)
class MathematicalBridgeFixtureResult:
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


def classify_mathematical_bridge_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "mathematical_bridge": "mathematical_bridge_source_boundary",
        "why_there_s_something_it_s_like": "why_there_s_something_it_s_like_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown mathematical_bridge component") from exc


def mathematical_bridge_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Mathematical Bridge:",
        "source_span": "P0R03427-P0R03439",
        "component_count": "2",
        "next_boundary": "P0R03440",
        "component_1": "Mathematical Bridge:",
        "component_2": "Why There's Something It's Like:",
    }


def validate_mathematical_bridge_fixture(
    config: MathematicalBridgeConfig | None = None,
) -> MathematicalBridgeFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or MathematicalBridgeConfig()
    components = ("mathematical_bridge", "why_there_s_something_it_s_like")
    return MathematicalBridgeFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_mathematical_bridge_component(component)
            for component in components
        },
        labels=mathematical_bridge_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "mathematical_bridge_is_not_empirical_validation_evidence": 1.0,
            "why_there_s_something_it_s_like_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3427, 3440)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_mathematical_bridge_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "MathematicalBridgeConfig",
    "MathematicalBridgeFixtureResult",
    "classify_mathematical_bridge_component",
    "mathematical_bridge_labels",
    "validate_mathematical_bridge_fixture",
]
