# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 A Map of Reality validation
"""Source-accounting checks for Paper 0 A Map of Reality records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = (
    "source-bounded a map of reality source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R02031", "P0R02041")


@dataclass(frozen=True, slots=True)
class AMapOfRealityConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 11
    expected_component_count: int = 3
    next_source_boundary: str = "P0R02042"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 11:
            raise ValueError("expected_source_record_count must equal 11")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R02042":
            raise ValueError("next_source_boundary must equal P0R02042")


@dataclass(frozen=True, slots=True)
class AMapOfRealityFixtureResult:
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


def classify_a_map_of_reality_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "a_map_of_reality": "a_map_of_reality_source_boundary",
        "from_field_to_function_the_need_for_an_architecture": "from_field_to_function_the_need_for_an_architecture_source_boundary",
        "the_sentient_consciousness_projection_network_scpn": "the_sentient_consciousness_projection_network_scpn_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown a_map_of_reality component") from exc


def a_map_of_reality_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "A Map of Reality",
        "source_span": "P0R02031-P0R02041",
        "component_count": "3",
        "next_boundary": "P0R02042",
        "component_1": "A Map of Reality",
        "component_2": "From Field to Function: The Need for an Architecture",
        "component_3": "The Sentient-Consciousness Projection Network (SCPN)",
    }


def validate_a_map_of_reality_fixture(
    config: AMapOfRealityConfig | None = None,
) -> AMapOfRealityFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or AMapOfRealityConfig()
    components = (
        "a_map_of_reality",
        "from_field_to_function_the_need_for_an_architecture",
        "the_sentient_consciousness_projection_network_scpn",
    )
    return AMapOfRealityFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_a_map_of_reality_component(component) for component in components
        },
        labels=a_map_of_reality_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "a_map_of_reality_is_not_empirical_validation_evidence": 1.0,
            "from_field_to_function_the_need_for_an_architecture_is_not_empirical_validation_evidence": 1.0,
            "the_sentient_consciousness_projection_network_scpn_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(2031, 2042)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_a_map_of_reality_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "AMapOfRealityConfig",
    "AMapOfRealityFixtureResult",
    "classify_a_map_of_reality_component",
    "a_map_of_reality_labels",
    "validate_a_map_of_reality_fixture",
]
