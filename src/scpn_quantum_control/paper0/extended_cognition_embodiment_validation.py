# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Extended Cognition & Embodiment validation
"""Source-accounting checks for Paper 0  Extended Cognition & Embodiment records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded extended cognition embodiment source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05894", "P0R05901")


@dataclass(frozen=True, slots=True)
class ExtendedCognitionEmbodimentConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 2
    next_source_boundary: str = "P0R05902"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R05902":
            raise ValueError("next_source_boundary must equal P0R05902")


@dataclass(frozen=True, slots=True)
class ExtendedCognitionEmbodimentFixtureResult:
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


def classify_extended_cognition_embodiment_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "extended_cognition_embodiment": "extended_cognition_embodiment_source_boundary",
        "information_geometry_statistical_foundations": "information_geometry_statistical_foundations_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown extended_cognition_embodiment component") from exc


def extended_cognition_embodiment_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": " Extended Cognition & Embodiment",
        "source_span": "P0R05894-P0R05901",
        "component_count": "2",
        "next_boundary": "P0R05902",
        "component_1": "Extended Cognition & Embodiment",
        "component_2": "Information Geometry & Statistical Foundations",
    }


def validate_extended_cognition_embodiment_fixture(
    config: ExtendedCognitionEmbodimentConfig | None = None,
) -> ExtendedCognitionEmbodimentFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or ExtendedCognitionEmbodimentConfig()
    components = ("extended_cognition_embodiment", "information_geometry_statistical_foundations")
    return ExtendedCognitionEmbodimentFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_extended_cognition_embodiment_component(component)
            for component in components
        },
        labels=extended_cognition_embodiment_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "extended_cognition_embodiment_is_not_empirical_validation_evidence": 1.0,
            "information_geometry_statistical_foundations_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5894, 5902)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_extended_cognition_embodiment_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "ExtendedCognitionEmbodimentConfig",
    "ExtendedCognitionEmbodimentFixtureResult",
    "classify_extended_cognition_embodiment_component",
    "extended_cognition_embodiment_labels",
    "validate_extended_cognition_embodiment_fixture",
]
