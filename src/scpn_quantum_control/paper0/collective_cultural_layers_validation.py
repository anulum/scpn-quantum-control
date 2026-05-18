# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Collective & Cultural Layers validation
"""Source-accounting checks for Paper 0  Collective & Cultural Layers records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = (
    "source-bounded collective cultural layers source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05713", "P0R05720")


@dataclass(frozen=True, slots=True)
class CollectiveCulturalLayersConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 2
    next_source_boundary: str = "P0R05721"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R05721":
            raise ValueError("next_source_boundary must equal P0R05721")


@dataclass(frozen=True, slots=True)
class CollectiveCulturalLayersFixtureResult:
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


def classify_collective_cultural_layers_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "collective_cultural_layers": "collective_cultural_layers_source_boundary",
        "thermodynamics_information": "thermodynamics_information_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown collective_cultural_layers component") from exc


def collective_cultural_layers_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": " Collective & Cultural Layers",
        "source_span": "P0R05713-P0R05720",
        "component_count": "2",
        "next_boundary": "P0R05721",
        "component_1": "Collective & Cultural Layers",
        "component_2": "Thermodynamics & Information",
    }


def validate_collective_cultural_layers_fixture(
    config: CollectiveCulturalLayersConfig | None = None,
) -> CollectiveCulturalLayersFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or CollectiveCulturalLayersConfig()
    components = ("collective_cultural_layers", "thermodynamics_information")
    return CollectiveCulturalLayersFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_collective_cultural_layers_component(component)
            for component in components
        },
        labels=collective_cultural_layers_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "collective_cultural_layers_is_not_empirical_validation_evidence": 1.0,
            "thermodynamics_information_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5713, 5721)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_collective_cultural_layers_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "CollectiveCulturalLayersConfig",
    "CollectiveCulturalLayersFixtureResult",
    "classify_collective_cultural_layers_component",
    "collective_cultural_layers_labels",
    "validate_collective_cultural_layers_fixture",
]
