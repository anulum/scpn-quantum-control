# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Topology & Geometry in Consciousness Models validation
"""Source-accounting checks for Paper 0  Topology & Geometry in Consciousness Models records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded topology geometry in consciousness models source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05876", "P0R05885")


@dataclass(frozen=True, slots=True)
class TopologyGeometryInConsciousnessModelsConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 10
    expected_component_count: int = 3
    next_source_boundary: str = "P0R05886"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 10:
            raise ValueError("expected_source_record_count must equal 10")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R05886":
            raise ValueError("next_source_boundary must equal P0R05886")


@dataclass(frozen=True, slots=True)
class TopologyGeometryInConsciousnessModelsFixtureResult:
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


def classify_topology_geometry_in_consciousness_models_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "topology_geometry_in_consciousness_models": "topology_geometry_in_consciousness_models_source_boundary",
        "source_component": "source_component_source_boundary",
        "cybernetics_control": "cybernetics_control_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown topology_geometry_in_consciousness_models component") from exc


def topology_geometry_in_consciousness_models_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": " Topology & Geometry in Consciousness Models",
        "source_span": "P0R05876-P0R05885",
        "component_count": "3",
        "next_boundary": "P0R05886",
        "component_1": "Topology & Geometry in Consciousness Models",
        "component_2": "",
        "component_3": "Cybernetics & Control",
    }


def validate_topology_geometry_in_consciousness_models_fixture(
    config: TopologyGeometryInConsciousnessModelsConfig | None = None,
) -> TopologyGeometryInConsciousnessModelsFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TopologyGeometryInConsciousnessModelsConfig()
    components = (
        "topology_geometry_in_consciousness_models",
        "source_component",
        "cybernetics_control",
    )
    return TopologyGeometryInConsciousnessModelsFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_topology_geometry_in_consciousness_models_component(component)
            for component in components
        },
        labels=topology_geometry_in_consciousness_models_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "topology_geometry_in_consciousness_models_is_not_empirical_validation_evidence": 1.0,
            "source_component_is_not_empirical_validation_evidence": 1.0,
            "cybernetics_control_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5876, 5886)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_topology_geometry_in_consciousness_models_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TopologyGeometryInConsciousnessModelsConfig",
    "TopologyGeometryInConsciousnessModelsFixtureResult",
    "classify_topology_geometry_in_consciousness_models_component",
    "topology_geometry_in_consciousness_models_labels",
    "validate_topology_geometry_in_consciousness_models_fixture",
]
