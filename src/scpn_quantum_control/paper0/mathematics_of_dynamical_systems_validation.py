# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Mathematics of Dynamical Systems validation
"""Source-accounting checks for Paper 0  Mathematics of Dynamical Systems records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded mathematics of dynamical systems source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05826", "P0R05835")


@dataclass(frozen=True, slots=True)
class MathematicsOfDynamicalSystemsConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 10
    expected_component_count: int = 3
    next_source_boundary: str = "P0R05836"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 10:
            raise ValueError("expected_source_record_count must equal 10")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R05836":
            raise ValueError("next_source_boundary must equal P0R05836")


@dataclass(frozen=True, slots=True)
class MathematicsOfDynamicalSystemsFixtureResult:
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


def classify_mathematics_of_dynamical_systems_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "mathematics_of_dynamical_systems": "mathematics_of_dynamical_systems_source_boundary",
        "source_component": "source_component_source_boundary",
        "mathematics_of_geometry_topology": "mathematics_of_geometry_topology_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown mathematics_of_dynamical_systems component") from exc


def mathematics_of_dynamical_systems_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": " Mathematics of Dynamical Systems",
        "source_span": "P0R05826-P0R05835",
        "component_count": "3",
        "next_boundary": "P0R05836",
        "component_1": "Mathematics of Dynamical Systems",
        "component_2": "",
        "component_3": "Mathematics of Geometry & Topology",
    }


def validate_mathematics_of_dynamical_systems_fixture(
    config: MathematicsOfDynamicalSystemsConfig | None = None,
) -> MathematicsOfDynamicalSystemsFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or MathematicsOfDynamicalSystemsConfig()
    components = (
        "mathematics_of_dynamical_systems",
        "source_component",
        "mathematics_of_geometry_topology",
    )
    return MathematicsOfDynamicalSystemsFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_mathematics_of_dynamical_systems_component(component)
            for component in components
        },
        labels=mathematics_of_dynamical_systems_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "mathematics_of_dynamical_systems_is_not_empirical_validation_evidence": 1.0,
            "source_component_is_not_empirical_validation_evidence": 1.0,
            "mathematics_of_geometry_topology_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5826, 5836)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_mathematics_of_dynamical_systems_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "MathematicsOfDynamicalSystemsConfig",
    "MathematicsOfDynamicalSystemsFixtureResult",
    "classify_mathematics_of_dynamical_systems_component",
    "mathematics_of_dynamical_systems_labels",
    "validate_mathematics_of_dynamical_systems_fixture",
]
