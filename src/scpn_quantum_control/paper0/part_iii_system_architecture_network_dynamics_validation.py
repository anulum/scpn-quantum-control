# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Part III: System Architecture & Network Dynamics validation
"""Source-accounting checks for Paper 0 Part III: System Architecture & Network Dynamics records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded part iii system architecture network dynamics source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R02011", "P0R02030")


@dataclass(frozen=True, slots=True)
class PartIiiSystemArchitectureNetworkDynamicsConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 20
    expected_component_count: int = 2
    next_source_boundary: str = "P0R02031"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 20:
            raise ValueError("expected_source_record_count must equal 20")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R02031":
            raise ValueError("next_source_boundary must equal P0R02031")


@dataclass(frozen=True, slots=True)
class PartIiiSystemArchitectureNetworkDynamicsFixtureResult:
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


def classify_part_iii_system_architecture_network_dynamics_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "part_iii_system_architecture_network_dynamics": "part_iii_system_architecture_network_dynamics_source_boundary",
        "3_1_the_master_diagram_visualising_the_15_layers_6_domains": "3_1_the_master_diagram_visualising_the_15_layers_6_domains_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown part_iii_system_architecture_network_dynamics component"
        ) from exc


def part_iii_system_architecture_network_dynamics_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Part III: System Architecture & Network Dynamics",
        "source_span": "P0R02011-P0R02030",
        "component_count": "2",
        "next_boundary": "P0R02031",
        "component_1": "Part III: System Architecture & Network Dynamics",
        "component_2": "3.1 The Master Diagram: Visualising the 15 Layers & 6 Domains",
    }


def validate_part_iii_system_architecture_network_dynamics_fixture(
    config: PartIiiSystemArchitectureNetworkDynamicsConfig | None = None,
) -> PartIiiSystemArchitectureNetworkDynamicsFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or PartIiiSystemArchitectureNetworkDynamicsConfig()
    components = (
        "part_iii_system_architecture_network_dynamics",
        "3_1_the_master_diagram_visualising_the_15_layers_6_domains",
    )
    return PartIiiSystemArchitectureNetworkDynamicsFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_part_iii_system_architecture_network_dynamics_component(component)
            for component in components
        },
        labels=part_iii_system_architecture_network_dynamics_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "part_iii_system_architecture_network_dynamics_is_not_empirical_validation_evidence": 1.0,
            "3_1_the_master_diagram_visualising_the_15_layers_6_domains_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(2011, 2031)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_part_iii_system_architecture_network_dynamics_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "PartIiiSystemArchitectureNetworkDynamicsConfig",
    "PartIiiSystemArchitectureNetworkDynamicsFixtureResult",
    "classify_part_iii_system_architecture_network_dynamics_component",
    "part_iii_system_architecture_network_dynamics_labels",
    "validate_part_iii_system_architecture_network_dynamics_fixture",
]
