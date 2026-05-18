# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Mechanism and Bidirectional Causality validation
"""Source-accounting checks for Paper 0 Mechanism and Bidirectional Causality records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded mechanism and bidirectional causality source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04404", "P0R04412")


@dataclass(frozen=True, slots=True)
class MechanismAndBidirectionalCausalityConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 3
    next_source_boundary: str = "P0R04413"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R04413":
            raise ValueError("next_source_boundary must equal P0R04413")


@dataclass(frozen=True, slots=True)
class MechanismAndBidirectionalCausalityFixtureResult:
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


def classify_mechanism_and_bidirectional_causality_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "mechanism_and_bidirectional_causality": "mechanism_and_bidirectional_causality_source_boundary",
        "iv_the_geometry_of_networks_and_dynamics_domain_i_l4": "iv_the_geometry_of_networks_and_dynamics_domain_i_l4_source_boundary",
        "1_the_connectome_topology_the_optimised_scaffold": "1_the_connectome_topology_the_optimised_scaffold_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown mechanism_and_bidirectional_causality component") from exc


def mechanism_and_bidirectional_causality_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Mechanism and Bidirectional Causality",
        "source_span": "P0R04404-P0R04412",
        "component_count": "3",
        "next_boundary": "P0R04413",
        "component_1": "Mechanism and Bidirectional Causality",
        "component_2": "IV. The Geometry of Networks and Dynamics (Domain I: L4)",
        "component_3": "1. The Connectome Topology (The Optimised Scaffold):",
    }


def validate_mechanism_and_bidirectional_causality_fixture(
    config: MechanismAndBidirectionalCausalityConfig | None = None,
) -> MechanismAndBidirectionalCausalityFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or MechanismAndBidirectionalCausalityConfig()
    components = (
        "mechanism_and_bidirectional_causality",
        "iv_the_geometry_of_networks_and_dynamics_domain_i_l4",
        "1_the_connectome_topology_the_optimised_scaffold",
    )
    return MechanismAndBidirectionalCausalityFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_mechanism_and_bidirectional_causality_component(component)
            for component in components
        },
        labels=mechanism_and_bidirectional_causality_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "mechanism_and_bidirectional_causality_is_not_empirical_validation_evidence": 1.0,
            "iv_the_geometry_of_networks_and_dynamics_domain_i_l4_is_not_empirical_validation_evidence": 1.0,
            "1_the_connectome_topology_the_optimised_scaffold_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4404, 4413)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_mechanism_and_bidirectional_causality_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "MechanismAndBidirectionalCausalityConfig",
    "MechanismAndBidirectionalCausalityFixtureResult",
    "classify_mechanism_and_bidirectional_causality_component",
    "mechanism_and_bidirectional_causality_labels",
    "validate_mechanism_and_bidirectional_causality_fixture",
]
