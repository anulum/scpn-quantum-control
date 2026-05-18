# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Dynamics and Evolution of the Embodied SCPN validation
"""Source-accounting checks for Paper 0 The Dynamics and Evolution of the Embodied SCPN records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded the dynamics and evolution of the embodied scpn source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04956", "P0R04966")


@dataclass(frozen=True, slots=True)
class TheDynamicsAndEvolutionOfTheEmbodiedScpnConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 11
    expected_component_count: int = 3
    next_source_boundary: str = "P0R04967"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 11:
            raise ValueError("expected_source_record_count must equal 11")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R04967":
            raise ValueError("next_source_boundary must equal P0R04967")


@dataclass(frozen=True, slots=True)
class TheDynamicsAndEvolutionOfTheEmbodiedScpnFixtureResult:
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


def classify_the_dynamics_and_evolution_of_the_embodied_scpn_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_dynamics_and_evolution_of_the_embodied_scpn": "the_dynamics_and_evolution_of_the_embodied_scpn_source_boundary",
        "i_the_dynamics_of_consciousness_in_the_brain_body_system": "i_the_dynamics_of_consciousness_in_the_brain_body_system_source_boundary",
        "1_the_embodied_upde_the_symphony_of_the_self": "1_the_embodied_upde_the_symphony_of_the_self_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown the_dynamics_and_evolution_of_the_embodied_scpn component"
        ) from exc


def the_dynamics_and_evolution_of_the_embodied_scpn_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Dynamics and Evolution of the Embodied SCPN",
        "source_span": "P0R04956-P0R04966",
        "component_count": "3",
        "next_boundary": "P0R04967",
        "component_1": "The Dynamics and Evolution of the Embodied SCPN",
        "component_2": "I. The Dynamics of Consciousness in the Brain-Body System",
        "component_3": "1. The Embodied UPDE (The Symphony of the Self)",
    }


def validate_the_dynamics_and_evolution_of_the_embodied_scpn_fixture(
    config: TheDynamicsAndEvolutionOfTheEmbodiedScpnConfig | None = None,
) -> TheDynamicsAndEvolutionOfTheEmbodiedScpnFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TheDynamicsAndEvolutionOfTheEmbodiedScpnConfig()
    components = (
        "the_dynamics_and_evolution_of_the_embodied_scpn",
        "i_the_dynamics_of_consciousness_in_the_brain_body_system",
        "1_the_embodied_upde_the_symphony_of_the_self",
    )
    return TheDynamicsAndEvolutionOfTheEmbodiedScpnFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_dynamics_and_evolution_of_the_embodied_scpn_component(
                component
            )
            for component in components
        },
        labels=the_dynamics_and_evolution_of_the_embodied_scpn_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_dynamics_and_evolution_of_the_embodied_scpn_is_not_empirical_validation_evidence": 1.0,
            "i_the_dynamics_of_consciousness_in_the_brain_body_system_is_not_empirical_validation_evidence": 1.0,
            "1_the_embodied_upde_the_symphony_of_the_self_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4956, 4967)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_dynamics_and_evolution_of_the_embodied_scpn_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TheDynamicsAndEvolutionOfTheEmbodiedScpnConfig",
    "TheDynamicsAndEvolutionOfTheEmbodiedScpnFixtureResult",
    "classify_the_dynamics_and_evolution_of_the_embodied_scpn_component",
    "the_dynamics_and_evolution_of_the_embodied_scpn_labels",
    "validate_the_dynamics_and_evolution_of_the_embodied_scpn_fixture",
]
