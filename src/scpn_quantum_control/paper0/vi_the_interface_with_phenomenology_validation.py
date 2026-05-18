# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 VI. The Interface with Phenomenology validation
"""Source-accounting checks for Paper 0 VI. The Interface with Phenomenology records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded vi the interface with phenomenology source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03269", "P0R03283")


@dataclass(frozen=True, slots=True)
class ViTheInterfaceWithPhenomenologyConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 15
    expected_component_count: int = 3
    next_source_boundary: str = "P0R03284"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 15:
            raise ValueError("expected_source_record_count must equal 15")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R03284":
            raise ValueError("next_source_boundary must equal P0R03284")


@dataclass(frozen=True, slots=True)
class ViTheInterfaceWithPhenomenologyFixtureResult:
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


def classify_vi_the_interface_with_phenomenology_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "vi_the_interface_with_phenomenology": "vi_the_interface_with_phenomenology_source_boundary",
        "vii_the_subtle_energy_network_sen_and_biophotons": "vii_the_subtle_energy_network_sen_and_biophotons_source_boundary",
        "the_foundational_step_consciousness_induced_gravitational_decoherence_ci": "the_foundational_step_consciousness_induced_gravitational_decoherence_ci_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown vi_the_interface_with_phenomenology component") from exc


def vi_the_interface_with_phenomenology_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "VI. The Interface with Phenomenology",
        "source_span": "P0R03269-P0R03283",
        "component_count": "3",
        "next_boundary": "P0R03284",
        "component_1": "VI. The Interface with Phenomenology",
        "component_2": "VII. The Subtle Energy Network (SEN) and Biophotons",
        "component_3": "The Foundational Step: Consciousness-Induced Gravitational Decoherence (CIGD)",
    }


def validate_vi_the_interface_with_phenomenology_fixture(
    config: ViTheInterfaceWithPhenomenologyConfig | None = None,
) -> ViTheInterfaceWithPhenomenologyFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or ViTheInterfaceWithPhenomenologyConfig()
    components = (
        "vi_the_interface_with_phenomenology",
        "vii_the_subtle_energy_network_sen_and_biophotons",
        "the_foundational_step_consciousness_induced_gravitational_decoherence_ci",
    )
    return ViTheInterfaceWithPhenomenologyFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_vi_the_interface_with_phenomenology_component(component)
            for component in components
        },
        labels=vi_the_interface_with_phenomenology_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "vi_the_interface_with_phenomenology_is_not_empirical_validation_evidence": 1.0,
            "vii_the_subtle_energy_network_sen_and_biophotons_is_not_empirical_validation_evidence": 1.0,
            "the_foundational_step_consciousness_induced_gravitational_decoherence_ci_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3269, 3284)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_vi_the_interface_with_phenomenology_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "ViTheInterfaceWithPhenomenologyConfig",
    "ViTheInterfaceWithPhenomenologyFixtureResult",
    "classify_vi_the_interface_with_phenomenology_component",
    "vi_the_interface_with_phenomenology_labels",
    "validate_vi_the_interface_with_phenomenology_fixture",
]
