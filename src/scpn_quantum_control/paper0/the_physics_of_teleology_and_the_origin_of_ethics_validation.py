# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Physics of Teleology and the Origin of Ethics validation
"""Source-accounting checks for Paper 0 The Physics of Teleology and the Origin of Ethics records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded the physics of teleology and the origin of ethics source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R06088", "P0R06098")


@dataclass(frozen=True, slots=True)
class ThePhysicsOfTeleologyAndTheOriginOfEthicsConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 11
    expected_component_count: int = 3
    next_source_boundary: str = "P0R06099"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 11:
            raise ValueError("expected_source_record_count must equal 11")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R06099":
            raise ValueError("next_source_boundary must equal P0R06099")


@dataclass(frozen=True, slots=True)
class ThePhysicsOfTeleologyAndTheOriginOfEthicsFixtureResult:
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


def classify_the_physics_of_teleology_and_the_origin_of_ethics_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_physics_of_teleology_and_the_origin_of_ethics": "the_physics_of_teleology_and_the_origin_of_ethics_source_boundary",
        "i_the_ontological_origin_of_ethics_gauge_theory_derivation": "i_the_ontological_origin_of_ethics_gauge_theory_derivation_source_boundary",
        "ii_the_principle_of_ethical_least_action_pela": "ii_the_principle_of_ethical_least_action_pela_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown the_physics_of_teleology_and_the_origin_of_ethics component"
        ) from exc


def the_physics_of_teleology_and_the_origin_of_ethics_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Physics of Teleology and the Origin of Ethics",
        "source_span": "P0R06088-P0R06098",
        "component_count": "3",
        "next_boundary": "P0R06099",
        "component_1": "The Physics of Teleology and the Origin of Ethics",
        "component_2": "I. The Ontological Origin of Ethics (Gauge Theory Derivation):",
        "component_3": "II. The Principle of Ethical Least Action (PELA):",
    }


def validate_the_physics_of_teleology_and_the_origin_of_ethics_fixture(
    config: ThePhysicsOfTeleologyAndTheOriginOfEthicsConfig | None = None,
) -> ThePhysicsOfTeleologyAndTheOriginOfEthicsFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or ThePhysicsOfTeleologyAndTheOriginOfEthicsConfig()
    components = (
        "the_physics_of_teleology_and_the_origin_of_ethics",
        "i_the_ontological_origin_of_ethics_gauge_theory_derivation",
        "ii_the_principle_of_ethical_least_action_pela",
    )
    return ThePhysicsOfTeleologyAndTheOriginOfEthicsFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_physics_of_teleology_and_the_origin_of_ethics_component(
                component
            )
            for component in components
        },
        labels=the_physics_of_teleology_and_the_origin_of_ethics_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_physics_of_teleology_and_the_origin_of_ethics_is_not_empirical_validation_evidence": 1.0,
            "i_the_ontological_origin_of_ethics_gauge_theory_derivation_is_not_empirical_validation_evidence": 1.0,
            "ii_the_principle_of_ethical_least_action_pela_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(6088, 6099)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_physics_of_teleology_and_the_origin_of_ethics_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "ThePhysicsOfTeleologyAndTheOriginOfEthicsConfig",
    "ThePhysicsOfTeleologyAndTheOriginOfEthicsFixtureResult",
    "classify_the_physics_of_teleology_and_the_origin_of_ethics_component",
    "the_physics_of_teleology_and_the_origin_of_ethics_labels",
    "validate_the_physics_of_teleology_and_the_origin_of_ethics_fixture",
]
