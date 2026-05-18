# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Hypothesis: Qualia as the Geometry of the Consciousness Manifold validation
"""Source-accounting checks for Paper 0 The Hypothesis: Qualia as the Geometry of the Consciousness Manifold records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded the hypothesis qualia as the geometry of the consciousness manifold source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03453", "P0R03461")


@dataclass(frozen=True, slots=True)
class TheHypothesisQualiaAsTheGeometryOfTheConsciousnessManifoldConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 3
    next_source_boundary: str = "P0R03462"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R03462":
            raise ValueError("next_source_boundary must equal P0R03462")


@dataclass(frozen=True, slots=True)
class TheHypothesisQualiaAsTheGeometryOfTheConsciousnessManifoldFixtureResult:
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


def classify_the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold": "the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_source_boundary",
        "meta_framework_integrations": "meta_framework_integrations_source_boundary",
        "predictive_coding_integration": "predictive_coding_integration_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold component"
        ) from exc


def the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Hypothesis: Qualia as the Geometry of the Consciousness Manifold",
        "source_span": "P0R03453-P0R03461",
        "component_count": "3",
        "next_boundary": "P0R03462",
        "component_1": "The Hypothesis: Qualia as the Geometry of the Consciousness Manifold",
        "component_2": "Meta-Framework Integrations",
        "component_3": "Predictive Coding Integration",
    }


def validate_the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_fixture(
    config: TheHypothesisQualiaAsTheGeometryOfTheConsciousnessManifoldConfig | None = None,
) -> TheHypothesisQualiaAsTheGeometryOfTheConsciousnessManifoldFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TheHypothesisQualiaAsTheGeometryOfTheConsciousnessManifoldConfig()
    components = (
        "the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold",
        "meta_framework_integrations",
        "predictive_coding_integration",
    )
    return TheHypothesisQualiaAsTheGeometryOfTheConsciousnessManifoldFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_component(
                component
            )
            for component in components
        },
        labels=the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_is_not_empirical_validation_evidence": 1.0,
            "meta_framework_integrations_is_not_empirical_validation_evidence": 1.0,
            "predictive_coding_integration_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3453, 3462)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TheHypothesisQualiaAsTheGeometryOfTheConsciousnessManifoldConfig",
    "TheHypothesisQualiaAsTheGeometryOfTheConsciousnessManifoldFixtureResult",
    "classify_the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_component",
    "the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_labels",
    "validate_the_hypothesis_qualia_as_the_geometry_of_the_consciousness_manifold_fixture",
]
