# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 1. Psychedelics (The Expanded Manifold): validation
"""Source-accounting checks for Paper 0 1. Psychedelics (The Expanded Manifold): records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 1 psychedelics the expanded manifold source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05026", "P0R05038")


@dataclass(frozen=True, slots=True)
class Section1PsychedelicsTheExpandedManifoldConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 13
    expected_component_count: int = 4
    next_source_boundary: str = "P0R05039"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 13:
            raise ValueError("expected_source_record_count must equal 13")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R05039":
            raise ValueError("next_source_boundary must equal P0R05039")


@dataclass(frozen=True, slots=True)
class Section1PsychedelicsTheExpandedManifoldFixtureResult:
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


def classify_section_1_psychedelics_the_expanded_manifold_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "1_psychedelics_the_expanded_manifold": "1_psychedelics_the_expanded_manifold_source_boundary",
        "2_meditation_and_flow_states_optimised_criticality": "2_meditation_and_flow_states_optimised_criticality_source_boundary",
        "3_anaesthesia_the_decoupling": "3_anaesthesia_the_decoupling_source_boundary",
        "vii_the_embodied_brain_and_the_extended_environment_l6_coupling": "vii_the_embodied_brain_and_the_extended_environment_l6_coupling_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown section_1_psychedelics_the_expanded_manifold component") from exc


def section_1_psychedelics_the_expanded_manifold_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "1. Psychedelics (The Expanded Manifold):",
        "source_span": "P0R05026-P0R05038",
        "component_count": "4",
        "next_boundary": "P0R05039",
        "component_1": "1. Psychedelics (The Expanded Manifold):",
        "component_2": "2. Meditation and Flow States (Optimised Criticality):",
        "component_3": "3. Anaesthesia (The Decoupling):",
        "component_4": "VII. The Embodied Brain and the Extended Environment (L6 Coupling)",
    }


def validate_section_1_psychedelics_the_expanded_manifold_fixture(
    config: Section1PsychedelicsTheExpandedManifoldConfig | None = None,
) -> Section1PsychedelicsTheExpandedManifoldFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section1PsychedelicsTheExpandedManifoldConfig()
    components = (
        "1_psychedelics_the_expanded_manifold",
        "2_meditation_and_flow_states_optimised_criticality",
        "3_anaesthesia_the_decoupling",
        "vii_the_embodied_brain_and_the_extended_environment_l6_coupling",
    )
    return Section1PsychedelicsTheExpandedManifoldFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_1_psychedelics_the_expanded_manifold_component(component)
            for component in components
        },
        labels=section_1_psychedelics_the_expanded_manifold_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "1_psychedelics_the_expanded_manifold_is_not_empirical_validation_evidence": 1.0,
            "2_meditation_and_flow_states_optimised_criticality_is_not_empirical_validation_evidence": 1.0,
            "3_anaesthesia_the_decoupling_is_not_empirical_validation_evidence": 1.0,
            "vii_the_embodied_brain_and_the_extended_environment_l6_coupling_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5026, 5039)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_1_psychedelics_the_expanded_manifold_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section1PsychedelicsTheExpandedManifoldConfig",
    "Section1PsychedelicsTheExpandedManifoldFixtureResult",
    "classify_section_1_psychedelics_the_expanded_manifold_component",
    "section_1_psychedelics_the_expanded_manifold_labels",
    "validate_section_1_psychedelics_the_expanded_manifold_fixture",
]
