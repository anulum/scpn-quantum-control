# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 VIII. The Synthesis of Subjectivity (The Triadic Solution) validation
"""Source-accounting checks for Paper 0 VIII. The Synthesis of Subjectivity (The Triadic Solution) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded viii the synthesis of subjectivity the triadic solution source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R06132", "P0R06146")


@dataclass(frozen=True, slots=True)
class ViiiTheSynthesisOfSubjectivityTheTriadicSolutionConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 15
    expected_component_count: int = 4
    next_source_boundary: str = "P0R06147"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 15:
            raise ValueError("expected_source_record_count must equal 15")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R06147":
            raise ValueError("next_source_boundary must equal P0R06147")


@dataclass(frozen=True, slots=True)
class ViiiTheSynthesisOfSubjectivityTheTriadicSolutionFixtureResult:
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


def classify_viii_the_synthesis_of_subjectivity_the_triadic_solution_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "viii_the_synthesis_of_subjectivity_the_triadic_solution": "viii_the_synthesis_of_subjectivity_the_triadic_solution_source_boundary",
        "ix_the_physics_of_teleology_and_ethics_the_teleological_engine": "ix_the_physics_of_teleology_and_ethics_the_teleological_engine_source_boundary",
        "x_the_dynamics_of_dissolution_death_and_transcendence": "x_the_dynamics_of_dissolution_death_and_transcendence_source_boundary",
        "the_physics_of_information_weaving_space_time_and_consciousness": "the_physics_of_information_weaving_space_time_and_consciousness_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown viii_the_synthesis_of_subjectivity_the_triadic_solution component"
        ) from exc


def viii_the_synthesis_of_subjectivity_the_triadic_solution_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "VIII. The Synthesis of Subjectivity (The Triadic Solution)",
        "source_span": "P0R06132-P0R06146",
        "component_count": "4",
        "next_boundary": "P0R06147",
        "component_1": "VIII. The Synthesis of Subjectivity (The Triadic Solution)",
        "component_2": "IX. The Physics of Teleology and Ethics (The Teleological Engine)",
        "component_3": "X. The Dynamics of Dissolution (Death and Transcendence)",
        "component_4": "The Physics of Information: Weaving Space-Time and Consciousness",
    }


def validate_viii_the_synthesis_of_subjectivity_the_triadic_solution_fixture(
    config: ViiiTheSynthesisOfSubjectivityTheTriadicSolutionConfig | None = None,
) -> ViiiTheSynthesisOfSubjectivityTheTriadicSolutionFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or ViiiTheSynthesisOfSubjectivityTheTriadicSolutionConfig()
    components = (
        "viii_the_synthesis_of_subjectivity_the_triadic_solution",
        "ix_the_physics_of_teleology_and_ethics_the_teleological_engine",
        "x_the_dynamics_of_dissolution_death_and_transcendence",
        "the_physics_of_information_weaving_space_time_and_consciousness",
    )
    return ViiiTheSynthesisOfSubjectivityTheTriadicSolutionFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_viii_the_synthesis_of_subjectivity_the_triadic_solution_component(
                component
            )
            for component in components
        },
        labels=viii_the_synthesis_of_subjectivity_the_triadic_solution_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "viii_the_synthesis_of_subjectivity_the_triadic_solution_is_not_empirical_validation_evidence": 1.0,
            "ix_the_physics_of_teleology_and_ethics_the_teleological_engine_is_not_empirical_validation_evidence": 1.0,
            "x_the_dynamics_of_dissolution_death_and_transcendence_is_not_empirical_validation_evidence": 1.0,
            "the_physics_of_information_weaving_space_time_and_consciousness_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(6132, 6147)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_viii_the_synthesis_of_subjectivity_the_triadic_solution_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "ViiiTheSynthesisOfSubjectivityTheTriadicSolutionConfig",
    "ViiiTheSynthesisOfSubjectivityTheTriadicSolutionFixtureResult",
    "classify_viii_the_synthesis_of_subjectivity_the_triadic_solution_component",
    "viii_the_synthesis_of_subjectivity_the_triadic_solution_labels",
    "validate_viii_the_synthesis_of_subjectivity_the_triadic_solution_fixture",
]
