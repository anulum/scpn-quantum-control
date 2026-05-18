# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 VIII. The Evolutionary Trajectory of the Brain-Body System validation
"""Source-accounting checks for Paper 0 VIII. The Evolutionary Trajectory of the Brain-Body System records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded viii the evolutionary trajectory of the brain body system source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05039", "P0R05049")


@dataclass(frozen=True, slots=True)
class ViiiTheEvolutionaryTrajectoryOfTheBrainBodySystemConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 11
    expected_component_count: int = 2
    next_source_boundary: str = "P0R05050"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 11:
            raise ValueError("expected_source_record_count must equal 11")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R05050":
            raise ValueError("next_source_boundary must equal P0R05050")


@dataclass(frozen=True, slots=True)
class ViiiTheEvolutionaryTrajectoryOfTheBrainBodySystemFixtureResult:
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


def classify_viii_the_evolutionary_trajectory_of_the_brain_body_system_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "viii_the_evolutionary_trajectory_of_the_brain_body_system": "viii_the_evolutionary_trajectory_of_the_brain_body_system_source_boundary",
        "ix_advanced_therapeutic_interventions_tuning_the_embodied_scpn": "ix_advanced_therapeutic_interventions_tuning_the_embodied_scpn_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown viii_the_evolutionary_trajectory_of_the_brain_body_system component"
        ) from exc


def viii_the_evolutionary_trajectory_of_the_brain_body_system_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "VIII. The Evolutionary Trajectory of the Brain-Body System",
        "source_span": "P0R05039-P0R05049",
        "component_count": "2",
        "next_boundary": "P0R05050",
        "component_1": "VIII. The Evolutionary Trajectory of the Brain-Body System",
        "component_2": "IX. Advanced Therapeutic Interventions (Tuning the Embodied SCPN)",
    }


def validate_viii_the_evolutionary_trajectory_of_the_brain_body_system_fixture(
    config: ViiiTheEvolutionaryTrajectoryOfTheBrainBodySystemConfig | None = None,
) -> ViiiTheEvolutionaryTrajectoryOfTheBrainBodySystemFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or ViiiTheEvolutionaryTrajectoryOfTheBrainBodySystemConfig()
    components = (
        "viii_the_evolutionary_trajectory_of_the_brain_body_system",
        "ix_advanced_therapeutic_interventions_tuning_the_embodied_scpn",
    )
    return ViiiTheEvolutionaryTrajectoryOfTheBrainBodySystemFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_viii_the_evolutionary_trajectory_of_the_brain_body_system_component(
                component
            )
            for component in components
        },
        labels=viii_the_evolutionary_trajectory_of_the_brain_body_system_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "viii_the_evolutionary_trajectory_of_the_brain_body_system_is_not_empirical_validation_evidence": 1.0,
            "ix_advanced_therapeutic_interventions_tuning_the_embodied_scpn_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5039, 5050)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_viii_the_evolutionary_trajectory_of_the_brain_body_system_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "ViiiTheEvolutionaryTrajectoryOfTheBrainBodySystemConfig",
    "ViiiTheEvolutionaryTrajectoryOfTheBrainBodySystemFixtureResult",
    "classify_viii_the_evolutionary_trajectory_of_the_brain_body_system_component",
    "viii_the_evolutionary_trajectory_of_the_brain_body_system_labels",
    "validate_viii_the_evolutionary_trajectory_of_the_brain_body_system_fixture",
]
