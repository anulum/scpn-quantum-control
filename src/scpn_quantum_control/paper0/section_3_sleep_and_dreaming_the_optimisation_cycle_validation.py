# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 3. Sleep and Dreaming (The Optimisation Cycle): validation
"""Source-accounting checks for Paper 0 3. Sleep and Dreaming (The Optimisation Cycle): records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 3 sleep and dreaming the optimisation cycle source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05017", "P0R05025")


@dataclass(frozen=True, slots=True)
class Section3SleepAndDreamingTheOptimisationCycleConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 2
    next_source_boundary: str = "P0R05026"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R05026":
            raise ValueError("next_source_boundary must equal P0R05026")


@dataclass(frozen=True, slots=True)
class Section3SleepAndDreamingTheOptimisationCycleFixtureResult:
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


def classify_section_3_sleep_and_dreaming_the_optimisation_cycle_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "3_sleep_and_dreaming_the_optimisation_cycle": "3_sleep_and_dreaming_the_optimisation_cycle_source_boundary",
        "vi_altered_states_of_consciousness_ascs_perturbations_of_the_architectur": "vi_altered_states_of_consciousness_ascs_perturbations_of_the_architectur_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_3_sleep_and_dreaming_the_optimisation_cycle component"
        ) from exc


def section_3_sleep_and_dreaming_the_optimisation_cycle_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "3. Sleep and Dreaming (The Optimisation Cycle):",
        "source_span": "P0R05017-P0R05025",
        "component_count": "2",
        "next_boundary": "P0R05026",
        "component_1": "3. Sleep and Dreaming (The Optimisation Cycle):",
        "component_2": "VI. Altered States of Consciousness (ASCs): Perturbations of the Architecture",
    }


def validate_section_3_sleep_and_dreaming_the_optimisation_cycle_fixture(
    config: Section3SleepAndDreamingTheOptimisationCycleConfig | None = None,
) -> Section3SleepAndDreamingTheOptimisationCycleFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section3SleepAndDreamingTheOptimisationCycleConfig()
    components = (
        "3_sleep_and_dreaming_the_optimisation_cycle",
        "vi_altered_states_of_consciousness_ascs_perturbations_of_the_architectur",
    )
    return Section3SleepAndDreamingTheOptimisationCycleFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_3_sleep_and_dreaming_the_optimisation_cycle_component(
                component
            )
            for component in components
        },
        labels=section_3_sleep_and_dreaming_the_optimisation_cycle_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "3_sleep_and_dreaming_the_optimisation_cycle_is_not_empirical_validation_evidence": 1.0,
            "vi_altered_states_of_consciousness_ascs_perturbations_of_the_architectur_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5017, 5026)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_3_sleep_and_dreaming_the_optimisation_cycle_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section3SleepAndDreamingTheOptimisationCycleConfig",
    "Section3SleepAndDreamingTheOptimisationCycleFixtureResult",
    "classify_section_3_sleep_and_dreaming_the_optimisation_cycle_component",
    "section_3_sleep_and_dreaming_the_optimisation_cycle_labels",
    "validate_section_3_sleep_and_dreaming_the_optimisation_cycle_fixture",
]
