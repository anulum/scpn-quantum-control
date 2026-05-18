# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Quantum-to-Classical Transition: Amplification of Intent validation
"""Source-accounting checks for Paper 0 The Quantum-to-Classical Transition: Amplification of Intent records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded the quantum to classical transition amplification of intent source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03332", "P0R03342")


@dataclass(frozen=True, slots=True)
class TheQuantumToClassicalTransitionAmplificationOfIntentConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 11
    expected_component_count: int = 2
    next_source_boundary: str = "P0R03343"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 11:
            raise ValueError("expected_source_record_count must equal 11")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R03343":
            raise ValueError("next_source_boundary must equal P0R03343")


@dataclass(frozen=True, slots=True)
class TheQuantumToClassicalTransitionAmplificationOfIntentFixtureResult:
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


def classify_the_quantum_to_classical_transition_amplification_of_intent_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "the_quantum_to_classical_transition_amplification_of_intent": "the_quantum_to_classical_transition_amplification_of_intent_source_boundary",
        "mechanism_1_guided_einselection": "mechanism_1_guided_einselection_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown the_quantum_to_classical_transition_amplification_of_intent component"
        ) from exc


def the_quantum_to_classical_transition_amplification_of_intent_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "The Quantum-to-Classical Transition: Amplification of Intent",
        "source_span": "P0R03332-P0R03342",
        "component_count": "2",
        "next_boundary": "P0R03343",
        "component_1": "The Quantum-to-Classical Transition: Amplification of Intent",
        "component_2": "Mechanism 1: Guided Einselection",
    }


def validate_the_quantum_to_classical_transition_amplification_of_intent_fixture(
    config: TheQuantumToClassicalTransitionAmplificationOfIntentConfig | None = None,
) -> TheQuantumToClassicalTransitionAmplificationOfIntentFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or TheQuantumToClassicalTransitionAmplificationOfIntentConfig()
    components = (
        "the_quantum_to_classical_transition_amplification_of_intent",
        "mechanism_1_guided_einselection",
    )
    return TheQuantumToClassicalTransitionAmplificationOfIntentFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_the_quantum_to_classical_transition_amplification_of_intent_component(
                component
            )
            for component in components
        },
        labels=the_quantum_to_classical_transition_amplification_of_intent_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "the_quantum_to_classical_transition_amplification_of_intent_is_not_empirical_validation_evidence": 1.0,
            "mechanism_1_guided_einselection_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3332, 3343)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_the_quantum_to_classical_transition_amplification_of_intent_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "TheQuantumToClassicalTransitionAmplificationOfIntentConfig",
    "TheQuantumToClassicalTransitionAmplificationOfIntentFixtureResult",
    "classify_the_quantum_to_classical_transition_amplification_of_intent_component",
    "the_quantum_to_classical_transition_amplification_of_intent_labels",
    "validate_the_quantum_to_classical_transition_amplification_of_intent_fixture",
]
