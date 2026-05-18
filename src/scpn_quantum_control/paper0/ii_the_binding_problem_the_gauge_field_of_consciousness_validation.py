# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 II. The Binding Problem: The Gauge Field of Consciousness validation
"""Source-accounting checks for Paper 0 II. The Binding Problem: The Gauge Field of Consciousness records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded ii the binding problem the gauge field of consciousness source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03208", "P0R03215")


@dataclass(frozen=True, slots=True)
class IiTheBindingProblemTheGaugeFieldOfConsciousnessConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 3
    next_source_boundary: str = "P0R03216"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R03216":
            raise ValueError("next_source_boundary must equal P0R03216")


@dataclass(frozen=True, slots=True)
class IiTheBindingProblemTheGaugeFieldOfConsciousnessFixtureResult:
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


def classify_ii_the_binding_problem_the_gauge_field_of_consciousness_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "ii_the_binding_problem_the_gauge_field_of_consciousness": "ii_the_binding_problem_the_gauge_field_of_consciousness_source_boundary",
        "1_the_connection_the_psi_field": "1_the_connection_the_psi_field_source_boundary",
        "2_local_gauge_invariance_and_coherence": "2_local_gauge_invariance_and_coherence_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown ii_the_binding_problem_the_gauge_field_of_consciousness component"
        ) from exc


def ii_the_binding_problem_the_gauge_field_of_consciousness_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "II. The Binding Problem: The Gauge Field of Consciousness",
        "source_span": "P0R03208-P0R03215",
        "component_count": "3",
        "next_boundary": "P0R03216",
        "component_1": "II. The Binding Problem: The Gauge Field of Consciousness",
        "component_2": "1. The Connection (The Psi-Field):",
        "component_3": "2. Local Gauge Invariance and Coherence:",
    }


def validate_ii_the_binding_problem_the_gauge_field_of_consciousness_fixture(
    config: IiTheBindingProblemTheGaugeFieldOfConsciousnessConfig | None = None,
) -> IiTheBindingProblemTheGaugeFieldOfConsciousnessFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or IiTheBindingProblemTheGaugeFieldOfConsciousnessConfig()
    components = (
        "ii_the_binding_problem_the_gauge_field_of_consciousness",
        "1_the_connection_the_psi_field",
        "2_local_gauge_invariance_and_coherence",
    )
    return IiTheBindingProblemTheGaugeFieldOfConsciousnessFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_ii_the_binding_problem_the_gauge_field_of_consciousness_component(
                component
            )
            for component in components
        },
        labels=ii_the_binding_problem_the_gauge_field_of_consciousness_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "ii_the_binding_problem_the_gauge_field_of_consciousness_is_not_empirical_validation_evidence": 1.0,
            "1_the_connection_the_psi_field_is_not_empirical_validation_evidence": 1.0,
            "2_local_gauge_invariance_and_coherence_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3208, 3216)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_ii_the_binding_problem_the_gauge_field_of_consciousness_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "IiTheBindingProblemTheGaugeFieldOfConsciousnessConfig",
    "IiTheBindingProblemTheGaugeFieldOfConsciousnessFixtureResult",
    "classify_ii_the_binding_problem_the_gauge_field_of_consciousness_component",
    "ii_the_binding_problem_the_gauge_field_of_consciousness_labels",
    "validate_ii_the_binding_problem_the_gauge_field_of_consciousness_fixture",
]
