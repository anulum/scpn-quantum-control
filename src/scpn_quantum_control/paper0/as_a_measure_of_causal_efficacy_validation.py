# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  as a Measure of Causal Efficacy: validation
"""Source-accounting checks for Paper 0  as a Measure of Causal Efficacy: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded as a measure of causal efficacy source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03295", "P0R03306")


@dataclass(frozen=True, slots=True)
class AsAMeasureOfCausalEfficacyConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 12
    expected_component_count: int = 3
    next_source_boundary: str = "P0R03307"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 12:
            raise ValueError("expected_source_record_count must equal 12")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R03307":
            raise ValueError("next_source_boundary must equal P0R03307")


@dataclass(frozen=True, slots=True)
class AsAMeasureOfCausalEfficacyFixtureResult:
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


def classify_as_a_measure_of_causal_efficacy_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "as_a_measure_of_causal_efficacy": "as_a_measure_of_causal_efficacy_source_boundary",
        "the_quantum_gravity_interface_and_cigd": "the_quantum_gravity_interface_and_cigd_source_boundary",
        "consciousness_induced_gravitational_decoherence_cigd": "consciousness_induced_gravitational_decoherence_cigd_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown as_a_measure_of_causal_efficacy component") from exc


def as_a_measure_of_causal_efficacy_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": " as a Measure of Causal Efficacy:",
        "source_span": "P0R03295-P0R03306",
        "component_count": "3",
        "next_boundary": "P0R03307",
        "component_1": "as a Measure of Causal Efficacy:",
        "component_2": "The Quantum-Gravity Interface and CIGD",
        "component_3": "Consciousness-Induced Gravitational Decoherence (CIGD):",
    }


def validate_as_a_measure_of_causal_efficacy_fixture(
    config: AsAMeasureOfCausalEfficacyConfig | None = None,
) -> AsAMeasureOfCausalEfficacyFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or AsAMeasureOfCausalEfficacyConfig()
    components = (
        "as_a_measure_of_causal_efficacy",
        "the_quantum_gravity_interface_and_cigd",
        "consciousness_induced_gravitational_decoherence_cigd",
    )
    return AsAMeasureOfCausalEfficacyFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_as_a_measure_of_causal_efficacy_component(component)
            for component in components
        },
        labels=as_a_measure_of_causal_efficacy_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "as_a_measure_of_causal_efficacy_is_not_empirical_validation_evidence": 1.0,
            "the_quantum_gravity_interface_and_cigd_is_not_empirical_validation_evidence": 1.0,
            "consciousness_induced_gravitational_decoherence_cigd_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3295, 3307)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_as_a_measure_of_causal_efficacy_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "AsAMeasureOfCausalEfficacyConfig",
    "AsAMeasureOfCausalEfficacyFixtureResult",
    "classify_as_a_measure_of_causal_efficacy_component",
    "as_a_measure_of_causal_efficacy_labels",
    "validate_as_a_measure_of_causal_efficacy_fixture",
]
