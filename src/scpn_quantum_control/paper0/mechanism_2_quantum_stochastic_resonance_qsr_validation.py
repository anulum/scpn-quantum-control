# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Mechanism 2: Quantum Stochastic Resonance (QSR) validation
"""Source-accounting checks for Paper 0 Mechanism 2: Quantum Stochastic Resonance (QSR) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded mechanism 2 quantum stochastic resonance qsr source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R03343", "P0R03359")


@dataclass(frozen=True, slots=True)
class Mechanism2QuantumStochasticResonanceQsrConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 17
    expected_component_count: int = 1
    next_source_boundary: str = "P0R03360"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 17:
            raise ValueError("expected_source_record_count must equal 17")
        if self.expected_component_count != 1:
            raise ValueError("expected_component_count must equal 1")
        if self.next_source_boundary != "P0R03360":
            raise ValueError("next_source_boundary must equal P0R03360")


@dataclass(frozen=True, slots=True)
class Mechanism2QuantumStochasticResonanceQsrFixtureResult:
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


def classify_mechanism_2_quantum_stochastic_resonance_qsr_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "mechanism_2_quantum_stochastic_resonance_qsr": "mechanism_2_quantum_stochastic_resonance_qsr_source_boundary"
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown mechanism_2_quantum_stochastic_resonance_qsr component") from exc


def mechanism_2_quantum_stochastic_resonance_qsr_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Mechanism 2: Quantum Stochastic Resonance (QSR)",
        "source_span": "P0R03343-P0R03359",
        "component_count": "1",
        "next_boundary": "P0R03360",
        "component_1": "Mechanism 2: Quantum Stochastic Resonance (QSR)",
    }


def validate_mechanism_2_quantum_stochastic_resonance_qsr_fixture(
    config: Mechanism2QuantumStochasticResonanceQsrConfig | None = None,
) -> Mechanism2QuantumStochasticResonanceQsrFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Mechanism2QuantumStochasticResonanceQsrConfig()
    components = ("mechanism_2_quantum_stochastic_resonance_qsr",)
    return Mechanism2QuantumStochasticResonanceQsrFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_mechanism_2_quantum_stochastic_resonance_qsr_component(component)
            for component in components
        },
        labels=mechanism_2_quantum_stochastic_resonance_qsr_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "mechanism_2_quantum_stochastic_resonance_qsr_is_not_empirical_validation_evidence": 1.0
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(3343, 3360)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_mechanism_2_quantum_stochastic_resonance_qsr_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Mechanism2QuantumStochasticResonanceQsrConfig",
    "Mechanism2QuantumStochasticResonanceQsrFixtureResult",
    "classify_mechanism_2_quantum_stochastic_resonance_qsr_component",
    "mechanism_2_quantum_stochastic_resonance_qsr_labels",
    "validate_mechanism_2_quantum_stochastic_resonance_qsr_fixture",
]
