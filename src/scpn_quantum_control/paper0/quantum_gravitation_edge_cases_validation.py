# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Quantum & Gravitation Edge Cases validation
"""Source-accounting checks for Paper 0  Quantum & Gravitation Edge Cases records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded quantum gravitation edge cases source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05689", "P0R05696")


@dataclass(frozen=True, slots=True)
class QuantumGravitationEdgeCasesConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 2
    next_source_boundary: str = "P0R05697"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R05697":
            raise ValueError("next_source_boundary must equal P0R05697")


@dataclass(frozen=True, slots=True)
class QuantumGravitationEdgeCasesFixtureResult:
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


def classify_quantum_gravitation_edge_cases_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "quantum_gravitation_edge_cases": "quantum_gravitation_edge_cases_source_boundary",
        "quantum_foundations_info": "quantum_foundations_info_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown quantum_gravitation_edge_cases component") from exc


def quantum_gravitation_edge_cases_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": " Quantum & Gravitation Edge Cases",
        "source_span": "P0R05689-P0R05696",
        "component_count": "2",
        "next_boundary": "P0R05697",
        "component_1": "Quantum & Gravitation Edge Cases",
        "component_2": "Quantum Foundations & Info",
    }


def validate_quantum_gravitation_edge_cases_fixture(
    config: QuantumGravitationEdgeCasesConfig | None = None,
) -> QuantumGravitationEdgeCasesFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or QuantumGravitationEdgeCasesConfig()
    components = ("quantum_gravitation_edge_cases", "quantum_foundations_info")
    return QuantumGravitationEdgeCasesFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_quantum_gravitation_edge_cases_component(component)
            for component in components
        },
        labels=quantum_gravitation_edge_cases_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "quantum_gravitation_edge_cases_is_not_empirical_validation_evidence": 1.0,
            "quantum_foundations_info_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5689, 5697)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_quantum_gravitation_edge_cases_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "QuantumGravitationEdgeCasesConfig",
    "QuantumGravitationEdgeCasesFixtureResult",
    "classify_quantum_gravitation_edge_cases_component",
    "quantum_gravitation_edge_cases_labels",
    "validate_quantum_gravitation_edge_cases_fixture",
]
