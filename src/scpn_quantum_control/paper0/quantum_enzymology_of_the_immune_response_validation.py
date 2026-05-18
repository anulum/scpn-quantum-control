# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Quantum Enzymology of the Immune Response validation
"""Source-accounting checks for Paper 0 Quantum Enzymology of the Immune Response records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded quantum enzymology of the immune response source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R05508", "P0R05516")


@dataclass(frozen=True, slots=True)
class QuantumEnzymologyOfTheImmuneResponseConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 3
    next_source_boundary: str = "P0R05517"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R05517":
            raise ValueError("next_source_boundary must equal P0R05517")


@dataclass(frozen=True, slots=True)
class QuantumEnzymologyOfTheImmuneResponseFixtureResult:
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


def classify_quantum_enzymology_of_the_immune_response_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "quantum_enzymology_of_the_immune_response": "quantum_enzymology_of_the_immune_response_source_boundary",
        "mechanism_of_nuclear_quantum_tunnelling": "mechanism_of_nuclear_quantum_tunnelling_source_boundary",
        "formalism_for_tunnelling_enhanced_reaction_rates": "formalism_for_tunnelling_enhanced_reaction_rates_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown quantum_enzymology_of_the_immune_response component") from exc


def quantum_enzymology_of_the_immune_response_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "Quantum Enzymology of the Immune Response",
        "source_span": "P0R05508-P0R05516",
        "component_count": "3",
        "next_boundary": "P0R05517",
        "component_1": "Quantum Enzymology of the Immune Response",
        "component_2": "Mechanism of Nuclear Quantum Tunnelling",
        "component_3": "Formalism for Tunnelling-Enhanced Reaction Rates",
    }


def validate_quantum_enzymology_of_the_immune_response_fixture(
    config: QuantumEnzymologyOfTheImmuneResponseConfig | None = None,
) -> QuantumEnzymologyOfTheImmuneResponseFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or QuantumEnzymologyOfTheImmuneResponseConfig()
    components = (
        "quantum_enzymology_of_the_immune_response",
        "mechanism_of_nuclear_quantum_tunnelling",
        "formalism_for_tunnelling_enhanced_reaction_rates",
    )
    return QuantumEnzymologyOfTheImmuneResponseFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_quantum_enzymology_of_the_immune_response_component(component)
            for component in components
        },
        labels=quantum_enzymology_of_the_immune_response_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "quantum_enzymology_of_the_immune_response_is_not_empirical_validation_evidence": 1.0,
            "mechanism_of_nuclear_quantum_tunnelling_is_not_empirical_validation_evidence": 1.0,
            "formalism_for_tunnelling_enhanced_reaction_rates_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(5508, 5517)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_quantum_enzymology_of_the_immune_response_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "QuantumEnzymologyOfTheImmuneResponseConfig",
    "QuantumEnzymologyOfTheImmuneResponseFixtureResult",
    "classify_quantum_enzymology_of_the_immune_response_component",
    "quantum_enzymology_of_the_immune_response_labels",
    "validate_quantum_enzymology_of_the_immune_response_fixture",
]
