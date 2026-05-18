# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 1. The Presynaptic Terminal (The Quantum Lever): validation
"""Source-accounting checks for Paper 0 1. The Presynaptic Terminal (The Quantum Lever): records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 1 the presynaptic terminal the quantum lever source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04737", "P0R04745")


@dataclass(frozen=True, slots=True)
class Section1ThePresynapticTerminalTheQuantumLeverConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 4
    next_source_boundary: str = "P0R04746"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R04746":
            raise ValueError("next_source_boundary must equal P0R04746")


@dataclass(frozen=True, slots=True)
class Section1ThePresynapticTerminalTheQuantumLeverFixtureResult:
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


def classify_section_1_the_presynaptic_terminal_the_quantum_lever_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "1_the_presynaptic_terminal_the_quantum_lever": "1_the_presynaptic_terminal_the_quantum_lever_source_boundary",
        "vi_the_nucleus_and_the_genomic_interface_l3": "vi_the_nucleus_and_the_genomic_interface_l3_source_boundary",
        "the_deepest_interface_molecular_and_quantum_foundations": "the_deepest_interface_molecular_and_quantum_foundations_source_boundary",
        "i_the_neuronal_membrane_a_liquid_crystal_interface_l2_l3": "i_the_neuronal_membrane_a_liquid_crystal_interface_l2_l3_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_1_the_presynaptic_terminal_the_quantum_lever component"
        ) from exc


def section_1_the_presynaptic_terminal_the_quantum_lever_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "1. The Presynaptic Terminal (The Quantum Lever):",
        "source_span": "P0R04737-P0R04745",
        "component_count": "4",
        "next_boundary": "P0R04746",
        "component_1": "1. The Presynaptic Terminal (The Quantum Lever):",
        "component_2": "VI. The Nucleus and the Genomic Interface (L3)",
        "component_3": "The Deepest Interface: Molecular and Quantum Foundations",
        "component_4": "I. The Neuronal Membrane: A Liquid Crystal Interface (L2/L3)",
    }


def validate_section_1_the_presynaptic_terminal_the_quantum_lever_fixture(
    config: Section1ThePresynapticTerminalTheQuantumLeverConfig | None = None,
) -> Section1ThePresynapticTerminalTheQuantumLeverFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section1ThePresynapticTerminalTheQuantumLeverConfig()
    components = (
        "1_the_presynaptic_terminal_the_quantum_lever",
        "vi_the_nucleus_and_the_genomic_interface_l3",
        "the_deepest_interface_molecular_and_quantum_foundations",
        "i_the_neuronal_membrane_a_liquid_crystal_interface_l2_l3",
    )
    return Section1ThePresynapticTerminalTheQuantumLeverFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_1_the_presynaptic_terminal_the_quantum_lever_component(
                component
            )
            for component in components
        },
        labels=section_1_the_presynaptic_terminal_the_quantum_lever_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "1_the_presynaptic_terminal_the_quantum_lever_is_not_empirical_validation_evidence": 1.0,
            "vi_the_nucleus_and_the_genomic_interface_l3_is_not_empirical_validation_evidence": 1.0,
            "the_deepest_interface_molecular_and_quantum_foundations_is_not_empirical_validation_evidence": 1.0,
            "i_the_neuronal_membrane_a_liquid_crystal_interface_l2_l3_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4737, 4746)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_1_the_presynaptic_terminal_the_quantum_lever_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section1ThePresynapticTerminalTheQuantumLeverConfig",
    "Section1ThePresynapticTerminalTheQuantumLeverFixtureResult",
    "classify_section_1_the_presynaptic_terminal_the_quantum_lever_component",
    "section_1_the_presynaptic_terminal_the_quantum_lever_labels",
    "validate_section_1_the_presynaptic_terminal_the_quantum_lever_fixture",
]
