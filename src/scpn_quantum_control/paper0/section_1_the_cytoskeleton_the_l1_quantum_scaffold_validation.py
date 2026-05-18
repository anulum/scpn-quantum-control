# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 1. The Cytoskeleton: The L1 Quantum Scaffold validation
"""Source-accounting checks for Paper 0 1. The Cytoskeleton: The L1 Quantum Scaffold records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 1 the cytoskeleton the l1 quantum scaffold source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04728", "P0R04736")


@dataclass(frozen=True, slots=True)
class Section1TheCytoskeletonTheL1QuantumScaffoldConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 3
    next_source_boundary: str = "P0R04737"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 3:
            raise ValueError("expected_component_count must equal 3")
        if self.next_source_boundary != "P0R04737":
            raise ValueError("next_source_boundary must equal P0R04737")


@dataclass(frozen=True, slots=True)
class Section1TheCytoskeletonTheL1QuantumScaffoldFixtureResult:
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


def classify_section_1_the_cytoskeleton_the_l1_quantum_scaffold_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "1_the_cytoskeleton_the_l1_quantum_scaffold": "1_the_cytoskeleton_the_l1_quantum_scaffold_source_boundary",
        "2_mitochondria_the_bioenergetic_generators_l1_l3": "2_mitochondria_the_bioenergetic_generators_l1_l3_source_boundary",
        "3_endoplasmic_reticulum_er_calcium_dynamics": "3_endoplasmic_reticulum_er_calcium_dynamics_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown section_1_the_cytoskeleton_the_l1_quantum_scaffold component"
        ) from exc


def section_1_the_cytoskeleton_the_l1_quantum_scaffold_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "1. The Cytoskeleton: The L1 Quantum Scaffold",
        "source_span": "P0R04728-P0R04736",
        "component_count": "3",
        "next_boundary": "P0R04737",
        "component_1": "1. The Cytoskeleton: The L1 Quantum Scaffold",
        "component_2": "2. Mitochondria: The Bioenergetic Generators (L1/L3)",
        "component_3": "3. Endoplasmic Reticulum (ER): Calcium Dynamics",
    }


def validate_section_1_the_cytoskeleton_the_l1_quantum_scaffold_fixture(
    config: Section1TheCytoskeletonTheL1QuantumScaffoldConfig | None = None,
) -> Section1TheCytoskeletonTheL1QuantumScaffoldFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section1TheCytoskeletonTheL1QuantumScaffoldConfig()
    components = (
        "1_the_cytoskeleton_the_l1_quantum_scaffold",
        "2_mitochondria_the_bioenergetic_generators_l1_l3",
        "3_endoplasmic_reticulum_er_calcium_dynamics",
    )
    return Section1TheCytoskeletonTheL1QuantumScaffoldFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_1_the_cytoskeleton_the_l1_quantum_scaffold_component(
                component
            )
            for component in components
        },
        labels=section_1_the_cytoskeleton_the_l1_quantum_scaffold_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "1_the_cytoskeleton_the_l1_quantum_scaffold_is_not_empirical_validation_evidence": 1.0,
            "2_mitochondria_the_bioenergetic_generators_l1_l3_is_not_empirical_validation_evidence": 1.0,
            "3_endoplasmic_reticulum_er_calcium_dynamics_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4728, 4737)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_1_the_cytoskeleton_the_l1_quantum_scaffold_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section1TheCytoskeletonTheL1QuantumScaffoldConfig",
    "Section1TheCytoskeletonTheL1QuantumScaffoldFixtureResult",
    "classify_section_1_the_cytoskeleton_the_l1_quantum_scaffold_component",
    "section_1_the_cytoskeleton_the_l1_quantum_scaffold_labels",
    "validate_section_1_the_cytoskeleton_the_l1_quantum_scaffold_fixture",
]
