# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 II. Micro-Scale Geometry: The Quantum and Molecular Scaffold (L1-L3) validation
"""Source-accounting checks for Paper 0 II. Micro-Scale Geometry: The Quantum and Molecular Scaffold (L1-L3) records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded ii micro scale geometry the quantum and molecular scaffold l1 l3 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04813", "P0R04823")


@dataclass(frozen=True, slots=True)
class IiMicroScaleGeometryTheQuantumAndMolecularScaffoldL1L3Config:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 11
    expected_component_count: int = 2
    next_source_boundary: str = "P0R04824"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 11:
            raise ValueError("expected_source_record_count must equal 11")
        if self.expected_component_count != 2:
            raise ValueError("expected_component_count must equal 2")
        if self.next_source_boundary != "P0R04824":
            raise ValueError("next_source_boundary must equal P0R04824")


@dataclass(frozen=True, slots=True)
class IiMicroScaleGeometryTheQuantumAndMolecularScaffoldL1L3FixtureResult:
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


def classify_ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_component(
    component: str,
) -> str:
    """Classify source-defined components."""
    mapping = {
        "ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3": "ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_source_boundary",
        "1_the_geometry_of_the_quantum_substrate_l1": "1_the_geometry_of_the_quantum_substrate_l1_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            "unknown ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3 component"
        ) from exc


def ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "II. Micro-Scale Geometry: The Quantum and Molecular Scaffold (L1-L3)",
        "source_span": "P0R04813-P0R04823",
        "component_count": "2",
        "next_boundary": "P0R04824",
        "component_1": "II. Micro-Scale Geometry: The Quantum and Molecular Scaffold (L1-L3)",
        "component_2": "1. The Geometry of the Quantum Substrate (L1):",
    }


def validate_ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_fixture(
    config: IiMicroScaleGeometryTheQuantumAndMolecularScaffoldL1L3Config | None = None,
) -> IiMicroScaleGeometryTheQuantumAndMolecularScaffoldL1L3FixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or IiMicroScaleGeometryTheQuantumAndMolecularScaffoldL1L3Config()
    components = (
        "ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3",
        "1_the_geometry_of_the_quantum_substrate_l1",
    )
    return IiMicroScaleGeometryTheQuantumAndMolecularScaffoldL1L3FixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_component(
                component
            )
            for component in components
        },
        labels=ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_is_not_empirical_validation_evidence": 1.0,
            "1_the_geometry_of_the_quantum_substrate_l1_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4813, 4824)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "IiMicroScaleGeometryTheQuantumAndMolecularScaffoldL1L3Config",
    "IiMicroScaleGeometryTheQuantumAndMolecularScaffoldL1L3FixtureResult",
    "classify_ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_component",
    "ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_labels",
    "validate_ii_micro_scale_geometry_the_quantum_and_molecular_scaffold_l1_l3_fixture",
]
