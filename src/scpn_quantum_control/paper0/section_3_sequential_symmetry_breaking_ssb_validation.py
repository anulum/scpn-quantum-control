# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 3. Sequential Symmetry Breaking (SSB): validation
"""Source-accounting checks for Paper 0 3. Sequential Symmetry Breaking (SSB): records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 3 sequential symmetry breaking ssb source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04388", "P0R04395")


@dataclass(frozen=True, slots=True)
class Section3SequentialSymmetryBreakingSsbConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 8
    expected_component_count: int = 4
    next_source_boundary: str = "P0R04396"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 8:
            raise ValueError("expected_source_record_count must equal 8")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R04396":
            raise ValueError("next_source_boundary must equal P0R04396")


@dataclass(frozen=True, slots=True)
class Section3SequentialSymmetryBreakingSsbFixtureResult:
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


def classify_section_3_sequential_symmetry_breaking_ssb_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "3_sequential_symmetry_breaking_ssb": "3_sequential_symmetry_breaking_ssb_source_boundary",
        "iii_the_geometry_of_the_quantum_biological_interface_domain_i_l1_l3": "iii_the_geometry_of_the_quantum_biological_interface_domain_i_l1_l3_source_boundary",
        "1_quantum_geometry_and_topological_order_l1": "1_quantum_geometry_and_topological_order_l1_source_boundary",
        "2_molecular_geometry_chirality_and_iet_l2_l3": "2_molecular_geometry_chirality_and_iet_l2_l3_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown section_3_sequential_symmetry_breaking_ssb component") from exc


def section_3_sequential_symmetry_breaking_ssb_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "3. Sequential Symmetry Breaking (SSB):",
        "source_span": "P0R04388-P0R04395",
        "component_count": "4",
        "next_boundary": "P0R04396",
        "component_1": "3. Sequential Symmetry Breaking (SSB):",
        "component_2": "III. The Geometry of the Quantum-Biological Interface (Domain I: L1-L3)",
        "component_3": "1. Quantum Geometry and Topological Order (L1):",
        "component_4": "2. Molecular Geometry, Chirality, and IET (L2/L3):",
    }


def validate_section_3_sequential_symmetry_breaking_ssb_fixture(
    config: Section3SequentialSymmetryBreakingSsbConfig | None = None,
) -> Section3SequentialSymmetryBreakingSsbFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section3SequentialSymmetryBreakingSsbConfig()
    components = (
        "3_sequential_symmetry_breaking_ssb",
        "iii_the_geometry_of_the_quantum_biological_interface_domain_i_l1_l3",
        "1_quantum_geometry_and_topological_order_l1",
        "2_molecular_geometry_chirality_and_iet_l2_l3",
    )
    return Section3SequentialSymmetryBreakingSsbFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_3_sequential_symmetry_breaking_ssb_component(component)
            for component in components
        },
        labels=section_3_sequential_symmetry_breaking_ssb_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "3_sequential_symmetry_breaking_ssb_is_not_empirical_validation_evidence": 1.0,
            "iii_the_geometry_of_the_quantum_biological_interface_domain_i_l1_l3_is_not_empirical_validation_evidence": 1.0,
            "1_quantum_geometry_and_topological_order_l1_is_not_empirical_validation_evidence": 1.0,
            "2_molecular_geometry_chirality_and_iet_l2_l3_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4388, 4396)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_3_sequential_symmetry_breaking_ssb_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section3SequentialSymmetryBreakingSsbConfig",
    "Section3SequentialSymmetryBreakingSsbFixtureResult",
    "classify_section_3_sequential_symmetry_breaking_ssb_component",
    "section_3_sequential_symmetry_breaking_ssb_labels",
    "validate_section_3_sequential_symmetry_breaking_ssb_fixture",
]
