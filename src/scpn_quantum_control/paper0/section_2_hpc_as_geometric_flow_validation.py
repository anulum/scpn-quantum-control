# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. HPC as Geometric Flow: validation
"""Source-accounting checks for Paper 0 2. HPC as Geometric Flow: records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CLAIM_BOUNDARY = "source-bounded section 2 hpc as geometric flow source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
SOURCE_LEDGER_SPAN = ("P0R04849", "P0R04857")


@dataclass(frozen=True, slots=True)
class Section2HpcAsGeometricFlowConfig:
    """Configuration for this Paper 0 source-accounting fixture."""

    expected_source_record_count: int = 9
    expected_component_count: int = 4
    next_source_boundary: str = "P0R04858"

    def __post_init__(self) -> None:
        if self.expected_source_record_count != 9:
            raise ValueError("expected_source_record_count must equal 9")
        if self.expected_component_count != 4:
            raise ValueError("expected_component_count must equal 4")
        if self.next_source_boundary != "P0R04858":
            raise ValueError("next_source_boundary must equal P0R04858")


@dataclass(frozen=True, slots=True)
class Section2HpcAsGeometricFlowFixtureResult:
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


def classify_section_2_hpc_as_geometric_flow_component(component: str) -> str:
    """Classify source-defined components."""
    mapping = {
        "2_hpc_as_geometric_flow": "2_hpc_as_geometric_flow_source_boundary",
        "3_the_geometry_of_the_self_the_strange_loop_and_the_dmn": "3_the_geometry_of_the_self_the_strange_loop_and_the_dmn_source_boundary",
        "vi_synthesis_the_brain_as_a_geometric_transducer": "vi_synthesis_the_brain_as_a_geometric_transducer_source_boundary",
        "the_integrative_physiology_of_the_scpn_the_embodied_brain": "the_integrative_physiology_of_the_scpn_the_embodied_brain_source_boundary",
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError("unknown section_2_hpc_as_geometric_flow component") from exc


def section_2_hpc_as_geometric_flow_labels() -> dict[str, str]:
    """Return source-bounded labels for this slice."""
    return {
        "section": "2. HPC as Geometric Flow:",
        "source_span": "P0R04849-P0R04857",
        "component_count": "4",
        "next_boundary": "P0R04858",
        "component_1": "2. HPC as Geometric Flow:",
        "component_2": "3. The Geometry of the Self (The Strange Loop and the DMN):",
        "component_3": "VI. Synthesis: The Brain as a Geometric Transducer",
        "component_4": "The Integrative Physiology of the SCPN: The Embodied Brain",
    }


def validate_section_2_hpc_as_geometric_flow_fixture(
    config: Section2HpcAsGeometricFlowConfig | None = None,
) -> Section2HpcAsGeometricFlowFixtureResult:
    """Validate source accounting for this Paper 0 slice."""
    cfg = config or Section2HpcAsGeometricFlowConfig()
    components = (
        "2_hpc_as_geometric_flow",
        "3_the_geometry_of_the_self_the_strange_loop_and_the_dmn",
        "vi_synthesis_the_brain_as_a_geometric_transducer",
        "the_integrative_physiology_of_the_scpn_the_embodied_brain",
    )
    return Section2HpcAsGeometricFlowFixtureResult(
        source_ledger_span=SOURCE_LEDGER_SPAN,
        hardware_status=HARDWARE_STATUS,
        claim_boundary=CLAIM_BOUNDARY,
        components={
            component: classify_section_2_hpc_as_geometric_flow_component(component)
            for component in components
        },
        labels=section_2_hpc_as_geometric_flow_labels(),
        source_record_count=cfg.expected_source_record_count,
        component_count=cfg.expected_component_count,
        next_source_boundary=cfg.next_source_boundary,
        null_controls={
            "2_hpc_as_geometric_flow_is_not_empirical_validation_evidence": 1.0,
            "3_the_geometry_of_the_self_the_strange_loop_and_the_dmn_is_not_empirical_validation_evidence": 1.0,
            "vi_synthesis_the_brain_as_a_geometric_transducer_is_not_empirical_validation_evidence": 1.0,
            "the_integrative_physiology_of_the_scpn_the_embodied_brain_is_not_empirical_validation_evidence": 1.0,
        },
        problem_metadata={
            "source_ledger_span": SOURCE_LEDGER_SPAN,
            "source_ledger_ids": tuple(f"P0R{number:05d}" for number in range(4849, 4858)),
            "claim_boundary": CLAIM_BOUNDARY,
            "protocol_state": "source_section_2_hpc_as_geometric_flow_only_no_experiment",
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HARDWARE_STATUS",
    "SOURCE_LEDGER_SPAN",
    "Section2HpcAsGeometricFlowConfig",
    "Section2HpcAsGeometricFlowFixtureResult",
    "classify_section_2_hpc_as_geometric_flow_component",
    "section_2_hpc_as_geometric_flow_labels",
    "validate_section_2_hpc_as_geometric_flow_fixture",
]
