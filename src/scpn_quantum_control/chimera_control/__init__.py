# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Chimera-control public facade
"""Synthetic chimera and hierarchical synchronisation-control surfaces."""

from .evidence import (
    CHIMERA_CONTROL_EVIDENCE_DATE,
    CHIMERA_CONTROL_EVIDENCE_SCHEMA,
    ChimeraMultiscaleEvidence,
    ChimeraSupportRow,
    SyntheticRegimeEvidence,
    build_chimera_multiscale_evidence,
    render_chimera_multiscale_markdown,
    write_chimera_multiscale_evidence,
)
from .objectives import (
    PhaseControlProposal,
    build_chimera_control_objective,
    propose_phase_control_step,
)
from .observables import (
    LevelOrderParameterSummary,
    MultiscaleOrderParameterReport,
    measure_multiscale_order_parameters,
)
from .schema import (
    CHIMERA_CONTROL_CLAIM_BOUNDARY,
    ChimeraControlSpecification,
    HierarchyLevel,
    HierarchyTarget,
    MultiscaleHierarchy,
    SyntheticRegime,
    two_population_hierarchy,
)
from .synthetic import (
    SYNTHETIC_CHIMERA_SOURCE,
    SyntheticChimeraConfig,
    SyntheticChimeraRun,
    build_two_population_coupling,
    generate_two_population_chimera,
)
from .topology import (
    HierarchyCouplingSummary,
    TopologyProjectionReport,
    project_chimera_coupling,
)

__all__ = [
    "CHIMERA_CONTROL_EVIDENCE_DATE",
    "CHIMERA_CONTROL_EVIDENCE_SCHEMA",
    "CHIMERA_CONTROL_CLAIM_BOUNDARY",
    "SYNTHETIC_CHIMERA_SOURCE",
    "ChimeraControlSpecification",
    "ChimeraMultiscaleEvidence",
    "ChimeraSupportRow",
    "HierarchyCouplingSummary",
    "HierarchyLevel",
    "HierarchyTarget",
    "LevelOrderParameterSummary",
    "MultiscaleHierarchy",
    "MultiscaleOrderParameterReport",
    "PhaseControlProposal",
    "SyntheticChimeraConfig",
    "SyntheticChimeraRun",
    "SyntheticRegime",
    "SyntheticRegimeEvidence",
    "TopologyProjectionReport",
    "build_chimera_control_objective",
    "build_chimera_multiscale_evidence",
    "build_two_population_coupling",
    "generate_two_population_chimera",
    "measure_multiscale_order_parameters",
    "project_chimera_coupling",
    "propose_phase_control_step",
    "render_chimera_multiscale_markdown",
    "two_population_hierarchy",
    "write_chimera_multiscale_evidence",
]
