# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Topology Control
"""Constrained persistent-H1 optimisation for coupling graphs."""

from .artefacts import TopologyOptimisationArtifact, export_topology_optimisation_artifact
from .complexes import (
    H1Summary,
    NetworkCycleBackend,
    PersistenceDiagram,
    PersistentHomologyBackend,
    RipserPHBackend,
    build_correlation_distance_matrix,
    build_coupling_distance_matrix,
    spike_trace_correlation_distance,
)
from .constraints import (
    CouplingGraphBounds,
    HardwareEmbeddingConstraint,
    TopologyConstraintLedger,
    algebraic_connectivity,
)
from .hardware_integration import TopologyHardwareManifest, validate_topology_hardware_manifest
from .objectives import CouplingTopologyObjective, DegeneracyMode, ObjectiveBreakdown
from .optimizers import (
    ProjectedScipyOptimizer,
    ProjectedSPSAOptimizer,
    TopologyOptimisationStep,
    TopologyOptimisationTrace,
)
from .qsnn_integration import TopologicalDynamicCouplingPolicy

__all__ = [
    "CouplingGraphBounds",
    "CouplingTopologyObjective",
    "DegeneracyMode",
    "H1Summary",
    "HardwareEmbeddingConstraint",
    "NetworkCycleBackend",
    "ObjectiveBreakdown",
    "PersistenceDiagram",
    "PersistentHomologyBackend",
    "ProjectedSPSAOptimizer",
    "ProjectedScipyOptimizer",
    "RipserPHBackend",
    "TopologicalDynamicCouplingPolicy",
    "TopologyConstraintLedger",
    "TopologyHardwareManifest",
    "TopologyOptimisationArtifact",
    "TopologyOptimisationStep",
    "TopologyOptimisationTrace",
    "algebraic_connectivity",
    "build_correlation_distance_matrix",
    "build_coupling_distance_matrix",
    "export_topology_optimisation_artifact",
    "spike_trace_correlation_distance",
    "validate_topology_hardware_manifest",
]
