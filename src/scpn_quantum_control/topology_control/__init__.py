# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Topology Control
"""Constrained persistent-H1 optimisation for coupling graphs."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "TopologyOptimisationArtifact": (
        "scpn_quantum_control.topology_control.artefacts",
        "TopologyOptimisationArtifact",
    ),
    "export_topology_optimisation_artifact": (
        "scpn_quantum_control.topology_control.artefacts",
        "export_topology_optimisation_artifact",
    ),
    "H1Summary": ("scpn_quantum_control.topology_control.complexes", "H1Summary"),
    "NetworkCycleBackend": (
        "scpn_quantum_control.topology_control.complexes",
        "NetworkCycleBackend",
    ),
    "PersistenceDiagram": (
        "scpn_quantum_control.topology_control.complexes",
        "PersistenceDiagram",
    ),
    "PersistentHomologyBackend": (
        "scpn_quantum_control.topology_control.complexes",
        "PersistentHomologyBackend",
    ),
    "RipserPHBackend": ("scpn_quantum_control.topology_control.complexes", "RipserPHBackend"),
    "build_correlation_distance_matrix": (
        "scpn_quantum_control.topology_control.complexes",
        "build_correlation_distance_matrix",
    ),
    "build_coupling_distance_matrix": (
        "scpn_quantum_control.topology_control.complexes",
        "build_coupling_distance_matrix",
    ),
    "spike_trace_correlation_distance": (
        "scpn_quantum_control.topology_control.complexes",
        "spike_trace_correlation_distance",
    ),
    "CouplingGraphBounds": (
        "scpn_quantum_control.topology_control.constraints",
        "CouplingGraphBounds",
    ),
    "HardwareEmbeddingConstraint": (
        "scpn_quantum_control.topology_control.constraints",
        "HardwareEmbeddingConstraint",
    ),
    "TopologyConstraintLedger": (
        "scpn_quantum_control.topology_control.constraints",
        "TopologyConstraintLedger",
    ),
    "algebraic_connectivity": (
        "scpn_quantum_control.topology_control.constraints",
        "algebraic_connectivity",
    ),
    "TopologyHardwareManifest": (
        "scpn_quantum_control.topology_control.hardware_integration",
        "TopologyHardwareManifest",
    ),
    "validate_topology_hardware_manifest": (
        "scpn_quantum_control.topology_control.hardware_integration",
        "validate_topology_hardware_manifest",
    ),
    "CouplingTopologyObjective": (
        "scpn_quantum_control.topology_control.objectives",
        "CouplingTopologyObjective",
    ),
    "DegeneracyMode": ("scpn_quantum_control.topology_control.objectives", "DegeneracyMode"),
    "ObjectiveBreakdown": (
        "scpn_quantum_control.topology_control.objectives",
        "ObjectiveBreakdown",
    ),
    "ProjectedScipyOptimizer": (
        "scpn_quantum_control.topology_control.optimizers",
        "ProjectedScipyOptimizer",
    ),
    "ProjectedSPSAOptimizer": (
        "scpn_quantum_control.topology_control.optimizers",
        "ProjectedSPSAOptimizer",
    ),
    "TopologyOptimisationStep": (
        "scpn_quantum_control.topology_control.optimizers",
        "TopologyOptimisationStep",
    ),
    "TopologyOptimisationTrace": (
        "scpn_quantum_control.topology_control.optimizers",
        "TopologyOptimisationTrace",
    ),
    "TopologicalDynamicCouplingPolicy": (
        "scpn_quantum_control.topology_control.qsnn_integration",
        "TopologicalDynamicCouplingPolicy",
    ),
}


def __getattr__(name: str) -> Any:
    """Resolve and cache a public export from its original owning module.

    Parameters
    ----------
    name
        Public export requested through this package.

    Returns
    -------
    Any
        Original object, including module-valued exports.

    Raises
    ------
    AttributeError
        If the name is undeclared or the original module lacks its attribute.
    ImportError
        If the owning module cannot be imported.

    """
    target = _PUBLIC_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    origin = import_module(target[0])
    value = origin if target[1] is None else getattr(origin, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List cached and deferred names for inspection tools.

    Returns
    -------
    list[str]
        Sorted package namespace and declared lazy export names.

    """
    return sorted(set(globals()) | set(_PUBLIC_EXPORTS))


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
