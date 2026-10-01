# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — DLA/topology constrained-control facade
"""Public finite synthetic DLA/topology differentiability facade."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .evidence import (
        TOPOLOGY_CONTROL_EVIDENCE_DATE,
        TOPOLOGY_CONTROL_EVIDENCE_SCHEMA,
        DlaTopologyControlEvidence,
        build_dla_topology_control_evidence,
        render_dla_topology_control_markdown,
        write_dla_topology_control_evidence,
    )
    from .objectives import (
        ParityProtectedObjectiveEvaluation,
        ParityProtectedQuadraticObjective,
    )
    from .optimizer import (
        ParityProjectedOptimisationTrace,
        ProjectedGradientConfig,
        ProjectedGradientStep,
        optimise_parity_protected_state,
    )
    from .parity import ParityLeakageEvaluation, ParitySectorProjector
    from .projection import (
        TopologyProjectionDifferential,
        topology_projection_jvp,
        topology_projection_support,
        topology_projection_vjp,
    )
    from .schema import (
        DLA_TOPOLOGY_CLAIM_BOUNDARY,
        ConstraintSupportRow,
        DifferentiabilityKind,
        DifferentiabilityReport,
        ParitySector,
        UnsupportedDifferentiableConstraintError,
    )

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "TOPOLOGY_CONTROL_EVIDENCE_DATE": (
        "scpn_quantum_control.dla_topology_control.evidence",
        "TOPOLOGY_CONTROL_EVIDENCE_DATE",
    ),
    "TOPOLOGY_CONTROL_EVIDENCE_SCHEMA": (
        "scpn_quantum_control.dla_topology_control.evidence",
        "TOPOLOGY_CONTROL_EVIDENCE_SCHEMA",
    ),
    "DlaTopologyControlEvidence": (
        "scpn_quantum_control.dla_topology_control.evidence",
        "DlaTopologyControlEvidence",
    ),
    "build_dla_topology_control_evidence": (
        "scpn_quantum_control.dla_topology_control.evidence",
        "build_dla_topology_control_evidence",
    ),
    "render_dla_topology_control_markdown": (
        "scpn_quantum_control.dla_topology_control.evidence",
        "render_dla_topology_control_markdown",
    ),
    "write_dla_topology_control_evidence": (
        "scpn_quantum_control.dla_topology_control.evidence",
        "write_dla_topology_control_evidence",
    ),
    "ParityProtectedObjectiveEvaluation": (
        "scpn_quantum_control.dla_topology_control.objectives",
        "ParityProtectedObjectiveEvaluation",
    ),
    "ParityProtectedQuadraticObjective": (
        "scpn_quantum_control.dla_topology_control.objectives",
        "ParityProtectedQuadraticObjective",
    ),
    "ParityProjectedOptimisationTrace": (
        "scpn_quantum_control.dla_topology_control.optimizer",
        "ParityProjectedOptimisationTrace",
    ),
    "ProjectedGradientConfig": (
        "scpn_quantum_control.dla_topology_control.optimizer",
        "ProjectedGradientConfig",
    ),
    "ProjectedGradientStep": (
        "scpn_quantum_control.dla_topology_control.optimizer",
        "ProjectedGradientStep",
    ),
    "optimise_parity_protected_state": (
        "scpn_quantum_control.dla_topology_control.optimizer",
        "optimise_parity_protected_state",
    ),
    "ParityLeakageEvaluation": (
        "scpn_quantum_control.dla_topology_control.parity",
        "ParityLeakageEvaluation",
    ),
    "ParitySectorProjector": (
        "scpn_quantum_control.dla_topology_control.parity",
        "ParitySectorProjector",
    ),
    "TopologyProjectionDifferential": (
        "scpn_quantum_control.dla_topology_control.projection",
        "TopologyProjectionDifferential",
    ),
    "topology_projection_jvp": (
        "scpn_quantum_control.dla_topology_control.projection",
        "topology_projection_jvp",
    ),
    "topology_projection_support": (
        "scpn_quantum_control.dla_topology_control.projection",
        "topology_projection_support",
    ),
    "topology_projection_vjp": (
        "scpn_quantum_control.dla_topology_control.projection",
        "topology_projection_vjp",
    ),
    "DLA_TOPOLOGY_CLAIM_BOUNDARY": (
        "scpn_quantum_control.dla_topology_control.schema",
        "DLA_TOPOLOGY_CLAIM_BOUNDARY",
    ),
    "ConstraintSupportRow": (
        "scpn_quantum_control.dla_topology_control.schema",
        "ConstraintSupportRow",
    ),
    "DifferentiabilityKind": (
        "scpn_quantum_control.dla_topology_control.schema",
        "DifferentiabilityKind",
    ),
    "DifferentiabilityReport": (
        "scpn_quantum_control.dla_topology_control.schema",
        "DifferentiabilityReport",
    ),
    "ParitySector": ("scpn_quantum_control.dla_topology_control.schema", "ParitySector"),
    "UnsupportedDifferentiableConstraintError": (
        "scpn_quantum_control.dla_topology_control.schema",
        "UnsupportedDifferentiableConstraintError",
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
    "TOPOLOGY_CONTROL_EVIDENCE_DATE",
    "TOPOLOGY_CONTROL_EVIDENCE_SCHEMA",
    "DLA_TOPOLOGY_CLAIM_BOUNDARY",
    "ConstraintSupportRow",
    "DifferentiabilityKind",
    "DifferentiabilityReport",
    "DlaTopologyControlEvidence",
    "ParityLeakageEvaluation",
    "ParityProjectedOptimisationTrace",
    "ParityProtectedObjectiveEvaluation",
    "ParityProtectedQuadraticObjective",
    "ParitySector",
    "ParitySectorProjector",
    "ProjectedGradientConfig",
    "ProjectedGradientStep",
    "TopologyProjectionDifferential",
    "UnsupportedDifferentiableConstraintError",
    "build_dla_topology_control_evidence",
    "optimise_parity_protected_state",
    "render_dla_topology_control_markdown",
    "topology_projection_jvp",
    "topology_projection_support",
    "topology_projection_vjp",
    "write_dla_topology_control_evidence",
]
