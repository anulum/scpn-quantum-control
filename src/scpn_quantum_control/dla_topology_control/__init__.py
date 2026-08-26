# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — DLA/topology constrained-control facade
"""Public finite synthetic DLA/topology differentiability facade."""

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
