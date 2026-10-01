# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Chimera-control public facade
"""Synthetic chimera and hierarchical synchronisation-control surfaces."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "CHIMERA_CONTROL_EVIDENCE_DATE": (
        "scpn_quantum_control.chimera_control.evidence",
        "CHIMERA_CONTROL_EVIDENCE_DATE",
    ),
    "CHIMERA_CONTROL_EVIDENCE_SCHEMA": (
        "scpn_quantum_control.chimera_control.evidence",
        "CHIMERA_CONTROL_EVIDENCE_SCHEMA",
    ),
    "ChimeraMultiscaleEvidence": (
        "scpn_quantum_control.chimera_control.evidence",
        "ChimeraMultiscaleEvidence",
    ),
    "ChimeraSupportRow": ("scpn_quantum_control.chimera_control.evidence", "ChimeraSupportRow"),
    "SyntheticRegimeEvidence": (
        "scpn_quantum_control.chimera_control.evidence",
        "SyntheticRegimeEvidence",
    ),
    "build_chimera_multiscale_evidence": (
        "scpn_quantum_control.chimera_control.evidence",
        "build_chimera_multiscale_evidence",
    ),
    "render_chimera_multiscale_markdown": (
        "scpn_quantum_control.chimera_control.evidence",
        "render_chimera_multiscale_markdown",
    ),
    "write_chimera_multiscale_evidence": (
        "scpn_quantum_control.chimera_control.evidence",
        "write_chimera_multiscale_evidence",
    ),
    "PhaseControlProposal": (
        "scpn_quantum_control.chimera_control.objectives",
        "PhaseControlProposal",
    ),
    "build_chimera_control_objective": (
        "scpn_quantum_control.chimera_control.objectives",
        "build_chimera_control_objective",
    ),
    "propose_phase_control_step": (
        "scpn_quantum_control.chimera_control.objectives",
        "propose_phase_control_step",
    ),
    "LevelOrderParameterSummary": (
        "scpn_quantum_control.chimera_control.observables",
        "LevelOrderParameterSummary",
    ),
    "MultiscaleOrderParameterReport": (
        "scpn_quantum_control.chimera_control.observables",
        "MultiscaleOrderParameterReport",
    ),
    "measure_multiscale_order_parameters": (
        "scpn_quantum_control.chimera_control.observables",
        "measure_multiscale_order_parameters",
    ),
    "CHIMERA_CONTROL_CLAIM_BOUNDARY": (
        "scpn_quantum_control.chimera_control.schema",
        "CHIMERA_CONTROL_CLAIM_BOUNDARY",
    ),
    "ChimeraControlSpecification": (
        "scpn_quantum_control.chimera_control.schema",
        "ChimeraControlSpecification",
    ),
    "HierarchyLevel": ("scpn_quantum_control.chimera_control.schema", "HierarchyLevel"),
    "HierarchyTarget": ("scpn_quantum_control.chimera_control.schema", "HierarchyTarget"),
    "MultiscaleHierarchy": ("scpn_quantum_control.chimera_control.schema", "MultiscaleHierarchy"),
    "SyntheticRegime": ("scpn_quantum_control.chimera_control.schema", "SyntheticRegime"),
    "two_population_hierarchy": (
        "scpn_quantum_control.chimera_control.schema",
        "two_population_hierarchy",
    ),
    "SYNTHETIC_CHIMERA_SOURCE": (
        "scpn_quantum_control.chimera_control.synthetic",
        "SYNTHETIC_CHIMERA_SOURCE",
    ),
    "SyntheticChimeraConfig": (
        "scpn_quantum_control.chimera_control.synthetic",
        "SyntheticChimeraConfig",
    ),
    "SyntheticChimeraRun": (
        "scpn_quantum_control.chimera_control.synthetic",
        "SyntheticChimeraRun",
    ),
    "build_two_population_coupling": (
        "scpn_quantum_control.chimera_control.synthetic",
        "build_two_population_coupling",
    ),
    "generate_two_population_chimera": (
        "scpn_quantum_control.chimera_control.synthetic",
        "generate_two_population_chimera",
    ),
    "HierarchyCouplingSummary": (
        "scpn_quantum_control.chimera_control.topology",
        "HierarchyCouplingSummary",
    ),
    "TopologyProjectionReport": (
        "scpn_quantum_control.chimera_control.topology",
        "TopologyProjectionReport",
    ),
    "project_chimera_coupling": (
        "scpn_quantum_control.chimera_control.topology",
        "project_chimera_coupling",
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
