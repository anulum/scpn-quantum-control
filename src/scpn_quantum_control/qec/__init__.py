# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum Error Correction
"""Quantum error correction analysis surfaces.

The package exports toric-code and biological graph MWPM decoders, DLA-protected
logical-memory prototypes, surface-code resource estimates, error budgets,
fault-tolerant UPDE scaffolds, and repetition-code logical qubits. It does not
export a union-find decoder.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .biological_diagnostics import (
        BiologicalSurfaceDiagnostics,
        analyse_biological_surface_code,
    )
    from .biological_pipeline import (
        BiologicalQecBatchExecution,
        BiologicalQecExecution,
        run_biological_qec_batch_execution,
        run_biological_qec_execution,
    )
    from .biological_surface_code import BiologicalMWPMDecoder, BiologicalSurfaceCode
    from .control_qec import ControlQEC, MWPMDecoder, SurfaceCode
    from .dla_protected_scar import (
        DLAProtectedScarPrototype,
        DLAProtectedScarSimulationResult,
        DLAProtectedScarSpec,
        build_dla_protected_scar_prototype,
        evaluate_dla_protected_scar_counts,
        simulate_dla_protected_scar_memory,
    )
    from .dla_protected_subspace import (
        DLAProtectedLogicalSyncWitness,
        DLAProtectedMemoryPrototype,
        DLAProtectedSubspaceSpec,
        DLAProtectedWitnessResult,
        DLAProtectionCertificate,
        build_dla_protected_memory_prototype,
        certify_dla_protected_subspace,
        evaluate_dla_protected_memory,
        protected_memory_mask,
        sync_memory_mask,
    )
    from .error_budget import (
        ErrorBudget,
        compare_error_budgets,
        compute_error_budget,
        logical_error_rate,
        minimum_code_distance,
    )
    from .fault_tolerant import FaultTolerantUPDE, LogicalQubit, RepetitionCodeUPDE
    from .logical_dla_parity import (
        LogicalDLAParityRow,
        MultiscaleComparison,
        compare_flat_surface_code_to_multiscale,
        estimate_logical_dla_parity_row,
        estimate_s7_resource_table,
        logical_dla_parity_markdown,
        logical_dla_parity_payload,
        repetition_scaffold_physical_qubits,
        surface_code_physical_qubits,
    )
    from .multiscale_qec import (
        MultiscaleQECResult,
        QECLevel,
        build_multiscale_qec,
        concatenated_logical_rate,
    )
    from .surface_code_upde import SurfaceCodeSpec, SurfaceCodeUPDE
    from .syndrome_flow import (
        SyndromeFlow,
        syndrome_flow_analysis,
    )

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "BiologicalSurfaceDiagnostics": (
        "scpn_quantum_control.qec.biological_diagnostics",
        "BiologicalSurfaceDiagnostics",
    ),
    "analyse_biological_surface_code": (
        "scpn_quantum_control.qec.biological_diagnostics",
        "analyse_biological_surface_code",
    ),
    "BiologicalQecBatchExecution": (
        "scpn_quantum_control.qec.biological_pipeline",
        "BiologicalQecBatchExecution",
    ),
    "BiologicalQecExecution": (
        "scpn_quantum_control.qec.biological_pipeline",
        "BiologicalQecExecution",
    ),
    "run_biological_qec_batch_execution": (
        "scpn_quantum_control.qec.biological_pipeline",
        "run_biological_qec_batch_execution",
    ),
    "run_biological_qec_execution": (
        "scpn_quantum_control.qec.biological_pipeline",
        "run_biological_qec_execution",
    ),
    "BiologicalMWPMDecoder": (
        "scpn_quantum_control.qec.biological_surface_code",
        "BiologicalMWPMDecoder",
    ),
    "BiologicalSurfaceCode": (
        "scpn_quantum_control.qec.biological_surface_code",
        "BiologicalSurfaceCode",
    ),
    "ControlQEC": ("scpn_quantum_control.qec.control_qec", "ControlQEC"),
    "MWPMDecoder": ("scpn_quantum_control.qec.control_qec", "MWPMDecoder"),
    "SurfaceCode": ("scpn_quantum_control.qec.control_qec", "SurfaceCode"),
    "DLAProtectedScarPrototype": (
        "scpn_quantum_control.qec.dla_protected_scar",
        "DLAProtectedScarPrototype",
    ),
    "DLAProtectedScarSimulationResult": (
        "scpn_quantum_control.qec.dla_protected_scar",
        "DLAProtectedScarSimulationResult",
    ),
    "DLAProtectedScarSpec": (
        "scpn_quantum_control.qec.dla_protected_scar",
        "DLAProtectedScarSpec",
    ),
    "build_dla_protected_scar_prototype": (
        "scpn_quantum_control.qec.dla_protected_scar",
        "build_dla_protected_scar_prototype",
    ),
    "evaluate_dla_protected_scar_counts": (
        "scpn_quantum_control.qec.dla_protected_scar",
        "evaluate_dla_protected_scar_counts",
    ),
    "simulate_dla_protected_scar_memory": (
        "scpn_quantum_control.qec.dla_protected_scar",
        "simulate_dla_protected_scar_memory",
    ),
    "DLAProtectedLogicalSyncWitness": (
        "scpn_quantum_control.qec.dla_protected_subspace",
        "DLAProtectedLogicalSyncWitness",
    ),
    "DLAProtectedMemoryPrototype": (
        "scpn_quantum_control.qec.dla_protected_subspace",
        "DLAProtectedMemoryPrototype",
    ),
    "DLAProtectedSubspaceSpec": (
        "scpn_quantum_control.qec.dla_protected_subspace",
        "DLAProtectedSubspaceSpec",
    ),
    "DLAProtectedWitnessResult": (
        "scpn_quantum_control.qec.dla_protected_subspace",
        "DLAProtectedWitnessResult",
    ),
    "DLAProtectionCertificate": (
        "scpn_quantum_control.qec.dla_protected_subspace",
        "DLAProtectionCertificate",
    ),
    "build_dla_protected_memory_prototype": (
        "scpn_quantum_control.qec.dla_protected_subspace",
        "build_dla_protected_memory_prototype",
    ),
    "certify_dla_protected_subspace": (
        "scpn_quantum_control.qec.dla_protected_subspace",
        "certify_dla_protected_subspace",
    ),
    "evaluate_dla_protected_memory": (
        "scpn_quantum_control.qec.dla_protected_subspace",
        "evaluate_dla_protected_memory",
    ),
    "protected_memory_mask": (
        "scpn_quantum_control.qec.dla_protected_subspace",
        "protected_memory_mask",
    ),
    "sync_memory_mask": ("scpn_quantum_control.qec.dla_protected_subspace", "sync_memory_mask"),
    "ErrorBudget": ("scpn_quantum_control.qec.error_budget", "ErrorBudget"),
    "compare_error_budgets": ("scpn_quantum_control.qec.error_budget", "compare_error_budgets"),
    "compute_error_budget": ("scpn_quantum_control.qec.error_budget", "compute_error_budget"),
    "logical_error_rate": ("scpn_quantum_control.qec.error_budget", "logical_error_rate"),
    "minimum_code_distance": ("scpn_quantum_control.qec.error_budget", "minimum_code_distance"),
    "FaultTolerantUPDE": ("scpn_quantum_control.qec.fault_tolerant", "FaultTolerantUPDE"),
    "LogicalQubit": ("scpn_quantum_control.qec.fault_tolerant", "LogicalQubit"),
    "RepetitionCodeUPDE": ("scpn_quantum_control.qec.fault_tolerant", "RepetitionCodeUPDE"),
    "LogicalDLAParityRow": ("scpn_quantum_control.qec.logical_dla_parity", "LogicalDLAParityRow"),
    "MultiscaleComparison": (
        "scpn_quantum_control.qec.logical_dla_parity",
        "MultiscaleComparison",
    ),
    "compare_flat_surface_code_to_multiscale": (
        "scpn_quantum_control.qec.logical_dla_parity",
        "compare_flat_surface_code_to_multiscale",
    ),
    "estimate_logical_dla_parity_row": (
        "scpn_quantum_control.qec.logical_dla_parity",
        "estimate_logical_dla_parity_row",
    ),
    "estimate_s7_resource_table": (
        "scpn_quantum_control.qec.logical_dla_parity",
        "estimate_s7_resource_table",
    ),
    "logical_dla_parity_markdown": (
        "scpn_quantum_control.qec.logical_dla_parity",
        "logical_dla_parity_markdown",
    ),
    "logical_dla_parity_payload": (
        "scpn_quantum_control.qec.logical_dla_parity",
        "logical_dla_parity_payload",
    ),
    "repetition_scaffold_physical_qubits": (
        "scpn_quantum_control.qec.logical_dla_parity",
        "repetition_scaffold_physical_qubits",
    ),
    "surface_code_physical_qubits": (
        "scpn_quantum_control.qec.logical_dla_parity",
        "surface_code_physical_qubits",
    ),
    "MultiscaleQECResult": ("scpn_quantum_control.qec.multiscale_qec", "MultiscaleQECResult"),
    "QECLevel": ("scpn_quantum_control.qec.multiscale_qec", "QECLevel"),
    "build_multiscale_qec": ("scpn_quantum_control.qec.multiscale_qec", "build_multiscale_qec"),
    "concatenated_logical_rate": (
        "scpn_quantum_control.qec.multiscale_qec",
        "concatenated_logical_rate",
    ),
    "SurfaceCodeSpec": ("scpn_quantum_control.qec.surface_code_upde", "SurfaceCodeSpec"),
    "SurfaceCodeUPDE": ("scpn_quantum_control.qec.surface_code_upde", "SurfaceCodeUPDE"),
    "SyndromeFlow": ("scpn_quantum_control.qec.syndrome_flow", "SyndromeFlow"),
    "syndrome_flow_analysis": ("scpn_quantum_control.qec.syndrome_flow", "syndrome_flow_analysis"),
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
    "ControlQEC",
    "SurfaceCode",
    "MWPMDecoder",
    "BiologicalSurfaceCode",
    "BiologicalMWPMDecoder",
    "BiologicalSurfaceDiagnostics",
    "analyse_biological_surface_code",
    "BiologicalQecExecution",
    "BiologicalQecBatchExecution",
    "run_biological_qec_execution",
    "run_biological_qec_batch_execution",
    "DLAProtectedLogicalSyncWitness",
    "DLAProtectedMemoryPrototype",
    "DLAProtectedSubspaceSpec",
    "DLAProtectedWitnessResult",
    "DLAProtectionCertificate",
    "build_dla_protected_memory_prototype",
    "certify_dla_protected_subspace",
    "evaluate_dla_protected_memory",
    "protected_memory_mask",
    "sync_memory_mask",
    "DLAProtectedScarPrototype",
    "DLAProtectedScarSimulationResult",
    "DLAProtectedScarSpec",
    "build_dla_protected_scar_prototype",
    "evaluate_dla_protected_scar_counts",
    "simulate_dla_protected_scar_memory",
    "ErrorBudget",
    "compute_error_budget",
    "compare_error_budgets",
    "logical_error_rate",
    "minimum_code_distance",
    "FaultTolerantUPDE",
    "RepetitionCodeUPDE",
    "LogicalQubit",
    "LogicalDLAParityRow",
    "MultiscaleComparison",
    "compare_flat_surface_code_to_multiscale",
    "estimate_logical_dla_parity_row",
    "estimate_s7_resource_table",
    "logical_dla_parity_markdown",
    "logical_dla_parity_payload",
    "repetition_scaffold_physical_qubits",
    "surface_code_physical_qubits",
    "SurfaceCodeSpec",
    "SurfaceCodeUPDE",
    "MultiscaleQECResult",
    "QECLevel",
    "SyndromeFlow",
    "build_multiscale_qec",
    "concatenated_logical_rate",
    "syndrome_flow_analysis",
]
