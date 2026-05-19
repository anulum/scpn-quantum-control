# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum Error Correction
"""Quantum error correction: surface codes, MWPM decoding, error budgets,
fault-tolerant UPDE, and repetition code logical qubits.
"""

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

__all__ = [
    "ControlQEC",
    "SurfaceCode",
    "MWPMDecoder",
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
