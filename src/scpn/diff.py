# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — diff module
# SCPN compatibility differentiable namespace.
"""Compatibility import path for :mod:`scpn_quantum_control.diff`."""

from __future__ import annotations

from scpn_quantum_control.diff import (
    DIFFERENTIABLE_CIRCUIT_CONTRACT_CLAIM_BOUNDARY,
    DIFFERENTIABLE_CIRCUIT_SCHEMA,
    BackendCapabilityMetadata,
    DifferentiableCircuit,
    DifferentiableCircuitContractAuditResult,
    DifferentiableCircuitContractCheck,
    DifferentiableCircuitContractStatus,
    DifferentiableCircuitDiagnostics,
    EstimatorProvenance,
    JITExplanation,
    QuantumFunction,
    QuantumGradientTape,
    ShotPolicy,
    TapeGradientRecord,
    differentiable_circuit,
    grad,
    gradient_tape,
    hessian,
    jacfwd,
    jacobian,
    jacrev,
    jit_or_explain,
    jvp,
    namespace_metadata,
    run_differentiable_circuit_contract_audit,
    supported_transforms,
    value_and_grad,
    vjp,
    vmap,
)

__all__ = [
    "BackendCapabilityMetadata",
    "DIFFERENTIABLE_CIRCUIT_CONTRACT_CLAIM_BOUNDARY",
    "DIFFERENTIABLE_CIRCUIT_SCHEMA",
    "DifferentiableCircuit",
    "DifferentiableCircuitContractAuditResult",
    "DifferentiableCircuitContractCheck",
    "DifferentiableCircuitContractStatus",
    "DifferentiableCircuitDiagnostics",
    "EstimatorProvenance",
    "JITExplanation",
    "QuantumFunction",
    "QuantumGradientTape",
    "ShotPolicy",
    "TapeGradientRecord",
    "differentiable_circuit",
    "grad",
    "gradient_tape",
    "hessian",
    "jacfwd",
    "jacobian",
    "jacrev",
    "jit_or_explain",
    "jvp",
    "namespace_metadata",
    "run_differentiable_circuit_contract_audit",
    "supported_transforms",
    "value_and_grad",
    "vjp",
    "vmap",
]
