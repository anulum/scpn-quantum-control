# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Quantum Benchmarks
"""Quantum-gradient case builders for the differentiable conformance benchmark suite."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from ..phase.jax_bridge import (
    jax_phase_qnode_aot_export_audit,
    jax_phase_qnode_native_transform_audit,
    jax_phase_qnode_pytree_transform_audit,
    jax_phase_qnode_sharding_transform_audit,
)
from ..phase.param_shift import verify_parameter_shift_gradient
from ..phase.qnode_circuit import (
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeOperation,
    execute_phase_qnode_circuit,
    parameter_shift_phase_qnode_gradient,
)
from ..phase.torch_bridge import (
    torch_phase_qnode_compile_audit,
    torch_phase_qnode_compile_boundary_audit,
    torch_phase_qnode_transform_audit,
    torch_phase_qnode_value_and_grad,
)
from .differentiable_programming_contracts import QuantumGradientBenchmarkResult, _max_abs_error


def _single_rotation_quantum_gradient_case() -> QuantumGradientBenchmarkResult:
    values = np.array([0.4], dtype=np.float64)

    def objective(params: NDArray[np.float64]) -> float:
        return float(np.cos(params[0]))

    analytic = np.array([-math.sin(values[0])], dtype=np.float64)
    return _quantum_gradient_case(
        "single_rotation_parameter_shift",
        objective,
        values,
        analytic,
    )


def _two_parameter_quantum_gradient_case() -> QuantumGradientBenchmarkResult:
    values = np.array([0.2, -0.4], dtype=np.float64)

    def objective(params: NDArray[np.float64]) -> float:
        return float(np.cos(params[0]) + 0.25 * np.sin(params[1]))

    analytic = np.array(
        [-math.sin(values[0]), 0.25 * math.cos(values[1])],
        dtype=np.float64,
    )
    return _quantum_gradient_case(
        "two_parameter_phase_expectation",
        objective,
        values,
        analytic,
    )


def _sparse_ising_chain_quantum_gradient_case() -> QuantumGradientBenchmarkResult:
    values = np.array([0.15, -0.35, 0.55, -0.75, 0.95, -1.15], dtype=np.float64)
    local_fields = np.array([0.31, -0.17, 0.23, 0.41, -0.29, 0.37], dtype=np.float64)
    couplings = np.array([0.19, -0.11, 0.07, 0.13, -0.05], dtype=np.float64)

    def objective(params: NDArray[np.float64]) -> float:
        cos_terms = np.cos(params)
        local_energy = float(np.dot(local_fields, cos_terms))
        coupling_energy = float(np.dot(couplings, cos_terms[:-1] * cos_terms[1:]))
        return local_energy + coupling_energy

    analytic = -local_fields * np.sin(values)
    analytic[:-1] -= couplings * np.sin(values[:-1]) * np.cos(values[1:])
    analytic[1:] -= couplings * np.cos(values[:-1]) * np.sin(values[1:])
    return _quantum_gradient_case(
        "sparse_ising_chain_six_qubit_expectation",
        objective,
        values,
        analytic,
        claim_boundary=(
            "deterministic sparse Hamiltonian expectation-gradient conformance "
            "for a six-qubit nearest-neighbour Ising chain; no wall-clock "
            "performance, hardware, provider, or framework-autodiff claim"
        ),
    )


def _torch_registered_phase_qnode_statevector_case() -> QuantumGradientBenchmarkResult:
    values = np.array([0.37, -0.21], dtype=np.float64)
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("rx", (1,), parameter_index=1),
            PhaseQNodeOperation("cnot", (0, 1)),
        ),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )
    result = torch_phase_qnode_value_and_grad(circuit, values, tolerance=1.0e-8)
    finite_difference_certificate = verify_parameter_shift_gradient(
        lambda params: torch_phase_qnode_value_and_grad(circuit, params, tolerance=1.0e-8).value,
        values,
    )
    return QuantumGradientBenchmarkResult(
        case_id="torch_registered_phase_qnode_statevector_lowering",
        category="quantum-gradient",
        value=result.value,
        parameter_shift_gradient=result.gradient,
        finite_difference_gradient=finite_difference_certificate.finite_difference_gradient,
        analytic_gradient=result.parameter_shift_gradient,
        max_abs_reference_error=result.max_abs_error,
        max_abs_finite_difference_error=finite_difference_certificate.max_abs_error,
        verification_passed=result.passed and finite_difference_certificate.passed,
        evaluations=finite_difference_certificate.total_evaluations + (2 * values.size),
        claim_boundary=(
            "native PyTorch autograd statevector lowering for deterministic "
            "registered local Phase-QNode circuits compared with SCPN "
            "parameter-shift references; no wall-clock performance claim and "
            "no provider, hardware, isolated benchmark, or performance promotion"
        ),
    )


def _torch_registered_phase_qnode_func_transform_case() -> QuantumGradientBenchmarkResult:
    values = np.array([0.37, -0.21], dtype=np.float64)
    params_batch = np.array([[0.37, -0.21], [0.4, -0.25]], dtype=np.float64)
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("rx", (1,), parameter_index=1),
            PhaseQNodeOperation("cnot", (0, 1)),
        ),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )
    result = torch_phase_qnode_transform_audit(
        circuit,
        values,
        params_batch=params_batch,
        tolerance=1.0e-8,
    )
    finite_difference_certificate = verify_parameter_shift_gradient(
        lambda params: (
            torch_phase_qnode_transform_audit(
                circuit,
                params,
                params_batch=params[None, :],
                tolerance=1.0e-8,
            ).value
        ),
        values,
    )
    return QuantumGradientBenchmarkResult(
        case_id="torch_registered_phase_qnode_func_transform_lowering",
        category="quantum-gradient",
        value=result.value,
        parameter_shift_gradient=result.gradient,
        finite_difference_gradient=finite_difference_certificate.finite_difference_gradient,
        analytic_gradient=result.parameter_shift_gradient,
        max_abs_reference_error=result.max_abs_error,
        max_abs_finite_difference_error=finite_difference_certificate.max_abs_error,
        verification_passed=result.passed and finite_difference_certificate.passed,
        evaluations=finite_difference_certificate.total_evaluations + (4 * values.size),
        claim_boundary=(
            "native PyTorch torch.func grad, jacrev, and vmap statevector "
            "lowering for deterministic registered local Phase-QNode circuits "
            "compared with SCPN parameter-shift references; no wall-clock "
            "performance claim and no torch.compile, CUDA, provider, hardware, "
            "isolated benchmark, or performance promotion"
        ),
    )


def _torch_registered_phase_qnode_compile_case() -> QuantumGradientBenchmarkResult:
    values = np.array([0.37, -0.21], dtype=np.float64)
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("rx", (1,), parameter_index=1),
            PhaseQNodeOperation("cnot", (0, 1)),
        ),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )
    result = torch_phase_qnode_compile_audit(
        circuit,
        values,
        tolerance=1.0e-8,
        fullgraph=False,
    )
    finite_difference_certificate = verify_parameter_shift_gradient(
        lambda params: (
            torch_phase_qnode_compile_audit(
                circuit,
                params,
                tolerance=1.0e-8,
                fullgraph=False,
            ).value
        ),
        values,
    )
    return QuantumGradientBenchmarkResult(
        case_id="torch_registered_phase_qnode_compile_lowering",
        category="quantum-gradient",
        value=result.value,
        parameter_shift_gradient=result.gradient,
        finite_difference_gradient=finite_difference_certificate.finite_difference_gradient,
        analytic_gradient=result.parameter_shift_gradient,
        max_abs_reference_error=result.max_abs_error,
        max_abs_finite_difference_error=finite_difference_certificate.max_abs_error,
        verification_passed=result.passed and finite_difference_certificate.passed,
        evaluations=finite_difference_certificate.total_evaluations + (4 * values.size),
        claim_boundary=(
            "native PyTorch non-fullgraph torch.compile statevector lowering "
            "for deterministic registered local Phase-QNode circuits compared "
            "with SCPN parameter-shift references; no wall-clock performance "
            "claim and no fullgraph compile, CUDA, provider, hardware, "
            "isolated benchmark, or performance promotion"
        ),
    )


def _torch_registered_phase_qnode_compile_boundary_case() -> QuantumGradientBenchmarkResult:
    values = np.array([0.37, -0.21], dtype=np.float64)
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("rx", (1,), parameter_index=1),
            PhaseQNodeOperation("cnot", (0, 1)),
        ),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )
    result = torch_phase_qnode_compile_boundary_audit(
        circuit,
        values,
        tolerance=1.0e-8,
    )
    finite_difference_certificate = verify_parameter_shift_gradient(
        lambda params: execute_phase_qnode_circuit(circuit, params).value,
        values,
    )
    return QuantumGradientBenchmarkResult(
        case_id="torch_registered_phase_qnode_compile_boundary_diagnostic",
        category="quantum-gradient",
        value=result.non_fullgraph_value,
        parameter_shift_gradient=result.non_fullgraph_gradient,
        finite_difference_gradient=finite_difference_certificate.finite_difference_gradient,
        analytic_gradient=result.parameter_shift_gradient,
        max_abs_reference_error=result.max_abs_reference_error,
        max_abs_finite_difference_error=finite_difference_certificate.max_abs_error,
        verification_passed=result.passed and finite_difference_certificate.passed,
        evaluations=finite_difference_certificate.total_evaluations + (6 * values.size),
        claim_boundary=(
            "native PyTorch torch.compile boundary diagnostic for deterministic "
            "registered local Phase-QNode circuits compared with SCPN "
            "parameter-shift references; dynamic-shape, fullgraph compiled-frame, "
            "AOTAutograd/export, CUDA, provider, hardware, isolated benchmark, "
            "and performance promotion remain blocked, with no wall-clock "
            "performance claim"
        ),
    )


def _jax_registered_phase_qnode_native_transform_case() -> QuantumGradientBenchmarkResult:
    values = np.array([0.37, -0.21], dtype=np.float64)
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("rx", (1,), parameter_index=1),
            PhaseQNodeOperation("cnot", (0, 1)),
        ),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )
    result = jax_phase_qnode_native_transform_audit(circuit, values, tolerance=1.0e-8)
    finite_difference_certificate = verify_parameter_shift_gradient(
        lambda params: (
            jax_phase_qnode_native_transform_audit(
                circuit,
                params,
                tolerance=1.0e-8,
            ).value
        ),
        values,
    )
    return QuantumGradientBenchmarkResult(
        case_id="jax_registered_phase_qnode_native_transform_lowering",
        category="quantum-gradient",
        value=result.value,
        parameter_shift_gradient=result.gradient,
        finite_difference_gradient=finite_difference_certificate.finite_difference_gradient,
        analytic_gradient=result.parameter_shift_gradient,
        max_abs_reference_error=max(
            result.max_abs_gradient_error,
            result.max_abs_transform_error,
            result.max_abs_hessian_symmetry_error,
        ),
        max_abs_finite_difference_error=finite_difference_certificate.max_abs_error,
        verification_passed=result.passed and finite_difference_certificate.passed,
        evaluations=finite_difference_certificate.total_evaluations
        + (2 * values.size * (result.batch_params.shape[0] + 1)),
        claim_boundary=(
            "native JAX grad, value_and_grad, jacfwd, jacrev, hessian, jvp, "
            "vjp, vmap, and jit statevector lowering for deterministic "
            "registered local Phase-QNode circuits compared with SCPN "
            "parameter-shift references; no wall-clock performance claim and "
            "no provider, hardware, isolated benchmark, or performance promotion"
        ),
    )


def _jax_registered_phase_qnode_pytree_transform_case() -> QuantumGradientBenchmarkResult:
    values = np.array([0.37, -0.21], dtype=np.float64)
    params_pytree = {
        "parameter_0": np.array([values[0]], dtype=np.float64),
        "parameter_1": (np.array([values[1]], dtype=np.float64),),
    }
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("rx", (1,), parameter_index=1),
            PhaseQNodeOperation("cnot", (0, 1)),
        ),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )
    result = jax_phase_qnode_pytree_transform_audit(circuit, params_pytree, tolerance=1.0e-8)
    finite_difference_certificate = verify_parameter_shift_gradient(
        lambda params: (
            jax_phase_qnode_pytree_transform_audit(
                circuit,
                {
                    "parameter_0": np.array([params[0]], dtype=np.float64),
                    "parameter_1": (np.array([params[1]], dtype=np.float64),),
                },
                tolerance=1.0e-8,
            ).value
        ),
        values,
    )
    return QuantumGradientBenchmarkResult(
        case_id="jax_registered_phase_qnode_pytree_transform_lowering",
        category="quantum-gradient",
        value=result.value,
        parameter_shift_gradient=result.gradient,
        finite_difference_gradient=finite_difference_certificate.finite_difference_gradient,
        analytic_gradient=result.parameter_shift_gradient,
        max_abs_reference_error=max(
            result.max_abs_gradient_error,
            result.max_abs_transform_error,
            result.max_abs_hessian_symmetry_error,
        ),
        max_abs_finite_difference_error=finite_difference_certificate.max_abs_error,
        verification_passed=result.passed and finite_difference_certificate.passed,
        evaluations=finite_difference_certificate.total_evaluations
        + (2 * values.size * (result.batch_params.shape[0] + 1)),
        claim_boundary=(
            "native JAX PyTree grad, value_and_grad, jacfwd, jacrev, "
            "hessian, jvp, vjp, vmap, and jit statevector lowering for deterministic "
            "registered local Phase-QNode circuits compared with SCPN "
            "parameter-shift references; no wall-clock performance claim and "
            "no provider, hardware, isolated benchmark, or performance promotion"
        ),
    )


def _jax_registered_phase_qnode_sharding_transform_case() -> QuantumGradientBenchmarkResult:
    import jax

    local_device_count = int(jax.local_device_count())
    values = np.array([0.37, -0.21], dtype=np.float64)
    offsets = np.arange(local_device_count, dtype=np.float64)[:, None] * np.array(
        [[0.01, -0.015]],
        dtype=np.float64,
    )
    params_batch = values[None, :] + offsets
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("rx", (1,), parameter_index=1),
            PhaseQNodeOperation("cnot", (0, 1)),
        ),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )
    result = jax_phase_qnode_sharding_transform_audit(
        circuit,
        params_batch,
        tolerance=1.0e-8,
    )
    finite_difference_certificate = verify_parameter_shift_gradient(
        lambda params: jax_phase_qnode_sharding_transform_audit(
            circuit,
            params[None, :] + offsets,
            tolerance=1.0e-8,
        ).values[0],
        values,
    )
    return QuantumGradientBenchmarkResult(
        case_id="jax_registered_phase_qnode_pmap_sharding_lowering",
        category="quantum-gradient",
        value=float(result.values[0]),
        parameter_shift_gradient=result.gradients[0],
        finite_difference_gradient=finite_difference_certificate.finite_difference_gradient,
        analytic_gradient=result.parameter_shift_gradients[0],
        max_abs_reference_error=max(result.max_abs_value_error, result.max_abs_gradient_error),
        max_abs_finite_difference_error=finite_difference_certificate.max_abs_error,
        verification_passed=result.passed and finite_difference_certificate.passed,
        evaluations=finite_difference_certificate.total_evaluations + (2 * values.size),
        claim_boundary=(
            "native JAX pmap statevector value-and-gradient lowering with one "
            "registered local Phase-QNode parameter row per local device compared "
            "with SCPN parameter-shift references; single-device CPU runs are "
            "sharding smoke evidence only, with no wall-clock performance claim "
            "and no provider, hardware, isolated benchmark, or performance promotion"
        ),
    )


def _jax_registered_phase_qnode_aot_export_case() -> QuantumGradientBenchmarkResult:
    values = np.array([0.37, -0.21], dtype=np.float64)
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("rx", (1,), parameter_index=1),
            PhaseQNodeOperation("cnot", (0, 1)),
        ),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )
    result = jax_phase_qnode_aot_export_audit(circuit, values, tolerance=1.0e-8)
    parameter_shift = parameter_shift_phase_qnode_gradient(circuit, values)
    finite_difference_certificate = verify_parameter_shift_gradient(
        lambda params: execute_phase_qnode_circuit(circuit, params).value,
        values,
    )
    return QuantumGradientBenchmarkResult(
        case_id="jax_registered_phase_qnode_aot_export_lowering",
        category="quantum-gradient",
        value=result.deserialized_value,
        parameter_shift_gradient=parameter_shift.gradient,
        finite_difference_gradient=finite_difference_certificate.finite_difference_gradient,
        analytic_gradient=parameter_shift.gradient,
        max_abs_reference_error=result.max_abs_value_error,
        max_abs_finite_difference_error=finite_difference_certificate.max_abs_error,
        verification_passed=result.passed and finite_difference_certificate.passed,
        evaluations=finite_difference_certificate.total_evaluations + 1,
        claim_boundary=(
            "native JAX AOT lowering plus jax.export serialization/deserialization "
            "diagnostic for deterministic registered local Phase-QNode value routes "
            "checked against SCPN parameter-shift references; gradient fields remain "
            "parameter-shift/finite-difference references, with no exported VJP, "
            "persistent cross-platform execution, provider, hardware, isolated "
            "benchmark, or performance promotion, and no wall-clock performance claim"
        ),
    )


def _quantum_gradient_case(
    case_id: str,
    objective: Callable[[NDArray[np.float64]], float],
    values: NDArray[np.float64],
    analytic_gradient: NDArray[np.float64],
    claim_boundary: str | None = None,
) -> QuantumGradientBenchmarkResult:
    certificate = verify_parameter_shift_gradient(objective, values)
    value = float(objective(values.copy()))
    return QuantumGradientBenchmarkResult(
        case_id=case_id,
        category="quantum-gradient",
        value=value,
        parameter_shift_gradient=certificate.analytic_gradient,
        finite_difference_gradient=certificate.finite_difference_gradient,
        analytic_gradient=analytic_gradient,
        max_abs_reference_error=_max_abs_error(
            certificate.analytic_gradient,
            analytic_gradient,
        ),
        max_abs_finite_difference_error=certificate.max_abs_error,
        verification_passed=certificate.passed,
        evaluations=certificate.total_evaluations,
        claim_boundary=claim_boundary
        or (
            "deterministic local expectation-gradient conformance against analytic "
            "and finite-difference references; no wall-clock performance, hardware, "
            "provider, or framework-autodiff claim"
        ),
    )
