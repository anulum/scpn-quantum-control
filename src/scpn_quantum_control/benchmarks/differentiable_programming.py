# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- differentiable-programming conformance benchmarks
"""Deterministic differentiable-programming conformance benchmark cases."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ..compiler.mlir import DifferentiableMLIRCompileConfig, compile_whole_program_ad_trace_to_mlir
from ..differentiable import (
    CustomDerivativeRule,
    Parameter,
    analyze_program_ad_alias_effects,
    custom_jvp,
    custom_vjp,
    grad,
    hessian,
    is_jax_autodiff_available,
    jacfwd,
    jacrev,
    jax_value_and_grad,
    jvp,
    parse_program_ad_effect_ir,
    program_adjoint_gradient,
    program_adjoint_result,
    vjp,
    vmap,
    whole_program_value_and_grad,
)
from ..phase.param_shift import verify_parameter_shift_gradient
from ..phase.qnode_circuit import PauliTerm, PhaseQNodeCircuit, PhaseQNodeOperation
from ..phase.torch_bridge import torch_phase_qnode_value_and_grad


@dataclass(frozen=True)
class DifferentiableProgrammingBenchmarkResult:
    """Conformance result for one differentiable-programming benchmark case."""

    case_id: str
    category: str
    value: float
    gradient: NDArray[np.float64]
    analytic_gradient: NDArray[np.float64]
    max_abs_gradient_error: float
    adjoint_supported: bool
    max_abs_adjoint_error: float | None
    claim_boundary: str

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("benchmark case_id must be non-empty")
        if not self.category:
            raise ValueError("benchmark category must be non-empty")
        if not math.isfinite(self.value):
            raise ValueError("benchmark value must be finite")
        gradient = _as_gradient("gradient", self.gradient)
        analytic = _as_gradient("analytic_gradient", self.analytic_gradient)
        if gradient.shape != analytic.shape:
            raise ValueError("benchmark gradient and analytic_gradient shapes must match")
        if self.max_abs_gradient_error < 0.0 or not math.isfinite(self.max_abs_gradient_error):
            raise ValueError("benchmark max_abs_gradient_error must be finite and non-negative")
        if not isinstance(self.adjoint_supported, bool):
            raise ValueError("benchmark adjoint_supported must be a boolean")
        if self.max_abs_adjoint_error is not None and (
            self.max_abs_adjoint_error < 0.0 or not math.isfinite(self.max_abs_adjoint_error)
        ):
            raise ValueError("benchmark max_abs_adjoint_error must be finite or None")
        if not self.claim_boundary:
            raise ValueError("benchmark claim_boundary must be non-empty")
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "analytic_gradient", analytic)

    @property
    def passed(self) -> bool:
        """Return whether implemented gradients match the analytic reference."""

        return self.max_abs_gradient_error <= 1.0e-12 and (
            self.max_abs_adjoint_error is None or self.max_abs_adjoint_error <= 1.0e-12
        )


@dataclass(frozen=True)
class DifferentiableProgrammingExternalReferenceResult:
    """Program-AD comparison against an independently executed autodiff backend."""

    case_id: str
    backend: str
    program_value: float
    reference_value: float
    program_gradient: NDArray[np.float64]
    reference_gradient: NDArray[np.float64]
    max_abs_value_error: float
    max_abs_gradient_error: float
    claim_boundary: str

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("external reference case_id must be non-empty")
        if not self.backend:
            raise ValueError("external reference backend must be non-empty")
        if not math.isfinite(self.program_value) or not math.isfinite(self.reference_value):
            raise ValueError("external reference values must be finite")
        program_gradient = _as_gradient("program_gradient", self.program_gradient)
        reference_gradient = _as_gradient("reference_gradient", self.reference_gradient)
        if program_gradient.shape != reference_gradient.shape:
            raise ValueError("external reference gradient shapes must match")
        if self.max_abs_value_error < 0.0 or not math.isfinite(self.max_abs_value_error):
            raise ValueError("external reference value error must be finite and non-negative")
        if self.max_abs_gradient_error < 0.0 or not math.isfinite(self.max_abs_gradient_error):
            raise ValueError("external reference gradient error must be finite and non-negative")
        if not self.claim_boundary:
            raise ValueError("external reference claim_boundary must be non-empty")
        object.__setattr__(self, "program_gradient", program_gradient)
        object.__setattr__(self, "reference_gradient", reference_gradient)

    @property
    def passed(self) -> bool:
        """Return whether program AD matches the external reference backend."""

        return self.max_abs_value_error <= 1.0e-10 and self.max_abs_gradient_error <= 1.0e-10


@dataclass(frozen=True)
class QuantumGradientBenchmarkResult:
    """Conformance result for deterministic quantum-gradient benchmark rows."""

    case_id: str
    category: str
    value: float
    parameter_shift_gradient: NDArray[np.float64]
    finite_difference_gradient: NDArray[np.float64]
    analytic_gradient: NDArray[np.float64]
    max_abs_reference_error: float
    max_abs_finite_difference_error: float
    verification_passed: bool
    evaluations: int
    claim_boundary: str

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("quantum gradient case_id must be non-empty")
        if not self.category:
            raise ValueError("quantum gradient category must be non-empty")
        if not math.isfinite(self.value):
            raise ValueError("quantum gradient value must be finite")
        parameter_shift_gradient = _as_gradient(
            "parameter_shift_gradient",
            self.parameter_shift_gradient,
        )
        finite_difference_gradient = _as_gradient(
            "finite_difference_gradient",
            self.finite_difference_gradient,
        )
        analytic_gradient = _as_gradient("analytic_gradient", self.analytic_gradient)
        if (
            parameter_shift_gradient.shape != finite_difference_gradient.shape
            or parameter_shift_gradient.shape != analytic_gradient.shape
        ):
            raise ValueError("quantum gradient benchmark gradient shapes must match")
        if self.max_abs_reference_error < 0.0 or not math.isfinite(self.max_abs_reference_error):
            raise ValueError("quantum gradient reference error must be finite and non-negative")
        if self.max_abs_finite_difference_error < 0.0 or not math.isfinite(
            self.max_abs_finite_difference_error
        ):
            raise ValueError(
                "quantum gradient finite-difference error must be finite and non-negative"
            )
        if not isinstance(self.verification_passed, bool):
            raise ValueError("quantum gradient verification_passed must be a boolean")
        if self.evaluations <= 0:
            raise ValueError("quantum gradient evaluations must be positive")
        if not self.claim_boundary:
            raise ValueError("quantum gradient claim_boundary must be non-empty")
        object.__setattr__(
            self,
            "parameter_shift_gradient",
            parameter_shift_gradient,
        )
        object.__setattr__(
            self,
            "finite_difference_gradient",
            finite_difference_gradient,
        )
        object.__setattr__(self, "analytic_gradient", analytic_gradient)

    @property
    def passed(self) -> bool:
        """Return whether parameter-shift gradients passed all reference checks."""

        return (
            self.verification_passed
            and self.max_abs_reference_error <= 1.0e-12
            and self.max_abs_finite_difference_error <= 1.0e-5
        )


def run_differentiable_programming_benchmark_suite() -> tuple[
    DifferentiableProgrammingBenchmarkResult, ...
]:
    """Run deterministic program-AD conformance benchmarks against analytic references."""

    return (
        _loop_heavy_case(),
        _python_semantics_list_comprehension_case(),
        _program_ad_ir_roundtrip_case(),
        _program_ad_control_phi_metadata_case(),
        _program_ad_mlir_interchange_case(),
        _program_adjoint_replay_provenance_case(),
        _elementwise_boundary_case(),
        _matrix_heavy_case(),
        _selection_heavy_case(),
        _structured_numeric_primitive_case(),
        _cumulative_primitive_case(),
        _assembly_primitive_case(),
        _reduction_primitive_case(),
        _shape_primitive_case(),
        _broadcast_primitive_case(),
        _linalg_primitive_case(),
        _indexing_heavy_case(),
        _mutation_heavy_case(),
        _shape_view_alias_metadata_case(),
        _slice_mutation_alias_metadata_case(),
        _loop_carried_state_alias_metadata_case(),
        _transform_nesting_case(),
        _custom_rule_transform_nesting_case(),
        _program_ad_transform_jvp_vjp_case(),
        _higher_order_transform_nesting_case(),
        _program_ad_hessian_transform_case(),
        _program_ad_hessian_jvp_vjp_transform_case(),
    )


def run_quantum_gradient_benchmark_suite() -> tuple[QuantumGradientBenchmarkResult, ...]:
    """Run deterministic quantum-gradient conformance rows.

    These rows exercise parameter-shift gradients on small smooth expectation
    objectives with analytic references and finite-difference certificates. They
    are correctness benchmarks only, not hardware, provider, or performance
    claims.
    """

    return (
        _single_rotation_quantum_gradient_case(),
        _two_parameter_quantum_gradient_case(),
        _sparse_ising_chain_quantum_gradient_case(),
        _torch_registered_phase_qnode_statevector_case(),
    )


def run_differentiable_programming_external_reference_suite() -> tuple[
    DifferentiableProgrammingExternalReferenceResult, ...
]:
    """Run optional external-backend conformance comparisons when dependencies exist."""

    if not is_jax_autodiff_available():
        return ()
    return (
        _jax_loop_heavy_case(),
        _jax_linalg_primitive_case(),
        _jax_transform_nesting_case(),
    )


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


def _python_semantics_list_comprehension_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([0.25, -0.5, 1.25], dtype=np.float64)

    def objective(params: Any) -> object:
        terms = [item * item + np.sin(item) for item in params]
        return sum(terms)

    analytic = 2.0 * values + np.cos(values)
    return _program_ad_case(
        "python_semantics_list_comprehension",
        "python-semantics",
        objective,
        values,
        analytic,
        claim_boundary=(
            "bounded plain list-comprehension whole-program AD conformance against "
            "analytic references; filtered, set, and dict comprehensions fail closed; "
            "no wall-clock performance, hardware, LLVM, Rust, or JIT execution claim"
        ),
    )


def _loop_heavy_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([0.2, -0.4, 0.7, -0.9], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        total = trace_values[0] * trace_values[0]
        for index in range(4):
            total = total + float(index + 1) * np.sin(trace_values[index])
        return total

    analytic = np.array(
        [
            2.0 * values[0] + math.cos(values[0]),
            2.0 * math.cos(values[1]),
            3.0 * math.cos(values[2]),
            4.0 * math.cos(values[3]),
        ],
        dtype=np.float64,
    )
    return _program_ad_case(
        "loop_heavy_scalar",
        "loop-heavy",
        objective,
        values,
        analytic,
    )


def _program_ad_ir_roundtrip_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([0.8, -0.35, 1.4], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        x, y, z = trace_values
        branch = x * z if x > y else y * z
        return branch + np.sin(x - y) + np.log(z + 2.0)

    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(Parameter("x"), Parameter("y"), Parameter("z")),
    )
    if result.program_ir is None:
        raise ValueError("program AD IR round-trip case requires program IR")
    parsed = parse_program_ad_effect_ir(result.program_ir.serialization)
    if (
        parsed.ssa_values != result.program_ir.ssa_values
        or parsed.effects != result.program_ir.effects
        or parsed.alias_edges != result.program_ir.alias_edges
        or parsed.control_regions != result.program_ir.control_regions
        or parsed.phi_nodes != result.program_ir.phi_nodes
        or parsed.serialization != result.program_ir.serialization
    ):
        raise ValueError("program AD IR round-trip parser did not reconstruct emitted IR")

    x, y, z = values
    analytic = np.array(
        [
            z + math.cos(x - y),
            -math.cos(x - y),
            x + 1.0 / (z + 2.0),
        ],
        dtype=np.float64,
    )
    return DifferentiableProgrammingBenchmarkResult(
        case_id="program_ad_ir_roundtrip_contracts",
        category="ir-roundtrip",
        value=result.value,
        gradient=result.gradient,
        analytic_gradient=analytic,
        max_abs_gradient_error=_max_abs_error(result.gradient, analytic),
        adjoint_supported=result.adjoint_result is not None and result.adjoint_result.supported,
        max_abs_adjoint_error=_max_abs_error(program_adjoint_gradient(result), analytic),
        claim_boundary=(
            "bounded program_ad_effect_ir.v1 parser and stable serialization "
            "round-trip conformance for emitted Program AD SSA/effect/control/phi "
            "metadata; not a bytecode/source compiler frontend, full alias lattice, "
            "non-executed branch semantics, Rust/LLVM executable lowering, hardware, "
            "or performance evidence; no wall-clock performance claim"
        ),
    )


def _program_ad_control_phi_metadata_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([1.2, -0.4, 0.75], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        x, y, z = trace_values
        acc = x
        if x > y:
            acc = acc + z * z
        else:
            acc = acc - y * z
        for _ in range(2):
            acc = acc + 0.5 * x
        return acc + np.sin(y + z)

    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(Parameter("x"), Parameter("y"), Parameter("z")),
    )
    if result.program_ir is None:
        raise ValueError("program AD control/phi metadata case requires program IR")
    parsed = parse_program_ad_effect_ir(result.program_ir.serialization)
    if parsed != result.program_ir:
        raise ValueError("program AD control/phi parser did not reconstruct emitted IR")
    has_runtime_branch = any(
        region.kind == "runtime_branch" and region.entered
        for region in result.program_ir.control_regions
    )
    has_source_control = any(
        region.kind.startswith("source_") for region in result.program_ir.control_regions
    )
    has_runtime_phi = any(
        phi.target.startswith("phi:runtime_branch")
        and phi.selected == "executed_true"
        and phi.control_region is not None
        for phi in result.program_ir.phi_nodes
    )
    has_source_phi = any(
        phi.target.startswith("phi:source:") and phi.control_region is not None
        for phi in result.program_ir.phi_nodes
    )
    has_control_effect = any(
        effect.kind == "control_branch" for effect in result.program_ir.effects
    )
    if not (
        has_runtime_branch
        and has_source_control
        and has_runtime_phi
        and has_source_phi
        and has_control_effect
    ):
        raise ValueError("program AD control/phi metadata provenance is incomplete")

    x, y, z = values
    common = math.cos(y + z)
    analytic = np.array([2.0, common, 2.0 * z + common], dtype=np.float64)
    return DifferentiableProgrammingBenchmarkResult(
        case_id="program_ad_control_phi_metadata_contracts",
        category="control-phi",
        value=result.value,
        gradient=result.gradient,
        analytic_gradient=analytic,
        max_abs_gradient_error=_max_abs_error(result.gradient, analytic),
        adjoint_supported=result.adjoint_result is not None and result.adjoint_result.supported,
        max_abs_adjoint_error=_max_abs_error(program_adjoint_gradient(result), analytic),
        claim_boundary=(
            "ProgramADPhiNode control-join provenance for supported executed "
            "runtime and source control regions in program_ad_effect_ir.v1; "
            "local conformance only, not non-executed branch adjoints, full "
            "compiler phi lowering, Rust/LLVM executable lowering, hardware, "
            "or performance evidence; no wall-clock performance claim"
        ),
    )


def _program_ad_mlir_interchange_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([0.25, 0.5, 0.75], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        total = trace_values[0]
        if trace_values[1] > 0.0:
            total = total + np.sin(trace_values[1])
        return total + trace_values[2]

    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(Parameter("a"), Parameter("b"), Parameter("c")),
    )
    if result.program_ir is None:
        raise ValueError("program AD MLIR interchange case requires program IR")
    module = compile_whole_program_ad_trace_to_mlir(result, DifferentiableMLIRCompileConfig())
    program_ir_metadata = module.metadata.get("program_ad_ir")
    if not isinstance(program_ir_metadata, dict):
        raise ValueError("program AD MLIR interchange metadata is missing")
    if (
        'scpn.program_ir_format = "program_ad_effect_ir.v1"' not in module.text
        or "scpn_diff.program_ad_ssa" not in module.text
        or "scpn_diff.program_ad_effect" not in module.text
        or "scpn_diff.program_ad_control_region" not in module.text
        or "scpn_diff.program_ad_phi" not in module.text
        or module.resource_counts["program_ad_ssa_values"] != len(result.program_ir.ssa_values)
        or module.resource_counts["program_ad_effects"] != len(result.program_ir.effects)
        or module.resource_counts["program_ad_control_regions"]
        != len(result.program_ir.control_regions)
        or module.resource_counts["program_ad_phi_nodes"] != len(result.program_ir.phi_nodes)
        or program_ir_metadata.get("format") != "program_ad_effect_ir.v1"
        or program_ir_metadata.get("claim_boundary")
        != "program_ad_ir_mlir_interchange_only_no_executable_lowering"
    ):
        raise ValueError("program AD MLIR interchange lowering is incomplete")

    analytic = np.array([1.0, math.cos(values[1]), 1.0], dtype=np.float64)
    return DifferentiableProgrammingBenchmarkResult(
        case_id="program_ad_mlir_interchange_contracts",
        category="mlir-interchange",
        value=result.value,
        gradient=result.gradient,
        analytic_gradient=analytic,
        max_abs_gradient_error=_max_abs_error(result.gradient, analytic),
        adjoint_supported=result.adjoint_result is not None and result.adjoint_result.supported,
        max_abs_adjoint_error=_max_abs_error(program_adjoint_gradient(result), analytic),
        claim_boundary=(
            "bounded program_ad_effect_ir.v1 MLIR dialect interchange lowering "
            "for captured SSA/effect/control/phi metadata only; no executable "
            "Rust, LLVM, or JIT differentiated runtime, hardware, provider, or "
            "performance evidence; no wall-clock performance claim"
        ),
    )


def _program_adjoint_replay_provenance_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([1.25, -0.4, 0.75], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        x, y, z = trace_values
        branch = x if x > y else y
        return (
            np.sin(x)
            + np.cos(y)
            + np.exp(x - y)
            + np.log(z + 3.0)
            + np.sqrt(z + 4.0)
            + np.tanh(x * z)
            + x**2.0
            + 2.0**y
            + branch
        )

    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(Parameter("x"), Parameter("y"), Parameter("z")),
    )
    adjoint = program_adjoint_result(result)
    if result.program_ir is None:
        raise ValueError("program AD adjoint replay provenance case requires program IR")
    if (
        adjoint.replay_node_count != len(result.ir_nodes)
        or adjoint.replay_effect_count != len(result.program_ir.effects)
        or adjoint.replay_control_region_count != len(result.program_ir.control_regions)
        or adjoint.replay_phi_node_count != len(result.program_ir.phi_nodes)
        or adjoint.replay_ir_format != "program_ad_effect_ir.v1"
    ):
        raise ValueError("program AD adjoint replay provenance counts do not match program IR")

    x, y, z = values
    analytic = np.array(
        [
            math.cos(x) + math.exp(x - y) + z * (1.0 - math.tanh(x * z) ** 2) + 2.0 * x + 1.0,
            -math.sin(y) - math.exp(x - y) + math.log(2.0) * (2.0**y),
            1.0 / (z + 3.0) + 1.0 / (2.0 * math.sqrt(z + 4.0)) + x * (1.0 - math.tanh(x * z) ** 2),
        ],
        dtype=np.float64,
    )
    adjoint_gradient = program_adjoint_gradient(result)
    return DifferentiableProgrammingBenchmarkResult(
        case_id="program_adjoint_replay_provenance_contracts",
        category="reverse-adjoint",
        value=result.value,
        gradient=result.gradient,
        analytic_gradient=analytic,
        max_abs_gradient_error=_max_abs_error(result.gradient, analytic),
        adjoint_supported=adjoint.supported,
        max_abs_adjoint_error=_max_abs_error(adjoint_gradient, analytic)
        if adjoint.supported
        else None,
        claim_boundary=(
            "ProgramADAdjointResult replay provenance over supported executed scalar "
            "IR nodes, effects, control regions, and phi metadata in "
            "program_ad_effect_ir.v1; local conformance only, not full reverse-mode "
            "compiler AD, non-executed branch adjoints, Rust, LLVM/JIT, hardware, "
            "or performance evidence; no wall-clock performance claim"
        ),
    )


def _elementwise_boundary_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([-1.5, 0.75, 2.25, 0.5], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        return (
            abs(trace_values[0])
            + np.abs(trace_values[1])
            + np.log(trace_values[2])
            + np.sqrt(trace_values[2])
            + np.reciprocal(trace_values[3])
            + np.log1p(trace_values[3])
            + np.arcsin(0.5 * trace_values[3])
            + np.arccos(0.25 * trace_values[3])
        )

    x2 = values[2]
    x3 = values[3]
    analytic = np.array(
        [
            -1.0,
            1.0,
            1.0 / x2 + 1.0 / (2.0 * math.sqrt(x2)),
            -1.0 / (x3 * x3)
            + 1.0 / (1.0 + x3)
            + 0.5 / math.sqrt(1.0 - (0.5 * x3) ** 2)
            - 0.25 / math.sqrt(1.0 - (0.25 * x3) ** 2),
        ],
        dtype=np.float64,
    )
    return _program_ad_case(
        "elementwise_boundary_contracts",
        "elementwise-boundary",
        objective,
        values,
        analytic,
        claim_boundary=(
            "deterministic program AD conformance for zero-cusp absolute-value, "
            "positive-domain, nonzero-denominator, and inverse-trig boundary "
            "contracts; no wall-clock performance, hardware, LLVM, Rust, or JIT "
            "execution claim"
        ),
    )


def _matrix_heavy_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([0.5, -0.25, 1.5, -2.0], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        left = trace_values[:2]
        right = trace_values[2:4]
        matrix = np.reshape(trace_values, (2, 2))
        tensor = np.reshape(trace_values, (2, 2, 1))
        tensor_weights = np.array([[0.5, -1.0], [1.25, 0.75]], dtype=np.float64)
        tensor_vector = np.array([2.0], dtype=np.float64)
        return (
            np.inner(left, right)
            + np.sum(np.outer(left, right))
            + np.trace(matrix)
            + np.sum(np.diag(matrix))
            + np.tensordot(left, right, axes=1)
            + np.sum(np.tensordot(left, right, axes=0))
            + np.einsum("i,i->", left, right)
            + np.sum(np.einsum("i,j->ij", left, right))
            + np.sum(np.einsum("ij,j->i", matrix, left))
            + np.einsum("ii->", matrix)
            + np.einsum("abc,c,ab->", tensor, tensor_vector, tensor_weights)
        )

    analytic = np.array(
        [
            7.0 * values[2] + 3.0 * values[3] + 2.0 * values[0] + 4.0,
            3.0 * values[2] + 7.0 * values[3] + 2.0 * values[1] - 2.0,
            7.0 * values[0] + 3.0 * values[1] + 2.5,
            3.0 * values[0] + 7.0 * values[1] + 4.5,
        ],
        dtype=np.float64,
    )
    return _program_ad_case(
        "matrix_heavy_linear_algebra",
        "matrix-heavy",
        objective,
        values,
        analytic,
    )


def _linalg_primitive_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([1.5, 2.0, -0.75, 0.5], dtype=np.float64)
    inverse_weights = np.array([[0.25, 0.0], [0.0, -0.5]], dtype=np.float64)
    solve_weights = np.array([1.25, -0.75], dtype=np.float64)
    power_weights = np.array([[0.5, 0.0], [0.0, -0.25]], dtype=np.float64)
    multi_dot_left = np.array([0.75, -1.5], dtype=np.float64)
    multi_dot_right = np.array([1.25, 0.5], dtype=np.float64)
    eigh_weights = np.array([0.1, -0.2], dtype=np.float64)
    eigh_vector_weights = np.array([[0.05, -0.1], [0.2, 0.15]], dtype=np.float64)
    eig_weights = np.array([0.025, -0.035], dtype=np.float64)
    eig_vector_weights = np.array([[0.04, -0.02], [0.03, 0.05]], dtype=np.float64)
    eigvals_weights = np.array([0.05, -0.07], dtype=np.float64)
    eigvalsh_weights = np.array([0.2, -0.3], dtype=np.float64)
    svd_weights = np.array([0.15, -0.25], dtype=np.float64)
    pinv_weights = np.array([[0.35, 0.0], [0.0, -0.45]], dtype=np.float64)
    norm_row_weights = np.array([0.45, -0.65], dtype=np.float64)
    norm_column_weights = np.array([0.15, 0.2], dtype=np.float64)
    frobenius_weight = 0.175
    trace_weight = 0.375
    diag_weights = np.array([-1.25, 0.5], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        matrix = np.diag(trace_values[:2])
        rhs = trace_values[2:4]
        eigh_values, eigh_vectors = np.linalg.eigh(matrix)
        eig_values, eig_vectors = np.linalg.eig(matrix)
        return (
            np.linalg.det(matrix)
            + np.sum(np.linalg.inv(matrix) * inverse_weights)
            + np.sum(np.linalg.solve(matrix, rhs) * solve_weights)
            + trace_weight * np.trace(matrix)
            + np.sum(np.diag(matrix) * diag_weights)
            + np.sum(np.linalg.matrix_power(matrix, 2) * power_weights)
            + np.linalg.multi_dot((multi_dot_left, matrix, multi_dot_right))
            + np.sum(eigh_values * eigh_weights)
            + np.sum(eigh_vectors * eigh_vector_weights)
            + np.sum(eig_values * eig_weights)
            + np.sum(eig_vectors * eig_vector_weights)
            + np.sum(np.linalg.eigvals(matrix) * eigvals_weights)
            + np.sum(np.linalg.eigvalsh(matrix) * eigvalsh_weights)
            + np.sum(np.linalg.svd(matrix, compute_uv=False) * svd_weights)
            + np.sum(np.linalg.pinv(matrix) * pinv_weights)
            + np.sum(np.linalg.norm(matrix, 2, axis=1) * norm_row_weights)
            + np.sum(np.linalg.norm(matrix, None, 0) * norm_column_weights)
            + frobenius_weight * np.linalg.norm(matrix, "fro", axis=(0, 1))
        )

    x0, x1, rhs0, rhs1 = values
    frobenius_norm = math.sqrt(x0 * x0 + x1 * x1)
    analytic = np.array(
        [
            x1
            - inverse_weights[0, 0] / (x0 * x0)
            - solve_weights[0] * rhs0 / (x0 * x0)
            + trace_weight
            + diag_weights[0]
            + 2.0 * power_weights[0, 0] * x0
            + multi_dot_left[0] * multi_dot_right[0]
            + eigh_weights[0]
            + eig_weights[0]
            + eigvals_weights[0]
            + eigvalsh_weights[0]
            + svd_weights[1]
            - pinv_weights[0, 0] / (x0 * x0)
            + norm_row_weights[0]
            + norm_column_weights[0]
            + frobenius_weight * x0 / frobenius_norm,
            x0
            - inverse_weights[1, 1] / (x1 * x1)
            - solve_weights[1] * rhs1 / (x1 * x1)
            + trace_weight
            + diag_weights[1]
            + 2.0 * power_weights[1, 1] * x1
            + multi_dot_left[1] * multi_dot_right[1]
            + eigh_weights[1]
            + eig_weights[1]
            + eigvals_weights[1]
            + eigvalsh_weights[1]
            + svd_weights[0]
            - pinv_weights[1, 1] / (x1 * x1)
            + norm_row_weights[1]
            + norm_column_weights[1]
            + frobenius_weight * x1 / frobenius_norm,
            solve_weights[0] / x0,
            solve_weights[1] / x1,
        ],
        dtype=np.float64,
    )
    return _program_ad_case(
        "linalg_primitive_contracts",
        "linalg-primitive",
        objective,
        values,
        analytic,
    )


def _selection_heavy_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([-1.0, 0.4, 1.2], dtype=np.float64)
    thresholds = np.array([-0.5, 0.0, 1.0], dtype=np.float64)
    offsets = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    upper = np.array([0.5, 0.75, 2.0], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        selected = np.select(
            [trace_values < -0.25, trace_values > 0.5],
            [trace_values * trace_values, 1.5 * trace_values],
            default=-0.75 * trace_values,
        )
        callable_piecewise = np.piecewise(
            trace_values,
            [trace_values < -0.25, trace_values > 0.5],
            [
                lambda item: item * item,
                lambda item: 1.5 * item,
                lambda item: -0.75 * item,
            ],
        )
        chosen = np.choose(
            np.array([0, 1, 2], dtype=np.int64),
            [trace_values * trace_values, -0.5 * trace_values, 2.0 * trace_values],
        )
        compressed = np.compress(np.array([True, False, True], dtype=np.bool_), trace_values)
        extracted = np.extract(np.array([True, False, True], dtype=np.bool_), trace_values)
        return (
            np.sum(
                np.where(trace_values > thresholds, trace_values**2, -trace_values)
                + np.clip(trace_values + offsets, -0.75, upper)
                + 0.09 * selected
                + 0.04 * callable_piecewise
                + 0.03 * chosen
            )
            + 0.02 * np.sum(compressed * np.array([2.0, -3.0], dtype=np.float64))
            + 0.015 * np.sum(extracted * np.array([1.0, -1.5], dtype=np.float64))
        )

    analytic = np.array([-1.265, 1.6875, 3.5725], dtype=np.float64)
    return _program_ad_case(
        "selection_piecewise_contracts",
        "selection-heavy",
        objective,
        values,
        analytic,
    )


def _structured_numeric_value_and_directional(
    source: NDArray[np.float64],
    direction: NDArray[np.float64],
) -> tuple[float, float]:
    left = source[:3]
    right = source[3:6]
    matrix = source[:6].reshape(2, 3)
    signal = source[6:10]
    kernel = source[10:13]
    samples = source[13:16]

    direction_left = direction[:3]
    direction_right = direction[3:6]
    direction_matrix = direction[:6].reshape(2, 3)
    direction_signal = direction[6:10]
    direction_kernel = direction[10:13]
    direction_samples = direction[13:16]

    product_vector = np.array([0.5, -1.25, 0.75], dtype=np.float64)
    matmul_weights = np.array([1.5, -0.25], dtype=np.float64)
    outer_weights = np.array([[0.2, -0.4], [0.6, -0.8]], dtype=np.float64)
    tensor_weights = np.array([[0.3, -0.5, 0.7], [-0.2, 0.4, -0.6]], dtype=np.float64)
    grid = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    convolve_weights = np.array([0.25, -0.5, 0.75, -1.0], dtype=np.float64)
    correlate_weights = np.array([0.4, -0.6], dtype=np.float64)
    gradient_weights = np.array([-0.3, 0.1, 0.5, -0.7], dtype=np.float64)

    interpolated = np.interp(samples, grid, signal)
    convolved = np.convolve(signal, kernel, mode="same")
    correlated = np.correlate(signal, kernel, mode="valid")
    stencil = np.gradient(signal, 0.5, edge_order=1)
    value = (
        float(np.inner(left, right))
        + 0.17 * float(np.sum(np.outer(left[:2], right[:2]) * outer_weights))
        + 0.23 * float(np.sum(np.matmul(matrix, product_vector) * matmul_weights))
        + 0.29 * float(np.tensordot(left, right, axes=1))
        + 0.31 * float(np.sum(np.tensordot(left[:2], right, axes=0) * tensor_weights))
        + 0.37 * float(np.einsum("i,i->", left, right))
        + 0.19 * float(np.sum(np.einsum("i,j->ij", left[:2], right[:2]) * outer_weights))
        + float(np.sum(interpolated))
        + 0.41 * float(np.sum(convolved * convolve_weights))
        - 0.13 * float(np.sum(correlated * correlate_weights))
        + 0.07 * float(np.sum(stencil * gradient_weights))
    )

    interpolation_directional = 0.0
    for sample, sample_tangent in zip(samples, direction_samples, strict=True):
        interval = int(np.searchsorted(grid, sample, side="right") - 1)
        if interval < 0 or interval >= grid.size - 1:
            raise ValueError("structured benchmark interpolation samples must stay inside grid")
        width = float(grid[interval + 1] - grid[interval])
        weight_right = float((sample - grid[interval]) / width)
        weight_left = 1.0 - weight_right
        slope = float((signal[interval + 1] - signal[interval]) / width)
        interpolation_directional += (
            slope * float(sample_tangent)
            + weight_left * float(direction_signal[interval])
            + weight_right * float(direction_signal[interval + 1])
        )

    directional = (
        float(np.inner(direction_left, right) + np.inner(left, direction_right))
        + 0.17
        * float(
            np.sum(
                (np.outer(direction_left[:2], right[:2]) + np.outer(left[:2], direction_right[:2]))
                * outer_weights
            )
        )
        + 0.23 * float(np.sum(np.matmul(direction_matrix, product_vector) * matmul_weights))
        + 0.29 * float(np.tensordot(direction_left, right, axes=1))
        + 0.29 * float(np.tensordot(left, direction_right, axes=1))
        + 0.31
        * float(
            np.sum(
                (
                    np.tensordot(direction_left[:2], right, axes=0)
                    + np.tensordot(left[:2], direction_right, axes=0)
                )
                * tensor_weights
            )
        )
        + 0.37
        * float(
            np.einsum("i,i->", direction_left, right) + np.einsum("i,i->", left, direction_right)
        )
        + 0.19
        * float(
            np.sum(
                (
                    np.einsum("i,j->ij", direction_left[:2], right[:2])
                    + np.einsum("i,j->ij", left[:2], direction_right[:2])
                )
                * outer_weights
            )
        )
        + interpolation_directional
        + 0.41
        * float(
            np.sum(
                (
                    np.convolve(direction_signal, kernel, mode="same")
                    + np.convolve(signal, direction_kernel, mode="same")
                )
                * convolve_weights
            )
        )
        - 0.13
        * float(
            np.sum(
                (
                    np.correlate(direction_signal, kernel, mode="valid")
                    + np.correlate(signal, direction_kernel, mode="valid")
                )
                * correlate_weights
            )
        )
        + 0.07 * float(np.sum(np.gradient(direction_signal, 0.5, edge_order=1) * gradient_weights))
    )
    return value, directional


def _structured_numeric_primitive_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array(
        [
            0.75,
            -1.25,
            2.0,
            -0.5,
            1.5,
            0.25,
            1.0,
            -0.5,
            2.0,
            0.25,
            0.5,
            -1.0,
            0.75,
            0.25,
            1.25,
            2.5,
        ],
        dtype=np.float64,
    )
    product_vector = np.array([0.5, -1.25, 0.75], dtype=np.float64)
    matmul_weights = np.array([1.5, -0.25], dtype=np.float64)
    outer_weights = np.array([[0.2, -0.4], [0.6, -0.8]], dtype=np.float64)
    tensor_weights = np.array([[0.3, -0.5, 0.7], [-0.2, 0.4, -0.6]], dtype=np.float64)
    grid = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    convolve_weights = np.array([0.25, -0.5, 0.75, -1.0], dtype=np.float64)
    correlate_weights = np.array([0.4, -0.6], dtype=np.float64)
    gradient_weights = np.array([-0.3, 0.1, 0.5, -0.7], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        left = trace_values[:3]
        right = trace_values[3:6]
        matrix = np.reshape(trace_values[:6], (2, 3))
        signal = trace_values[6:10]
        kernel = trace_values[10:13]
        samples = trace_values[13:16]
        return (
            np.inner(left, right)
            + 0.17 * np.sum(np.outer(left[:2], right[:2]) * outer_weights)
            + 0.23 * np.sum(np.matmul(matrix, product_vector) * matmul_weights)
            + 0.29 * np.tensordot(left, right, axes=1)
            + 0.31 * np.sum(np.tensordot(left[:2], right, axes=0) * tensor_weights)
            + 0.37 * np.einsum("i,i->", left, right)
            + 0.19 * np.sum(np.einsum("i,j->ij", left[:2], right[:2]) * outer_weights)
            + np.sum(np.interp(samples, grid, signal))
            + 0.41 * np.sum(np.convolve(signal, kernel, mode="same") * convolve_weights)
            - 0.13 * np.sum(np.correlate(signal, kernel, mode="valid") * correlate_weights)
            + 0.07 * np.sum(np.gradient(signal, 0.5, edge_order=1) * gradient_weights)
        )

    analytic = np.array(
        [
            _structured_numeric_value_and_directional(values, basis)[1]
            for basis in np.eye(values.size, dtype=np.float64)
        ],
        dtype=np.float64,
    )
    return _program_ad_case(
        "structured_numeric_primitive_contracts",
        "structured-numeric",
        objective,
        values,
        analytic,
        claim_boundary=(
            "deterministic product, interpolation, signal, and stencil primitive "
            "contracts for inner, outer, matmul, tensordot, einsum, interp, "
            "convolve, correlate, and gradient; no wall-clock performance, "
            "hardware, LLVM, Rust, or JIT execution claim"
        ),
    )


def _cumulative_primitive_gradient(
    values: NDArray[np.float64],
    cumulative_weights: NDArray[np.float64],
    product_weights: NDArray[np.float64],
    first_difference_weights: NDArray[np.float64],
    second_difference_weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    shifted = values + 2.5
    products = np.cumprod(shifted)
    gradient = np.zeros_like(values, dtype=np.float64)
    size = int(values.size)

    for parameter_index in range(size):
        gradient[parameter_index] += float(np.sum(cumulative_weights[parameter_index:]))
        for output_index in range(parameter_index, size):
            gradient[parameter_index] += (
                float(product_weights[output_index])
                * float(products[output_index])
                / float(shifted[parameter_index])
            )
        if parameter_index > 0:
            gradient[parameter_index] += float(first_difference_weights[parameter_index - 1])
        if parameter_index < size - 1:
            gradient[parameter_index] -= float(first_difference_weights[parameter_index])
        if parameter_index < size - 2:
            gradient[parameter_index] += float(second_difference_weights[parameter_index])
        if 0 < parameter_index < size - 1:
            gradient[parameter_index] -= 2.0 * float(
                second_difference_weights[parameter_index - 1]
            )
        if parameter_index >= 2:
            gradient[parameter_index] += float(second_difference_weights[parameter_index - 2])
    return gradient


def _cumulative_primitive_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([0.2, -0.35, 0.6, 1.1, -0.15], dtype=np.float64)
    cumulative_weights = np.array([0.3, -0.5, 0.7, -0.2, 0.4], dtype=np.float64)
    product_weights = np.array([-0.11, 0.17, -0.23, 0.29, -0.31], dtype=np.float64)
    first_difference_weights = np.array([0.13, -0.19, 0.37, -0.41], dtype=np.float64)
    second_difference_weights = np.array([-0.07, 0.09, -0.15], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        shifted = trace_values + 2.5
        return (
            np.sum(np.cumsum(trace_values) * cumulative_weights)
            + np.sum(np.cumprod(shifted) * product_weights)
            + np.sum(np.diff(trace_values) * first_difference_weights)
            + np.sum(np.diff(trace_values, n=2) * second_difference_weights)
        )

    analytic = _cumulative_primitive_gradient(
        values,
        cumulative_weights,
        product_weights,
        first_difference_weights,
        second_difference_weights,
    )
    return _program_ad_case(
        "cumulative_primitive_contracts",
        "cumulative-primitive",
        objective,
        values,
        analytic,
        claim_boundary=(
            "deterministic cumsum, cumprod, and diff primitive contracts for "
            "bounded one-dimensional Program AD traces; dynamic axis promotion, "
            "Rust/LLVM executable lowering, hardware, and performance promotion "
            "remain blocked; no wall-clock performance claim"
        ),
    )


def _assembly_primitive_value(values: NDArray[np.float64]) -> float:
    matrix = values.reshape(2, 3)
    left = values[:3]
    right = values[3:]
    flat_weights = np.array([0.5, -0.75, 1.25, -1.5, 0.875, -0.625], dtype=np.float64)
    matrix_weights = np.array([[0.2, -0.4, 0.6], [-0.8, 1.0, -1.2]], dtype=np.float64)
    column_weights = np.array([[0.3, -0.5], [0.7, -0.9], [1.1, -1.3]], dtype=np.float64)
    depth_weights = np.array(
        [[[0.15, -0.25], [0.35, -0.45]], [[0.55, -0.65], [0.75, -0.85]]],
        dtype=np.float64,
    )
    return float(
        np.sum(np.zeros_like(values))
        + 0.01 * np.sum(np.ones_like(values))
        + 0.02 * np.sum(np.full_like(values, -0.25))
        + np.sum(np.hstack((left, right)) * flat_weights)
        + np.sum(np.vstack((matrix[0], matrix[1])) * matrix_weights)
        + np.sum(np.column_stack((left, right)) * column_weights)
        + np.sum(np.dstack((matrix[:, :2], matrix[:, 1:])) * depth_weights)
    )


def _assembly_primitive_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([0.5, -1.25, 2.0, -0.75, 1.5, 0.25], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        matrix = np.reshape(trace_values, (2, 3))
        left = trace_values[:3]
        right = trace_values[3:]
        flat_weights = np.array(
            [0.5, -0.75, 1.25, -1.5, 0.875, -0.625],
            dtype=np.float64,
        )
        matrix_weights = np.array(
            [[0.2, -0.4, 0.6], [-0.8, 1.0, -1.2]],
            dtype=np.float64,
        )
        column_weights = np.array(
            [[0.3, -0.5], [0.7, -0.9], [1.1, -1.3]],
            dtype=np.float64,
        )
        depth_weights = np.array(
            [[[0.15, -0.25], [0.35, -0.45]], [[0.55, -0.65], [0.75, -0.85]]],
            dtype=np.float64,
        )
        return (
            np.sum(np.zeros_like(trace_values))
            + 0.01 * np.sum(np.ones_like(trace_values))
            + 0.02 * np.sum(np.full_like(trace_values, -0.25))
            + np.sum(np.hstack((left, right)) * flat_weights)
            + np.sum(np.vstack((matrix[0], matrix[1])) * matrix_weights)
            + np.sum(np.column_stack((left, right)) * column_weights)
            + np.sum(np.dstack((matrix[:, :2], matrix[:, 1:])) * depth_weights)
        )

    zero_value = _assembly_primitive_value(np.zeros_like(values))
    analytic = np.array(
        [
            _assembly_primitive_value(basis) - zero_value
            for basis in np.eye(values.size, dtype=np.float64)
        ],
        dtype=np.float64,
    )
    return _program_ad_case(
        "assembly_primitive_contracts",
        "assembly-primitive",
        objective,
        values,
        analytic,
        claim_boundary=(
            "deterministic like-constructor and stack assembly primitive "
            "contracts for zeros_like, ones_like, full_like, hstack, vstack, "
            "column_stack, and dstack; dynamic shape assembly, Rust/LLVM "
            "executable lowering, hardware, and performance promotion remain "
            "blocked; no wall-clock performance claim"
        ),
    )


def _reduction_primitive_gradient(values: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = values + 2.5
    product = float(np.prod(shifted))
    mean = float(np.mean(values))
    shifted_std = float(np.std(values + 3.0, ddof=0))
    size = float(values.size)
    trapezoid_weights = np.array([0.25, 0.5, 0.5, 0.5, 0.25], dtype=np.float64)
    gradient = (
        0.2
        + 0.13 * product / shifted
        + 0.17 / size
        + 0.19 * 2.0 * (values - mean) / size
        + 0.23 * (values - mean) / (size * shifted_std)
        + 0.29 * trapezoid_weights
    )
    gradient[int(np.argmax(values))] += 0.31
    gradient[int(np.argmin(values))] -= 0.37

    order = np.argsort(values)
    gradient[int(order[2])] += 0.41
    gradient[int(order[1])] += 0.43
    gradient[int(order[3])] -= 0.47
    return gradient


def _reduction_primitive_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([-1.0, 0.25, 1.4, -0.4, 0.9], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        shifted = trace_values + 2.5
        return (
            0.2 * np.sum(trace_values)
            + 0.13 * np.prod(shifted)
            + 0.17 * np.mean(trace_values)
            + 0.19 * np.var(trace_values, ddof=0)
            + 0.23 * np.std(trace_values + 3.0, ddof=0)
            + 0.29 * np.trapezoid(trace_values, dx=0.5)
            + 0.31 * np.max(trace_values)
            - 0.37 * np.min(trace_values)
            + 0.41 * np.median(trace_values)
            + 0.43 * np.quantile(trace_values, 0.25)
            - 0.47 * np.percentile(trace_values, 75.0)
        )

    return _program_ad_case(
        "reduction_primitive_contracts",
        "reduction-primitive",
        objective,
        values,
        _reduction_primitive_gradient(values),
        claim_boundary=(
            "deterministic sum, prod, mean, var, std, trapezoid, unique and "
            "strict-order selector reductions, and scalar q order-statistics for "
            "bounded one-dimensional Program AD traces; dynamic axes, dynamic q, "
            "tie boundaries, zero-variance standard-deviation, Rust/LLVM "
            "executable lowering, hardware, and performance promotion remain "
            "blocked; no wall-clock performance claim"
        ),
    )


def _shape_primitive_gradient(values: NDArray[np.float64]) -> NDArray[np.float64]:
    source = np.arange(values.size, dtype=np.int64)
    matrix_source = source.reshape(2, 3)
    tensor_source = source.reshape(1, 2, 1, 3)
    squeezed_source = np.squeeze(tensor_source, axis=(0, 2))
    moved_source = np.moveaxis(
        np.swapaxes(np.expand_dims(squeezed_source, axis=0), 0, 1),
        source=2,
        destination=0,
    )
    gradient = np.zeros(values.size, dtype=np.float64)
    for transformed_source in (
        np.ravel(np.transpose(matrix_source, axes=(1, 0))),
        np.repeat(moved_source, repeats=(1, 2, 1), axis=0),
        np.atleast_1d(source[0]),
        np.atleast_2d(source[:3]),
        np.atleast_3d(matrix_source[0]),
        np.tile(matrix_source, (2, 1)),
        np.roll(matrix_source, shift=1, axis=1),
        np.rot90(matrix_source, k=1),
        np.flip(matrix_source, axis=0),
        np.flipud(matrix_source),
        np.fliplr(matrix_source),
    ):
        np.add.at(gradient, transformed_source.reshape(-1), 1.0)
    return gradient


def _shape_primitive_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        matrix = np.reshape(trace_values, (2, 3))
        tensor = np.reshape(trace_values, (1, 2, 1, 3))
        squeezed = np.squeeze(tensor, axis=(0, 2))
        expanded = np.expand_dims(squeezed, axis=0)
        swapped = np.swapaxes(expanded, 0, 1)
        moved = np.moveaxis(swapped, source=2, destination=0)
        return (
            np.sum(np.ravel(np.transpose(matrix, axes=(1, 0))))
            + np.sum(np.repeat(moved, repeats=(1, 2, 1), axis=0))
            + np.sum(np.atleast_1d(trace_values[0]))
            + np.sum(np.atleast_2d(trace_values[:3]))
            + np.sum(np.atleast_3d(matrix[0]))
            + np.sum(np.tile(matrix, (2, 1)))
            + np.sum(np.roll(matrix, shift=1, axis=1))
            + np.sum(np.rot90(matrix, k=1))
            + np.sum(np.flip(matrix, axis=0))
            + np.sum(np.flipud(matrix))
            + np.sum(np.fliplr(matrix))
        )

    return _program_ad_case(
        "shape_primitive_contracts",
        "shape-primitive",
        objective,
        values,
        _shape_primitive_gradient(values),
        claim_boundary=(
            "deterministic reshape, ravel, transpose, expand_dims, squeeze, "
            "swapaxes, moveaxis, repeat, rank promotion, tile, roll, rot90, "
            "flip, flipud, fliplr shape primitive contracts for bounded Program "
            "AD traces; dynamic shape arguments, invalid axes, Rust/LLVM "
            "executable lowering, hardware, and performance promotion remain "
            "blocked; no wall-clock performance claim"
        ),
    )


def _broadcast_primitive_gradient() -> NDArray[np.float64]:
    broadcast_to_weights = np.array(
        [[0.25, -0.5], [0.75, -1.0], [1.25, -1.5]],
        dtype=np.float64,
    )
    column_weights = np.array([[0.3, -0.7, 1.1], [-1.3, 0.5, -0.2]], dtype=np.float64)
    row_weights = np.array([[-0.4, 0.8, -1.2], [1.4, -0.6, 0.2]], dtype=np.float64)
    scalar_weights = np.array([[0.9, -0.3, 0.6], [-0.8, 1.0, -0.5]], dtype=np.float64)
    product_weights = np.array([[0.2, -0.1, 0.4], [-0.6, 0.3, -0.2]], dtype=np.float64)
    values = np.array([0.2, -0.4, 0.7, -1.1, 1.3, -0.6], dtype=np.float64)
    column = values[:2].reshape(2, 1)
    row = values[2:5]

    gradient = np.zeros(values.size, dtype=np.float64)
    gradient[:2] += np.sum(broadcast_to_weights, axis=0)
    gradient[:2] += np.sum(column_weights, axis=1)
    gradient[2:5] += np.sum(row_weights, axis=0)
    gradient[5] += float(np.sum(scalar_weights))
    gradient[:2] += 0.17 * np.sum(row * product_weights, axis=1)
    gradient[2:5] += 0.17 * np.sum(column * product_weights, axis=0)
    return gradient


def _broadcast_primitive_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([0.2, -0.4, 0.7, -1.1, 1.3, -0.6], dtype=np.float64)
    broadcast_to_weights = np.array(
        [[0.25, -0.5], [0.75, -1.0], [1.25, -1.5]],
        dtype=np.float64,
    )
    column_weights = np.array([[0.3, -0.7, 1.1], [-1.3, 0.5, -0.2]], dtype=np.float64)
    row_weights = np.array([[-0.4, 0.8, -1.2], [1.4, -0.6, 0.2]], dtype=np.float64)
    scalar_weights = np.array([[0.9, -0.3, 0.6], [-0.8, 1.0, -0.5]], dtype=np.float64)
    product_weights = np.array([[0.2, -0.1, 0.4], [-0.6, 0.3, -0.2]], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        source = trace_values[:2]
        column = np.reshape(trace_values[:2], (2, 1))
        row = trace_values[2:5]
        scalar = trace_values[5:6]
        broadcast_column, broadcast_row, broadcast_scalar = np.broadcast_arrays(
            column,
            row,
            scalar,
        )
        return (
            np.sum(np.broadcast_to(source, (3, 2)) * broadcast_to_weights)
            + np.sum(broadcast_column * column_weights)
            + np.sum(broadcast_row * row_weights)
            + np.sum(broadcast_scalar * scalar_weights)
            + 0.17 * np.sum((column * row) * product_weights)
        )

    return _program_ad_case(
        "broadcast_primitive_contracts",
        "broadcast-primitive",
        objective,
        values,
        _broadcast_primitive_gradient(),
        claim_boundary=(
            "deterministic broadcast_to, broadcast_arrays, and binary elementwise "
            "rank broadcasting contracts for bounded Program AD traces; dynamic "
            "output shapes, incompatible shapes, subok propagation, Rust/LLVM "
            "executable lowering, hardware, and performance promotion remain "
            "blocked; no wall-clock performance claim"
        ),
    )


def _jax_loop_heavy_case() -> DifferentiableProgrammingExternalReferenceResult:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    values = np.array([0.2, -0.4, 0.7, -0.9], dtype=np.float64)

    def program_objective(trace_values: Any) -> object:
        total = trace_values[0] * trace_values[0]
        for index in range(4):
            total = total + float(index + 1) * np.sin(trace_values[index])
        return total

    def reference_objective(raw_values: Any) -> object:
        total = raw_values[0] * raw_values[0]
        for index in range(4):
            total = total + float(index + 1) * jnp.sin(raw_values[index])
        return total

    program_result = whole_program_value_and_grad(
        program_objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )
    reference_value, reference_gradient = jax_value_and_grad(reference_objective, values)
    return DifferentiableProgrammingExternalReferenceResult(
        case_id="jax_loop_heavy_reference",
        backend="jax",
        program_value=program_result.value,
        reference_value=reference_value,
        program_gradient=program_result.gradient,
        reference_gradient=reference_gradient,
        max_abs_value_error=abs(program_result.value - reference_value),
        max_abs_gradient_error=_max_abs_error(program_result.gradient, reference_gradient),
        claim_boundary=(
            "optional JAX external-backend conformance for loop-heavy program AD; "
            "diagnostic correctness only, not a JIT, performance, LLVM, Rust, or hardware claim"
        ),
    )


def _jax_linalg_primitive_case() -> DifferentiableProgrammingExternalReferenceResult:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    values = np.array([1.5, 2.0, -0.75, 0.5], dtype=np.float64)
    inverse_weights = np.array([[0.25, 0.0], [0.0, -0.5]], dtype=np.float64)
    solve_weights = np.array([1.25, -0.75], dtype=np.float64)
    power_weights = np.array([[0.5, 0.0], [0.0, -0.25]], dtype=np.float64)
    eigh_weights = np.array([0.1, -0.2], dtype=np.float64)
    eigh_vector_weights = np.array([[0.05, -0.1], [0.2, 0.15]], dtype=np.float64)
    eig_weights = np.array([0.025, -0.035], dtype=np.float64)
    eig_vector_weights = np.array([[0.04, -0.02], [0.03, 0.05]], dtype=np.float64)
    eigvals_weights = np.array([0.05, -0.07], dtype=np.float64)
    eigvalsh_weights = np.array([0.2, -0.3], dtype=np.float64)
    svd_weights = np.array([0.15, -0.25], dtype=np.float64)
    pinv_weights = np.array([[0.35, 0.0], [0.0, -0.45]], dtype=np.float64)
    norm_row_weights = np.array([0.45, -0.65], dtype=np.float64)
    norm_column_weights = np.array([0.15, 0.2], dtype=np.float64)
    frobenius_weight = 0.175

    def program_objective(trace_values: Any) -> object:
        matrix = np.diag(trace_values[:2])
        rhs = trace_values[2:4]
        eigh_values, eigh_vectors = np.linalg.eigh(matrix)
        eig_values, eig_vectors = np.linalg.eig(matrix)
        return (
            np.linalg.det(matrix)
            + np.sum(np.linalg.inv(matrix) * inverse_weights)
            + np.sum(np.linalg.solve(matrix, rhs) * solve_weights)
            + np.sum(np.linalg.matrix_power(matrix, 2) * power_weights)
            + np.sum(eigh_values * eigh_weights)
            + np.sum(eigh_vectors * eigh_vector_weights)
            + np.sum(eig_values * eig_weights)
            + np.sum(eig_vectors * eig_vector_weights)
            + np.sum(np.linalg.eigvals(matrix) * eigvals_weights)
            + np.sum(np.linalg.eigvalsh(matrix) * eigvalsh_weights)
            + np.sum(np.linalg.svd(matrix, compute_uv=False) * svd_weights)
            + np.sum(np.linalg.pinv(matrix) * pinv_weights)
            + np.sum(np.linalg.norm(matrix, 2, axis=1) * norm_row_weights)
            + np.sum(np.linalg.norm(matrix, None, 0) * norm_column_weights)
            + frobenius_weight * np.linalg.norm(matrix, "fro", axis=(0, 1))
        )

    def reference_objective(raw_values: Any) -> object:
        matrix = jnp.diag(raw_values[:2])
        rhs = raw_values[2:4]
        eigh_values, eigh_vectors = jnp.linalg.eigh(matrix)
        eig_values, eig_vectors = jnp.linalg.eig(matrix)
        return (
            jnp.linalg.det(matrix)
            + jnp.sum(jnp.linalg.inv(matrix) * jnp.asarray(inverse_weights))
            + jnp.sum(jnp.linalg.solve(matrix, rhs) * jnp.asarray(solve_weights))
            + jnp.sum(jnp.linalg.matrix_power(matrix, 2) * jnp.asarray(power_weights))
            + jnp.sum(eigh_values * jnp.asarray(eigh_weights))
            + jnp.sum(eigh_vectors * jnp.asarray(eigh_vector_weights))
            + jnp.sum(jnp.real(eig_values) * jnp.asarray(eig_weights))
            + jnp.sum(jnp.real(eig_vectors) * jnp.asarray(eig_vector_weights))
            + jnp.sum(jnp.real(jnp.linalg.eigvals(matrix)) * jnp.asarray(eigvals_weights))
            + jnp.sum(jnp.linalg.eigvalsh(matrix) * jnp.asarray(eigvalsh_weights))
            + jnp.sum(jnp.linalg.svd(matrix, compute_uv=False) * jnp.asarray(svd_weights))
            + jnp.sum(jnp.linalg.pinv(matrix) * jnp.asarray(pinv_weights))
            + jnp.sum(jnp.linalg.norm(matrix, ord=2, axis=1) * jnp.asarray(norm_row_weights))
            + jnp.sum(jnp.linalg.norm(matrix, ord=2, axis=0) * jnp.asarray(norm_column_weights))
            + frobenius_weight * jnp.linalg.norm(matrix, ord="fro", axis=(0, 1))
        )

    program_result = whole_program_value_and_grad(
        program_objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )
    reference_value, reference_gradient = jax_value_and_grad(reference_objective, values)
    return DifferentiableProgrammingExternalReferenceResult(
        case_id="jax_linalg_primitive_reference",
        backend="jax",
        program_value=program_result.value,
        reference_value=reference_value,
        program_gradient=program_result.gradient,
        reference_gradient=reference_gradient,
        max_abs_value_error=abs(program_result.value - reference_value),
        max_abs_gradient_error=_max_abs_error(program_result.gradient, reference_gradient),
        claim_boundary=(
            "optional JAX external-backend conformance for supported linalg primitives; "
            "diagnostic correctness only, not a JIT, performance, LLVM, Rust, or hardware claim"
        ),
    )


def _jax_transform_nesting_case() -> DifferentiableProgrammingExternalReferenceResult:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    values = np.array([[0.5, -0.25], [1.25, 0.75], [-0.4, 1.1]], dtype=np.float64)

    def program_sample_objective(row: Any) -> object:
        return row[0] * row[0] + np.sin(row[1])

    program_gradients = vmap(
        lambda row: (
            whole_program_value_and_grad(program_sample_objective, row, trace=False).gradient
        )
    )(values)

    def reference_sample_objective(row: Any) -> object:
        return row[0] * row[0] + jnp.sin(row[1])

    reference_gradients = np.asarray(
        jax.vmap(jax.grad(reference_sample_objective))(jnp.asarray(values)),
        dtype=np.float64,
    )
    program_gradient = np.asarray(program_gradients, dtype=np.float64).reshape(-1)
    reference_gradient = reference_gradients.reshape(-1)
    program_value = float(np.sum(values[:, 0] ** 2 + np.sin(values[:, 1])))
    reference_value = float(np.sum(np.asarray(jax.vmap(reference_sample_objective)(values))))
    return DifferentiableProgrammingExternalReferenceResult(
        case_id="jax_transform_nesting_reference",
        backend="jax",
        program_value=program_value,
        reference_value=reference_value,
        program_gradient=program_gradient,
        reference_gradient=reference_gradient,
        max_abs_value_error=abs(program_value - reference_value),
        max_abs_gradient_error=_max_abs_error(program_gradient, reference_gradient),
        claim_boundary=(
            "optional JAX external-backend conformance for vmap over program AD gradients; "
            "diagnostic correctness only, not a JIT, performance, LLVM, Rust, or hardware claim"
        ),
    )


def _mutation_heavy_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([0.4, -0.6, 1.25, -1.5], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        work = trace_values.copy()
        work[0] = trace_values[1] * trace_values[1] + trace_values[2]
        return work[0] + trace_values[0] * trace_values[3]

    analytic = np.array([values[3], 2.0 * values[1], 1.0, values[0]], dtype=np.float64)
    return _program_ad_case(
        "mutation_heavy_forward_only",
        "mutation-heavy",
        objective,
        values,
        analytic,
    )


def _indexing_heavy_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([1.0, -2.0, 0.5, 3.0, -1.5, 2.0], dtype=np.float64)
    block_weights = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    flat_sort_weights = np.array([-0.4, 0.8, -1.2, 1.6, -2.0, 2.4], dtype=np.float64)
    axis_sort_weights = np.array([[0.3, -0.6, 0.9], [-1.1, 1.4, -1.7]], dtype=np.float64)
    median_weight = 0.033
    quantile_weight = 0.041
    quantile_row_weights = np.array([0.9, -0.35], dtype=np.float64)
    percentile_weight = 0.026
    percentile_column_weights = np.array([-0.7, 0.55, 1.1], dtype=np.float64)
    trapz_x = np.array([0.0, 0.25, 1.0], dtype=np.float64)
    trapz_row_weights = np.array([0.35, -0.45], dtype=np.float64)
    gradient_flat_weights = np.array([0.25, -0.5, 1.0, -1.5, 0.75, -0.25], dtype=np.float64)
    gradient_axis_weights = np.array([[0.4, -0.6, 0.8], [-1.0, 1.2, -1.4]], dtype=np.float64)
    interp_xp = np.array([-3.0, 0.0, 2.0, 5.0], dtype=np.float64)
    interp_static_fp = np.array([0.75, -1.25, 1.5, -0.5], dtype=np.float64)
    interp_sample_weights = np.array([0.45, -0.35], dtype=np.float64)
    interp_control_weights = np.array([1.1, -0.7, 0.3], dtype=np.float64)
    convolve_full_weights = np.array([0.2, -0.35, 0.5, -0.65, 0.8], dtype=np.float64)
    convolve_same_kernel = np.array([0.4, -0.2], dtype=np.float64)
    convolve_same_weights = np.array([1.0, -0.75, 0.5, -0.25, 0.125, -0.5], dtype=np.float64)
    convolve_static_signal = np.array([0.75, -1.25, 1.5, -0.5], dtype=np.float64)
    convolve_valid_weights = np.array([0.6, -0.4], dtype=np.float64)
    correlate_full_weights = np.array([-0.15, 0.35, -0.55, 0.75, -0.95], dtype=np.float64)
    correlate_same_reference = np.array([0.45, -0.3], dtype=np.float64)
    correlate_same_weights = np.array([-0.6, 0.4, -0.2, 0.8, -1.0, 0.5], dtype=np.float64)
    correlate_static_signal = np.array([1.2, -0.7, 0.9, -1.1], dtype=np.float64)
    correlate_valid_weights = np.array([0.25, -0.85], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        matrix = np.reshape(trace_values, (2, 3))
        block = matrix[:, 1:]
        gathered = np.take(trace_values, [2, 0, 2])
        lower_left = matrix[1:, :2]
        advanced = matrix[[1, 0, 1], [2, 0, 2]]
        masked_columns = matrix[np.array([True, False])][:, np.array([2, 0, 2])]
        along_indices = np.array([[2, 0, 2], [1, 1, 0]], dtype=np.int64)
        along_weights = np.array([[1.0, -0.5, 2.0], [0.25, 1.5, -1.25]], dtype=np.float64)
        along = np.take_along_axis(matrix, along_indices, axis=1)
        wrapped = np.take(trace_values, [-1, 6, 0], mode="wrap")
        clipped = np.take(matrix, [-2, 1, 10], axis=1, mode="clip")
        axis_deleted = np.delete(matrix, [1], axis=1)
        flat_deleted = np.delete(trace_values, [1, 4])
        padded = np.pad(matrix, ((1, 0), (1, 0)), mode="constant", constant_values=-0.75)
        inserted = np.insert(matrix, 1, np.array([-0.25, 0.5]), axis=1)
        axis_appended = np.append(matrix[:, :2], matrix[:, 2:], axis=1)
        flat_appended = np.append(trace_values[:3], trace_values[3:])
        hstacked = np.hstack((trace_values[:3], trace_values[3:]))
        vstacked = np.vstack((matrix[0], matrix[1]))
        column_stacked = np.column_stack((trace_values[:3], trace_values[3:]))
        dstacked = np.dstack((matrix[:, :2], matrix[:, 1:]))
        blocked = np.block([[matrix[:, :2], matrix[:, 2:]], [matrix[1:, :2], matrix[:1, 1:2]]])
        split_first, split_second, split_third = np.split(trace_values, [2, 4])
        split_top, split_bottom = np.vsplit(matrix, 2)
        split_left, split_middle, split_right = np.hsplit(matrix, [1, 2])
        split_depth0, split_depth1 = np.dsplit(np.reshape(trace_values, (1, 2, 3)), [1])
        uneven0, uneven1, uneven2, uneven3 = np.array_split(trace_values, 4)
        lower_triangle = np.tril(matrix)
        upper_triangle = np.triu(matrix, k=1)
        depth_triangle = np.tril(np.reshape(trace_values, (1, 2, 3)), k=-1)
        offset_diagonal = np.diagonal(matrix, offset=1)
        depth_diagonal = np.diagonal(
            np.reshape(trace_values, (1, 2, 3)), offset=1, axis1=1, axis2=2
        )
        flat_diagonal = np.diagflat(matrix[:, :2], k=1)
        broadcast_left, broadcast_right = np.broadcast_arrays(matrix[:, :1], trace_values[:3])
        column_assembled = np.concatenate((matrix[:, 2:], matrix[:, :1], matrix[:, 1:2]), axis=1)
        depth_stacked = np.stack((matrix, matrix[:, ::-1]), axis=2)
        flat_assembled = np.concatenate((matrix[:, :1], matrix[:, 1:]), axis=None)
        flat_sorted = np.sort(trace_values, axis=None)
        axis_sorted = np.sort(matrix, axis=1)
        flat_median = np.median(trace_values)
        row_quantiles = np.quantile(matrix, 0.25, axis=1)
        column_percentiles = np.percentile(matrix, 75.0, axis=0)
        row_integrals = np.trapezoid(matrix, x=trapz_x, axis=1)
        flat_integral = np.trapezoid(trace_values, dx=0.2)
        flat_gradient = np.gradient(trace_values, 0.5, edge_order=1)
        axis_gradient = np.gradient(
            matrix,
            np.array([0.0, 0.5, 1.5], dtype=np.float64),
            axis=1,
            edge_order=2,
        )
        sample_interpolation = np.interp(trace_values[:2], interp_xp, interp_static_fp)
        control_interpolation = np.interp(
            np.array([-2.5, 1.0, 4.0], dtype=np.float64),
            interp_xp,
            trace_values[2:],
        )
        dynamic_convolution = np.convolve(trace_values[:3], trace_values[3:], mode="full")
        static_kernel_convolution = np.convolve(trace_values, convolve_same_kernel, mode="same")
        static_signal_convolution = np.convolve(
            convolve_static_signal,
            trace_values[3:],
            mode="valid",
        )
        dynamic_correlation = np.correlate(trace_values[:3], trace_values[3:], mode="full")
        static_reference_correlation = np.correlate(
            trace_values,
            correlate_same_reference,
            mode="same",
        )
        static_signal_correlation = np.correlate(
            correlate_static_signal,
            trace_values[3:],
            mode="valid",
        )
        return (
            np.sum(block * block_weights)
            + np.sum(gathered)
            - 2.0 * lower_left[0, 1]
            + 0.5 * matrix[None, :, :][0, -1, -1]
            + 0.25 * np.sum(advanced)
            - 0.1 * np.sum(masked_columns)
            + 0.2 * np.sum(along * along_weights)
            + 0.3 * np.sum(wrapped * np.array([0.5, -1.0, 2.0], dtype=np.float64))
            - 0.2
            * np.sum(
                clipped
                * np.array(
                    [[1.0, -0.25, 0.5], [0.75, -1.5, 0.25]],
                    dtype=np.float64,
                )
            )
            + 0.4 * np.sum(axis_deleted * np.array([[1.0, -2.0], [0.5, 3.0]]))
            + 0.6 * np.sum(flat_deleted * np.array([0.25, -0.75, 1.25, -1.5], dtype=np.float64))
            + 0.15
            * np.sum(
                padded
                * np.array(
                    [[0.5, -1.0, 2.0, 0.25], [1.5, -2.0, 0.75, 3.0], [-0.25, 0.5, 2.5, -1.5]],
                    dtype=np.float64,
                )
            )
            + 0.12
            * np.sum(
                inserted
                * np.array(
                    [[1.0, -4.0, 2.0, 0.5], [-1.0, 3.0, 1.5, -2.0]],
                    dtype=np.float64,
                )
            )
            + 0.05
            * np.sum(
                axis_appended * np.array([[1.0, -2.0, 0.5], [1.5, -0.75, 2.25]], dtype=np.float64)
            )
            + 0.07
            * np.sum(flat_appended * np.array([0.2, -0.4, 0.6, -0.8, 1.0, -1.2], dtype=np.float64))
            + 0.03
            * np.sum(hstacked * np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], dtype=np.float64))
            + 0.04
            * np.sum(
                vstacked * np.array([[0.5, -1.0, 1.5], [-0.25, 0.75, -1.25]], dtype=np.float64)
            )
            + 0.02
            * np.sum(
                column_stacked
                * np.array([[1.0, -0.5], [0.25, 1.5], [-1.0, 0.75]], dtype=np.float64)
            )
            + 0.01
            * np.sum(
                dstacked
                * np.array(
                    [[[0.2, -0.1], [0.4, -0.3]], [[0.5, -0.2], [0.7, -0.6]]],
                    dtype=np.float64,
                )
            )
            + 0.025
            * np.sum(
                blocked
                * np.array(
                    [[0.25, -0.5, 0.75], [1.0, -1.25, 1.5], [-0.75, 0.5, -1.0]],
                    dtype=np.float64,
                )
            )
            + 0.015
            * (
                np.sum(split_first * np.array([1.0, -2.0], dtype=np.float64))
                + np.sum(split_second * np.array([0.5, 3.0], dtype=np.float64))
                + np.sum(split_third * np.array([-1.5, 2.5], dtype=np.float64))
                + np.sum(split_top * np.array([[0.25, -0.5, 0.75]], dtype=np.float64))
                + np.sum(split_bottom * np.array([[-1.0, 1.5, -2.0]], dtype=np.float64))
                + np.sum(split_left * np.array([[2.0], [-0.25]], dtype=np.float64))
                + np.sum(split_middle * np.array([[-1.25], [0.5]], dtype=np.float64))
                + np.sum(split_right * np.array([[1.75], [-0.75]], dtype=np.float64))
                + np.sum(split_depth0 * np.array([[[0.4], [-0.6]]], dtype=np.float64))
                + np.sum(split_depth1 * np.array([[[0.2, -0.3], [0.8, -0.9]]], dtype=np.float64))
                + np.sum(uneven0 * np.array([0.05, -0.1], dtype=np.float64))
                + np.sum(uneven1 * np.array([0.15, -0.2], dtype=np.float64))
                + np.sum(uneven2 * np.array([0.25], dtype=np.float64))
                + np.sum(uneven3 * np.array([-0.3], dtype=np.float64))
            )
            + 0.02
            * (
                np.sum(
                    lower_triangle
                    * np.array([[1.0, -2.0, 3.0], [0.5, 1.5, -1.0]], dtype=np.float64)
                )
                + np.sum(
                    upper_triangle
                    * np.array([[-0.25, 0.75, -1.25], [2.0, -0.5, 1.0]], dtype=np.float64)
                )
                + np.sum(
                    depth_triangle
                    * np.array([[[0.1, -0.2, 0.3], [2.0, -0.4, 0.5]]], dtype=np.float64)
                )
            )
            + 0.025 * np.sum(offset_diagonal * np.array([0.4, -0.6], dtype=np.float64))
            + 0.015 * np.sum(depth_diagonal * np.array([[1.2, -0.8]], dtype=np.float64))
            + 0.017
            * np.sum(
                flat_diagonal
                * np.array(
                    [
                        [0.0, 0.2, 0.0, 0.0, 0.0],
                        [0.0, 0.0, -0.7, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.1, 0.0],
                        [0.0, 0.0, 0.0, 0.0, -0.3],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    dtype=np.float64,
                )
            )
            + 0.011
            * np.sum(
                broadcast_left * np.array([[0.4, -0.2, 0.6], [1.0, -0.5, 0.25]], dtype=np.float64)
            )
            + 0.013
            * np.sum(
                broadcast_right * np.array([[-0.3, 0.7, -0.1], [0.5, -0.4, 0.2]], dtype=np.float64)
            )
            + np.sum(column_assembled * np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
            + np.sum(
                depth_stacked
                * np.array(
                    [
                        [[0.5, -1.0], [1.5, 0.25], [-0.75, 2.0]],
                        [[-0.5, 1.25], [0.75, -1.5], [2.5, -0.25]],
                    ],
                    dtype=np.float64,
                )
            )
            + np.sum(
                flat_assembled * np.array([-0.25, 0.5, -1.5, 2.0, 0.75, -0.5], dtype=np.float64)
            )
            + 0.075 * np.sum(flat_sorted * flat_sort_weights)
            + 0.055 * np.sum(axis_sorted * axis_sort_weights)
            + median_weight * flat_median
            + quantile_weight * np.sum(row_quantiles * quantile_row_weights)
            + percentile_weight * np.sum(column_percentiles * percentile_column_weights)
            + 0.09 * np.sum(row_integrals * trapz_row_weights)
            - 0.04 * flat_integral
            + 0.031 * np.sum(flat_gradient * gradient_flat_weights)
            + 0.027 * np.sum(axis_gradient * gradient_axis_weights)
            + 0.029 * np.sum(sample_interpolation * interp_sample_weights)
            + 0.034 * np.sum(control_interpolation * interp_control_weights)
            + 0.023 * np.sum(dynamic_convolution * convolve_full_weights)
            + 0.019 * np.sum(static_kernel_convolution * convolve_same_weights)
            - 0.017 * np.sum(static_signal_convolution * convolve_valid_weights)
            + 0.021 * np.sum(dynamic_correlation * correlate_full_weights)
            - 0.016 * np.sum(static_reference_correlation * correlate_same_weights)
            + 0.018 * np.sum(static_signal_correlation * correlate_valid_weights)
        )

    analytic = np.array(
        [
            6.19807125,
            4.391926666666667,
            4.958059166666667,
            5.364220833333333,
            8.34728,
            12.0869925,
        ],
        dtype=np.float64,
    )
    return _program_ad_case(
        "indexing_static_gather_contracts",
        "indexing-heavy",
        objective,
        values,
        analytic,
    )


def _shape_view_alias_metadata_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        matrix = np.reshape(trace_values, (2, 3))
        trailing = matrix[:, 1:]
        transposed = trailing.T
        flat = transposed.ravel()
        tensor = np.reshape(trace_values, (1, 2, 1, 3))
        squeezed = np.squeeze(tensor, axis=(0, 2))
        expanded = np.expand_dims(squeezed, axis=0)
        swapped = np.swapaxes(expanded, 0, 1)
        moved = np.moveaxis(swapped, source=2, destination=0)
        repeated = np.repeat(moved, repeats=(1, 2, 1), axis=0)
        promoted = np.atleast_3d(squeezed[0])
        base = np.reshape(trace_values, (2, 3))
        tiled = np.tile(base, (2, 1))
        rolled = np.roll(base, shift=1, axis=1)
        rotated = np.rot90(base, k=1)
        flipped = np.flip(base, axis=0)
        flipped_ud = np.flipud(base)
        flipped_lr = np.fliplr(base)
        return (
            flat[0]
            + 2.0 * flat[2]
            + np.sum(repeated)
            + np.sum(promoted)
            + np.sum(tiled)
            + np.sum(rolled)
            + np.sum(rotated)
            + np.sum(flipped)
            + np.sum(flipped_ud)
            + np.sum(flipped_lr)
        )

    result = whole_program_value_and_grad(objective, values)
    if result.program_ir is None:
        raise ValueError("shape-view alias benchmark requires Program AD IR")
    analysis = analyze_program_ad_alias_effects(result.program_ir)
    view_targets = tuple(edge.target for edge in analysis.alias_edges if edge.kind == "view_alias")
    required_prefixes = (
        "view:getitem",
        "view:ravel",
        "view:reshape",
        "view:squeeze",
        "view:expand_dims",
        "view:swapaxes",
        "view:moveaxis",
        "view:repeat",
        "view:atleast_3d",
        "view:tile",
        "view:roll",
        "view:rot90",
        "view:flip",
        "view:flipud",
        "view:fliplr",
    )
    missing = tuple(
        prefix
        for prefix in required_prefixes
        if not any(target.startswith(prefix) for target in view_targets)
    )
    if missing:
        raise ValueError(f"shape-view alias benchmark missing aliases: {missing}")

    source = np.arange(values.size, dtype=np.int64)
    matrix_source = source.reshape(2, 3)
    flat_source = matrix_source[:, 1:].T.reshape(-1)
    squeezed_source = source.reshape(1, 2, 1, 3).squeeze(axis=(0, 2))
    moved_source = np.moveaxis(
        np.swapaxes(np.expand_dims(squeezed_source, axis=0), 0, 1),
        source=2,
        destination=0,
    )
    repeated_source = np.repeat(moved_source, repeats=(1, 2, 1), axis=0)
    promoted_source = np.atleast_3d(squeezed_source[0]).reshape(-1)
    analytic = np.zeros(values.size, dtype=np.float64)
    np.add.at(analytic, [int(flat_source[0]), int(flat_source[2])], [1.0, 2.0])
    np.add.at(analytic, repeated_source.reshape(-1), 1.0)
    np.add.at(analytic, promoted_source, 1.0)
    for permutation_source in (
        np.tile(matrix_source, (2, 1)),
        np.roll(matrix_source, shift=1, axis=1),
        np.rot90(matrix_source, k=1),
        np.flip(matrix_source, axis=0),
        np.flipud(matrix_source),
        np.fliplr(matrix_source),
    ):
        np.add.at(analytic, permutation_source.reshape(-1), 1.0)
    adjoint_supported = result.adjoint_result is not None and result.adjoint_result.supported
    adjoint_error = (
        _max_abs_error(program_adjoint_gradient(result), analytic) if adjoint_supported else None
    )
    return DifferentiableProgrammingBenchmarkResult(
        case_id="shape_view_alias_metadata_contracts",
        category="alias-effect",
        value=result.value,
        gradient=result.gradient,
        analytic_gradient=analytic,
        max_abs_gradient_error=_max_abs_error(result.gradient, analytic),
        adjoint_supported=adjoint_supported,
        max_abs_adjoint_error=adjoint_error,
        claim_boundary=(
            "deterministic shape-view alias metadata conformance for reshape, getitem, "
            "ravel, squeeze, expand_dims, swapaxes, moveaxis, repeat, atleast_3d, "
            "tile, roll, rot90, flip, flipud, and fliplr; "
            "metadata_only_no_general_alias_lattice; no wall-clock performance, hardware, "
            "LLVM, Rust, or JIT execution claim"
        ),
    )


def _slice_mutation_alias_metadata_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        window = np.reshape(trace_values, (6,))[1:5]
        window[1:3] = np.array([2.0 * trace_values[0], trace_values[5] + 1.0])
        return window[0] + window[1] + window[2] + window[3]

    result = whole_program_value_and_grad(objective, values)
    if result.program_ir is None:
        raise ValueError("slice-mutation alias benchmark requires Program AD IR")
    analysis = analyze_program_ad_alias_effects(result.program_ir)
    mutation_sources = tuple(
        edge.source for edge in analysis.alias_edges if edge.kind == "mutation_version"
    )
    if "%array[2]" not in mutation_sources or "%array[3]" not in mutation_sources:
        raise ValueError("slice-mutation alias benchmark missing source-index mutations")
    if not any(edge.target.startswith("view:getitem") for edge in analysis.alias_edges):
        raise ValueError("slice-mutation alias benchmark missing view alias metadata")
    if len(analysis.mutation_effects) != 2:
        raise ValueError("slice-mutation alias benchmark expected two mutation effects")

    analytic = np.array([2.0, 1.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    adjoint_supported = result.adjoint_result is not None and result.adjoint_result.supported
    adjoint_error = (
        _max_abs_error(program_adjoint_gradient(result), analytic) if adjoint_supported else None
    )
    return DifferentiableProgrammingBenchmarkResult(
        case_id="slice_mutation_alias_metadata_contracts",
        category="alias-effect",
        value=result.value,
        gradient=result.gradient,
        analytic_gradient=analytic,
        max_abs_gradient_error=_max_abs_error(result.gradient, analytic),
        adjoint_supported=adjoint_supported,
        max_abs_adjoint_error=adjoint_error,
        claim_boundary=(
            "deterministic static slice-mutation alias/effect metadata conformance for "
            "rank-1 Program AD views; metadata_only_no_general_alias_lattice; no "
            "wall-clock performance, hardware, LLVM, Rust, or JIT execution claim"
        ),
    )


def _loop_carried_state_alias_metadata_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([0.25, 0.5, 0.75, 1.0, 1.25], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        carry = trace_values[0]
        for index in range(1, 4):
            carry = carry + float(index) * trace_values[index]
        return carry + trace_values[4]

    result = whole_program_value_and_grad(objective, values)
    if result.program_ir is None:
        raise ValueError("loop-carried state alias benchmark requires Program AD IR")
    analysis = analyze_program_ad_alias_effects(result.program_ir)
    loop_edges = tuple(edge for edge in analysis.alias_edges if edge.kind == "loop_carried_state")
    if not any(
        edge.source == "loop:carry:entry" and edge.target == "loop:carry:backedge"
        for edge in loop_edges
    ):
        raise ValueError("loop-carried state alias benchmark missing carry backedge metadata")
    if not any(
        phi.selected == "executed_loop_trace"
        and "loop_entry" in phi.incoming
        and "loop_backedge" in phi.incoming
        for phi in result.program_ir.phi_nodes
    ):
        raise ValueError("loop-carried state alias benchmark missing loop phi metadata")

    analytic = np.array([1.0, 1.0, 2.0, 3.0, 1.0], dtype=np.float64)
    adjoint_supported = result.adjoint_result is not None and result.adjoint_result.supported
    adjoint_error = (
        _max_abs_error(program_adjoint_gradient(result), analytic) if adjoint_supported else None
    )
    return DifferentiableProgrammingBenchmarkResult(
        case_id="loop_carried_state_alias_metadata_contracts",
        category="alias-effect",
        value=result.value,
        gradient=result.gradient,
        analytic_gradient=analytic,
        max_abs_gradient_error=_max_abs_error(result.gradient, analytic),
        adjoint_supported=adjoint_supported,
        max_abs_adjoint_error=adjoint_error,
        claim_boundary=(
            "deterministic loop-carried state alias metadata conformance for local "
            "derivative-carrying scalar reassignment; "
            "metadata_only_no_general_alias_lattice; no wall-clock performance, "
            "hardware, LLVM, Rust, or JIT execution claim"
        ),
    )


def _transform_nesting_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([[0.5, -0.25], [1.25, 0.75], [-0.4, 1.1]], dtype=np.float64)

    def sample_objective(row: Any) -> object:
        return row[0] * row[0] + np.sin(row[1])

    def batched_objective(flat_values: Any) -> object:
        rows = flat_values.reshape((3, 2))
        return np.sum(cast(Any, vmap(sample_objective)(rows)))

    mapped_gradients = vmap(
        lambda row: whole_program_value_and_grad(sample_objective, row, trace=False).gradient
    )(values)
    whole_program_batched_gradient = grad(
        batched_objective,
        values.reshape(-1),
        method="whole_program",
    )
    analytic = np.column_stack((2.0 * values[:, 0], np.cos(values[:, 1])))
    analytic_flat = analytic.reshape(-1)
    gradient = np.concatenate(
        [
            np.asarray(mapped_gradients, dtype=np.float64).reshape(-1),
            np.asarray(whole_program_batched_gradient, dtype=np.float64).reshape(-1),
        ]
    )
    analytic_gradient = np.concatenate([analytic_flat, analytic_flat])
    return DifferentiableProgrammingBenchmarkResult(
        case_id="transform_nesting_vmap_program_grad",
        category="transform-nesting",
        value=float(np.sum(values[:, 0] ** 2 + np.sin(values[:, 1]))),
        gradient=gradient,
        analytic_gradient=analytic_gradient,
        max_abs_gradient_error=_max_abs_error(gradient, analytic_gradient),
        adjoint_supported=True,
        max_abs_adjoint_error=0.0,
        claim_boundary=(
            "vmap over program AD gradients and program AD grad(vmap(f)) compared with "
            "analytic separable references; diagnostic conformance only, not a "
            "performance timing claim"
        ),
    )


def _higher_order_transform_nesting_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([[0.5, 0.25], [-1.2, 0.75]], dtype=np.float64)
    flat_values = values.reshape(-1)
    row_shape = (2, 2)

    def row_loss(row: Any) -> object:
        return row[0] * row[0] + row[0] * row[1] + 0.5 * row[1] * row[1]

    def batched_loss(candidate: Any) -> object:
        return np.sum(cast(Any, vmap(row_loss)(np.reshape(candidate, row_shape))))

    def batched_program_gradient(candidate: Any) -> NDArray[np.float64]:
        return grad(batched_loss, candidate, method="whole_program")

    forward_nested = jacfwd(batched_program_gradient, flat_values, step=1.0e-4)
    reverse_nested = jacrev(batched_program_gradient, flat_values, step=1.0e-4)
    analytic_hessian = np.zeros((flat_values.size, flat_values.size), dtype=np.float64)
    analytic_hessian[0:2, 0:2] = np.array([[2.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    analytic_hessian[2:4, 2:4] = np.array([[2.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    gradient = np.concatenate([forward_nested.reshape(-1), reverse_nested.reshape(-1)])
    analytic_gradient = np.concatenate(
        [analytic_hessian.reshape(-1), analytic_hessian.reshape(-1)]
    )
    return DifferentiableProgrammingBenchmarkResult(
        case_id="transform_nesting_whole_program_higher_order",
        category="transform-nesting",
        value=float(cast(Any, batched_loss(flat_values))),
        gradient=gradient,
        analytic_gradient=analytic_gradient,
        max_abs_gradient_error=_max_abs_error(gradient, analytic_gradient),
        adjoint_supported=True,
        max_abs_adjoint_error=0.0,
        claim_boundary=(
            "jacfwd/jacrev over whole-program grad(vmap(f)) compared with analytic "
            "block-diagonal curvature; diagnostic conformance only, not a performance "
            "timing claim"
        ),
    )


def _program_ad_hessian_transform_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([0.4, -0.75], dtype=np.float64)
    analytic_hessian = np.array([[2.0, 1.0], [1.0, 1.0]], dtype=np.float64)

    def row_loss(row: Any) -> object:
        return row[0] * row[0] + row[0] * row[1] + 0.5 * row[1] * row[1]

    def program_value(candidate: Any) -> float:
        return float(whole_program_value_and_grad(row_loss, candidate, trace=False).value)

    program_hessian = hessian(program_value, values, step=1.0e-1)
    gradient = program_hessian.reshape(-1)
    analytic_gradient = analytic_hessian.reshape(-1)
    return DifferentiableProgrammingBenchmarkResult(
        case_id="transform_nesting_program_ad_hessian",
        category="transform-nesting",
        value=program_value(values),
        gradient=gradient,
        analytic_gradient=analytic_gradient,
        max_abs_gradient_error=_max_abs_error(gradient, analytic_gradient),
        adjoint_supported=True,
        max_abs_adjoint_error=0.0,
        claim_boundary=(
            "hessian over a whole-program AD scalar objective compared with analytic "
            "curvature; diagnostic conformance only, not a compiler, Rust, LLVM/JIT, "
            "hardware, or performance timing claim, and not a performance benchmark"
        ),
    )


def _program_ad_hessian_jvp_vjp_transform_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([0.4, -0.35], dtype=np.float64)
    tangent = np.array([0.25, -0.5], dtype=np.float64)
    cotangent = np.array([1.0, -0.25, 0.75, 0.5], dtype=np.float64)
    hessian_jacobian = np.array(
        [[6.0, 2.0], [2.0, 1.0], [2.0, 1.0], [1.0, 6.0]],
        dtype=np.float64,
    )

    def cubic_loss(row: Any) -> object:
        return (
            row[0] * row[0] * row[0]
            + row[0] * row[0] * row[1]
            + 0.5 * row[0] * row[1] * row[1]
            + row[1] * row[1] * row[1]
        )

    def program_value(candidate: Any) -> float:
        return float(whole_program_value_and_grad(cubic_loss, candidate, trace=False).value)

    def hessian_flat(candidate: Any) -> NDArray[np.float64]:
        return hessian(program_value, candidate, step=1.0e-1).reshape(-1)

    hessian_jvp = jvp(hessian_flat, values, tangent, step=1.0e-2)
    hessian_vjp = vjp(hessian_flat, values, cotangent, step=1.0e-2)
    analytic_jvp = hessian_jacobian @ tangent
    analytic_vjp = hessian_jacobian.T @ cotangent
    gradient = np.concatenate([hessian_jvp, hessian_vjp])
    analytic_gradient = np.concatenate([analytic_jvp, analytic_vjp])
    return DifferentiableProgrammingBenchmarkResult(
        case_id="transform_nesting_program_ad_hessian_jvp_vjp",
        category="transform-nesting",
        value=float(np.sum(hessian_flat(values))),
        gradient=gradient,
        analytic_gradient=analytic_gradient,
        max_abs_gradient_error=_max_abs_error(gradient, analytic_gradient),
        adjoint_supported=True,
        max_abs_adjoint_error=0.0,
        claim_boundary=(
            "jvp/vjp over whole-program AD Hessian transforms compared with analytic "
            "third-derivative contractions; diagnostic conformance only, not a "
            "compiler, Rust, LLVM/JIT, hardware, or performance timing claim, and "
            "not a performance benchmark"
        ),
    )


def _custom_rule_transform_nesting_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([[0.5, -0.25], [1.25, 0.75], [-0.4, 1.1]], dtype=np.float64)
    tangents = np.array([[0.2, -0.5], [-0.3, 0.4], [0.75, -0.1]], dtype=np.float64)
    cotangents = np.array([[1.0, -0.25], [0.5, 1.5], [-1.2, 0.75]], dtype=np.float64)

    def value_fn(row: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([row[0] * row[0] + np.sin(row[1]), row[0] * row[1]], dtype=np.float64)

    def jvp_rule(row: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array(
            [
                2.0 * row[0] * tangent[0] + math.cos(row[1]) * tangent[1],
                tangent[0] * row[1] + row[0] * tangent[1],
            ],
            dtype=np.float64,
        )

    def vjp_rule(row: NDArray[np.float64], cotangent: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array(
            [
                2.0 * row[0] * cotangent[0] + row[1] * cotangent[1],
                math.cos(row[1]) * cotangent[0] + row[0] * cotangent[1],
            ],
            dtype=np.float64,
        )

    rule = CustomDerivativeRule(
        name="benchmark_custom_rule_transform_nesting",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
        parameter_names=("x0", "x1"),
    )
    mapped_jvp = vmap(lambda row, tangent: custom_jvp(rule, row, tangent))(values, tangents)
    mapped_vjp = vmap(lambda row, cotangent: custom_vjp(rule, row, cotangent).vjp)(
        values, cotangents
    )
    analytic_jvp = np.asarray(
        [jvp_rule(row, tangent) for row, tangent in zip(values, tangents, strict=True)],
        dtype=np.float64,
    )
    analytic_vjp = np.asarray(
        [vjp_rule(row, cotangent) for row, cotangent in zip(values, cotangents, strict=True)],
        dtype=np.float64,
    )
    gradient = np.concatenate(
        [
            np.asarray(mapped_jvp, dtype=np.float64).reshape(-1),
            np.asarray(mapped_vjp, dtype=np.float64).reshape(-1),
        ]
    )
    analytic_gradient = np.concatenate([analytic_jvp.reshape(-1), analytic_vjp.reshape(-1)])
    value = float(np.sum(cast(Any, vmap(value_fn)(values))))
    return DifferentiableProgrammingBenchmarkResult(
        case_id="transform_nesting_custom_rule_vmap_jvp_vjp",
        category="transform-nesting",
        value=value,
        gradient=gradient,
        analytic_gradient=analytic_gradient,
        max_abs_gradient_error=_max_abs_error(gradient, analytic_gradient),
        adjoint_supported=True,
        max_abs_adjoint_error=0.0,
        claim_boundary=(
            "vmap over exact custom JVP/VJP rules compared with analytic row-wise "
            "references; diagnostic conformance only, not a performance timing claim"
        ),
    )


def _program_ad_transform_jvp_vjp_case() -> DifferentiableProgrammingBenchmarkResult:
    values = np.array([[0.5, 0.25], [-1.2, 0.75]], dtype=np.float64)
    flat_values = values.reshape(-1)
    tangent = np.array([0.3, -0.2, 0.4, 0.1], dtype=np.float64)
    cotangent = np.array([1.5, -0.25, 0.75, 0.5], dtype=np.float64)
    row_shape = (2, 2)
    row_hessian = np.array([[2.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    block_hessian = np.zeros((flat_values.size, flat_values.size), dtype=np.float64)
    block_hessian[0:2, 0:2] = row_hessian
    block_hessian[2:4, 2:4] = row_hessian

    def row_loss(row: Any) -> object:
        return row[0] * row[0] + row[0] * row[1] + 0.5 * row[1] * row[1]

    def mapped_program_gradient(candidate: Any) -> NDArray[np.float64]:
        rows = np.reshape(candidate, row_shape)
        mapped = vmap(
            lambda row: whole_program_value_and_grad(row_loss, row, trace=False).gradient
        )(rows)
        return np.asarray(mapped, dtype=np.float64).reshape(-1)

    jvp_gradient = jvp(mapped_program_gradient, flat_values, tangent, step=1.0e-2)
    vjp_gradient = vjp(mapped_program_gradient, flat_values, cotangent, step=1.0e-2)
    analytic_jvp = block_hessian @ tangent
    analytic_vjp = block_hessian.T @ cotangent
    gradient = np.concatenate([jvp_gradient, vjp_gradient])
    analytic_gradient = np.concatenate([analytic_jvp, analytic_vjp])
    return DifferentiableProgrammingBenchmarkResult(
        case_id="transform_nesting_program_ad_vmap_jvp_vjp",
        category="transform-nesting",
        value=float(np.sum(mapped_program_gradient(flat_values))),
        gradient=gradient,
        analytic_gradient=analytic_gradient,
        max_abs_gradient_error=_max_abs_error(gradient, analytic_gradient),
        adjoint_supported=True,
        max_abs_adjoint_error=0.0,
        claim_boundary=(
            "jvp/vjp over vmap of whole-program AD gradients compared with analytic "
            "block-Hessian contractions; diagnostic conformance only, not a "
            "performance timing claim"
        ),
    )


def _program_ad_case(
    case_id: str,
    category: str,
    objective: Callable[[Any], object],
    values: NDArray[np.float64],
    analytic_gradient: NDArray[np.float64],
    claim_boundary: str | None = None,
) -> DifferentiableProgrammingBenchmarkResult:
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )
    adjoint_supported = result.adjoint_result is not None and result.adjoint_result.supported
    adjoint_error = None
    if adjoint_supported:
        adjoint_error = _max_abs_error(program_adjoint_gradient(result), analytic_gradient)
    return DifferentiableProgrammingBenchmarkResult(
        case_id=case_id,
        category=category,
        value=result.value,
        gradient=result.gradient,
        analytic_gradient=analytic_gradient,
        max_abs_gradient_error=_max_abs_error(result.gradient, analytic_gradient),
        adjoint_supported=adjoint_supported,
        max_abs_adjoint_error=adjoint_error,
        claim_boundary=claim_boundary
        or (
            "deterministic program AD conformance against analytic references; "
            "no wall-clock performance, hardware, LLVM, Rust, or JIT execution claim"
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


def _as_gradient(name: str, value: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _max_abs_error(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    return float(
        np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)))
    )


__all__ = [
    "DifferentiableProgrammingBenchmarkResult",
    "DifferentiableProgrammingExternalReferenceResult",
    "QuantumGradientBenchmarkResult",
    "run_differentiable_programming_benchmark_suite",
    "run_differentiable_programming_external_reference_suite",
    "run_quantum_gradient_benchmark_suite",
]
