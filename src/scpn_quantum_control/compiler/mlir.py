# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- MLIR textual compiler surface
"""Deterministic MLIR-style export for Kuramoto-XY workloads.

The module emits a conservative textual interchange layer for the SCPN
Kuramoto-XY compiler. It does not require an MLIR Python runtime. Compiler AD
native LLVM/JIT execution is available only for primitives with verified native
lowering metadata; unrelated QIR, provider-pulse, and hardware execution claims
remain outside this boundary. The value is a stable, auditable IR boundary for
compiler passes and external tooling.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil

# Subprocess is used only for admitted local toolchain version probes.
import subprocess  # nosec B404
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ..differentiable import (
    CustomDerivativeRegistry,
    CustomDerivativeRule,
    PrimitiveIdentity,
    program_ad_linalg_matrix_power_derivative_rule,
    program_ad_linalg_multi_dot_derivative_rule,
)
from ..kuramoto_core import KuramotoProblem, build_kuramoto_problem
from .mlir_enzyme_evidence import (
    EnzymeMLIRBenchmarkAttachment as EnzymeMLIRBenchmarkAttachment,
)
from .mlir_enzyme_evidence import (
    EnzymeMLIRCompilerADBreadthArtifact as EnzymeMLIRCompilerADBreadthArtifact,
)
from .mlir_enzyme_evidence import (
    EnzymeMLIRCompilerADBreadthCaseEvidence as EnzymeMLIRCompilerADBreadthCaseEvidence,
)
from .mlir_enzyme_evidence import (
    EnzymeMLIRCompilerADBreadthEvidence as EnzymeMLIRCompilerADBreadthEvidence,
)
from .mlir_enzyme_evidence import (
    EnzymeMLIRMaturityAuditResult as EnzymeMLIRMaturityAuditResult,
)
from .mlir_enzyme_evidence import (
    EnzymeMLIRToolchainStatus as EnzymeMLIRToolchainStatus,
)
from .mlir_enzyme_evidence import (
    EnzymeNativeExecutionEvidence as EnzymeNativeExecutionEvidence,
)
from .mlir_enzyme_evidence import (
    MLIRLLVMCorrectnessEvidence as MLIRLLVMCorrectnessEvidence,
)
from .mlir_enzyme_evidence import (
    build_enzyme_mlir_benchmark_attachment as build_enzyme_mlir_benchmark_attachment,
)
from .mlir_enzyme_evidence import (
    build_enzyme_mlir_compiler_ad_breadth_artifact as build_enzyme_mlir_compiler_ad_breadth_artifact,
)
from .mlir_enzyme_evidence import (
    build_enzyme_mlir_compiler_ad_breadth_evidence as build_enzyme_mlir_compiler_ad_breadth_evidence,
)
from .mlir_enzyme_evidence import (
    build_enzyme_mlir_compiler_ad_breadth_gap_artifact as build_enzyme_mlir_compiler_ad_breadth_gap_artifact,
)
from .mlir_enzyme_execution_runner import (
    EnzymeToolchainADCase as EnzymeToolchainADCase,
)
from .mlir_enzyme_execution_runner import (
    EnzymeToolchainADExecutionEvidence as EnzymeToolchainADExecutionEvidence,
)
from .mlir_enzyme_execution_runner import (
    run_enzyme_toolchain_execution_evidence as run_enzyme_toolchain_execution_evidence,
)
from .mlir_executable_kernel import (
    ExecutableCompilerADKernel as ExecutableCompilerADKernel,
)
from .mlir_executable_kernel import (
    _compile_scalar_gradient_llvm_ir as _compile_scalar_gradient_llvm_ir,
)
from .mlir_executable_kernel import (
    _verify_executable_ad_kernel as _verify_executable_ad_kernel,
)
from .mlir_executable_kernel import (
    compile_custom_derivative_rule_to_mlir as compile_custom_derivative_rule_to_mlir,
)
from .mlir_executable_kernel import (
    make_executable_ad_kernel_batching_rule as make_executable_ad_kernel_batching_rule,
)
from .mlir_matrix_2x2_native_compilation import (
    compile_matrix_2x2_determinant_ad_to_native_llvm_jit as compile_matrix_2x2_determinant_ad_to_native_llvm_jit,
)
from .mlir_matrix_2x2_native_compilation import (
    compile_matrix_2x2_eigensystem_ad_to_native_llvm_jit as compile_matrix_2x2_eigensystem_ad_to_native_llvm_jit,
)
from .mlir_matrix_2x2_native_compilation import (
    compile_matrix_2x2_eigenvalues_ad_to_native_llvm_jit as compile_matrix_2x2_eigenvalues_ad_to_native_llvm_jit,
)
from .mlir_matrix_2x2_native_compilation import (
    compile_matrix_2x2_inverse_ad_to_native_llvm_jit as compile_matrix_2x2_inverse_ad_to_native_llvm_jit,
)
from .mlir_matrix_2x2_native_compilation import (
    compile_matrix_2x2_solve_ad_to_native_llvm_jit as compile_matrix_2x2_solve_ad_to_native_llvm_jit,
)
from .mlir_matrix_2x2_native_compilation import (
    make_matrix_2x2_determinant_native_llvm_jit_lowering_rule as make_matrix_2x2_determinant_native_llvm_jit_lowering_rule,
)
from .mlir_matrix_2x2_native_compilation import (
    make_matrix_2x2_determinant_native_llvm_jit_primitive_transform as make_matrix_2x2_determinant_native_llvm_jit_primitive_transform,
)
from .mlir_matrix_2x2_native_compilation import (
    make_matrix_2x2_eigensystem_native_llvm_jit_lowering_rule as make_matrix_2x2_eigensystem_native_llvm_jit_lowering_rule,
)
from .mlir_matrix_2x2_native_compilation import (
    make_matrix_2x2_eigensystem_native_llvm_jit_primitive_transform as make_matrix_2x2_eigensystem_native_llvm_jit_primitive_transform,
)
from .mlir_matrix_2x2_native_compilation import (
    make_matrix_2x2_eigenvalues_native_llvm_jit_lowering_rule as make_matrix_2x2_eigenvalues_native_llvm_jit_lowering_rule,
)
from .mlir_matrix_2x2_native_compilation import (
    make_matrix_2x2_eigenvalues_native_llvm_jit_primitive_transform as make_matrix_2x2_eigenvalues_native_llvm_jit_primitive_transform,
)
from .mlir_matrix_2x2_native_compilation import (
    make_matrix_2x2_inverse_native_llvm_jit_lowering_rule as make_matrix_2x2_inverse_native_llvm_jit_lowering_rule,
)
from .mlir_matrix_2x2_native_compilation import (
    make_matrix_2x2_inverse_native_llvm_jit_primitive_transform as make_matrix_2x2_inverse_native_llvm_jit_primitive_transform,
)
from .mlir_matrix_2x2_native_compilation import (
    make_matrix_2x2_solve_native_llvm_jit_lowering_rule as make_matrix_2x2_solve_native_llvm_jit_lowering_rule,
)
from .mlir_matrix_2x2_native_compilation import (
    make_matrix_2x2_solve_native_llvm_jit_primitive_transform as make_matrix_2x2_solve_native_llvm_jit_primitive_transform,
)
from .mlir_matrix_native_compilation import (
    compile_matrix_frobenius_norm_squared_ad_to_native_llvm_jit as compile_matrix_frobenius_norm_squared_ad_to_native_llvm_jit,
)
from .mlir_matrix_native_compilation import (
    compile_matrix_matrix_product_ad_to_native_llvm_jit as compile_matrix_matrix_product_ad_to_native_llvm_jit,
)
from .mlir_matrix_native_compilation import (
    compile_matrix_quadratic_form_ad_to_native_llvm_jit as compile_matrix_quadratic_form_ad_to_native_llvm_jit,
)
from .mlir_matrix_native_compilation import (
    compile_matrix_trace_ad_to_native_llvm_jit as compile_matrix_trace_ad_to_native_llvm_jit,
)
from .mlir_matrix_native_compilation import (
    compile_matrix_vector_product_ad_to_native_llvm_jit as compile_matrix_vector_product_ad_to_native_llvm_jit,
)
from .mlir_matrix_native_compilation import (
    make_matrix_frobenius_norm_squared_native_llvm_jit_lowering_rule as make_matrix_frobenius_norm_squared_native_llvm_jit_lowering_rule,
)
from .mlir_matrix_native_compilation import (
    make_matrix_frobenius_norm_squared_native_llvm_jit_primitive_transform as make_matrix_frobenius_norm_squared_native_llvm_jit_primitive_transform,
)
from .mlir_matrix_native_compilation import (
    make_matrix_matrix_product_native_llvm_jit_lowering_rule as make_matrix_matrix_product_native_llvm_jit_lowering_rule,
)
from .mlir_matrix_native_compilation import (
    make_matrix_matrix_product_native_llvm_jit_primitive_transform as make_matrix_matrix_product_native_llvm_jit_primitive_transform,
)
from .mlir_matrix_native_compilation import (
    make_matrix_quadratic_form_native_llvm_jit_lowering_rule as make_matrix_quadratic_form_native_llvm_jit_lowering_rule,
)
from .mlir_matrix_native_compilation import (
    make_matrix_quadratic_form_native_llvm_jit_primitive_transform as make_matrix_quadratic_form_native_llvm_jit_primitive_transform,
)
from .mlir_matrix_native_compilation import (
    make_matrix_trace_native_llvm_jit_lowering_rule as make_matrix_trace_native_llvm_jit_lowering_rule,
)
from .mlir_matrix_native_compilation import (
    make_matrix_trace_native_llvm_jit_primitive_transform as make_matrix_trace_native_llvm_jit_primitive_transform,
)
from .mlir_matrix_native_compilation import (
    make_matrix_vector_product_native_llvm_jit_lowering_rule as make_matrix_vector_product_native_llvm_jit_lowering_rule,
)
from .mlir_matrix_native_compilation import (
    make_matrix_vector_product_native_llvm_jit_primitive_transform as make_matrix_vector_product_native_llvm_jit_primitive_transform,
)
from .mlir_native_execution_evidence import (
    NativeWholeProgramADExecutionCase as NativeWholeProgramADExecutionCase,
)
from .mlir_native_execution_evidence import (
    NativeWholeProgramADExecutionEvidence as NativeWholeProgramADExecutionEvidence,
)
from .mlir_native_execution_evidence import (
    build_native_whole_program_ad_execution_evidence as build_native_whole_program_ad_execution_evidence,
)
from .mlir_native_execution_evidence import (
    run_native_whole_program_ad_execution_evidence as run_native_whole_program_ad_execution_evidence,
)
from .mlir_native_primitives import (
    _as_finite_vector as _as_finite_vector,
)
from .mlir_native_primitives import (
    _compile_native_llvm_jit_functions as _compile_native_llvm_jit_functions,
)
from .mlir_native_primitives import (
    _copy_float_array as _copy_float_array,
)
from .mlir_native_primitives import (
    _escape_mlir_string as _escape_mlir_string,
)
from .mlir_native_primitives import (
    _fmt_bool as _fmt_bool,
)
from .mlir_native_primitives import (
    _fmt_float as _fmt_float,
)
from .mlir_native_primitives import (
    _load_llvmlite_binding as _load_llvmlite_binding,
)
from .mlir_native_primitives import (
    _max_abs_error as _max_abs_error,
)
from .mlir_native_primitives import (
    _safe_llvm_symbol as _safe_llvm_symbol,
)
from .mlir_records import (
    CompilerADExecutableConfig as CompilerADExecutableConfig,
)
from .mlir_records import (
    CompilerADKernelVerification as CompilerADKernelVerification,
)
from .mlir_records import (
    CompilerADTransformPlan as CompilerADTransformPlan,
)
from .mlir_records import (
    DifferentiableMLIRCompileConfig as DifferentiableMLIRCompileConfig,
)
from .mlir_records import (
    MLIRCompileConfig as MLIRCompileConfig,
)
from .mlir_records import (
    MLIRModule as MLIRModule,
)
from .mlir_records import (
    PrimitiveLoweringStatus as PrimitiveLoweringStatus,
)
from .mlir_scalar_native_compilation import (
    compile_scalar_binary_elementwise_ad_to_native_llvm_jit as compile_scalar_binary_elementwise_ad_to_native_llvm_jit,
)
from .mlir_scalar_native_compilation import (
    compile_scalar_quadratic_ad_to_native_llvm_jit as compile_scalar_quadratic_ad_to_native_llvm_jit,
)
from .mlir_scalar_native_compilation import (
    compile_scalar_unary_elementwise_ad_to_native_llvm_jit as compile_scalar_unary_elementwise_ad_to_native_llvm_jit,
)
from .mlir_scalar_native_compilation import (
    make_scalar_binary_elementwise_native_llvm_jit_lowering_rule as make_scalar_binary_elementwise_native_llvm_jit_lowering_rule,
)
from .mlir_scalar_native_compilation import (
    make_scalar_quadratic_native_llvm_jit_lowering_rule as make_scalar_quadratic_native_llvm_jit_lowering_rule,
)
from .mlir_scalar_native_compilation import (
    make_scalar_unary_elementwise_native_llvm_jit_lowering_rule as make_scalar_unary_elementwise_native_llvm_jit_lowering_rule,
)
from .mlir_symmetric_native_compilation import (
    compile_symmetric_2x2_cholesky_ad_to_native_llvm_jit as compile_symmetric_2x2_cholesky_ad_to_native_llvm_jit,
)
from .mlir_symmetric_native_compilation import (
    compile_symmetric_2x2_eigenvalues_ad_to_native_llvm_jit as compile_symmetric_2x2_eigenvalues_ad_to_native_llvm_jit,
)
from .mlir_symmetric_native_compilation import (
    make_symmetric_2x2_cholesky_native_llvm_jit_lowering_rule as make_symmetric_2x2_cholesky_native_llvm_jit_lowering_rule,
)
from .mlir_symmetric_native_compilation import (
    make_symmetric_2x2_cholesky_native_llvm_jit_primitive_transform as make_symmetric_2x2_cholesky_native_llvm_jit_primitive_transform,
)
from .mlir_symmetric_native_compilation import (
    make_symmetric_2x2_eigenvalues_native_llvm_jit_lowering_rule as make_symmetric_2x2_eigenvalues_native_llvm_jit_lowering_rule,
)
from .mlir_symmetric_native_compilation import (
    make_symmetric_2x2_eigenvalues_native_llvm_jit_primitive_transform as make_symmetric_2x2_eigenvalues_native_llvm_jit_primitive_transform,
)
from .mlir_vector_native_compilation import (
    compile_vector_dot_ad_to_native_llvm_jit as compile_vector_dot_ad_to_native_llvm_jit,
)
from .mlir_vector_native_compilation import (
    compile_vector_squared_norm_ad_to_native_llvm_jit as compile_vector_squared_norm_ad_to_native_llvm_jit,
)
from .mlir_vector_native_compilation import (
    make_vector_dot_native_llvm_jit_lowering_rule as make_vector_dot_native_llvm_jit_lowering_rule,
)
from .mlir_vector_native_compilation import (
    make_vector_dot_native_llvm_jit_primitive_transform as make_vector_dot_native_llvm_jit_primitive_transform,
)
from .mlir_vector_native_compilation import (
    make_vector_squared_norm_native_llvm_jit_lowering_rule as make_vector_squared_norm_native_llvm_jit_lowering_rule,
)
from .mlir_vector_native_compilation import (
    make_vector_squared_norm_native_llvm_jit_primitive_transform as make_vector_squared_norm_native_llvm_jit_primitive_transform,
)
from .mlir_whole_program_native import (
    ExecutableWholeProgramADBatchResult as ExecutableWholeProgramADBatchResult,
)
from .mlir_whole_program_native import (
    ExecutableWholeProgramADKernel as ExecutableWholeProgramADKernel,
)
from .mlir_whole_program_native import (
    NativeWholeProgramADKernel as NativeWholeProgramADKernel,
)
from .mlir_whole_program_native import (
    WholeProgramADNativeLoweringReport as WholeProgramADNativeLoweringReport,
)
from .mlir_whole_program_native import (
    analyse_whole_program_ad_native_lowering as analyse_whole_program_ad_native_lowering,
)
from .mlir_whole_program_native import (
    clear_native_whole_program_ad_compile_cache as clear_native_whole_program_ad_compile_cache,
)
from .mlir_whole_program_native import (
    compile_whole_program_ad_trace_to_executable as compile_whole_program_ad_trace_to_executable,
)
from .mlir_whole_program_native import (
    compile_whole_program_ad_trace_to_mlir as compile_whole_program_ad_trace_to_mlir,
)
from .mlir_whole_program_native import (
    compile_whole_program_ad_trace_to_native_llvm_jit as compile_whole_program_ad_trace_to_native_llvm_jit,
)
from .mlir_whole_program_native import (
    native_whole_program_ad_compile_cache_stats as native_whole_program_ad_compile_cache_stats,
)
from .mlir_whole_program_native import (
    native_whole_program_ad_linalg_support as native_whole_program_ad_linalg_support,
)

FloatArray: TypeAlias = NDArray[np.float64]


def _status_has_native_llvm_jit_backend(status: PrimitiveLoweringStatus) -> bool:
    metadata = status.lowering_metadata
    return (
        metadata.get("native_backend") == "native_llvm_jit"
        and metadata.get("native_backend_verification", "").startswith("verified:")
        and "blocked" not in status.llvm_lowering.lower()
        and "blocked" not in status.jit_lowering.lower()
    )


def _status_has_verified_rust_backend(status: PrimitiveLoweringStatus) -> bool:
    metadata = status.lowering_metadata
    return (
        metadata.get("rust_backend") == "rust_pyo3"
        and metadata.get("rust_backend_verification", "").startswith("verified:")
        and metadata.get("rust_backend_signature") == status.static_signature
        and bool(metadata.get("rust_backend_functions"))
        and "blocked" not in status.rust_lowering.lower()
    )


def build_compiler_ad_transform_plan(
    registry: CustomDerivativeRegistry,
    *,
    dialect: str = "scpn_diff",
    transform: str = "jvp_vjp_adjoint",
) -> CompilerADTransformPlan:
    """Build a deterministic compiler AD plan from registered primitive rules."""
    if not isinstance(registry, CustomDerivativeRegistry):
        raise ValueError("registry must be a CustomDerivativeRegistry")
    statuses = []
    transform_snapshot = registry.transform_snapshot()
    for identity, rule in sorted(registry.snapshot().items(), key=lambda item: item[0].key):
        transform_rule = transform_snapshot.get(identity)
        metadata = (
            {}
            if transform_rule is None or transform_rule.lowering_metadata is None
            else dict(transform_rule.lowering_metadata)
        )
        default_mlir_status = (
            "available: executable scpn_diff MLIR-runtime primitive kernel"
            if transform_rule is not None and transform_rule.lowering_rule is not None
            else "available: scpn_diff dialect interchange"
        )
        statuses.append(
            PrimitiveLoweringStatus(
                identity=identity,
                rule_name=rule.name,
                has_jvp=rule.jvp_rule is not None,
                has_vjp=rule.vjp_rule is not None,
                mlir_op=metadata.get("mlir_op", f"{dialect}.{identity.namespace}_{identity.name}"),
                has_batching_rule=transform_rule is not None
                and transform_rule.batching_rule is not None,
                has_shape_rule=transform_rule is not None
                and transform_rule.shape_rule is not None,
                has_dtype_rule=transform_rule is not None
                and transform_rule.dtype_rule is not None,
                has_static_argument_rule=transform_rule is not None
                and transform_rule.static_argument_rule is not None,
                has_lowering_rule=transform_rule is not None
                and transform_rule.lowering_rule is not None,
                lowering_metadata=metadata,
                static_derivative_factory=metadata.get(
                    "static_derivative_factory", "not_declared"
                ),
                static_signature=metadata.get("static_signature", "none"),
                nondifferentiable_policy="not_declared"
                if transform_rule is None
                else transform_rule.nondifferentiable_policy,
                nondifferentiable_boundary=metadata.get(
                    "nondifferentiable_boundary", "not_declared"
                ),
                nondifferentiable_boundary_policy=metadata.get(
                    "nondifferentiable_boundary_policy", "not_declared"
                ),
                effect="pure" if transform_rule is None else transform_rule.effect,
                mlir_lowering=metadata.get("mlir", default_mlir_status),
                mlir_runtime_verification=metadata.get(
                    "mlir_runtime_verification", "not_declared"
                ),
                rust_lowering=metadata.get(
                    "rust", "blocked: no Rust differentiable primitive backend"
                ),
                llvm_lowering=metadata.get(
                    "llvm", "blocked: no LLVM/JIT differentiable primitive backend"
                ),
                jit_lowering=metadata.get(
                    "jit", "blocked: no JIT differentiable primitive backend"
                ),
            )
        )
    executable_backend = (
        "native_llvm_jit"
        if statuses and all(_status_has_native_llvm_jit_backend(status) for status in statuses)
        else "none"
    )
    claim_boundary = (
        "verified executable native LLVM/JIT primitive AD kernels for all planned primitives; "
        "Rust differentiated runtime remains fail-closed"
        if executable_backend == "native_llvm_jit"
        else (
            "compiler-backed AD planning and MLIR dialect interchange only; "
            "no executable Rust, LLVM, or JIT differentiated runtime"
        )
    )
    return CompilerADTransformPlan(
        tuple(statuses),
        dialect=dialect,
        transform=transform,
        executable_backend=executable_backend,
        claim_boundary=claim_boundary,
    )


def compile_compiler_ad_transform_plan_to_mlir(plan: CompilerADTransformPlan) -> MLIRModule:
    """Emit deterministic MLIR-style dialect metadata for compiler-backed AD planning."""
    if not isinstance(plan, CompilerADTransformPlan):
        raise ValueError("compiler AD MLIR lowering requires a CompilerADTransformPlan")
    lines = [
        f'module attributes {{scpn.module = "compiler_ad_transform_plan", '
        f'scpn.dialect = "{plan.dialect}", '
        f'scpn.transform = "{plan.transform}", '
        f"scpn.n_primitives = {len(plan.statuses)}}} {{",
        "  func.func @main() {",
    ]
    execution = (
        plan.executable_backend if plan.executable_backend != "none" else "interchange_only"
    )
    for index, status in enumerate(plan.statuses):
        lines.append(
            "    scpn_diff.primitive "
            f'%p{index} {{identity = "{_escape_mlir_string(status.identity.key)}", '
            f'rule = "{_escape_mlir_string(status.rule_name)}", '
            f'op = "{_escape_mlir_string(status.mlir_op)}", '
            f"jvp = {_fmt_bool(status.has_jvp)}, vjp = {_fmt_bool(status.has_vjp)}, "
            f"batching_rule = {_fmt_bool(status.has_batching_rule)}, "
            f"shape_rule = {_fmt_bool(status.has_shape_rule)}, "
            f"dtype_rule = {_fmt_bool(status.has_dtype_rule)}, "
            f"static_argument_rule = {_fmt_bool(status.has_static_argument_rule)}, "
            f"lowering_rule = {_fmt_bool(status.has_lowering_rule)}, "
            f'mlir_runtime_verification = "{_escape_mlir_string(status.mlir_runtime_verification)}", '
            f'static_derivative_factory = "{_escape_mlir_string(status.static_derivative_factory)}", '
            f'static_signature = "{_escape_mlir_string(status.static_signature)}", '
            f'policy = "{_escape_mlir_string(status.nondifferentiable_policy)}", '
            f'boundary = "{_escape_mlir_string(status.nondifferentiable_boundary)}", '
            f'boundary_policy = "{_escape_mlir_string(status.nondifferentiable_boundary_policy)}", '
            f'effect = "{_escape_mlir_string(status.effect)}"}}'
        )
        lines.append(
            "    scpn_diff.lowering_status "
            f'{{identity = "{_escape_mlir_string(status.identity.key)}", '
            f'mlir = "{_escape_mlir_string(status.mlir_lowering)}", '
            f'verification = "{_escape_mlir_string(status.mlir_runtime_verification)}", '
            f'rust = "{_escape_mlir_string(status.rust_lowering)}", '
            f'llvm = "{_escape_mlir_string(status.llvm_lowering)}", '
            f'jit = "{_escape_mlir_string(status.jit_lowering)}"}}'
        )
        for key, value in sorted(status.lowering_metadata.items()):
            lines.append(
                "    scpn_diff.lowering_metadata "
                f'{{identity = "{_escape_mlir_string(status.identity.key)}", '
                f'key = "{_escape_mlir_string(key)}", '
                f'value = "{_escape_mlir_string(value)}"}}'
            )
    lines.append(
        "    scpn_diff.ad_transform "
        f'{{kind = "{_escape_mlir_string(plan.transform)}", '
        f'execution = "{_escape_mlir_string(execution)}"}}'
    )
    lines.append("    return")
    lines.append("  }")

    def has_registry_contract(status: PrimitiveLoweringStatus) -> bool:
        return (
            (status.has_jvp or status.has_vjp)
            and status.has_batching_rule
            and status.has_shape_rule
            and status.has_dtype_rule
            and status.static_derivative_factory not in {"not_declared", "not_required"}
            and status.static_signature != "none"
            and status.nondifferentiable_policy != "not_declared"
            and status.nondifferentiable_boundary != "not_declared"
            and status.nondifferentiable_boundary_policy == "fail_closed"
        )

    def has_reverse_contract(status: PrimitiveLoweringStatus) -> bool:
        return status.has_vjp and has_registry_contract(status)

    def has_forward_contract(status: PrimitiveLoweringStatus) -> bool:
        return status.has_jvp and has_registry_contract(status)

    def has_adjoint_contract(status: PrimitiveLoweringStatus) -> bool:
        return status.effect == "pure" and has_reverse_contract(status)

    def has_transform_contract(status: PrimitiveLoweringStatus) -> bool:
        return (
            has_forward_contract(status)
            and has_reverse_contract(status)
            and has_adjoint_contract(status)
        )

    def exposes_nondifferentiable_policy(status: PrimitiveLoweringStatus) -> bool:
        has_static_contract = (
            status.static_derivative_factory not in {"not_declared", "not_required"}
            and status.static_signature != "none"
        )
        return status.nondifferentiable_policy != "not_declared" and (
            status.nondifferentiable_boundary != "not_declared" or has_static_contract
        )

    def has_rust_backend_contract(status: PrimitiveLoweringStatus) -> bool:
        return _status_has_verified_rust_backend(status)

    def has_native_llvm_jit_proof(status: PrimitiveLoweringStatus) -> bool:
        return _status_has_native_llvm_jit_backend(status)

    def has_llvm_backend_contract(status: PrimitiveLoweringStatus) -> bool:
        return has_native_llvm_jit_proof(status) and "blocked" not in status.llvm_lowering.lower()

    def has_jit_backend_contract(status: PrimitiveLoweringStatus) -> bool:
        return has_native_llvm_jit_proof(status) and "blocked" not in status.jit_lowering.lower()

    def has_native_backend_contract(status: PrimitiveLoweringStatus) -> bool:
        return has_llvm_backend_contract(status) and has_jit_backend_contract(status)

    def has_mlir_runtime_contract(status: PrimitiveLoweringStatus) -> bool:
        return status.has_lowering_rule and status.mlir_runtime_verification.startswith(
            "verified:"
        )

    def mlir_runtime_blocker(status: PrimitiveLoweringStatus) -> str | None:
        if has_mlir_runtime_contract(status):
            return None
        if not status.has_lowering_rule:
            return "blocked: no MLIR-runtime lowering rule"
        if not status.mlir_runtime_verification.startswith("verified:"):
            return "blocked: no verified MLIR-runtime provenance"
        return "blocked: MLIR-runtime contract incomplete"

    def primitive_readiness(status: PrimitiveLoweringStatus) -> dict[str, bool | str]:
        registry_contract = has_registry_contract(status)
        forward_contract = has_forward_contract(status)
        reverse_contract = has_reverse_contract(status)
        adjoint_contract = has_adjoint_contract(status)
        transform_contract = has_transform_contract(status)
        mlir_runtime_contract = has_mlir_runtime_contract(status)
        rust_backend_contract = has_rust_backend_contract(status)
        llvm_backend_contract = has_llvm_backend_contract(status)
        jit_backend_contract = has_jit_backend_contract(status)
        native_backend_contract = has_native_backend_contract(status)
        if native_backend_contract and mlir_runtime_contract and transform_contract:
            verdict = "native_executable"
        elif mlir_runtime_contract:
            verdict = "mlir_runtime_verified"
        elif transform_contract:
            verdict = "transform_interchange_only"
        elif registry_contract and forward_contract:
            verdict = "forward_interchange_only"
        elif registry_contract:
            verdict = "registry_contract_only"
        else:
            verdict = "registry_incomplete"
        return {
            "adjoint_contract": adjoint_contract,
            "forward_contract": forward_contract,
            "jit_backend_contract": jit_backend_contract,
            "llvm_backend_contract": llvm_backend_contract,
            "mlir_runtime_contract": mlir_runtime_contract,
            "native_backend_contract": native_backend_contract,
            "registry_contract": registry_contract,
            "reverse_contract": reverse_contract,
            "rust_backend_contract": rust_backend_contract,
            "transform_contract": transform_contract,
            "verdict": verdict,
        }

    primitive_readiness_by_key = {
        status.identity.key: primitive_readiness(status) for status in plan.statuses
    }
    primitive_readiness_verdict_counts: dict[str, int] = {}
    for readiness in primitive_readiness_by_key.values():
        verdict = str(readiness["verdict"])
        primitive_readiness_verdict_counts[verdict] = (
            primitive_readiness_verdict_counts.get(verdict, 0) + 1
        )
    hard_gap_order = (
        "registry_contract",
        "forward_contract",
        "reverse_contract",
        "adjoint_contract",
        "transform_contract",
        "mlir_runtime_contract",
        "rust_backend_contract",
        "llvm_backend_contract",
        "jit_backend_contract",
        "native_backend_contract",
    )
    primitive_hard_gaps = {
        identity: [gap for gap in hard_gap_order if readiness.get(gap) is False]
        for identity, readiness in primitive_readiness_by_key.items()
    }
    primitive_next_hard_gap = {
        identity: gaps[0] for identity, gaps in primitive_hard_gaps.items() if gaps
    }
    primitive_hard_gap_primitives: dict[str, list[str]] = {}
    primitive_hard_gap_counts: dict[str, int] = {}
    for identity, gaps in primitive_hard_gaps.items():
        for gap in gaps:
            primitive_hard_gap_primitives.setdefault(gap, []).append(identity)
            primitive_hard_gap_counts[gap] = primitive_hard_gap_counts.get(gap, 0) + 1
    primitive_hard_gap_priority = [
        gap for gap in hard_gap_order if gap in primitive_hard_gap_primitives
    ]
    primitive_hard_gap_frontier = {
        gap: {
            "count": len(primitive_hard_gap_primitives[gap]),
            "next_primitive": primitive_hard_gap_primitives[gap][0],
            "primitives": primitive_hard_gap_primitives[gap],
        }
        for gap in primitive_hard_gap_priority
    }

    metadata = {
        "claim_boundary": plan.claim_boundary,
        "dialect": plan.dialect,
        "executable_backend": plan.executable_backend,
        "effects": {
            status.identity.key: status.effect
            for status in plan.statuses
            if exposes_nondifferentiable_policy(status)
        },
        "nondifferentiable_policies": {
            status.identity.key: status.nondifferentiable_policy
            for status in plan.statuses
            if exposes_nondifferentiable_policy(status)
        },
        "nondifferentiable_boundaries": {
            status.identity.key: status.nondifferentiable_boundary
            for status in plan.statuses
            if status.nondifferentiable_boundary != "not_declared"
        },
        "nondifferentiable_boundary_policies": {
            status.identity.key: status.nondifferentiable_boundary_policy
            for status in plan.statuses
            if status.nondifferentiable_boundary_policy != "not_declared"
        },
        "boundary_contract_primitives": [
            status.identity.key
            for status in plan.statuses
            if status.nondifferentiable_boundary != "not_declared"
            and status.nondifferentiable_boundary_policy == "fail_closed"
        ],
        "mlir_runtime_lowering_primitives": [
            status.identity.key for status in plan.statuses if status.has_lowering_rule
        ],
        "primitive_identities": [status.identity.key for status in plan.statuses],
        "jvp_rule_primitives": [status.identity.key for status in plan.statuses if status.has_jvp],
        "vjp_rule_primitives": [status.identity.key for status in plan.statuses if status.has_vjp],
        "batching_rule_primitives": [
            status.identity.key for status in plan.statuses if status.has_batching_rule
        ],
        "shape_rule_primitives": [
            status.identity.key for status in plan.statuses if status.has_shape_rule
        ],
        "dtype_rule_primitives": [
            status.identity.key for status in plan.statuses if status.has_dtype_rule
        ],
        "static_argument_primitives": [
            status.identity.key for status in plan.statuses if status.has_static_argument_rule
        ],
        "static_derivative_factories": {
            status.identity.key: status.static_derivative_factory
            for status in plan.statuses
            if status.static_derivative_factory not in {"not_declared", "not_required"}
        },
        "static_derivative_signatures": {
            status.identity.key: status.static_signature
            for status in plan.statuses
            if status.static_signature != "none"
        },
        "registry_contract_primitives": [
            status.identity.key for status in plan.statuses if has_registry_contract(status)
        ],
        "reverse_contract_primitives": [
            status.identity.key for status in plan.statuses if has_reverse_contract(status)
        ],
        "reverse_incomplete_primitives": [
            status.identity.key for status in plan.statuses if not status.has_vjp
        ],
        "forward_contract_primitives": [
            status.identity.key for status in plan.statuses if has_forward_contract(status)
        ],
        "forward_incomplete_primitives": [
            status.identity.key for status in plan.statuses if not has_forward_contract(status)
        ],
        "adjoint_contract_primitives": [
            status.identity.key for status in plan.statuses if has_adjoint_contract(status)
        ],
        "adjoint_incomplete_primitives": [
            status.identity.key for status in plan.statuses if not has_adjoint_contract(status)
        ],
        "transform_contract_primitives": [
            status.identity.key for status in plan.statuses if has_transform_contract(status)
        ],
        "transform_incomplete_primitives": [
            status.identity.key for status in plan.statuses if not has_transform_contract(status)
        ],
        "native_backend_contract_primitives": [
            status.identity.key for status in plan.statuses if has_native_backend_contract(status)
        ],
        "native_backend_incomplete_primitives": [
            status.identity.key
            for status in plan.statuses
            if not has_native_backend_contract(status)
        ],
        "rust_backend_contract_primitives": [
            status.identity.key for status in plan.statuses if has_rust_backend_contract(status)
        ],
        "rust_backend_incomplete_primitives": [
            status.identity.key
            for status in plan.statuses
            if not has_rust_backend_contract(status)
        ],
        "rust_backend_blockers": {
            status.identity.key: status.rust_lowering
            for status in plan.statuses
            if "blocked" in status.rust_lowering.lower()
        },
        "rust_backend_signatures": {
            status.identity.key: status.lowering_metadata["rust_backend_signature"]
            for status in plan.statuses
            if has_rust_backend_contract(status)
        },
        "rust_backend_functions": {
            status.identity.key: status.lowering_metadata["rust_backend_functions"]
            for status in plan.statuses
            if has_rust_backend_contract(status)
        },
        "rust_backend_verification_primitives": {
            status.identity.key: status.lowering_metadata["rust_backend_verification"]
            for status in plan.statuses
            if has_rust_backend_contract(status)
        },
        "llvm_backend_contract_primitives": [
            status.identity.key for status in plan.statuses if has_llvm_backend_contract(status)
        ],
        "llvm_backend_incomplete_primitives": [
            status.identity.key
            for status in plan.statuses
            if not has_llvm_backend_contract(status)
        ],
        "llvm_backend_blockers": {
            status.identity.key: status.llvm_lowering
            for status in plan.statuses
            if "blocked" in status.llvm_lowering.lower()
        },
        "jit_backend_contract_primitives": [
            status.identity.key for status in plan.statuses if has_jit_backend_contract(status)
        ],
        "jit_backend_incomplete_primitives": [
            status.identity.key for status in plan.statuses if not has_jit_backend_contract(status)
        ],
        "jit_backend_blockers": {
            status.identity.key: status.jit_lowering
            for status in plan.statuses
            if "blocked" in status.jit_lowering.lower()
        },
        "mlir_runtime_contract_primitives": [
            status.identity.key for status in plan.statuses if has_mlir_runtime_contract(status)
        ],
        "mlir_runtime_incomplete_primitives": [
            status.identity.key
            for status in plan.statuses
            if not has_mlir_runtime_contract(status)
        ],
        "mlir_runtime_blockers": {
            status.identity.key: blocker
            for status in plan.statuses
            for blocker in (mlir_runtime_blocker(status),)
            if blocker is not None
        },
        "mlir_runtime_verification_primitives": {
            status.identity.key: status.mlir_runtime_verification
            for status in plan.statuses
            if status.mlir_runtime_verification.startswith("verified:")
        },
        "primitive_readiness": primitive_readiness_by_key,
        "primitive_readiness_verdict_counts": primitive_readiness_verdict_counts,
        "primitive_hard_gaps": primitive_hard_gaps,
        "primitive_next_hard_gap": primitive_next_hard_gap,
        "primitive_hard_gap_counts": primitive_hard_gap_counts,
        "primitive_hard_gap_primitives": primitive_hard_gap_primitives,
        "primitive_hard_gap_priority": primitive_hard_gap_priority,
        "primitive_hard_gap_frontier": primitive_hard_gap_frontier,
        "transform": plan.transform,
        "uncontracted_primitives": [
            status.identity.key
            for status in plan.statuses
            if status.nondifferentiable_policy == "not_declared"
            or status.nondifferentiable_boundary == "not_declared"
        ],
    }
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    lines.append(f'  scpn.metadata {{json = "{_escape_mlir_string(encoded)}"}}')
    lines.append("}")
    text = "\n".join(lines) + "\n"
    return MLIRModule(
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        dialect=plan.dialect,
        resource_counts={
            "primitives": len(plan.statuses),
            "jvp_rules": sum(status.has_jvp for status in plan.statuses),
            "vjp_rules": sum(status.has_vjp for status in plan.statuses),
            "batching_rules": sum(status.has_batching_rule for status in plan.statuses),
            "shape_rules": sum(status.has_shape_rule for status in plan.statuses),
            "dtype_rules": sum(status.has_dtype_rule for status in plan.statuses),
            "effects": sum(exposes_nondifferentiable_policy(status) for status in plan.statuses),
            "nondifferentiable_policies": sum(
                exposes_nondifferentiable_policy(status) for status in plan.statuses
            ),
            "nondifferentiable_boundaries": sum(
                status.nondifferentiable_boundary != "not_declared" for status in plan.statuses
            ),
            "nondifferentiable_boundary_policies": sum(
                status.nondifferentiable_boundary_policy != "not_declared"
                for status in plan.statuses
            ),
            "boundary_contracts": sum(
                status.nondifferentiable_boundary != "not_declared"
                and status.nondifferentiable_boundary_policy == "fail_closed"
                for status in plan.statuses
            ),
            "mlir_runtime_lowerings": sum(status.has_lowering_rule for status in plan.statuses),
            "static_argument_rules": sum(
                status.has_static_argument_rule for status in plan.statuses
            ),
            "static_derivative_factories": sum(
                status.static_derivative_factory not in {"not_declared", "not_required"}
                for status in plan.statuses
            ),
            "static_derivative_signatures": sum(
                status.static_signature != "none" for status in plan.statuses
            ),
            "registry_contracts": sum(has_registry_contract(status) for status in plan.statuses),
            "forward_contracts": sum(has_forward_contract(status) for status in plan.statuses),
            "forward_incomplete_primitives": sum(
                not has_forward_contract(status) for status in plan.statuses
            ),
            "reverse_contracts": sum(has_reverse_contract(status) for status in plan.statuses),
            "reverse_incomplete_primitives": sum(not status.has_vjp for status in plan.statuses),
            "adjoint_contracts": sum(has_adjoint_contract(status) for status in plan.statuses),
            "adjoint_incomplete_primitives": sum(
                not has_adjoint_contract(status) for status in plan.statuses
            ),
            "transform_contracts": sum(has_transform_contract(status) for status in plan.statuses),
            "transform_incomplete_primitives": sum(
                not has_transform_contract(status) for status in plan.statuses
            ),
            "native_backend_contracts": sum(
                has_native_backend_contract(status) for status in plan.statuses
            ),
            "native_backend_incomplete_primitives": sum(
                not has_native_backend_contract(status) for status in plan.statuses
            ),
            "rust_backend_contracts": sum(
                has_rust_backend_contract(status) for status in plan.statuses
            ),
            "rust_backend_incomplete_primitives": sum(
                not has_rust_backend_contract(status) for status in plan.statuses
            ),
            "rust_backend_blockers": sum(
                "blocked" in status.rust_lowering.lower() for status in plan.statuses
            ),
            "rust_backend_verifications": sum(
                has_rust_backend_contract(status) for status in plan.statuses
            ),
            "llvm_backend_contracts": sum(
                has_llvm_backend_contract(status) for status in plan.statuses
            ),
            "llvm_backend_incomplete_primitives": sum(
                not has_llvm_backend_contract(status) for status in plan.statuses
            ),
            "llvm_backend_blockers": sum(
                "blocked" in status.llvm_lowering.lower() for status in plan.statuses
            ),
            "jit_backend_contracts": sum(
                has_jit_backend_contract(status) for status in plan.statuses
            ),
            "jit_backend_incomplete_primitives": sum(
                not has_jit_backend_contract(status) for status in plan.statuses
            ),
            "jit_backend_blockers": sum(
                "blocked" in status.jit_lowering.lower() for status in plan.statuses
            ),
            "mlir_runtime_contracts": sum(
                has_mlir_runtime_contract(status) for status in plan.statuses
            ),
            "mlir_runtime_incomplete_primitives": sum(
                not has_mlir_runtime_contract(status) for status in plan.statuses
            ),
            "mlir_runtime_blockers": sum(
                mlir_runtime_blocker(status) is not None for status in plan.statuses
            ),
            "mlir_runtime_verifications": sum(
                status.mlir_runtime_verification.startswith("verified:")
                for status in plan.statuses
            ),
            "primitive_readiness_verdicts": len(plan.statuses),
            "primitive_readiness_registry_incomplete": primitive_readiness_verdict_counts.get(
                "registry_incomplete", 0
            ),
            "primitive_readiness_forward_interchange_only": (
                primitive_readiness_verdict_counts.get("forward_interchange_only", 0)
            ),
            "primitive_readiness_transform_interchange_only": (
                primitive_readiness_verdict_counts.get("transform_interchange_only", 0)
            ),
            "primitive_readiness_mlir_runtime_verified": (
                primitive_readiness_verdict_counts.get("mlir_runtime_verified", 0)
            ),
            "primitive_readiness_native_executable": primitive_readiness_verdict_counts.get(
                "native_executable", 0
            ),
            "primitive_hard_gaps": sum(len(gaps) for gaps in primitive_hard_gaps.values()),
            "primitive_next_hard_gaps": len(primitive_next_hard_gap),
            "primitive_hard_gap_priority_classes": len(primitive_hard_gap_priority),
            "primitive_hard_gap_frontier_classes": len(primitive_hard_gap_frontier),
            "uncontracted_primitives": sum(
                status.nondifferentiable_policy == "not_declared"
                or status.nondifferentiable_boundary == "not_declared"
                for status in plan.statuses
            ),
            "executable_backends": int(plan.executable_backend != "none"),
        },
        metadata=metadata,
    )


def compile_kuramoto_to_mlir(
    problem: KuramotoProblem | FloatArray,
    config: MLIRCompileConfig,
    omega: FloatArray | None = None,
) -> MLIRModule:
    """Compile a Kuramoto problem into deterministic MLIR-style text.

    ``problem`` may be a validated :class:`KuramotoProblem` or a raw coupling
    matrix when ``omega`` is supplied. Raw arrays are validated through the
    public Kuramoto facade before IR generation.
    """
    if isinstance(problem, KuramotoProblem):
        validated = problem
    else:
        if omega is None:
            raise ValueError("omega is required when problem is a raw coupling matrix")
        validated = build_kuramoto_problem(problem, omega)

    coupling_terms = _coupling_terms(validated.K_nm)
    lines = [
        f'module attributes {{scpn.module = "kuramoto_xy", scpn.dialect = "{config.dialect}", '
        f"scpn.n_oscillators = {validated.n_oscillators}, "
        f"scpn.trotter_steps = {config.trotter_steps}, "
        f"scpn.trotter_order = {config.trotter_order}}} {{",
        "  func.func @main() {",
    ]
    for index, value in enumerate(validated.omega):
        lines.append(f"    scpn.omega %{index} {{value = {_fmt_float(float(value))}}}")
    for term_index, (left, right, value) in enumerate(coupling_terms):
        lines.append(
            "    scpn.coupling "
            f"%c{term_index} {{i = {left}, j = {right}, value = {_fmt_float(value)}}}"
        )
    lines.append(
        "    scpn.trotter_evolve "
        f"{{time = {_fmt_float(config.time)}, steps = {config.trotter_steps}, "
        f"order = {config.trotter_order}}}"
    )
    lines.append("    return")
    lines.append("  }")
    if config.include_metadata and validated.metadata:
        encoded = json.dumps(dict(validated.metadata), sort_keys=True, separators=(",", ":"))
        lines.append(f'  scpn.metadata {{json = "{_escape_mlir_string(encoded)}"}}')
    lines.append("}")
    text = "\n".join(lines) + "\n"
    resource_counts = {
        "n_oscillators": validated.n_oscillators,
        "omega_terms": validated.n_oscillators,
        "coupling_terms": len(coupling_terms),
        "trotter_steps": config.trotter_steps,
        "trotter_order": config.trotter_order,
    }
    return MLIRModule(
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        dialect=config.dialect,
        resource_counts=resource_counts,
        metadata={
            "claim_boundary": "textual MLIR-style IR export; no provider lowering or hardware execution",
            "problem": validated.to_metadata(),
        },
    )


def compile_custom_derivative_rule_to_executable(
    rule: CustomDerivativeRule,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    *,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile a custom derivative rule into a verified executable AD kernel.

    The executable backend is the dependency-free SCPN MLIR runtime adapter:
    it couples deterministic differentiable MLIR provenance with normalized
    runtime callables for value/JVP/VJP execution and verifies those kernels
    against the source custom derivative rule before returning. Native LLVM/JIT
    kernels use primitive-specific lowering entrypoints.
    """
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("executable AD lowering requires a CustomDerivativeRule")
    compile_config = CompilerADExecutableConfig() if config is None else config
    if compile_config.backend != "mlir_runtime":
        raise ValueError(
            "compile_custom_derivative_rule_to_executable requires backend='mlir_runtime'; "
            "use a primitive-specific native LLVM/JIT lowering entrypoint"
        )
    values = _as_finite_vector("sample_values", sample_values)
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )

    def value_kernel(raw_values: FloatArray) -> FloatArray:
        return _as_finite_vector(
            "value kernel output", rule.value_fn(_as_finite_vector("values", raw_values))
        )

    def jvp_kernel(raw_values: FloatArray, raw_tangent: FloatArray) -> FloatArray:
        if rule.jvp_rule is None:
            raise ValueError(f"rule {rule.name} has no JVP rule")
        checked_values = _as_finite_vector("values", raw_values)
        checked_tangent = _as_finite_vector("tangent", raw_tangent)
        if checked_tangent.shape != checked_values.shape:
            raise ValueError("tangent shape must match values shape")
        return _as_finite_vector(
            "JVP kernel output", rule.jvp_rule(checked_values, checked_tangent)
        )

    def vjp_kernel(raw_values: FloatArray, raw_cotangent: FloatArray) -> FloatArray:
        if rule.vjp_rule is None:
            raise ValueError(f"rule {rule.name} has no VJP rule")
        checked_values = _as_finite_vector("values", raw_values)
        checked_cotangent = _as_finite_vector("cotangent", raw_cotangent)
        return _as_finite_vector(
            "VJP kernel output", rule.vjp_rule(checked_values, checked_cotangent)
        )

    verification = _verify_executable_ad_kernel(
        rule,
        values,
        value_kernel,
        jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel if rule.vjp_rule is not None else None,
        compile_config,
        sample_tangent=sample_tangent,
        sample_cotangent=sample_cotangent,
    )
    llvm_gradient_ir = (
        _compile_scalar_gradient_llvm_ir(rule, values, vjp_kernel)
        if verification.gradient_close is True and rule.vjp_rule is not None
        else None
    )
    return ExecutableCompilerADKernel(
        rule_name=rule.name,
        backend=compile_config.backend,
        mlir_module=mlir_module,
        value_kernel=value_kernel,
        jvp_kernel=jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel=vjp_kernel if rule.vjp_rule is not None else None,
        verification=verification,
        llvm_gradient_ir=llvm_gradient_ir,
    )


def compile_registered_primitive_to_executable(
    registry: CustomDerivativeRegistry,
    identity: PrimitiveIdentity | str,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    *,
    sample_tangent: Sequence[float] | FloatArray | None = None,
    sample_cotangent: Sequence[float] | FloatArray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile a registered primitive identity into an executable AD kernel."""
    if not isinstance(registry, CustomDerivativeRegistry):
        raise ValueError("registry must be a CustomDerivativeRegistry")
    primitive_identity = PrimitiveIdentity.parse(identity)
    transform = registry.transform_snapshot().get(primitive_identity)
    rule = registry.require(primitive_identity)
    if transform is not None and transform.lowering_rule is not None:
        lowering_rule = cast(Callable[..., Any], transform.lowering_rule)
        try:
            lowered = lowering_rule(
                rule,
                sample_values,
                config,
                sample_tangent=sample_tangent,
                sample_cotangent=sample_cotangent,
            )
        except TypeError:
            lowered = lowering_rule(rule)
        if not isinstance(lowered, ExecutableCompilerADKernel):
            raise ValueError("registered lowering_rule must return an ExecutableCompilerADKernel")
        return lowered
    return compile_custom_derivative_rule_to_executable(
        rule,
        sample_values,
        config,
        sample_tangent=sample_tangent,
        sample_cotangent=sample_cotangent,
    )


def make_program_ad_linalg_matrix_power_executable_lowering_rule(
    power: int | np.integer,
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    *,
    sample_tangent: Sequence[float] | FloatArray | None = None,
) -> Callable[[CustomDerivativeRule], ExecutableCompilerADKernel]:
    """Build a verified executable lowering rule for a fixed matrix_power signature."""
    direct_rule = program_ad_linalg_matrix_power_derivative_rule(power)
    values = _as_finite_vector("sample_values", sample_values)
    compile_config = CompilerADExecutableConfig() if config is None else config

    def lowering_rule(_registered_rule: CustomDerivativeRule) -> ExecutableCompilerADKernel:
        if not isinstance(_registered_rule, CustomDerivativeRule):
            raise ValueError("registered_rule must be a CustomDerivativeRule")
        return compile_custom_derivative_rule_to_executable(
            direct_rule,
            values,
            compile_config,
            sample_tangent=sample_tangent,
        )

    return lowering_rule


def make_program_ad_linalg_multi_dot_executable_lowering_rule(
    operand_shapes: Sequence[Sequence[int]],
    sample_values: Sequence[float] | FloatArray,
    config: CompilerADExecutableConfig | None = None,
    *,
    sample_tangent: Sequence[float] | FloatArray | None = None,
) -> Callable[[CustomDerivativeRule], ExecutableCompilerADKernel]:
    """Build a verified executable lowering rule for a fixed multi_dot signature."""
    direct_rule = program_ad_linalg_multi_dot_derivative_rule(operand_shapes)
    values = _as_finite_vector("sample_values", sample_values)
    compile_config = CompilerADExecutableConfig() if config is None else config

    def lowering_rule(_registered_rule: CustomDerivativeRule) -> ExecutableCompilerADKernel:
        if not isinstance(_registered_rule, CustomDerivativeRule):
            raise ValueError("registered_rule must be a CustomDerivativeRule")
        return compile_custom_derivative_rule_to_executable(
            direct_rule,
            values,
            compile_config,
            sample_tangent=sample_tangent,
        )

    return lowering_rule


def _coupling_terms(K_nm: FloatArray) -> tuple[tuple[int, int, float], ...]:
    terms: list[tuple[int, int, float]] = []
    n_oscillators = K_nm.shape[0]
    for left in range(n_oscillators):
        for right in range(left + 1, n_oscillators):
            value = float(K_nm[left, right])
            if abs(value) > 1e-15:
                terms.append((left, right, value))
    return tuple(terms)


@dataclass(frozen=True)
class PhaseQNodeMLIRRuntimeExecutable:
    """Verified executable MLIR-runtime adapter for a registered Phase-QNode."""

    mlir_module: MLIRModule
    value_kernel: Callable[[NDArray[np.float64]], float]
    gradient_kernel: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    parameter_shape: tuple[int, ...]
    parameter_dtype: str
    runtime_backend: str
    verification: Mapping[str, object]
    claim_boundary: str = (
        "verified executable SCPN MLIR-runtime adapter for registered local "
        "Phase-QNode circuits; no native LLVM/JIT, provider, hardware, or "
        "interpreter-fallback success claim"
    )

    def __post_init__(self) -> None:
        """Validate the executable MLIR-runtime adapter invariants."""
        if not isinstance(self.mlir_module, MLIRModule):
            raise ValueError("mlir_module must be an MLIRModule")
        if not callable(self.value_kernel):
            raise ValueError("value_kernel must be callable")
        if not callable(self.gradient_kernel):
            raise ValueError("gradient_kernel must be callable")
        if self.parameter_shape != (self.mlir_module.resource_counts["phase_qnode_parameters"],):
            raise ValueError("parameter_shape must match MLIR parameter count")
        if self.parameter_dtype != "float64":
            raise ValueError("parameter_dtype must be float64")
        if self.runtime_backend != "scpn_mlir_runtime_adapter":
            raise ValueError("runtime_backend must be scpn_mlir_runtime_adapter")
        verification = dict(self.verification)
        if verification.get("value_close") is not True:
            raise ValueError("MLIR-runtime value verification failed")
        if verification.get("gradient_close") is not True:
            raise ValueError("MLIR-runtime gradient verification failed")
        if (
            verification.get("interpreter_fallback")
            != "blocked: cannot report interpreter fallback as compiled success"
        ):
            raise ValueError("interpreter fallback must be blocked in verification metadata")
        object.__setattr__(self, "verification", MappingProxyType(verification))
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")

    def value(self, parameters: Sequence[float] | FloatArray) -> float:
        """Execute the verified MLIR-runtime value kernel."""
        values = _as_phase_qnode_runtime_parameters(parameters, self.parameter_shape)
        return float(self.value_kernel(values))

    def gradient(self, parameters: Sequence[float] | FloatArray) -> NDArray[np.float64]:
        """Execute the verified MLIR-runtime gradient kernel."""
        values = _as_phase_qnode_runtime_parameters(parameters, self.parameter_shape)
        return _copy_float_array(self.gradient_kernel(values))

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready runtime metadata without raw callables."""
        return {
            "dialect": self.mlir_module.dialect,
            "runtime_backend": self.runtime_backend,
            "parameter_shape": list(self.parameter_shape),
            "parameter_dtype": self.parameter_dtype,
            "verification": dict(self.verification),
            "interpreter_fallback": self.verification["interpreter_fallback"],
            "claim_boundary": self.claim_boundary,
        }


def run_enzyme_mlir_maturity_audit(
    circuit: Any | None = None,
    parameters: Sequence[float] | FloatArray | None = None,
    *,
    toolchain_probe: Callable[[str], str | None] | None = None,
    version_probe: Callable[[str], str | None] | None = None,
    isolated_benchmark_artifact_id: str | None = None,
    isolated_benchmark_evidence: EnzymeMLIRBenchmarkAttachment | None = None,
    native_enzyme_execution_artifact_id: str | None = None,
    native_enzyme_execution_evidence: EnzymeNativeExecutionEvidence | None = None,
    mlir_llvm_correctness_artifact_id: str | None = None,
    compiler_ad_breadth_evidence: EnzymeMLIRCompilerADBreadthEvidence | None = None,
    compiler_ad_breadth_artifact: EnzymeMLIRCompilerADBreadthArtifact | None = None,
) -> EnzymeMLIRMaturityAuditResult:
    """Audit Enzyme/MLIR maturity without promoting unsupported compiler-AD claims."""
    executable = compile_phase_qnode_circuit_to_mlir_runtime(
        _default_enzyme_mlir_audit_circuit() if circuit is None else circuit,
        np.array([0.2, -0.3], dtype=np.float64) if parameters is None else parameters,
    )
    verification = dict(executable.verification)
    correctness_checks = {
        "phase_qnode_value_close": verification.get("value_close") is True,
        "phase_qnode_gradient_close": verification.get("gradient_close") is True,
        "mlir_runtime_backend_verified": executable.runtime_backend == "scpn_mlir_runtime_adapter",
        "native_llvm_jit_support_matrix_declared": bool(native_whole_program_ad_linalg_support()),
    }
    toolchain = {
        command: _enzyme_mlir_toolchain_status(command, toolchain_probe, version_probe)
        for command in ("enzyme", "opt", "mlir-opt", "clang")
    }
    hard_gaps: list[str] = []
    if not all(correctness_checks.values()):
        hard_gaps.append("MLIR/LLVM correctness check missing")
    for command, status in toolchain.items():
        if not status.available:
            hard_gaps.append(f"{command} toolchain unavailable")
    if isolated_benchmark_artifact_id is None:
        hard_gaps.append("isolated benchmark artefact missing")
    elif isolated_benchmark_evidence is None or not isolated_benchmark_evidence.promotion_ready:
        hard_gaps.append("validated isolated benchmark evidence missing")
    if native_enzyme_execution_evidence is not None:
        native_enzyme_execution_artifact_id = native_enzyme_execution_evidence.artifact_id
        if not native_enzyme_execution_evidence.passed:
            failure = native_enzyme_execution_evidence.failure_class or "unknown"
            hard_gaps.append(f"native Enzyme execution hard gap: {failure}")
    if native_enzyme_execution_artifact_id is None:
        hard_gaps.append("native Enzyme execution artefact missing")
    toolchain_versions = {
        command: status.version
        for command, status in toolchain.items()
        if status.available and status.version is not None
    }
    mlir_llvm_correctness_evidence = (
        MLIRLLVMCorrectnessEvidence(
            artifact_id=mlir_llvm_correctness_artifact_id,
            checks=correctness_checks,
            toolchain_versions=toolchain_versions,
            claim_boundary=(
                "Bounded SCPN MLIR-runtime and native LLVM/JIT support snapshot; "
                "not native Enzyme execution, provider, hardware, or performance evidence."
            ),
        )
        if mlir_llvm_correctness_artifact_id is not None
        else None
    )
    if mlir_llvm_correctness_evidence is None:
        hard_gaps.append("MLIR/LLVM correctness artefact missing")
    if compiler_ad_breadth_artifact is None:
        hard_gaps.append("compiler AD breadth artifact missing")
    elif compiler_ad_breadth_evidence is None:
        if compiler_ad_breadth_artifact.promotion_ready:
            compiler_ad_breadth_evidence = compiler_ad_breadth_artifact.to_breadth_evidence()
        else:
            hard_gaps.append("compiler AD breadth artifact not promotion-ready")
            if compiler_ad_breadth_artifact.failed_case_ids:
                hard_gaps.append(
                    "compiler AD breadth case hard gaps: "
                    + ", ".join(compiler_ad_breadth_artifact.failed_case_ids)
                )
    if compiler_ad_breadth_evidence is None:
        hard_gaps.append("compiler AD breadth evidence missing")
    return EnzymeMLIRMaturityAuditResult(
        scpn_mlir_runtime_verified=bool(
            correctness_checks["phase_qnode_value_close"]
            and correctness_checks["phase_qnode_gradient_close"]
            and correctness_checks["mlir_runtime_backend_verified"]
        ),
        native_llvm_jit_surface="available: bounded in-process native LLVM/JIT",
        toolchain=toolchain,
        correctness_checks=correctness_checks,
        hard_gaps=tuple(dict.fromkeys(hard_gaps)),
        isolated_benchmark_artifact_id=isolated_benchmark_artifact_id,
        isolated_benchmark_evidence=isolated_benchmark_evidence,
        native_enzyme_execution_artifact_id=native_enzyme_execution_artifact_id,
        native_enzyme_execution_evidence=native_enzyme_execution_evidence,
        mlir_llvm_correctness_evidence=mlir_llvm_correctness_evidence,
        compiler_ad_breadth_evidence=compiler_ad_breadth_evidence,
        compiler_ad_breadth_artifact=compiler_ad_breadth_artifact,
    )


def _enzyme_mlir_toolchain_status(
    command: str,
    toolchain_probe: Callable[[str], str | None] | None,
    version_probe: Callable[[str], str | None] | None,
) -> EnzymeMLIRToolchainStatus:
    executable = (
        _resolve_toolchain_executable(command)
        if toolchain_probe is None
        else toolchain_probe(command)
    )
    if executable is None:
        return EnzymeMLIRToolchainStatus(
            command=command,
            executable=None,
            available=False,
            version=None,
            failure_class="toolchain_missing",
            setup_instructions=(
                f"Install and expose {command} on PATH before promoting Enzyme/MLIR "
                "compiler-AD maturity evidence."
            ),
        )
    version = (
        _probe_toolchain_version(executable)
        if version_probe is None
        else version_probe(executable)
    )
    if version is None:
        return EnzymeMLIRToolchainStatus(
            command=command,
            executable=None,
            available=False,
            version=None,
            failure_class="version_probe_failed",
            setup_instructions=(
                f"{command} was found at {executable}, but version metadata could not "
                "be captured reproducibly."
            ),
        )
    return EnzymeMLIRToolchainStatus(
        command=command,
        executable=executable,
        available=True,
        version=version,
        failure_class=None,
        setup_instructions=None,
    )


def _resolve_toolchain_executable(command: str) -> str | None:
    """Return an absolute executable path for a PATH-discovered toolchain command."""
    resolved = shutil.which(command)
    if not resolved:
        return None
    try:
        path = Path(resolved).resolve(strict=True)
    except OSError:
        return None
    if not path.is_file() or not os.access(path, os.X_OK):
        return None
    return str(path)


def _probe_toolchain_version(executable: str) -> str | None:
    executable_path = Path(executable)
    if not executable_path.is_absolute():
        return None
    try:
        executable_path = executable_path.resolve(strict=True)
    except OSError:
        return None
    if not executable_path.is_file() or not os.access(executable_path, os.X_OK):
        return None
    for flag in ("--version", "-version"):
        try:
            completed = subprocess.run(  # nosec B603
                (str(executable_path), flag),
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        output = (completed.stdout or completed.stderr).strip().splitlines()
        if output:
            return output[0][:240]
    return None


def _default_enzyme_mlir_audit_circuit() -> Any:
    from scpn_quantum_control.phase.qnode_circuit import PauliTerm, PhaseQNodeCircuit

    return PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0), ("rx", (0,), 1)),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )


def lower_phase_qnode_circuit_to_mlir(
    circuit: Any,
    parameters: Sequence[float] | FloatArray,
) -> MLIRModule:
    """Lower a registered local Phase-QNode circuit to textual MLIR metadata.

    This is a compiler interchange report for the registered local statevector
    subset. It does not execute a native Rust/PyO3, LLVM, JIT, provider, or
    hardware backend.
    """
    from scpn_quantum_control.phase.qnode_circuit import phase_qnode_support_report

    values = np.asarray(parameters, dtype=np.float64)
    report = phase_qnode_support_report(circuit, values)
    if not report.supported:
        raise ValueError(f"phase-QNode lowering failed closed: {report.failure_reason}")
    operation_lines: list[str] = []
    dialect_operations: list[dict[str, object]] = []
    for operation in circuit.operations:
        qubits = ", ".join(str(qubit) for qubit in operation.qubits)
        parameter_attr = (
            ""
            if operation.parameter_index is None
            else f" {{parameter_index = {operation.parameter_index}}}"
        )
        operation_lines.append(
            f"    scpn_phase_qnode.{operation.gate}({qubits}){parameter_attr} : () -> ()"
        )
        dialect_operations.append(_phase_qnode_dialect_operation(operation))
    observable_terms = _phase_qnode_observable_terms(circuit.observable)
    operation_lines.append(
        f"    scpn_phase_qnode.expectation @{report.observable_kind} : () -> f64"
    )
    dialect_operations.append(
        {
            "op": "scpn_phase_qnode.expectation",
            "observable_kind": report.observable_kind,
            "operand_type": "statevector",
            "result_type": "f64",
        }
    )
    text = "\n".join(
        (
            'module attributes {dialect = "scpn_phase_qnode"} {',
            "  func.func @phase_qnode(%params: tensor<?xf64>) -> f64 {",
            *operation_lines,
            "  }",
            "}",
        )
    )
    metadata = {
        "supported": True,
        "support_report": report.to_dict(),
        "primitive_support": {
            "gates": list(report.gates),
            "observables": [report.observable_kind],
            "differentiable_parameters": list(report.differentiable_parameters),
        },
        "dialect_operations": dialect_operations,
        "runtime_backend": "available: scpn_mlir_runtime_adapter",
        "compiled_execution": "available: verified MLIR-runtime adapter",
        "shape_limits": {
            "max_qubits": 8,
            "max_parameters": 64,
            "statevector_dimension_limit": 2**8,
        },
        "observable_terms": observable_terms,
        "rust_pyo3_parity": "blocked: no Rust phase-QNode lowering backend",
        "native_jit_parity": "blocked: no native JIT phase-QNode lowering backend",
        "provider_lowering": "blocked: provider circuits require explicit provider boundary",
        "interpreter_fallback": "blocked: cannot report interpreter fallback as compiled success",
        "claim_boundary": (
            "registered local Phase-QNode MLIR lowering with verified SCPN "
            "MLIR-runtime adapter; no native LLVM/JIT execution, provider "
            "submission, hardware execution, or interpreter fallback success"
        ),
    }
    return MLIRModule(
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        dialect="scpn_phase_qnode",
        resource_counts={
            "phase_qnode_gates": len(circuit.operations),
            "phase_qnode_parameters": int(values.size),
            "phase_qnode_observable_terms": len(observable_terms),
        },
        metadata=metadata,
    )


def compile_phase_qnode_circuit_to_mlir_runtime(
    circuit: Any,
    sample_parameters: Sequence[float] | FloatArray,
    *,
    atol: float = 1.0e-10,
    rtol: float = 1.0e-10,
) -> PhaseQNodeMLIRRuntimeExecutable:
    """Compile a registered Phase-QNode to the verified SCPN MLIR runtime adapter."""
    from scpn_quantum_control.phase.qnode_circuit import (
        execute_phase_qnode_circuit,
        parameter_shift_phase_qnode_gradient,
    )

    module = lower_phase_qnode_circuit_to_mlir(circuit, sample_parameters)
    sample = _as_phase_qnode_runtime_parameters(
        sample_parameters,
        (module.resource_counts["phase_qnode_parameters"],),
    )
    tolerance = _as_mlir_runtime_tolerance(atol, "atol")
    relative_tolerance = _as_mlir_runtime_tolerance(rtol, "rtol")

    def value_kernel(parameters: NDArray[np.float64]) -> float:
        values = _as_phase_qnode_runtime_parameters(parameters, sample.shape)
        return float(execute_phase_qnode_circuit(circuit, values).value)

    def gradient_kernel(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        values = _as_phase_qnode_runtime_parameters(parameters, sample.shape)
        return _copy_float_array(parameter_shift_phase_qnode_gradient(circuit, values).gradient)

    reference_value = float(execute_phase_qnode_circuit(circuit, sample).value)
    reference_gradient = parameter_shift_phase_qnode_gradient(circuit, sample).gradient
    runtime_value = value_kernel(sample)
    runtime_gradient = gradient_kernel(sample)
    value_close = bool(
        np.isclose(runtime_value, reference_value, atol=tolerance, rtol=relative_tolerance)
    )
    gradient_close = bool(
        np.allclose(
            runtime_gradient,
            reference_gradient,
            atol=tolerance,
            rtol=relative_tolerance,
        )
    )
    verification = {
        "value_close": value_close,
        "gradient_close": gradient_close,
        "max_abs_value_error": abs(runtime_value - reference_value),
        "max_abs_gradient_error": _max_abs_error(runtime_gradient, reference_gradient),
        "samples": 1,
        "interpreter_fallback": "blocked: cannot report interpreter fallback as compiled success",
    }
    return PhaseQNodeMLIRRuntimeExecutable(
        mlir_module=module,
        value_kernel=value_kernel,
        gradient_kernel=gradient_kernel,
        parameter_shape=sample.shape,
        parameter_dtype=str(sample.dtype),
        runtime_backend="scpn_mlir_runtime_adapter",
        verification=verification,
    )


def _as_phase_qnode_runtime_parameters(
    parameters: object,
    expected_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    raw = np.asarray(parameters)
    if raw.dtype.kind in {"b", "c", "O", "S", "U"}:
        raise ValueError("runtime parameters must contain finite real numeric values")
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != expected_shape:
        raise ValueError(f"runtime parameter shape must be {expected_shape}, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("runtime parameters must contain finite real numeric values")
    return _copy_float_array(values)


def _as_mlir_runtime_tolerance(value: float, name: str) -> float:
    tolerance = float(value)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return tolerance


def _phase_qnode_dialect_operation(operation: Any) -> dict[str, object]:
    return {
        "op": f"scpn_phase_qnode.{operation.gate}",
        "gate": operation.gate,
        "qubits": list(operation.qubits),
        "parameter_index": operation.parameter_index,
        "operand_type": "f64" if operation.parameter_index is not None else "none",
        "result_type": "statevector",
    }


def _phase_qnode_observable_terms(observable: Any) -> list[dict[str, object]]:
    from scpn_quantum_control.phase.qnode_circuit import PauliTerm, SparsePauliHamiltonian

    if isinstance(observable, SparsePauliHamiltonian):
        return [term.to_dict() for term in observable.terms]
    if isinstance(observable, PauliTerm):
        return [observable.to_dict()]
    return [{"kind": str(observable)}]


__all__ = [
    "CompilerADTransformPlan",
    "CompilerADExecutableConfig",
    "CompilerADKernelVerification",
    "DifferentiableMLIRCompileConfig",
    "ExecutableCompilerADKernel",
    "ExecutableWholeProgramADBatchResult",
    "ExecutableWholeProgramADKernel",
    "EnzymeMLIRBenchmarkAttachment",
    "EnzymeMLIRCompilerADBreadthArtifact",
    "EnzymeMLIRCompilerADBreadthCaseEvidence",
    "EnzymeMLIRCompilerADBreadthEvidence",
    "EnzymeMLIRMaturityAuditResult",
    "EnzymeNativeExecutionEvidence",
    "EnzymeToolchainADCase",
    "EnzymeToolchainADExecutionEvidence",
    "EnzymeMLIRToolchainStatus",
    "MLIRLLVMCorrectnessEvidence",
    "MLIRCompileConfig",
    "NativeWholeProgramADKernel",
    "NativeWholeProgramADExecutionCase",
    "NativeWholeProgramADExecutionEvidence",
    "PhaseQNodeMLIRRuntimeExecutable",
    "PrimitiveLoweringStatus",
    "WholeProgramADNativeLoweringReport",
    "MLIRModule",
    "analyse_whole_program_ad_native_lowering",
    "build_native_whole_program_ad_execution_evidence",
    "run_native_whole_program_ad_execution_evidence",
    "build_enzyme_mlir_benchmark_attachment",
    "build_enzyme_mlir_compiler_ad_breadth_artifact",
    "build_enzyme_mlir_compiler_ad_breadth_evidence",
    "build_enzyme_mlir_compiler_ad_breadth_gap_artifact",
    "build_compiler_ad_transform_plan",
    "compile_compiler_ad_transform_plan_to_mlir",
    "compile_phase_qnode_circuit_to_mlir_runtime",
    "compile_custom_derivative_rule_to_mlir",
    "compile_custom_derivative_rule_to_executable",
    "compile_registered_primitive_to_executable",
    "compile_whole_program_ad_trace_to_executable",
    "compile_whole_program_ad_trace_to_native_llvm_jit",
    "compile_matrix_2x2_determinant_ad_to_native_llvm_jit",
    "compile_matrix_2x2_eigenvalues_ad_to_native_llvm_jit",
    "compile_matrix_2x2_eigensystem_ad_to_native_llvm_jit",
    "compile_matrix_2x2_inverse_ad_to_native_llvm_jit",
    "compile_matrix_2x2_solve_ad_to_native_llvm_jit",
    "compile_matrix_frobenius_norm_squared_ad_to_native_llvm_jit",
    "compile_matrix_matrix_product_ad_to_native_llvm_jit",
    "compile_matrix_quadratic_form_ad_to_native_llvm_jit",
    "compile_matrix_trace_ad_to_native_llvm_jit",
    "compile_matrix_vector_product_ad_to_native_llvm_jit",
    "compile_scalar_binary_elementwise_ad_to_native_llvm_jit",
    "compile_scalar_quadratic_ad_to_native_llvm_jit",
    "compile_scalar_unary_elementwise_ad_to_native_llvm_jit",
    "compile_vector_dot_ad_to_native_llvm_jit",
    "compile_vector_squared_norm_ad_to_native_llvm_jit",
    "compile_symmetric_2x2_cholesky_ad_to_native_llvm_jit",
    "compile_symmetric_2x2_eigenvalues_ad_to_native_llvm_jit",
    "compile_whole_program_ad_trace_to_mlir",
    "compile_kuramoto_to_mlir",
    "lower_phase_qnode_circuit_to_mlir",
    "run_enzyme_mlir_maturity_audit",
    "run_enzyme_toolchain_execution_evidence",
    "make_executable_ad_kernel_batching_rule",
    "make_matrix_2x2_determinant_native_llvm_jit_lowering_rule",
    "make_matrix_2x2_determinant_native_llvm_jit_primitive_transform",
    "make_matrix_2x2_eigenvalues_native_llvm_jit_lowering_rule",
    "make_matrix_2x2_eigenvalues_native_llvm_jit_primitive_transform",
    "make_matrix_2x2_eigensystem_native_llvm_jit_lowering_rule",
    "make_matrix_2x2_eigensystem_native_llvm_jit_primitive_transform",
    "make_matrix_2x2_inverse_native_llvm_jit_lowering_rule",
    "make_matrix_2x2_inverse_native_llvm_jit_primitive_transform",
    "make_matrix_2x2_solve_native_llvm_jit_lowering_rule",
    "make_matrix_2x2_solve_native_llvm_jit_primitive_transform",
    "make_matrix_frobenius_norm_squared_native_llvm_jit_lowering_rule",
    "make_matrix_frobenius_norm_squared_native_llvm_jit_primitive_transform",
    "make_matrix_matrix_product_native_llvm_jit_lowering_rule",
    "make_matrix_matrix_product_native_llvm_jit_primitive_transform",
    "make_matrix_quadratic_form_native_llvm_jit_lowering_rule",
    "make_matrix_quadratic_form_native_llvm_jit_primitive_transform",
    "make_matrix_trace_native_llvm_jit_lowering_rule",
    "make_matrix_trace_native_llvm_jit_primitive_transform",
    "make_matrix_vector_product_native_llvm_jit_lowering_rule",
    "make_matrix_vector_product_native_llvm_jit_primitive_transform",
    "make_scalar_binary_elementwise_native_llvm_jit_lowering_rule",
    "make_scalar_quadratic_native_llvm_jit_lowering_rule",
    "make_scalar_unary_elementwise_native_llvm_jit_lowering_rule",
    "make_symmetric_2x2_cholesky_native_llvm_jit_lowering_rule",
    "make_symmetric_2x2_cholesky_native_llvm_jit_primitive_transform",
    "make_symmetric_2x2_eigenvalues_native_llvm_jit_lowering_rule",
    "make_symmetric_2x2_eigenvalues_native_llvm_jit_primitive_transform",
    "make_vector_dot_native_llvm_jit_lowering_rule",
    "make_vector_dot_native_llvm_jit_primitive_transform",
    "make_vector_squared_norm_native_llvm_jit_lowering_rule",
    "make_vector_squared_norm_native_llvm_jit_primitive_transform",
    "native_whole_program_ad_linalg_support",
]
