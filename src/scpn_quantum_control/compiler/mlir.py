# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR module
# scpn-quantum-control -- MLIR compiler facade
"""Stable MLIR compiler facade over focused implementation leaves."""

from __future__ import annotations

from .mlir_enzyme_audit import (
    _default_enzyme_mlir_audit_circuit as _default_enzyme_mlir_audit_circuit,
)
from .mlir_enzyme_audit import (
    _enzyme_mlir_toolchain_status as _enzyme_mlir_toolchain_status,
)
from .mlir_enzyme_audit import (
    _probe_toolchain_version as _probe_toolchain_version,
)
from .mlir_enzyme_audit import (
    _resolve_toolchain_executable as _resolve_toolchain_executable,
)
from .mlir_enzyme_audit import (
    run_enzyme_mlir_maturity_audit as run_enzyme_mlir_maturity_audit,
)
from .mlir_enzyme_evidence import (
    EnzymeMLIRBenchmarkAttachment as EnzymeMLIRBenchmarkAttachment,
)
from .mlir_enzyme_evidence import (
    EnzymeMLIRCompilerADBreadthArtifact as EnzymeMLIRCompilerADBreadthArtifact,
)
from .mlir_enzyme_evidence import (
    EnzymeMLIRCompilerADBreadthArtifactFiles as EnzymeMLIRCompilerADBreadthArtifactFiles,
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
from .mlir_enzyme_evidence import (
    render_enzyme_mlir_compiler_ad_breadth_artifact_markdown as render_enzyme_mlir_compiler_ad_breadth_artifact_markdown,
)
from .mlir_enzyme_evidence import (
    write_enzyme_mlir_compiler_ad_breadth_artifact as write_enzyme_mlir_compiler_ad_breadth_artifact,
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
from .mlir_llvm_jit_claim_gate import (
    LLVM_JIT_CLAIM_GATE_BOUNDARY as LLVM_JIT_CLAIM_GATE_BOUNDARY,
)
from .mlir_llvm_jit_claim_gate import (
    LLVMJITClaimGate as LLVMJITClaimGate,
)
from .mlir_llvm_jit_claim_gate import (
    build_llvm_jit_claim_gate as build_llvm_jit_claim_gate,
)
from .mlir_llvm_jit_claim_gate import (
    llvm_jit_claim_gate_from_dict as llvm_jit_claim_gate_from_dict,
)
from .mlir_llvm_jit_claim_gate import (
    render_llvm_jit_claim_gate_markdown as render_llvm_jit_claim_gate_markdown,
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
from .mlir_phase_qnode_runtime import (
    PhaseQNodeMLIRRuntimeExecutable as PhaseQNodeMLIRRuntimeExecutable,
)
from .mlir_phase_qnode_runtime import (
    _as_mlir_runtime_tolerance as _as_mlir_runtime_tolerance,
)
from .mlir_phase_qnode_runtime import (
    _as_phase_qnode_runtime_parameters as _as_phase_qnode_runtime_parameters,
)
from .mlir_phase_qnode_runtime import (
    _phase_qnode_dialect_operation as _phase_qnode_dialect_operation,
)
from .mlir_phase_qnode_runtime import (
    _phase_qnode_observable_terms as _phase_qnode_observable_terms,
)
from .mlir_phase_qnode_runtime import (
    compile_phase_qnode_circuit_to_mlir_runtime as compile_phase_qnode_circuit_to_mlir_runtime,
)
from .mlir_phase_qnode_runtime import (
    lower_phase_qnode_circuit_to_mlir as lower_phase_qnode_circuit_to_mlir,
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
from .mlir_transform_plan_assembly import (
    _status_has_native_llvm_jit_backend as _status_has_native_llvm_jit_backend,
)
from .mlir_transform_plan_assembly import (
    _status_has_verified_rust_backend as _status_has_verified_rust_backend,
)
from .mlir_transform_plan_assembly import (
    build_compiler_ad_transform_plan as build_compiler_ad_transform_plan,
)
from .mlir_transform_plan_assembly import (
    compile_compiler_ad_transform_plan_to_mlir as compile_compiler_ad_transform_plan_to_mlir,
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
from .mlir_workload_compilation import FloatArray as FloatArray
from .mlir_workload_compilation import _coupling_terms as _coupling_terms
from .mlir_workload_compilation import (
    compile_custom_derivative_rule_to_executable as compile_custom_derivative_rule_to_executable,
)
from .mlir_workload_compilation import (
    compile_kuramoto_to_mlir as compile_kuramoto_to_mlir,
)
from .mlir_workload_compilation import (
    compile_registered_primitive_to_executable as compile_registered_primitive_to_executable,
)
from .mlir_workload_compilation import (
    make_program_ad_linalg_matrix_power_executable_lowering_rule as make_program_ad_linalg_matrix_power_executable_lowering_rule,
)
from .mlir_workload_compilation import (
    make_program_ad_linalg_multi_dot_executable_lowering_rule as make_program_ad_linalg_multi_dot_executable_lowering_rule,
)

__all__ = [
    "CompilerADTransformPlan",
    "CompilerADExecutableConfig",
    "CompilerADKernelVerification",
    "DifferentiableMLIRCompileConfig",
    "ExecutableCompilerADKernel",
    "ExecutableWholeProgramADBatchResult",
    "ExecutableWholeProgramADKernel",
    "LLVM_JIT_CLAIM_GATE_BOUNDARY",
    "LLVMJITClaimGate",
    "EnzymeMLIRBenchmarkAttachment",
    "EnzymeMLIRCompilerADBreadthArtifact",
    "EnzymeMLIRCompilerADBreadthArtifactFiles",
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
    "build_llvm_jit_claim_gate",
    "build_native_whole_program_ad_execution_evidence",
    "run_native_whole_program_ad_execution_evidence",
    "build_enzyme_mlir_benchmark_attachment",
    "build_enzyme_mlir_compiler_ad_breadth_artifact",
    "build_enzyme_mlir_compiler_ad_breadth_evidence",
    "build_enzyme_mlir_compiler_ad_breadth_gap_artifact",
    "render_enzyme_mlir_compiler_ad_breadth_artifact_markdown",
    "write_enzyme_mlir_compiler_ad_breadth_artifact",
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
    "llvm_jit_claim_gate_from_dict",
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
    "render_llvm_jit_claim_gate_markdown",
]
