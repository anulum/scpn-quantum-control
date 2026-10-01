# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — compiler package exports
# scpn-quantum-control -- compiler exports
"""Compiler frontends and interchange formats."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .alias_activity_evidence import (
        COMPILER_ALIAS_ACTIVITY_EVIDENCE_ID,
        COMPILER_ALIAS_ACTIVITY_EVIDENCE_SCHEMA,
        CompilerAliasActivityCase,
        CompilerAliasActivityEvidence,
        build_compiler_alias_activity_evidence,
        render_compiler_alias_activity_evidence_markdown,
    )
    from .mlir import (
        LLVM_JIT_CLAIM_GATE_BOUNDARY,
        CompilerADExecutableConfig,
        CompilerADKernelVerification,
        CompilerADTransformPlan,
        DifferentiableMLIRCompileConfig,
        EnzymeMLIRBenchmarkAttachment,
        EnzymeMLIRCompilerADBreadthArtifact,
        EnzymeMLIRCompilerADBreadthArtifactFiles,
        EnzymeMLIRCompilerADBreadthCaseEvidence,
        EnzymeMLIRCompilerADBreadthEvidence,
        EnzymeMLIRMaturityAuditResult,
        EnzymeMLIRToolchainStatus,
        EnzymeNativeExecutionEvidence,
        EnzymeToolchainADCase,
        EnzymeToolchainADExecutionEvidence,
        ExecutableCompilerADKernel,
        ExecutableWholeProgramADBatchResult,
        ExecutableWholeProgramADKernel,
        LLVMJITClaimGate,
        MLIRCompileConfig,
        MLIRLLVMCorrectnessEvidence,
        MLIRModule,
        NativeWholeProgramADExecutionCase,
        NativeWholeProgramADExecutionEvidence,
        NativeWholeProgramADKernel,
        PhaseQNodeMLIRRuntimeExecutable,
        PrimitiveLoweringStatus,
        WholeProgramADNativeLoweringReport,
        analyse_whole_program_ad_native_lowering,
        build_compiler_ad_transform_plan,
        build_enzyme_mlir_benchmark_attachment,
        build_enzyme_mlir_compiler_ad_breadth_artifact,
        build_enzyme_mlir_compiler_ad_breadth_evidence,
        build_enzyme_mlir_compiler_ad_breadth_gap_artifact,
        build_llvm_jit_claim_gate,
        build_native_whole_program_ad_execution_evidence,
        clear_native_whole_program_ad_compile_cache,
        compile_compiler_ad_transform_plan_to_mlir,
        compile_custom_derivative_rule_to_executable,
        compile_custom_derivative_rule_to_mlir,
        compile_kuramoto_to_mlir,
        compile_matrix_2x2_determinant_ad_to_native_llvm_jit,
        compile_matrix_2x2_eigensystem_ad_to_native_llvm_jit,
        compile_matrix_2x2_eigenvalues_ad_to_native_llvm_jit,
        compile_matrix_2x2_inverse_ad_to_native_llvm_jit,
        compile_matrix_2x2_solve_ad_to_native_llvm_jit,
        compile_matrix_frobenius_norm_squared_ad_to_native_llvm_jit,
        compile_matrix_matrix_product_ad_to_native_llvm_jit,
        compile_matrix_quadratic_form_ad_to_native_llvm_jit,
        compile_matrix_trace_ad_to_native_llvm_jit,
        compile_matrix_vector_product_ad_to_native_llvm_jit,
        compile_phase_qnode_circuit_to_mlir_runtime,
        compile_registered_primitive_to_executable,
        compile_scalar_binary_elementwise_ad_to_native_llvm_jit,
        compile_scalar_quadratic_ad_to_native_llvm_jit,
        compile_scalar_unary_elementwise_ad_to_native_llvm_jit,
        compile_symmetric_2x2_cholesky_ad_to_native_llvm_jit,
        compile_symmetric_2x2_eigenvalues_ad_to_native_llvm_jit,
        compile_vector_dot_ad_to_native_llvm_jit,
        compile_vector_squared_norm_ad_to_native_llvm_jit,
        compile_whole_program_ad_trace_to_executable,
        compile_whole_program_ad_trace_to_mlir,
        compile_whole_program_ad_trace_to_native_llvm_jit,
        llvm_jit_claim_gate_from_dict,
        make_executable_ad_kernel_batching_rule,
        make_matrix_2x2_determinant_native_llvm_jit_lowering_rule,
        make_matrix_2x2_eigensystem_native_llvm_jit_lowering_rule,
        make_matrix_2x2_eigensystem_native_llvm_jit_primitive_transform,
        make_matrix_2x2_eigenvalues_native_llvm_jit_lowering_rule,
        make_matrix_2x2_inverse_native_llvm_jit_lowering_rule,
        make_matrix_2x2_solve_native_llvm_jit_lowering_rule,
        make_matrix_frobenius_norm_squared_native_llvm_jit_lowering_rule,
        make_matrix_matrix_product_native_llvm_jit_lowering_rule,
        make_matrix_quadratic_form_native_llvm_jit_lowering_rule,
        make_matrix_trace_native_llvm_jit_lowering_rule,
        make_matrix_vector_product_native_llvm_jit_lowering_rule,
        make_scalar_binary_elementwise_native_llvm_jit_lowering_rule,
        make_scalar_quadratic_native_llvm_jit_lowering_rule,
        make_scalar_unary_elementwise_native_llvm_jit_lowering_rule,
        make_symmetric_2x2_cholesky_native_llvm_jit_lowering_rule,
        make_symmetric_2x2_eigenvalues_native_llvm_jit_lowering_rule,
        make_vector_dot_native_llvm_jit_lowering_rule,
        make_vector_squared_norm_native_llvm_jit_lowering_rule,
        native_whole_program_ad_compile_cache_stats,
        native_whole_program_ad_linalg_support,
        render_enzyme_mlir_compiler_ad_breadth_artifact_markdown,
        render_llvm_jit_claim_gate_markdown,
        run_enzyme_mlir_maturity_audit,
        run_enzyme_toolchain_execution_evidence,
        run_native_whole_program_ad_execution_evidence,
        write_enzyme_mlir_compiler_ad_breadth_artifact,
    )
    from .promotion_batch import (
        COMPILER_PROMOTION_BATCH_ID,
        COMPILER_PROMOTION_BATCH_SCHEMA,
        CompilerPromotionBatch,
        CompilerPromotionBatchEvidenceFile,
        build_compiler_promotion_batch,
        render_compiler_promotion_batch_markdown,
    )

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "COMPILER_ALIAS_ACTIVITY_EVIDENCE_ID": (
        "scpn_quantum_control.compiler.alias_activity_evidence",
        "COMPILER_ALIAS_ACTIVITY_EVIDENCE_ID",
    ),
    "COMPILER_ALIAS_ACTIVITY_EVIDENCE_SCHEMA": (
        "scpn_quantum_control.compiler.alias_activity_evidence",
        "COMPILER_ALIAS_ACTIVITY_EVIDENCE_SCHEMA",
    ),
    "CompilerAliasActivityCase": (
        "scpn_quantum_control.compiler.alias_activity_evidence",
        "CompilerAliasActivityCase",
    ),
    "CompilerAliasActivityEvidence": (
        "scpn_quantum_control.compiler.alias_activity_evidence",
        "CompilerAliasActivityEvidence",
    ),
    "build_compiler_alias_activity_evidence": (
        "scpn_quantum_control.compiler.alias_activity_evidence",
        "build_compiler_alias_activity_evidence",
    ),
    "render_compiler_alias_activity_evidence_markdown": (
        "scpn_quantum_control.compiler.alias_activity_evidence",
        "render_compiler_alias_activity_evidence_markdown",
    ),
    "LLVM_JIT_CLAIM_GATE_BOUNDARY": (
        "scpn_quantum_control.compiler.mlir",
        "LLVM_JIT_CLAIM_GATE_BOUNDARY",
    ),
    "CompilerADExecutableConfig": (
        "scpn_quantum_control.compiler.mlir",
        "CompilerADExecutableConfig",
    ),
    "CompilerADKernelVerification": (
        "scpn_quantum_control.compiler.mlir",
        "CompilerADKernelVerification",
    ),
    "CompilerADTransformPlan": ("scpn_quantum_control.compiler.mlir", "CompilerADTransformPlan"),
    "DifferentiableMLIRCompileConfig": (
        "scpn_quantum_control.compiler.mlir",
        "DifferentiableMLIRCompileConfig",
    ),
    "EnzymeMLIRBenchmarkAttachment": (
        "scpn_quantum_control.compiler.mlir",
        "EnzymeMLIRBenchmarkAttachment",
    ),
    "EnzymeMLIRCompilerADBreadthArtifact": (
        "scpn_quantum_control.compiler.mlir",
        "EnzymeMLIRCompilerADBreadthArtifact",
    ),
    "EnzymeMLIRCompilerADBreadthArtifactFiles": (
        "scpn_quantum_control.compiler.mlir",
        "EnzymeMLIRCompilerADBreadthArtifactFiles",
    ),
    "EnzymeMLIRCompilerADBreadthCaseEvidence": (
        "scpn_quantum_control.compiler.mlir",
        "EnzymeMLIRCompilerADBreadthCaseEvidence",
    ),
    "EnzymeMLIRCompilerADBreadthEvidence": (
        "scpn_quantum_control.compiler.mlir",
        "EnzymeMLIRCompilerADBreadthEvidence",
    ),
    "EnzymeMLIRMaturityAuditResult": (
        "scpn_quantum_control.compiler.mlir",
        "EnzymeMLIRMaturityAuditResult",
    ),
    "EnzymeMLIRToolchainStatus": (
        "scpn_quantum_control.compiler.mlir",
        "EnzymeMLIRToolchainStatus",
    ),
    "EnzymeNativeExecutionEvidence": (
        "scpn_quantum_control.compiler.mlir",
        "EnzymeNativeExecutionEvidence",
    ),
    "EnzymeToolchainADCase": ("scpn_quantum_control.compiler.mlir", "EnzymeToolchainADCase"),
    "EnzymeToolchainADExecutionEvidence": (
        "scpn_quantum_control.compiler.mlir",
        "EnzymeToolchainADExecutionEvidence",
    ),
    "ExecutableCompilerADKernel": (
        "scpn_quantum_control.compiler.mlir",
        "ExecutableCompilerADKernel",
    ),
    "ExecutableWholeProgramADBatchResult": (
        "scpn_quantum_control.compiler.mlir",
        "ExecutableWholeProgramADBatchResult",
    ),
    "ExecutableWholeProgramADKernel": (
        "scpn_quantum_control.compiler.mlir",
        "ExecutableWholeProgramADKernel",
    ),
    "LLVMJITClaimGate": ("scpn_quantum_control.compiler.mlir", "LLVMJITClaimGate"),
    "MLIRCompileConfig": ("scpn_quantum_control.compiler.mlir", "MLIRCompileConfig"),
    "MLIRLLVMCorrectnessEvidence": (
        "scpn_quantum_control.compiler.mlir",
        "MLIRLLVMCorrectnessEvidence",
    ),
    "MLIRModule": ("scpn_quantum_control.compiler.mlir", "MLIRModule"),
    "NativeWholeProgramADExecutionCase": (
        "scpn_quantum_control.compiler.mlir",
        "NativeWholeProgramADExecutionCase",
    ),
    "NativeWholeProgramADExecutionEvidence": (
        "scpn_quantum_control.compiler.mlir",
        "NativeWholeProgramADExecutionEvidence",
    ),
    "NativeWholeProgramADKernel": (
        "scpn_quantum_control.compiler.mlir",
        "NativeWholeProgramADKernel",
    ),
    "PhaseQNodeMLIRRuntimeExecutable": (
        "scpn_quantum_control.compiler.mlir",
        "PhaseQNodeMLIRRuntimeExecutable",
    ),
    "PrimitiveLoweringStatus": ("scpn_quantum_control.compiler.mlir", "PrimitiveLoweringStatus"),
    "WholeProgramADNativeLoweringReport": (
        "scpn_quantum_control.compiler.mlir",
        "WholeProgramADNativeLoweringReport",
    ),
    "analyse_whole_program_ad_native_lowering": (
        "scpn_quantum_control.compiler.mlir",
        "analyse_whole_program_ad_native_lowering",
    ),
    "build_compiler_ad_transform_plan": (
        "scpn_quantum_control.compiler.mlir",
        "build_compiler_ad_transform_plan",
    ),
    "build_enzyme_mlir_benchmark_attachment": (
        "scpn_quantum_control.compiler.mlir",
        "build_enzyme_mlir_benchmark_attachment",
    ),
    "build_enzyme_mlir_compiler_ad_breadth_artifact": (
        "scpn_quantum_control.compiler.mlir",
        "build_enzyme_mlir_compiler_ad_breadth_artifact",
    ),
    "build_enzyme_mlir_compiler_ad_breadth_evidence": (
        "scpn_quantum_control.compiler.mlir",
        "build_enzyme_mlir_compiler_ad_breadth_evidence",
    ),
    "build_enzyme_mlir_compiler_ad_breadth_gap_artifact": (
        "scpn_quantum_control.compiler.mlir",
        "build_enzyme_mlir_compiler_ad_breadth_gap_artifact",
    ),
    "build_llvm_jit_claim_gate": (
        "scpn_quantum_control.compiler.mlir",
        "build_llvm_jit_claim_gate",
    ),
    "build_native_whole_program_ad_execution_evidence": (
        "scpn_quantum_control.compiler.mlir",
        "build_native_whole_program_ad_execution_evidence",
    ),
    "clear_native_whole_program_ad_compile_cache": (
        "scpn_quantum_control.compiler.mlir",
        "clear_native_whole_program_ad_compile_cache",
    ),
    "compile_compiler_ad_transform_plan_to_mlir": (
        "scpn_quantum_control.compiler.mlir",
        "compile_compiler_ad_transform_plan_to_mlir",
    ),
    "compile_custom_derivative_rule_to_executable": (
        "scpn_quantum_control.compiler.mlir",
        "compile_custom_derivative_rule_to_executable",
    ),
    "compile_custom_derivative_rule_to_mlir": (
        "scpn_quantum_control.compiler.mlir",
        "compile_custom_derivative_rule_to_mlir",
    ),
    "compile_kuramoto_to_mlir": ("scpn_quantum_control.compiler.mlir", "compile_kuramoto_to_mlir"),
    "compile_matrix_2x2_determinant_ad_to_native_llvm_jit": (
        "scpn_quantum_control.compiler.mlir",
        "compile_matrix_2x2_determinant_ad_to_native_llvm_jit",
    ),
    "compile_matrix_2x2_eigensystem_ad_to_native_llvm_jit": (
        "scpn_quantum_control.compiler.mlir",
        "compile_matrix_2x2_eigensystem_ad_to_native_llvm_jit",
    ),
    "compile_matrix_2x2_eigenvalues_ad_to_native_llvm_jit": (
        "scpn_quantum_control.compiler.mlir",
        "compile_matrix_2x2_eigenvalues_ad_to_native_llvm_jit",
    ),
    "compile_matrix_2x2_inverse_ad_to_native_llvm_jit": (
        "scpn_quantum_control.compiler.mlir",
        "compile_matrix_2x2_inverse_ad_to_native_llvm_jit",
    ),
    "compile_matrix_2x2_solve_ad_to_native_llvm_jit": (
        "scpn_quantum_control.compiler.mlir",
        "compile_matrix_2x2_solve_ad_to_native_llvm_jit",
    ),
    "compile_matrix_frobenius_norm_squared_ad_to_native_llvm_jit": (
        "scpn_quantum_control.compiler.mlir",
        "compile_matrix_frobenius_norm_squared_ad_to_native_llvm_jit",
    ),
    "compile_matrix_matrix_product_ad_to_native_llvm_jit": (
        "scpn_quantum_control.compiler.mlir",
        "compile_matrix_matrix_product_ad_to_native_llvm_jit",
    ),
    "compile_matrix_quadratic_form_ad_to_native_llvm_jit": (
        "scpn_quantum_control.compiler.mlir",
        "compile_matrix_quadratic_form_ad_to_native_llvm_jit",
    ),
    "compile_matrix_trace_ad_to_native_llvm_jit": (
        "scpn_quantum_control.compiler.mlir",
        "compile_matrix_trace_ad_to_native_llvm_jit",
    ),
    "compile_matrix_vector_product_ad_to_native_llvm_jit": (
        "scpn_quantum_control.compiler.mlir",
        "compile_matrix_vector_product_ad_to_native_llvm_jit",
    ),
    "compile_phase_qnode_circuit_to_mlir_runtime": (
        "scpn_quantum_control.compiler.mlir",
        "compile_phase_qnode_circuit_to_mlir_runtime",
    ),
    "compile_registered_primitive_to_executable": (
        "scpn_quantum_control.compiler.mlir",
        "compile_registered_primitive_to_executable",
    ),
    "compile_scalar_binary_elementwise_ad_to_native_llvm_jit": (
        "scpn_quantum_control.compiler.mlir",
        "compile_scalar_binary_elementwise_ad_to_native_llvm_jit",
    ),
    "compile_scalar_quadratic_ad_to_native_llvm_jit": (
        "scpn_quantum_control.compiler.mlir",
        "compile_scalar_quadratic_ad_to_native_llvm_jit",
    ),
    "compile_scalar_unary_elementwise_ad_to_native_llvm_jit": (
        "scpn_quantum_control.compiler.mlir",
        "compile_scalar_unary_elementwise_ad_to_native_llvm_jit",
    ),
    "compile_symmetric_2x2_cholesky_ad_to_native_llvm_jit": (
        "scpn_quantum_control.compiler.mlir",
        "compile_symmetric_2x2_cholesky_ad_to_native_llvm_jit",
    ),
    "compile_symmetric_2x2_eigenvalues_ad_to_native_llvm_jit": (
        "scpn_quantum_control.compiler.mlir",
        "compile_symmetric_2x2_eigenvalues_ad_to_native_llvm_jit",
    ),
    "compile_vector_dot_ad_to_native_llvm_jit": (
        "scpn_quantum_control.compiler.mlir",
        "compile_vector_dot_ad_to_native_llvm_jit",
    ),
    "compile_vector_squared_norm_ad_to_native_llvm_jit": (
        "scpn_quantum_control.compiler.mlir",
        "compile_vector_squared_norm_ad_to_native_llvm_jit",
    ),
    "compile_whole_program_ad_trace_to_executable": (
        "scpn_quantum_control.compiler.mlir",
        "compile_whole_program_ad_trace_to_executable",
    ),
    "compile_whole_program_ad_trace_to_mlir": (
        "scpn_quantum_control.compiler.mlir",
        "compile_whole_program_ad_trace_to_mlir",
    ),
    "compile_whole_program_ad_trace_to_native_llvm_jit": (
        "scpn_quantum_control.compiler.mlir",
        "compile_whole_program_ad_trace_to_native_llvm_jit",
    ),
    "llvm_jit_claim_gate_from_dict": (
        "scpn_quantum_control.compiler.mlir",
        "llvm_jit_claim_gate_from_dict",
    ),
    "make_executable_ad_kernel_batching_rule": (
        "scpn_quantum_control.compiler.mlir",
        "make_executable_ad_kernel_batching_rule",
    ),
    "make_matrix_2x2_determinant_native_llvm_jit_lowering_rule": (
        "scpn_quantum_control.compiler.mlir",
        "make_matrix_2x2_determinant_native_llvm_jit_lowering_rule",
    ),
    "make_matrix_2x2_eigensystem_native_llvm_jit_lowering_rule": (
        "scpn_quantum_control.compiler.mlir",
        "make_matrix_2x2_eigensystem_native_llvm_jit_lowering_rule",
    ),
    "make_matrix_2x2_eigensystem_native_llvm_jit_primitive_transform": (
        "scpn_quantum_control.compiler.mlir",
        "make_matrix_2x2_eigensystem_native_llvm_jit_primitive_transform",
    ),
    "make_matrix_2x2_eigenvalues_native_llvm_jit_lowering_rule": (
        "scpn_quantum_control.compiler.mlir",
        "make_matrix_2x2_eigenvalues_native_llvm_jit_lowering_rule",
    ),
    "make_matrix_2x2_inverse_native_llvm_jit_lowering_rule": (
        "scpn_quantum_control.compiler.mlir",
        "make_matrix_2x2_inverse_native_llvm_jit_lowering_rule",
    ),
    "make_matrix_2x2_solve_native_llvm_jit_lowering_rule": (
        "scpn_quantum_control.compiler.mlir",
        "make_matrix_2x2_solve_native_llvm_jit_lowering_rule",
    ),
    "make_matrix_frobenius_norm_squared_native_llvm_jit_lowering_rule": (
        "scpn_quantum_control.compiler.mlir",
        "make_matrix_frobenius_norm_squared_native_llvm_jit_lowering_rule",
    ),
    "make_matrix_matrix_product_native_llvm_jit_lowering_rule": (
        "scpn_quantum_control.compiler.mlir",
        "make_matrix_matrix_product_native_llvm_jit_lowering_rule",
    ),
    "make_matrix_quadratic_form_native_llvm_jit_lowering_rule": (
        "scpn_quantum_control.compiler.mlir",
        "make_matrix_quadratic_form_native_llvm_jit_lowering_rule",
    ),
    "make_matrix_trace_native_llvm_jit_lowering_rule": (
        "scpn_quantum_control.compiler.mlir",
        "make_matrix_trace_native_llvm_jit_lowering_rule",
    ),
    "make_matrix_vector_product_native_llvm_jit_lowering_rule": (
        "scpn_quantum_control.compiler.mlir",
        "make_matrix_vector_product_native_llvm_jit_lowering_rule",
    ),
    "make_scalar_binary_elementwise_native_llvm_jit_lowering_rule": (
        "scpn_quantum_control.compiler.mlir",
        "make_scalar_binary_elementwise_native_llvm_jit_lowering_rule",
    ),
    "make_scalar_quadratic_native_llvm_jit_lowering_rule": (
        "scpn_quantum_control.compiler.mlir",
        "make_scalar_quadratic_native_llvm_jit_lowering_rule",
    ),
    "make_scalar_unary_elementwise_native_llvm_jit_lowering_rule": (
        "scpn_quantum_control.compiler.mlir",
        "make_scalar_unary_elementwise_native_llvm_jit_lowering_rule",
    ),
    "make_symmetric_2x2_cholesky_native_llvm_jit_lowering_rule": (
        "scpn_quantum_control.compiler.mlir",
        "make_symmetric_2x2_cholesky_native_llvm_jit_lowering_rule",
    ),
    "make_symmetric_2x2_eigenvalues_native_llvm_jit_lowering_rule": (
        "scpn_quantum_control.compiler.mlir",
        "make_symmetric_2x2_eigenvalues_native_llvm_jit_lowering_rule",
    ),
    "make_vector_dot_native_llvm_jit_lowering_rule": (
        "scpn_quantum_control.compiler.mlir",
        "make_vector_dot_native_llvm_jit_lowering_rule",
    ),
    "make_vector_squared_norm_native_llvm_jit_lowering_rule": (
        "scpn_quantum_control.compiler.mlir",
        "make_vector_squared_norm_native_llvm_jit_lowering_rule",
    ),
    "native_whole_program_ad_compile_cache_stats": (
        "scpn_quantum_control.compiler.mlir",
        "native_whole_program_ad_compile_cache_stats",
    ),
    "native_whole_program_ad_linalg_support": (
        "scpn_quantum_control.compiler.mlir",
        "native_whole_program_ad_linalg_support",
    ),
    "render_enzyme_mlir_compiler_ad_breadth_artifact_markdown": (
        "scpn_quantum_control.compiler.mlir",
        "render_enzyme_mlir_compiler_ad_breadth_artifact_markdown",
    ),
    "render_llvm_jit_claim_gate_markdown": (
        "scpn_quantum_control.compiler.mlir",
        "render_llvm_jit_claim_gate_markdown",
    ),
    "run_enzyme_mlir_maturity_audit": (
        "scpn_quantum_control.compiler.mlir",
        "run_enzyme_mlir_maturity_audit",
    ),
    "run_enzyme_toolchain_execution_evidence": (
        "scpn_quantum_control.compiler.mlir",
        "run_enzyme_toolchain_execution_evidence",
    ),
    "run_native_whole_program_ad_execution_evidence": (
        "scpn_quantum_control.compiler.mlir",
        "run_native_whole_program_ad_execution_evidence",
    ),
    "write_enzyme_mlir_compiler_ad_breadth_artifact": (
        "scpn_quantum_control.compiler.mlir",
        "write_enzyme_mlir_compiler_ad_breadth_artifact",
    ),
    "COMPILER_PROMOTION_BATCH_ID": (
        "scpn_quantum_control.compiler.promotion_batch",
        "COMPILER_PROMOTION_BATCH_ID",
    ),
    "COMPILER_PROMOTION_BATCH_SCHEMA": (
        "scpn_quantum_control.compiler.promotion_batch",
        "COMPILER_PROMOTION_BATCH_SCHEMA",
    ),
    "CompilerPromotionBatch": (
        "scpn_quantum_control.compiler.promotion_batch",
        "CompilerPromotionBatch",
    ),
    "CompilerPromotionBatchEvidenceFile": (
        "scpn_quantum_control.compiler.promotion_batch",
        "CompilerPromotionBatchEvidenceFile",
    ),
    "build_compiler_promotion_batch": (
        "scpn_quantum_control.compiler.promotion_batch",
        "build_compiler_promotion_batch",
    ),
    "render_compiler_promotion_batch_markdown": (
        "scpn_quantum_control.compiler.promotion_batch",
        "render_compiler_promotion_batch_markdown",
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
    "CompilerADExecutableConfig",
    "CompilerADKernelVerification",
    "CompilerADTransformPlan",
    "COMPILER_ALIAS_ACTIVITY_EVIDENCE_ID",
    "COMPILER_ALIAS_ACTIVITY_EVIDENCE_SCHEMA",
    "COMPILER_PROMOTION_BATCH_ID",
    "COMPILER_PROMOTION_BATCH_SCHEMA",
    "CompilerAliasActivityCase",
    "CompilerAliasActivityEvidence",
    "CompilerPromotionBatch",
    "CompilerPromotionBatchEvidenceFile",
    "DifferentiableMLIRCompileConfig",
    "EnzymeMLIRBenchmarkAttachment",
    "EnzymeMLIRCompilerADBreadthArtifact",
    "EnzymeMLIRCompilerADBreadthArtifactFiles",
    "EnzymeMLIRCompilerADBreadthCaseEvidence",
    "EnzymeMLIRCompilerADBreadthEvidence",
    "EnzymeMLIRMaturityAuditResult",
    "EnzymeMLIRToolchainStatus",
    "EnzymeNativeExecutionEvidence",
    "EnzymeToolchainADCase",
    "EnzymeToolchainADExecutionEvidence",
    "ExecutableCompilerADKernel",
    "ExecutableWholeProgramADBatchResult",
    "ExecutableWholeProgramADKernel",
    "LLVM_JIT_CLAIM_GATE_BOUNDARY",
    "LLVMJITClaimGate",
    "MLIRLLVMCorrectnessEvidence",
    "MLIRCompileConfig",
    "NativeWholeProgramADExecutionCase",
    "NativeWholeProgramADExecutionEvidence",
    "NativeWholeProgramADKernel",
    "PhaseQNodeMLIRRuntimeExecutable",
    "PrimitiveLoweringStatus",
    "WholeProgramADNativeLoweringReport",
    "MLIRModule",
    "analyse_whole_program_ad_native_lowering",
    "build_compiler_alias_activity_evidence",
    "build_compiler_promotion_batch",
    "build_llvm_jit_claim_gate",
    "build_enzyme_mlir_benchmark_attachment",
    "build_enzyme_mlir_compiler_ad_breadth_artifact",
    "build_enzyme_mlir_compiler_ad_breadth_evidence",
    "build_enzyme_mlir_compiler_ad_breadth_gap_artifact",
    "build_native_whole_program_ad_execution_evidence",
    "build_compiler_ad_transform_plan",
    "compile_compiler_ad_transform_plan_to_mlir",
    "compile_custom_derivative_rule_to_executable",
    "compile_custom_derivative_rule_to_mlir",
    "compile_phase_qnode_circuit_to_mlir_runtime",
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
    "compile_registered_primitive_to_executable",
    "compile_scalar_binary_elementwise_ad_to_native_llvm_jit",
    "compile_scalar_quadratic_ad_to_native_llvm_jit",
    "compile_scalar_unary_elementwise_ad_to_native_llvm_jit",
    "compile_vector_dot_ad_to_native_llvm_jit",
    "compile_vector_squared_norm_ad_to_native_llvm_jit",
    "compile_symmetric_2x2_cholesky_ad_to_native_llvm_jit",
    "compile_symmetric_2x2_eigenvalues_ad_to_native_llvm_jit",
    "compile_whole_program_ad_trace_to_executable",
    "compile_whole_program_ad_trace_to_native_llvm_jit",
    "compile_whole_program_ad_trace_to_mlir",
    "clear_native_whole_program_ad_compile_cache",
    "compile_kuramoto_to_mlir",
    "llvm_jit_claim_gate_from_dict",
    "make_executable_ad_kernel_batching_rule",
    "make_matrix_2x2_determinant_native_llvm_jit_lowering_rule",
    "make_matrix_2x2_eigenvalues_native_llvm_jit_lowering_rule",
    "make_matrix_2x2_eigensystem_native_llvm_jit_lowering_rule",
    "make_matrix_2x2_eigensystem_native_llvm_jit_primitive_transform",
    "make_matrix_2x2_inverse_native_llvm_jit_lowering_rule",
    "make_matrix_2x2_solve_native_llvm_jit_lowering_rule",
    "make_matrix_frobenius_norm_squared_native_llvm_jit_lowering_rule",
    "make_matrix_matrix_product_native_llvm_jit_lowering_rule",
    "make_matrix_quadratic_form_native_llvm_jit_lowering_rule",
    "make_matrix_trace_native_llvm_jit_lowering_rule",
    "make_matrix_vector_product_native_llvm_jit_lowering_rule",
    "make_scalar_binary_elementwise_native_llvm_jit_lowering_rule",
    "make_scalar_quadratic_native_llvm_jit_lowering_rule",
    "make_scalar_unary_elementwise_native_llvm_jit_lowering_rule",
    "make_symmetric_2x2_cholesky_native_llvm_jit_lowering_rule",
    "make_symmetric_2x2_eigenvalues_native_llvm_jit_lowering_rule",
    "make_vector_dot_native_llvm_jit_lowering_rule",
    "make_vector_squared_norm_native_llvm_jit_lowering_rule",
    "native_whole_program_ad_compile_cache_stats",
    "native_whole_program_ad_linalg_support",
    "render_compiler_alias_activity_evidence_markdown",
    "render_compiler_promotion_batch_markdown",
    "render_enzyme_mlir_compiler_ad_breadth_artifact_markdown",
    "render_llvm_jit_claim_gate_markdown",
    "run_enzyme_mlir_maturity_audit",
    "run_enzyme_toolchain_execution_evidence",
    "run_native_whole_program_ad_execution_evidence",
    "write_enzyme_mlir_compiler_ad_breadth_artifact",
]
