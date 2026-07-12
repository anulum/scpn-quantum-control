# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable dashboard module
# scpn-quantum-control -- differentiable dashboard catalog
"""Claim-bounded capability catalog for differentiable dashboard consumers."""

from __future__ import annotations

from .differentiable_api_contracts import (
    CLAIM_BOUNDARY,
    DifferentiableDashboardCapabilityRow,
    DifferentiableDashboardStatus,
)
from .differentiable_benchmark_report import build_differentiable_benchmark_report
from .program_ad_registry import program_ad_registry_dispatch_coverage_report


def differentiable_dashboard_status(
    *,
    include_conformance: bool = False,
) -> DifferentiableDashboardStatus:
    """Return the claim-bounded status contract for a future GUI/dashboard."""
    conformance_passed: bool | None = None
    if include_conformance:
        conformance_passed = build_differentiable_benchmark_report().supported

    rows = [
        DifferentiableDashboardCapabilityRow(
            surface="unified_differentiable_api",
            state="executable",
            backing_api="differentiable_api",
            evidence=(
                "UnifiedDifferentiableAPIResult",
                "differentiable_value",
                "differentiable_gradient",
                "differentiable_jacobian",
                "differentiable_hessian",
            ),
            blocked_reasons=(),
            claim_boundary=CLAIM_BOUNDARY,
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_ir",
            state="metadata_only",
            backing_api="whole_program_value_and_grad",
            evidence=("ProgramADEffectIR", "WholeProgramADResult.program_ir"),
            blocked_reasons=("executable compiler lowering remains open",),
            claim_boundary=(
                "metadata and executed-trace evidence only; static frontend "
                "preflight is separate and not full arbitrary Python AD"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_bytecode_source_frontend",
            state="diagnostic",
            backing_api="compile_whole_program_frontend",
            evidence=(
                "WholeProgramCompilerFrontendReport",
                "WholeProgramBytecodeBasicBlock",
                "WholeProgramBytecodeInstruction",
                "WholeProgramSourceIRFeature",
                "WholeProgramSourceRegion",
                "WholeProgramSourceBytecodeLineMap",
                "WholeProgramSymbolScopeEntry",
                "WholeProgramSemanticsReport",
                "WholeProgramUnsupportedSemanticDiagnostic",
            ),
            blocked_reasons=(
                "static bytecode basic blocks, source regions, source-bytecode maps, "
                "symbol scopes, and unsupported-semantics diagnostics only; executable Rust, "
                "LLVM, JIT, provider, hardware, and benchmark promotion remain open",
            ),
            claim_boundary=(
                "first-class static bytecode/source compiler frontend preflight "
                "with bytecode basic blocks, source regions, normalized "
                "source-bytecode maps, symbol scopes, and unsupported-semantics "
                "diagnostics for supported "
                "Program AD Python semantics; no objective execution, no "
                "executable compiler lowering, no Rust/LLVM/JIT, provider, "
                "hardware, or performance claim"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_ir_roundtrip",
            state="metadata_only",
            backing_api="parse_program_ad_effect_ir",
            evidence=("parse_program_ad_effect_ir", "program_ad_effect_ir.v1"),
            blocked_reasons=("parser is metadata-only and not executable compiler lowering",),
            claim_boundary=(
                "bounded Program AD IR JSON metadata round-trip only; not a full "
                "compiler frontend, alias lattice, Rust interpreter, LLVM/JIT "
                "lowering, provider, or hardware evidence"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_ir_roundtrip_conformance",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_differentiable_programming_benchmark_suite",
            evidence=(
                "parse_program_ad_effect_ir",
                "program_ad_effect_ir.v1",
                "program_ad_ir_roundtrip_contracts",
            ),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=(
                "bounded program_ad_effect_ir.v1 parser and stable serialization "
                "round-trip conformance only; not executable compiler lowering, "
                "full alias lattice, non-executed branch semantics, "
                "Rust/LLVM executable lowering, hardware, or performance promotion"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_control_phi_metadata",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_differentiable_programming_benchmark_suite",
            evidence=(
                "ProgramADControlRegion",
                "ProgramADPhiNode",
                "program_ad_control_phi_metadata_contracts",
            ),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=(
                "bounded ProgramADPhiNode control-join provenance for supported "
                "executed runtime and source control regions only; not "
                "non-executed branch adjoints, full compiler phi lowering, "
                "Rust/LLVM executable lowering, hardware, or performance promotion"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_rust_scalar_interpreter",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_differentiable_programming_benchmark_suite",
            evidence=(
                "program_ad_effect_ir_interpret_forward",
                "interpret_program_ad_effect_ir_with_rust",
                "program_ad_rust_scalar_interpreter_contracts",
            ),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=(
                "bounded Rust scalar forward interpreter for opcode-bearing "
                "program_ad_effect_ir.v1 rows only; not reverse-mode Rust AD, "
                "general Program AD execution, LLVM/JIT lowering, provider, "
                "hardware, or performance promotion"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_mlir_interchange",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="compile_whole_program_ad_trace_to_mlir",
            evidence=(
                "program_ad_effect_ir.v1",
                "scpn_diff.program_ad_ssa",
                "scpn_diff.program_ad_effect",
                "scpn_diff.program_ad_phi",
                "program_ad_mlir_interchange_contracts",
            ),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=(
                "bounded Program AD IR MLIR dialect interchange lowering for "
                "captured SSA/effect/control/phi metadata only; no executable "
                "Rust, LLVM, or JIT differentiated runtime, hardware, provider, "
                "or performance promotion"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_alias_effects",
            state="metadata_only",
            backing_api="analyze_program_ad_alias_effects; program_ad_static_alias_lattice_report",
            evidence=(
                "ProgramADAliasEffectAnalysis",
                "ProgramADAliasSet",
                "ProgramADStaticAliasLatticeReport",
                "ProgramADUnknownAliasEdge",
                "ProgramADControlPathAliasProvenance",
                "ProgramADViewAliasProvenance",
                "ProgramADListAliasProvenance",
                "ProgramADLoopCarriedStateProvenance",
                "ProgramADRebindingAliasProvenance",
                "typed view-alias provenance metadata",
                "typed list-alias provenance metadata",
                "typed loop-carried state provenance metadata",
                "typed control-path alias provenance metadata",
                "typed rebinding-alias provenance metadata",
                "bounded local object-attribute alias metadata",
                "expression-rebinding alias metadata",
                "mutation-effect blocker metadata",
                "control-path alias blocker metadata",
                "unsupported-Python frontend diagnostic metadata",
                "captured/global object-attribute blocker metadata",
                "unknown alias-edge provenance blocker metadata",
                "shape_view_alias_metadata_contracts",
                "slice_mutation_alias_metadata_contracts",
                "loop_carried_state_alias_metadata_contracts",
                "program_ad_static_alias_lattice_contracts",
            ),
            blocked_reasons=(
                "captured/global object attributes require static object-model alias sets",
                "mutation effects require versioned alias semantics",
                "malformed view aliases require parseable source-to-view provenance",
                "malformed list aliases require parseable local-list provenance",
                "malformed loop-carried state aliases require parseable loop provenance",
                "malformed control-path aliases require parseable branch-local provenance",
                "control-path aliases require non-executed branch semantics",
                "unsupported Python semantics require executable frontend lowering",
                "unknown alias edges require static alias-kind support",
                "non-executed branch semantics remain blocked",
            ),
            claim_boundary=(
                "static alias-lattice readiness over emitted Program AD IR metadata; "
                "typed view-alias, list-alias, loop-carried state, and control-path provenance "
                "plus bounded local object-attribute and expression-rebinding aliases are "
                "metadata only; mutation effects, malformed view/list/loop/control-path aliases, "
                "unsupported "
                "Python semantics with source/region/bytecode diagnostics, and captured/global "
                "object-attribute diagnostics are explicit blockers; unknown alias-edge "
                "provenance is an explicit blocker, "
                "not promoted static object-model alias sets, mutation, branch, or "
                "unknown dynamic alias promotion, or arbitrary dynamic-Python frontend "
                "lowering"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_registry_dispatch_coverage",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="program_ad_registry_dispatch_coverage_report",
            evidence=(
                "ProgramADRegistryDispatchCoverageReport",
                "ProgramADRegistryDispatchCoverageRow",
                "program_ad_registry_dispatch_contracts",
            ),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=program_ad_registry_dispatch_coverage_report().claim_boundary,
        ),
        DifferentiableDashboardCapabilityRow(
            surface="jax_phase_qnode_aot_export_lowering",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="jax_phase_qnode_aot_export_audit",
            evidence=(
                "PhaseJAXPhaseQNodeAOTExportResult",
                "registered_phase_qnode_aot_export_lowering",
                "jax_registered_phase_qnode_aot_export_lowering",
            ),
            blocked_reasons=()
            if conformance_passed
            else (
                "conformance suite not run in this status call",
                "exported VJP and persistent cross-platform execution remain blocked",
                "provider, hardware, dynamic-circuit, and isolated benchmark promotion remain blocked",
            ),
            claim_boundary=(
                "native JAX AOT lowering and jax.export serialization diagnostics "
                "for deterministic registered local Phase-QNode value routes only; "
                "gradient fields remain parameter-shift references, with no exported "
                "VJP, persistent cross-platform execution, provider, hardware, "
                "isolated benchmark, or performance promotion"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="torch_phase_qnode_statevector_lowering",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="torch_phase_qnode_value_and_grad",
            evidence=(
                "PhaseTorchPhaseQNodeStatevectorResult",
                "registered_phase_qnode_torch_statevector_lowering",
                "torch_registered_phase_qnode_statevector_lowering",
            ),
            blocked_reasons=()
            if conformance_passed
            else (
                "conformance suite not run in this status call",
                "finite-shot Torch lowering remains blocked",
                "provider, hardware, dynamic-circuit, and isolated benchmark promotion remain blocked",
            ),
            claim_boundary=(
                "native PyTorch autograd statevector lowering for deterministic "
                "registered local Phase-QNode circuits only; no provider, "
                "hardware, isolated benchmark, or performance promotion"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="torch_phase_qnode_compile_boundary_diagnostic",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="torch_phase_qnode_compile_boundary_audit",
            evidence=(
                "PhaseTorchCompileBoundaryAuditResult",
                "registered_phase_qnode_torch_compile_boundary_diagnostic",
                "torch_registered_phase_qnode_compile_boundary_diagnostic",
            ),
            blocked_reasons=()
            if conformance_passed
            else (
                "conformance suite not run in this status call",
                "dynamic-shape compile promotion remains blocked",
                "fullgraph compile promotion remains blocked",
                "AOTAutograd/export persistent artifact remains blocked",
                "provider, hardware, CUDA, isolated benchmark, and performance promotion remain blocked",
            ),
            claim_boundary=(
                "native PyTorch compile-boundary diagnostic for deterministic "
                "registered local Phase-QNode circuits only; non-fullgraph CPU "
                "execution is compared with SCPN parameter-shift references, "
                "while dynamic-shape, fullgraph compiled-frame, AOTAutograd/export, "
                "provider, hardware, CUDA, isolated benchmark, and performance "
                "routes remain blocked with no persistent export or performance claim"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="torch_bounded_qnn_autograd_function_audit",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_torch_autograd_function_audit",
            evidence=(
                "PhaseTorchAutogradFunctionResult",
                "torch_autograd_function_qnn_loss",
                "Tensor.backward parameter-gradient parity",
                "torch.optim.SGD custom-backward integration",
            ),
            blocked_reasons=()
            if conformance_passed
            else (
                "conformance suite not run in this status call",
                "higher-order autograd graph transformation remains blocked",
                "CUDA custom-autograd execution remains blocked",
                "provider, hardware, arbitrary-simulator, isolated benchmark, and performance promotion remain blocked",
            ),
            claim_boundary=(
                "bounded PyTorch torch.autograd.Function route for the local "
                "phase-QNN classifier loss only; Tensor.backward and SGD "
                "optimizer integration are checked against SCPN "
                "parameter-shift references with no higher-order autograd, "
                "CUDA, provider, hardware, arbitrary-simulator, isolated "
                "benchmark, or performance claim"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="torch_bounded_qnn_module_state_audit",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_torch_module_state_audit",
            evidence=(
                "PhaseTorchModuleStateAuditResult",
                "PhaseTorchModuleStateValidationResult",
                "strict module state_dict round-trip",
                "Adam optimizer state_dict round-trip",
            ),
            blocked_reasons=()
            if conformance_passed
            else (
                "conformance suite not run in this status call",
                "CUDA device transfer and cross-runtime checkpoint portability remain blocked",
                "provider, hardware, CUDA, isolated benchmark, and performance promotion remain blocked",
            ),
            claim_boundary=(
                "bounded PyTorch module-state audit for the phase-QNN nn.Module "
                "route only; strict state_dict and Adam optimizer-state replay "
                "are local CPU-compatible evidence, while CUDA device transfer, "
                "cross-runtime checkpoint portability, provider, hardware, CUDA, "
                "isolated benchmark, and performance promotion remain blocked"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="torch_bounded_qnn_module_device_state_audit",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_torch_module_device_state_audit",
            evidence=(
                "PhaseTorchDeviceStateAuditResult",
                "module.to('cpu') state_dict round-trip",
                "CUDA smoke-gated module.to('cuda') classification",
            ),
            blocked_reasons=()
            if conformance_passed
            else (
                "conformance suite not run in this status call",
                "CUDA transfer remains blocked when the local PyTorch runtime cannot execute a CUDA smoke",
                "cross-runtime checkpoint portability, provider, hardware, isolated benchmark, and performance promotion remain blocked",
            ),
            claim_boundary=(
                "bounded PyTorch device-state audit for the phase-QNN nn.Module "
                "route only; CPU module.to(...) state replay is local functional "
                "evidence and CUDA replay is smoke-gated device evidence, while "
                "cross-runtime checkpoint portability, provider, hardware, "
                "isolated benchmark, and performance promotion remain blocked"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="torch_bounded_qnn_module_checkpoint_audit",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_torch_module_checkpoint_audit",
            evidence=(
                "PhaseTorchCheckpointAuditResult",
                "torch.save checkpoint round-trip",
                "torch.load(weights_only=True, map_location='cpu') replay",
                "Adam optimizer checkpoint replay",
            ),
            blocked_reasons=()
            if conformance_passed
            else (
                "conformance suite not run in this status call",
                "cross-runtime checkpoint portability remains blocked",
                "CUDA, provider, hardware, isolated benchmark, and performance promotion remain blocked",
            ),
            claim_boundary=(
                "bounded PyTorch checkpoint audit for the phase-QNN nn.Module "
                "route only; torch.save checkpoints are reloaded on CPU with "
                "weights_only=True and replayed through strict module plus Adam "
                "optimizer state loading, while cross-runtime portability, CUDA, "
                "provider, hardware, isolated benchmark, and performance "
                "promotion remain blocked"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="torch_bounded_qnn_long_lived_checkpoint_matrix",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_torch_long_lived_checkpoint_matrix",
            evidence=(
                "PhaseTorchCheckpointMatrixResult",
                "checkpoint tensor metadata manifest",
                "runtime fingerprint",
                "repeated weights_only CPU checkpoint loads",
            ),
            blocked_reasons=()
            if conformance_passed
            else (
                "conformance suite not run in this status call",
                "cross-runtime checkpoint replay remains blocked",
                "CUDA checkpoint replay remains blocked",
                "long-lived external checkpoint artifact remains blocked",
                "provider, hardware, isolated benchmark, and performance promotion remain blocked",
            ),
            claim_boundary=(
                "bounded PyTorch long-lived checkpoint matrix for the phase-QNN "
                "nn.Module route only; schema, tensor metadata, runtime "
                "fingerprint, and repeated local CPU weights_only loads are "
                "recorded with no cross-runtime, CUDA, provider, hardware, "
                "external long-lived artefact, isolated benchmark, or "
                "performance claim"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="torch_bounded_qnn_training_loop_matrix",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_torch_training_loop_matrix",
            evidence=(
                "PhaseTorchTrainingLoopMatrixResult",
                "multi-scenario training-loop parity",
                "loss descent and parameter-update records",
                "fullgraph/static and non-fullgraph dynamic-request compile rows",
            ),
            blocked_reasons=()
            if conformance_passed
            else (
                "conformance suite not run in this status call",
                "CUDA training-loop execution remains blocked",
                "provider and hardware training-loop execution remain blocked",
                "isolated benchmark and performance promotion remain blocked",
                "arbitrary QNN architecture training loops remain blocked",
            ),
            claim_boundary=(
                "bounded PyTorch training-loop matrix for the phase-QNN nn.Module "
                "route only; deterministic local CPU scenarios record loss "
                "descent, parameter updates, compile-mode coverage, and gradient "
                "parity against SCPN parameter-shift references with no CUDA, "
                "provider, hardware, arbitrary-architecture, isolated benchmark, "
                "or performance claim"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="torch_bounded_qnn_module_export_audit",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_torch_module_export_audit",
            evidence=(
                "PhaseTorchExportAuditResult",
                "torch.export.export ExportedProgram",
                "torch.export.save/load local value replay",
                "ExportedProgram graph signature",
            ),
            blocked_reasons=()
            if conformance_passed
            else (
                "conformance suite not run in this status call",
                "AOTAutograd gradient-export persistence remains blocked",
                "dynamic-shape export remains blocked",
                "CUDA, provider, hardware, isolated benchmark, and performance promotion remain blocked",
            ),
            claim_boundary=(
                "bounded PyTorch export audit for the phase-QNN nn.Module route "
                "only; torch.export.export, torch.export.save, and "
                "torch.export.load are checked for a local CPU value route, while "
                "AOTAutograd gradient export, dynamic shapes, CUDA, provider, "
                "hardware, isolated benchmark, cross-runtime deployment, and "
                "performance promotion remain blocked"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="torch_bounded_qnn_export_shape_matrix",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_torch_export_shape_matrix",
            evidence=(
                "PhaseTorchExportShapeMatrixResult",
                "multi-static-shape torch.export replay",
                "per-shape ExportedProgram artifact metadata",
                "dynamic-shape blocker routes",
            ),
            blocked_reasons=()
            if conformance_passed
            else (
                "conformance suite not run in this status call",
                "dynamic-shape export constraints remain blocked",
                "dynamic-shape export replay remains blocked",
                "AOTAutograd gradient-export persistence remains blocked",
                "CUDA, provider, hardware, isolated benchmark, and performance promotion remain blocked",
            ),
            claim_boundary=(
                "bounded PyTorch static-shape export matrix for the phase-QNN "
                "nn.Module route only; multiple static feature shapes are "
                "exported, saved, loaded, and replayed on a local CPU value "
                "route with no dynamic-shape, AOTAutograd gradient-export, "
                "CUDA, provider, hardware, isolated benchmark, cross-runtime "
                "deployment, or performance claim"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="torch_bounded_qnn_dynamic_shape_export_audit",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_torch_dynamic_shape_export_audit",
            evidence=(
                "PhaseTorchDynamicShapeExportResult",
                "input-driven torch.export dynamic batch constraints",
                "single ExportedProgram replayed across batch sizes",
                "torch.export.save/load local dynamic-batch value replay",
            ),
            blocked_reasons=()
            if conformance_passed
            else (
                "conformance suite not run in this status call",
                "AOTAutograd gradient-export persistence remains blocked",
                "CUDA dynamic-shape export replay remains blocked",
                "cross-runtime dynamic-shape export replay remains blocked",
                "dynamic feature-width export remains blocked",
                "isolated benchmark and performance promotion remain blocked",
            ),
            claim_boundary=(
                "bounded PyTorch dynamic batch torch.export audit for the "
                "phase-QNN nn.Module route only; one input-driven ExportedProgram "
                "is exported with symbolic batch constraints, saved, loaded, and "
                "replayed on a local CPU value route across multiple batch sizes "
                "with no dynamic feature-width, AOTAutograd gradient-export, CUDA, "
                "provider, hardware, isolated benchmark, cross-runtime deployment, "
                "or performance claim"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="torch_bounded_qnn_aot_autograd_export_audit",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_torch_aot_autograd_export_audit",
            evidence=(
                "PhaseTorchAOTAutogradExportResult",
                "persisted AOTAutograd forward/backward FX graphs",
                "loaded backward FX graph gradient replay",
                "SCPN parameter-shift gradient agreement",
            ),
            blocked_reasons=()
            if conformance_passed
            else (
                "conformance suite not run in this status call",
                "cross-runtime AOTAutograd execution remains blocked",
                "CUDA AOTAutograd execution remains blocked",
                "dynamic-shape AOTAutograd export remains blocked",
                "isolated benchmark and performance promotion remain blocked",
            ),
            claim_boundary=(
                "bounded PyTorch AOTAutograd FX graph persistence audit for the "
                "local phase-QNN loss route only; forward/backward FX "
                "GraphModules are saved, loaded, and replayed locally against "
                "SCPN parameter-shift references with no stable cross-runtime "
                "export, CUDA, provider, hardware, dynamic-shape, isolated "
                "benchmark, or performance claim"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_python_semantics",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="whole_program_value_and_grad",
            evidence=(
                "WholeProgramSemanticsReport",
                "python_semantics_list_comprehension",
            ),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=(
                "bounded plain list-comprehension whole-program AD semantics only; "
                "filtered, set, and dict comprehensions remain fail-closed; no "
                "compiler, Rust, LLVM, JIT, hardware, or performance claim"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_reverse_adjoint_replay",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_differentiable_programming_benchmark_suite",
            evidence=(
                "ProgramADAdjointResult",
                "ProgramADAdjointStep",
                "WholeProgramADResult replay contract",
                "program_adjoint_result",
                "program_adjoint_gradient",
                "program_adjoint_replay_provenance_contracts",
            ),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=(
                "bounded Program AD reverse adjoint generation over stabilized "
                "program_ad_effect_ir.v1 for supported executed scalar IR only, "
                "with generated adjoint steps, finite local pullback scales, "
                "cotangent-flow rows, reverse effect-order rows, and "
                "node/effect/runtime control/phi row bindings, with blocked "
                "non-executed phi inputs, validated at WholeProgramADResult "
                "construction and exposed through caller-visible replay; not "
                "full reverse-mode compiler AD, non-executed branch adjoints, "
                "Rust/LLVM executable lowering, hardware, or performance "
                "promotion"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_elementwise_primitives",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_differentiable_programming_benchmark_suite",
            evidence=(
                "scpn.program_ad.elementwise:{abs,absolute,log,sqrt,reciprocal,log1p,arcsin,arccos}",
                "elementwise_boundary_contracts",
            ),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=(
                "bounded Program AD elementwise primitive conformance for "
                "zero-cusp absolute-value, positive-domain, nonzero-denominator, "
                "and inverse-trig boundary contracts only; unsupported domain "
                "boundaries, derivative-losing sign/heaviside kernels, Rust/LLVM "
                "executable lowering, hardware, and performance promotion remain blocked"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_array_indexing",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_differentiable_programming_benchmark_suite",
            evidence=(
                "scpn.program_ad.array:{getitem,take,take_along_axis,delete,pad,insert}",
                "indexing_static_gather_contracts",
            ),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=(
                "bounded static array indexing, gather, delete, constant-pad, and "
                "constant-insert Program AD semantics only; dynamic indices, "
                "dynamic insertion values, Rust, LLVM/JIT, hardware, and "
                "performance promotion remain blocked"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_linalg_primitives",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_differentiable_programming_benchmark_suite",
            evidence=(
                "scpn.program_ad.linalg:{trace,diag,diagflat,det,inv,solve,matrix_power,multi_dot}",
                "linalg_primitive_contracts",
            ),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=(
                "bounded local Program AD linalg primitive conformance only; "
                "spectral multiplicity, rank-threshold crossings, wider native "
                "LLVM/JIT kernels, Rust interpreter promotion, hardware, and "
                "performance promotion remain blocked"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_structured_primitives",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_differentiable_programming_benchmark_suite",
            evidence=(
                "scpn.program_ad.product:{inner,outer,matmul,tensordot,einsum}",
                "scpn.program_ad.interpolation:interp",
                "scpn.program_ad.signal:{convolve,correlate}",
                "scpn.program_ad.stencil:gradient",
                "structured_numeric_primitive_contracts",
            ),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=(
                "structured numeric Program AD primitive conformance only; "
                "bounded static-grid interp, static rank-1 signal, and static stencil "
                "gradient Program AD Rust replay are covered; dynamic interpolation grids, "
                "dynamic signal metadata, singular stencil spacing, broad Rust/LLVM/JIT "
                "lowering, hardware, and performance promotion remain blocked"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_cumulative_primitives",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_differentiable_programming_benchmark_suite",
            evidence=(
                "scpn.program_ad.cumulative:{cumsum,cumprod,diff}",
                "cumulative_primitive_contracts",
            ),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=(
                "bounded cumsum, cumprod, and diff Program AD primitive "
                "conformance only; dynamic axis promotion, Rust/LLVM executable "
                "lowering, hardware, and performance promotion remain blocked"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_assembly_primitives",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_differentiable_programming_benchmark_suite",
            evidence=(
                "scpn.program_ad.assembly:{zeros_like,ones_like,full_like,hstack,vstack,column_stack,dstack}",
                "assembly_primitive_contracts",
            ),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=(
                "bounded like-constructor and stack assembly Program AD "
                "primitive conformance only; dynamic shape assembly, "
                "Rust/LLVM executable lowering, hardware, and performance "
                "promotion remain blocked"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_reduction_primitives",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_differentiable_programming_benchmark_suite",
            evidence=(
                "scpn.program_ad.reduction:{sum,prod,mean,var,std,trapezoid,max,min,median,quantile,percentile}",
                "reduction_primitive_contracts",
            ),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=(
                "bounded Program AD reduction primitive conformance only; dynamic "
                "axes, dynamic q, strict-order selectors, zero-variance standard "
                "deviation, Rust/LLVM executable lowering, hardware, and "
                "performance promotion remain blocked; scalar q order-statistics "
                "are covered only under local deterministic conformance"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_shape_primitives",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_differentiable_programming_benchmark_suite",
            evidence=(
                "scpn.program_ad.shape:{reshape,ravel,transpose,expand_dims,squeeze,swapaxes,moveaxis,repeat,atleast_1d,atleast_2d,atleast_3d,tile,roll,rot90,flip,flipud,fliplr}",
                "shape_primitive_contracts",
            ),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=(
                "bounded Program AD shape primitive conformance only; dynamic "
                "shape arguments, invalid axes, Rust/LLVM executable lowering, "
                "hardware, and performance promotion remain blocked; rank "
                "promotion is covered only under local deterministic conformance"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_broadcast_primitives",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_differentiable_programming_benchmark_suite",
            evidence=(
                "scpn.program_ad.assembly:{broadcast_to,broadcast_arrays}",
                "scpn.program_ad.elementwise:binary_rank_broadcasting",
                "broadcast_primitive_contracts",
            ),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=(
                "bounded Program AD broadcast primitive conformance only; "
                "dynamic output shapes, incompatible shapes, subok propagation, "
                "Rust/LLVM executable lowering, hardware, and performance "
                "promotion remain blocked; rank broadcasting is covered only "
                "under local deterministic conformance"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="program_ad_selection_primitives",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_differentiable_programming_benchmark_suite",
            evidence=(
                "scpn.program_ad.selection:{where,clip,sort,select,piecewise,choose,compress,extract,argmax,argmin,argsort}",
                "selection_piecewise_contracts",
            ),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=(
                "bounded Program AD static selection folds, strict sort, where, "
                "and clip conformance only; dynamic masks, dynamic selectors, "
                "tie boundaries, integer-output selectors, Rust/LLVM executable "
                "lowering, hardware, and performance promotion remain blocked"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="primitive_contracts",
            state="executable",
            backing_api="primitive_complete_contract_for",
            evidence=("PrimitiveContract", "PrimitiveTransformRule"),
            blocked_reasons=(),
            claim_boundary="registered primitive contracts only; unknown primitives fail closed",
        ),
        DifferentiableDashboardCapabilityRow(
            surface="higher_order_transform_algebra",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="run_differentiable_programming_benchmark_suite",
            evidence=(
                "transform_nesting_vmap_program_grad",
                "transform_nesting_custom_rule_vmap_jvp_vjp",
                "transform_nesting_program_ad_vmap_jvp_vjp",
                "transform_nesting_whole_program_higher_order",
                "transform_nesting_program_ad_hessian",
                "transform_nesting_program_ad_hessian_jvp_vjp",
            ),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=(
                "local transform-algebra conformance for vmap, exact custom JVP/VJP "
                "rules, whole-program grad, JVP/VJP, jacfwd, jacrev, and local "
                "Hessian transforms including JVP/VJP over Hessian transforms only; "
                "no compiler, JIT, hardware, or performance claim"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="nondifferentiability_diagnostics",
            state="diagnostic",
            backing_api="primitive_contract_for",
            evidence=(
                "ProgramADLinalgConditioningDiagnostic",
                "program_ad_elementwise:sign",
                "program_ad_elementwise:heaviside",
            ),
            blocked_reasons=("diagnostic rows do not execute or promote derivative kernels",),
            claim_boundary="local diagnostic evidence only; no provider, hardware, or benchmark claim",
        ),
        DifferentiableDashboardCapabilityRow(
            surface="benchmark_conformance",
            state="conformance_backed" if conformance_passed else "diagnostic",
            backing_api="differentiable_benchmark_report",
            evidence=("run_differentiable_programming_benchmark_suite",),
            blocked_reasons=()
            if conformance_passed
            else ("conformance suite not run in this status call",),
            claim_boundary=(
                "local deterministic conformance evidence; not isolated performance, "
                "hardware, or provider execution evidence"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="polyglot_compiler_chain",
            state="blocked",
            backing_api="differentiable_compile_report",
            evidence=(
                "compile_compiler_ad_transform_plan_to_mlir",
                "compile_whole_program_ad_trace_to_mlir",
                "Rust Program AD IR metadata parser",
                "program_ad_effect_ir_metadata_summary",
                "program_ad_effect_ir_interpret_forward",
                "program_ad_effect_ir_interpret_value_and_gradient",
            ),
            blocked_reasons=(
                "only bounded scalar opcode-bearing Rust value+gradient replay is promoted",
                "native LLVM/JIT differentiated kernels remain blocked until runtime verified",
            ),
            claim_boundary=(
                "compiler/interchange planning plus Rust "
                "metadata parsing plus bounded scalar value+gradient Program AD IR "
                "replay only; no general Program AD execution, control-flow replay, "
                "array adjoints, LLVM/JIT execution, hardware, provider, or performance claim"
            ),
        ),
        DifferentiableDashboardCapabilityRow(
            surface="provider_and_hardware_gradients",
            state="blocked",
            backing_api="explain_differentiability",
            evidence=("DifferentiabilityDiagnosticReport",),
            blocked_reasons=(
                "live provider and hardware gradient execution require explicit policy evidence",
            ),
            claim_boundary="planning and diagnostic evidence only; no hardware job submission",
        ),
        DifferentiableDashboardCapabilityRow(
            surface="gui_frontend",
            state="planned",
            backing_api="differentiable_dashboard_status",
            evidence=("DifferentiableDashboardStatus",),
            blocked_reasons=("frontend implementation is planned after this status contract",),
            claim_boundary="dashboard backing contract only; no user-interface implementation claim",
        ),
    ]
    return DifferentiableDashboardStatus(
        rows=tuple(rows),
        status_api_ready=True,
        generated_from=(
            "differentiable_api",
            "program_ad_capability_contracts",
            "program_ad_effect_ir.v1",
            "compiler_ad_transform_plan",
            "gradient_support_matrix",
        ),
        claim_boundary=(
            "machine-readable status for audit dashboards; row states must be displayed "
            "without upgrading planned, metadata-only, diagnostic, blocked, or unsupported routes"
        ),
    )


__all__ = [
    "DifferentiableDashboardCapabilityRow",
    "DifferentiableDashboardStatus",
    "differentiable_dashboard_status",
]
