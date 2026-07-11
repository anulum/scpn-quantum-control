# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- differentiable dashboard tests
"""Tests for the claim-bounded differentiable dashboard catalog."""

from __future__ import annotations

from typing import Any, cast

from _differentiable_api_test_helpers import _require_torch_backend

from scpn_quantum_control.differentiable_dashboard import (
    DifferentiableDashboardStatus,
    differentiable_dashboard_status,
)


def test_differentiable_dashboard_status_is_claim_bounded_for_gui_consumers() -> None:
    status = differentiable_dashboard_status()

    assert isinstance(status, DifferentiableDashboardStatus)
    assert status.status_api_ready is True
    payload = status.to_dict()
    payload_rows = cast(list[dict[str, Any]], payload["rows"])
    rows = {str(row["surface"]): row for row in payload_rows}

    assert rows["unified_differentiable_api"]["state"] == "executable"
    assert rows["unified_differentiable_api"]["fail_closed"] is False
    assert rows["program_ad_ir"]["state"] == "metadata_only"
    assert rows["program_ad_ir"]["fail_closed"] is True
    assert rows["program_ad_bytecode_source_frontend"]["state"] == "diagnostic"
    assert rows["program_ad_bytecode_source_frontend"]["fail_closed"] is True
    assert (
        rows["program_ad_bytecode_source_frontend"]["backing_api"]
        == "compile_whole_program_frontend"
    )
    assert (
        "WholeProgramCompilerFrontendReport"
        in rows["program_ad_bytecode_source_frontend"]["evidence"]
    )
    assert (
        "WholeProgramBytecodeBasicBlock" in rows["program_ad_bytecode_source_frontend"]["evidence"]
    )
    assert "WholeProgramSourceRegion" in rows["program_ad_bytecode_source_frontend"]["evidence"]
    assert (
        "WholeProgramSourceBytecodeLineMap"
        in rows["program_ad_bytecode_source_frontend"]["evidence"]
    )
    assert (
        "WholeProgramSymbolScopeEntry" in rows["program_ad_bytecode_source_frontend"]["evidence"]
    )
    assert (
        "WholeProgramUnsupportedSemanticDiagnostic"
        in rows["program_ad_bytecode_source_frontend"]["evidence"]
    )
    assert (
        "static bytecode/source compiler frontend preflight"
        in rows["program_ad_bytecode_source_frontend"]["claim_boundary"]
    )
    assert "bytecode basic blocks" in rows["program_ad_bytecode_source_frontend"]["claim_boundary"]
    assert "source-bytecode maps" in rows["program_ad_bytecode_source_frontend"]["claim_boundary"]
    assert (
        "unsupported-semantics diagnostics"
        in rows["program_ad_bytecode_source_frontend"]["claim_boundary"]
    )
    assert rows["program_ad_ir_roundtrip"]["state"] == "metadata_only"
    assert rows["program_ad_ir_roundtrip"]["backing_api"] == "parse_program_ad_effect_ir"
    assert rows["program_ad_ir_roundtrip"]["fail_closed"] is True
    assert "program_ad_effect_ir.v1" in rows["program_ad_ir_roundtrip"]["evidence"]
    assert "not executable compiler lowering" in str(
        rows["program_ad_ir_roundtrip"]["blocked_reasons"]
    )
    assert rows["program_ad_ir_roundtrip_conformance"]["state"] == "diagnostic"
    assert rows["program_ad_ir_roundtrip_conformance"]["fail_closed"] is True
    assert (
        "program_ad_ir_roundtrip_contracts"
        in rows["program_ad_ir_roundtrip_conformance"]["evidence"]
    )
    assert (
        "stable serialization" in (rows["program_ad_ir_roundtrip_conformance"]["claim_boundary"])
    )
    assert (
        "not executable compiler lowering"
        in (rows["program_ad_ir_roundtrip_conformance"]["claim_boundary"])
    )
    assert rows["program_ad_control_phi_metadata"]["state"] == "diagnostic"
    assert rows["program_ad_control_phi_metadata"]["fail_closed"] is True
    assert (
        "program_ad_control_phi_metadata_contracts"
        in rows["program_ad_control_phi_metadata"]["evidence"]
    )
    assert "ProgramADPhiNode" in rows["program_ad_control_phi_metadata"]["evidence"]
    assert (
        "runtime and source control regions"
        in (rows["program_ad_control_phi_metadata"]["claim_boundary"])
    )
    assert (
        "not non-executed branch adjoints"
        in (rows["program_ad_control_phi_metadata"]["claim_boundary"])
    )
    assert rows["program_ad_mlir_interchange"]["state"] == "diagnostic"
    assert rows["program_ad_mlir_interchange"]["fail_closed"] is True
    assert (
        "program_ad_mlir_interchange_contracts" in rows["program_ad_mlir_interchange"]["evidence"]
    )
    assert (
        rows["program_ad_mlir_interchange"]["backing_api"]
        == "compile_whole_program_ad_trace_to_mlir"
    )
    assert (
        "no executable Rust, LLVM, or JIT" in rows["program_ad_mlir_interchange"]["claim_boundary"]
    )
    assert "static alias-lattice readiness" in rows["program_ad_alias_effects"]["claim_boundary"]
    assert "bounded local object-attribute" in rows["program_ad_alias_effects"]["claim_boundary"]
    assert "expression-rebinding aliases" in rows["program_ad_alias_effects"]["claim_boundary"]
    assert "control-path aliases" in rows["program_ad_alias_effects"]["claim_boundary"]
    assert "loop-carried state" in rows["program_ad_alias_effects"]["claim_boundary"]
    assert "unsupported Python semantics" in rows["program_ad_alias_effects"]["claim_boundary"]
    assert (
        "source/region/bytecode diagnostics"
        in (rows["program_ad_alias_effects"]["claim_boundary"])
    )
    assert (
        "captured/global object-attribute diagnostics are explicit blockers"
        in rows["program_ad_alias_effects"]["claim_boundary"]
    )
    assert (
        "unknown alias-edge provenance is an explicit blocker"
        in rows["program_ad_alias_effects"]["claim_boundary"]
    )
    assert "unknown dynamic alias promotion" in rows["program_ad_alias_effects"]["claim_boundary"]
    assert "ProgramADStaticAliasLatticeReport" in rows["program_ad_alias_effects"]["evidence"]
    assert "ProgramADUnknownAliasEdge" in rows["program_ad_alias_effects"]["evidence"]
    assert "ProgramADControlPathAliasProvenance" in rows["program_ad_alias_effects"]["evidence"]
    assert "ProgramADViewAliasProvenance" in rows["program_ad_alias_effects"]["evidence"]
    assert "ProgramADLoopCarriedStateProvenance" in rows["program_ad_alias_effects"]["evidence"]
    assert "ProgramADRebindingAliasProvenance" in rows["program_ad_alias_effects"]["evidence"]
    assert "typed view-alias provenance metadata" in rows["program_ad_alias_effects"]["evidence"]
    assert (
        "typed loop-carried state provenance metadata"
        in rows["program_ad_alias_effects"]["evidence"]
    )
    assert (
        "typed rebinding-alias provenance metadata" in rows["program_ad_alias_effects"]["evidence"]
    )
    assert (
        "typed control-path alias provenance metadata"
        in rows["program_ad_alias_effects"]["evidence"]
    )
    assert (
        "malformed view aliases require parseable source-to-view provenance"
        in rows["program_ad_alias_effects"]["blocked_reasons"]
    )
    assert (
        "malformed control-path aliases require parseable branch-local provenance"
        in rows["program_ad_alias_effects"]["blocked_reasons"]
    )
    assert (
        "malformed loop-carried state aliases require parseable loop provenance"
        in rows["program_ad_alias_effects"]["blocked_reasons"]
    )
    assert (
        "bounded local object-attribute alias metadata"
        in rows["program_ad_alias_effects"]["evidence"]
    )
    assert "expression-rebinding alias metadata" in rows["program_ad_alias_effects"]["evidence"]
    assert "control-path alias blocker metadata" in rows["program_ad_alias_effects"]["evidence"]
    assert (
        "unsupported-Python frontend diagnostic metadata"
        in rows["program_ad_alias_effects"]["evidence"]
    )
    assert (
        "captured/global object-attribute blocker metadata"
        in rows["program_ad_alias_effects"]["evidence"]
    )
    assert (
        "unknown alias-edge provenance blocker metadata"
        in rows["program_ad_alias_effects"]["evidence"]
    )
    assert "shape_view_alias_metadata_contracts" in rows["program_ad_alias_effects"]["evidence"]
    assert (
        "slice_mutation_alias_metadata_contracts" in rows["program_ad_alias_effects"]["evidence"]
    )
    assert (
        "loop_carried_state_alias_metadata_contracts"
        in rows["program_ad_alias_effects"]["evidence"]
    )
    assert (
        "program_ad_static_alias_lattice_contracts" in rows["program_ad_alias_effects"]["evidence"]
    )
    assert (
        "captured/global object attributes require static object-model alias sets"
        in (rows["program_ad_alias_effects"]["blocked_reasons"])
    )
    assert (
        "control-path aliases require non-executed branch semantics"
        in (rows["program_ad_alias_effects"]["blocked_reasons"])
    )
    assert (
        "unsupported Python semantics require executable frontend lowering"
        in (rows["program_ad_alias_effects"]["blocked_reasons"])
    )
    assert (
        "unknown alias edges require static alias-kind support"
        in (rows["program_ad_alias_effects"]["blocked_reasons"])
    )
    assert (
        "non-executed branch semantics remain blocked"
        in (rows["program_ad_alias_effects"]["blocked_reasons"])
    )
    assert rows["program_ad_registry_dispatch_coverage"]["state"] == "diagnostic"
    assert rows["program_ad_registry_dispatch_coverage"]["fail_closed"] is True
    assert (
        "ProgramADRegistryDispatchCoverageReport"
        in rows["program_ad_registry_dispatch_coverage"]["evidence"]
    )
    assert (
        "program_ad_registry_dispatch_contracts"
        in rows["program_ad_registry_dispatch_coverage"]["evidence"]
    )
    assert (
        "not executable Rust, LLVM, JIT"
        in (rows["program_ad_registry_dispatch_coverage"]["claim_boundary"])
    )
    assert rows["program_ad_python_semantics"]["state"] == "diagnostic"
    assert rows["program_ad_python_semantics"]["fail_closed"] is True
    assert "python_semantics_list_comprehension" in rows["program_ad_python_semantics"]["evidence"]
    assert (
        "filtered, set, and dict comprehensions remain fail-closed"
        in rows["program_ad_python_semantics"]["claim_boundary"]
    )
    assert rows["program_ad_reverse_adjoint_replay"]["state"] == "diagnostic"
    assert rows["program_ad_reverse_adjoint_replay"]["fail_closed"] is True
    assert (
        "program_adjoint_replay_provenance_contracts"
        in rows["program_ad_reverse_adjoint_replay"]["evidence"]
    )
    assert "ProgramADAdjointResult" in rows["program_ad_reverse_adjoint_replay"]["evidence"]
    assert "ProgramADAdjointStep" in rows["program_ad_reverse_adjoint_replay"]["evidence"]
    assert (
        "WholeProgramADResult replay contract"
        in rows["program_ad_reverse_adjoint_replay"]["evidence"]
    )
    assert "generated adjoint steps" in rows["program_ad_reverse_adjoint_replay"]["claim_boundary"]
    assert (
        "validated at WholeProgramADResult construction"
        in rows["program_ad_reverse_adjoint_replay"]["claim_boundary"]
    )
    assert "caller-visible replay" in rows["program_ad_reverse_adjoint_replay"]["claim_boundary"]
    assert (
        "finite local pullback scales"
        in rows["program_ad_reverse_adjoint_replay"]["claim_boundary"]
    )
    assert "cotangent-flow rows" in rows["program_ad_reverse_adjoint_replay"]["claim_boundary"]
    assert (
        "reverse effect-order rows" in rows["program_ad_reverse_adjoint_replay"]["claim_boundary"]
    )
    assert (
        "runtime control/phi row bindings"
        in rows["program_ad_reverse_adjoint_replay"]["claim_boundary"]
    )
    assert (
        "not full reverse-mode compiler AD"
        in (rows["program_ad_reverse_adjoint_replay"]["claim_boundary"])
    )
    assert rows["program_ad_elementwise_primitives"]["state"] == "diagnostic"
    assert rows["program_ad_elementwise_primitives"]["fail_closed"] is True
    assert (
        "elementwise_boundary_contracts" in rows["program_ad_elementwise_primitives"]["evidence"]
    )
    assert (
        "zero-cusp absolute-value" in rows["program_ad_elementwise_primitives"]["claim_boundary"]
    )
    assert "domain boundaries" in rows["program_ad_elementwise_primitives"]["claim_boundary"]
    assert rows["program_ad_array_indexing"]["state"] == "diagnostic"
    assert rows["program_ad_array_indexing"]["fail_closed"] is True
    assert "indexing_static_gather_contracts" in rows["program_ad_array_indexing"]["evidence"]
    assert "dynamic indices" in rows["program_ad_array_indexing"]["claim_boundary"]
    assert rows["program_ad_linalg_primitives"]["state"] == "diagnostic"
    assert rows["program_ad_linalg_primitives"]["fail_closed"] is True
    assert "linalg_primitive_contracts" in rows["program_ad_linalg_primitives"]["evidence"]
    assert "rank-threshold" in rows["program_ad_linalg_primitives"]["claim_boundary"]
    assert rows["program_ad_structured_primitives"]["state"] == "diagnostic"
    assert rows["program_ad_structured_primitives"]["fail_closed"] is True
    assert (
        "structured_numeric_primitive_contracts"
        in rows["program_ad_structured_primitives"]["evidence"]
    )
    assert "Rust/LLVM" in rows["program_ad_structured_primitives"]["claim_boundary"]
    assert rows["program_ad_cumulative_primitives"]["state"] == "diagnostic"
    assert rows["program_ad_cumulative_primitives"]["fail_closed"] is True
    assert "cumulative_primitive_contracts" in rows["program_ad_cumulative_primitives"]["evidence"]
    assert "dynamic axis promotion" in rows["program_ad_cumulative_primitives"]["claim_boundary"]
    assert rows["program_ad_assembly_primitives"]["state"] == "diagnostic"
    assert rows["program_ad_assembly_primitives"]["fail_closed"] is True
    assert "assembly_primitive_contracts" in rows["program_ad_assembly_primitives"]["evidence"]
    assert "dynamic shape assembly" in rows["program_ad_assembly_primitives"]["claim_boundary"]
    assert rows["program_ad_reduction_primitives"]["state"] == "diagnostic"
    assert rows["program_ad_reduction_primitives"]["fail_closed"] is True
    assert "reduction_primitive_contracts" in rows["program_ad_reduction_primitives"]["evidence"]
    assert "strict-order selectors" in rows["program_ad_reduction_primitives"]["claim_boundary"]
    assert rows["program_ad_shape_primitives"]["state"] == "diagnostic"
    assert rows["program_ad_shape_primitives"]["fail_closed"] is True
    assert "shape_primitive_contracts" in rows["program_ad_shape_primitives"]["evidence"]
    assert "dynamic shape arguments" in rows["program_ad_shape_primitives"]["claim_boundary"]
    assert rows["program_ad_broadcast_primitives"]["state"] == "diagnostic"
    assert rows["program_ad_broadcast_primitives"]["fail_closed"] is True
    assert "broadcast_primitive_contracts" in rows["program_ad_broadcast_primitives"]["evidence"]
    assert "subok" in rows["program_ad_broadcast_primitives"]["claim_boundary"]
    assert rows["program_ad_selection_primitives"]["state"] == "diagnostic"
    assert rows["program_ad_selection_primitives"]["fail_closed"] is True
    assert "selection_piecewise_contracts" in rows["program_ad_selection_primitives"]["evidence"]
    assert "dynamic masks" in rows["program_ad_selection_primitives"]["claim_boundary"]
    assert "integer-output selectors" in rows["program_ad_selection_primitives"]["claim_boundary"]
    assert rows["nondifferentiability_diagnostics"]["state"] == "diagnostic"
    assert "program_ad_elementwise:sign" in rows["nondifferentiability_diagnostics"]["evidence"]
    assert (
        "program_ad_elementwise:heaviside" in rows["nondifferentiability_diagnostics"]["evidence"]
    )
    assert rows["higher_order_transform_algebra"]["state"] == "diagnostic"
    assert rows["higher_order_transform_algebra"]["fail_closed"] is True
    assert (
        "transform_nesting_program_ad_hessian"
        in rows["higher_order_transform_algebra"]["evidence"]
    )
    assert (
        "transform_nesting_program_ad_hessian_jvp_vjp"
        in rows["higher_order_transform_algebra"]["evidence"]
    )
    assert (
        "transform_nesting_whole_program_higher_order"
        in rows["higher_order_transform_algebra"]["evidence"]
    )
    assert rows["polyglot_compiler_chain"]["state"] == "blocked"
    assert (
        "only bounded scalar opcode-bearing Rust value+gradient replay is promoted"
        in rows["polyglot_compiler_chain"]["blocked_reasons"]
    )
    assert rows["program_ad_rust_scalar_interpreter"]["state"] == "diagnostic"
    assert (
        "program_ad_rust_scalar_interpreter_contracts"
        in rows["program_ad_rust_scalar_interpreter"]["evidence"]
    )
    assert "Rust Program AD IR metadata parser" in rows["polyglot_compiler_chain"]["evidence"]
    assert "compile_whole_program_ad_trace_to_mlir" in rows["polyglot_compiler_chain"]["evidence"]
    assert "program_ad_effect_ir_interpret_forward" in rows["polyglot_compiler_chain"]["evidence"]
    assert (
        "program_ad_effect_ir_interpret_value_and_gradient"
        in rows["polyglot_compiler_chain"]["evidence"]
    )
    assert (
        "bounded scalar value+gradient Program AD IR replay"
        in rows["polyglot_compiler_chain"]["claim_boundary"]
    )
    assert rows["jax_phase_qnode_aot_export_lowering"]["state"] == "diagnostic"
    assert rows["jax_phase_qnode_aot_export_lowering"]["fail_closed"] is True
    assert (
        rows["jax_phase_qnode_aot_export_lowering"]["backing_api"]
        == "jax_phase_qnode_aot_export_audit"
    )
    assert (
        "PhaseJAXPhaseQNodeAOTExportResult"
        in rows["jax_phase_qnode_aot_export_lowering"]["evidence"]
    )
    assert (
        "jax_registered_phase_qnode_aot_export_lowering"
        in rows["jax_phase_qnode_aot_export_lowering"]["evidence"]
    )
    assert (
        "exported VJP and persistent cross-platform execution remain blocked"
        in rows["jax_phase_qnode_aot_export_lowering"]["blocked_reasons"]
    )
    assert (
        "jax.export serialization diagnostics"
        in (rows["jax_phase_qnode_aot_export_lowering"]["claim_boundary"])
    )
    assert "no exported VJP" in rows["jax_phase_qnode_aot_export_lowering"]["claim_boundary"]
    assert rows["torch_phase_qnode_statevector_lowering"]["state"] == "diagnostic"
    assert rows["torch_phase_qnode_statevector_lowering"]["fail_closed"] is True
    assert (
        rows["torch_phase_qnode_statevector_lowering"]["backing_api"]
        == "torch_phase_qnode_value_and_grad"
    )
    assert (
        "torch_registered_phase_qnode_statevector_lowering"
        in rows["torch_phase_qnode_statevector_lowering"]["evidence"]
    )
    assert (
        "finite-shot Torch lowering remains blocked"
        in rows["torch_phase_qnode_statevector_lowering"]["blocked_reasons"]
    )
    assert (
        "no provider, hardware, isolated benchmark, or performance promotion"
        in rows["torch_phase_qnode_statevector_lowering"]["claim_boundary"]
    )
    assert rows["torch_phase_qnode_compile_boundary_diagnostic"]["state"] == "diagnostic"
    assert rows["torch_phase_qnode_compile_boundary_diagnostic"]["fail_closed"] is True
    assert (
        rows["torch_phase_qnode_compile_boundary_diagnostic"]["backing_api"]
        == "torch_phase_qnode_compile_boundary_audit"
    )
    assert (
        "PhaseTorchCompileBoundaryAuditResult"
        in rows["torch_phase_qnode_compile_boundary_diagnostic"]["evidence"]
    )
    assert (
        "torch_registered_phase_qnode_compile_boundary_diagnostic"
        in rows["torch_phase_qnode_compile_boundary_diagnostic"]["evidence"]
    )
    assert (
        "fullgraph compile promotion remains blocked"
        in rows["torch_phase_qnode_compile_boundary_diagnostic"]["blocked_reasons"]
    )
    assert (
        "no persistent export"
        in rows["torch_phase_qnode_compile_boundary_diagnostic"]["claim_boundary"]
    )
    assert rows["torch_bounded_qnn_autograd_function_audit"]["state"] == "diagnostic"
    assert rows["torch_bounded_qnn_autograd_function_audit"]["fail_closed"] is True
    assert (
        rows["torch_bounded_qnn_autograd_function_audit"]["backing_api"]
        == "run_torch_autograd_function_audit"
    )
    assert (
        "PhaseTorchAutogradFunctionResult"
        in rows["torch_bounded_qnn_autograd_function_audit"]["evidence"]
    )
    assert (
        "Tensor.backward parameter-gradient parity"
        in rows["torch_bounded_qnn_autograd_function_audit"]["evidence"]
    )
    assert (
        "higher-order autograd graph transformation remains blocked"
        in rows["torch_bounded_qnn_autograd_function_audit"]["blocked_reasons"]
    )
    assert (
        "torch.autograd.Function"
        in rows["torch_bounded_qnn_autograd_function_audit"]["claim_boundary"]
    )
    assert rows["torch_bounded_qnn_module_state_audit"]["state"] == "diagnostic"
    assert rows["torch_bounded_qnn_module_state_audit"]["fail_closed"] is True
    assert (
        rows["torch_bounded_qnn_module_state_audit"]["backing_api"]
        == "run_torch_module_state_audit"
    )
    assert (
        "PhaseTorchModuleStateAuditResult"
        in rows["torch_bounded_qnn_module_state_audit"]["evidence"]
    )
    assert (
        "strict module state_dict round-trip"
        in rows["torch_bounded_qnn_module_state_audit"]["evidence"]
    )
    assert (
        "provider, hardware, CUDA, isolated benchmark, and performance promotion remain blocked"
        in rows["torch_bounded_qnn_module_state_audit"]["blocked_reasons"]
    )
    assert "strict state_dict" in rows["torch_bounded_qnn_module_state_audit"]["claim_boundary"]
    assert rows["torch_bounded_qnn_module_device_state_audit"]["state"] == "diagnostic"
    assert rows["torch_bounded_qnn_module_device_state_audit"]["fail_closed"] is True
    assert (
        rows["torch_bounded_qnn_module_device_state_audit"]["backing_api"]
        == "run_torch_module_device_state_audit"
    )
    assert (
        "PhaseTorchDeviceStateAuditResult"
        in rows["torch_bounded_qnn_module_device_state_audit"]["evidence"]
    )
    assert (
        "module.to('cpu') state_dict round-trip"
        in rows["torch_bounded_qnn_module_device_state_audit"]["evidence"]
    )
    assert (
        "cross-runtime checkpoint portability, provider, hardware, isolated benchmark, and performance promotion remain blocked"
        in rows["torch_bounded_qnn_module_device_state_audit"]["blocked_reasons"]
    )
    assert "CPU module.to" in rows["torch_bounded_qnn_module_device_state_audit"]["claim_boundary"]
    assert rows["torch_bounded_qnn_module_checkpoint_audit"]["state"] == "diagnostic"
    assert rows["torch_bounded_qnn_module_checkpoint_audit"]["fail_closed"] is True
    assert (
        rows["torch_bounded_qnn_module_checkpoint_audit"]["backing_api"]
        == "run_torch_module_checkpoint_audit"
    )
    assert (
        "PhaseTorchCheckpointAuditResult"
        in rows["torch_bounded_qnn_module_checkpoint_audit"]["evidence"]
    )
    assert (
        "torch.load(weights_only=True, map_location='cpu') replay"
        in rows["torch_bounded_qnn_module_checkpoint_audit"]["evidence"]
    )
    assert (
        "cross-runtime checkpoint portability remains blocked"
        in rows["torch_bounded_qnn_module_checkpoint_audit"]["blocked_reasons"]
    )
    assert (
        "weights_only=True" in rows["torch_bounded_qnn_module_checkpoint_audit"]["claim_boundary"]
    )
    assert rows["torch_bounded_qnn_long_lived_checkpoint_matrix"]["state"] == "diagnostic"
    assert rows["torch_bounded_qnn_long_lived_checkpoint_matrix"]["fail_closed"] is True
    assert (
        rows["torch_bounded_qnn_long_lived_checkpoint_matrix"]["backing_api"]
        == "run_torch_long_lived_checkpoint_matrix"
    )
    assert (
        "PhaseTorchCheckpointMatrixResult"
        in rows["torch_bounded_qnn_long_lived_checkpoint_matrix"]["evidence"]
    )
    assert (
        "checkpoint tensor metadata manifest"
        in rows["torch_bounded_qnn_long_lived_checkpoint_matrix"]["evidence"]
    )
    assert (
        "long-lived external checkpoint artifact remains blocked"
        in rows["torch_bounded_qnn_long_lived_checkpoint_matrix"]["blocked_reasons"]
    )
    assert (
        "no cross-runtime"
        in rows["torch_bounded_qnn_long_lived_checkpoint_matrix"]["claim_boundary"]
    )
    assert rows["torch_bounded_qnn_training_loop_matrix"]["state"] == "diagnostic"
    assert rows["torch_bounded_qnn_training_loop_matrix"]["fail_closed"] is True
    assert (
        rows["torch_bounded_qnn_training_loop_matrix"]["backing_api"]
        == "run_torch_training_loop_matrix"
    )
    assert (
        "PhaseTorchTrainingLoopMatrixResult"
        in rows["torch_bounded_qnn_training_loop_matrix"]["evidence"]
    )
    assert (
        "multi-scenario training-loop parity"
        in rows["torch_bounded_qnn_training_loop_matrix"]["evidence"]
    )
    assert (
        "CUDA training-loop execution remains blocked"
        in rows["torch_bounded_qnn_training_loop_matrix"]["blocked_reasons"]
    )
    assert "no CUDA" in rows["torch_bounded_qnn_training_loop_matrix"]["claim_boundary"]
    assert rows["torch_bounded_qnn_module_export_audit"]["state"] == "diagnostic"
    assert rows["torch_bounded_qnn_module_export_audit"]["fail_closed"] is True
    assert (
        rows["torch_bounded_qnn_module_export_audit"]["backing_api"]
        == "run_torch_module_export_audit"
    )
    assert (
        "PhaseTorchExportAuditResult" in rows["torch_bounded_qnn_module_export_audit"]["evidence"]
    )
    assert (
        "torch.export.save/load local value replay"
        in rows["torch_bounded_qnn_module_export_audit"]["evidence"]
    )
    assert (
        "AOTAutograd gradient-export persistence remains blocked"
        in rows["torch_bounded_qnn_module_export_audit"]["blocked_reasons"]
    )
    assert "torch.export.export" in rows["torch_bounded_qnn_module_export_audit"]["claim_boundary"]
    assert rows["torch_bounded_qnn_export_shape_matrix"]["state"] == "diagnostic"
    assert rows["torch_bounded_qnn_export_shape_matrix"]["fail_closed"] is True
    assert (
        rows["torch_bounded_qnn_export_shape_matrix"]["backing_api"]
        == "run_torch_export_shape_matrix"
    )
    assert (
        "PhaseTorchExportShapeMatrixResult"
        in rows["torch_bounded_qnn_export_shape_matrix"]["evidence"]
    )
    assert (
        "multi-static-shape torch.export replay"
        in rows["torch_bounded_qnn_export_shape_matrix"]["evidence"]
    )
    assert (
        "dynamic-shape export replay remains blocked"
        in rows["torch_bounded_qnn_export_shape_matrix"]["blocked_reasons"]
    )
    assert "no dynamic-shape" in rows["torch_bounded_qnn_export_shape_matrix"]["claim_boundary"]
    assert rows["torch_bounded_qnn_dynamic_shape_export_audit"]["state"] == "diagnostic"
    assert rows["torch_bounded_qnn_dynamic_shape_export_audit"]["fail_closed"] is True
    assert (
        rows["torch_bounded_qnn_dynamic_shape_export_audit"]["backing_api"]
        == "run_torch_dynamic_shape_export_audit"
    )
    assert (
        "PhaseTorchDynamicShapeExportResult"
        in rows["torch_bounded_qnn_dynamic_shape_export_audit"]["evidence"]
    )
    assert (
        "single ExportedProgram replayed across batch sizes"
        in rows["torch_bounded_qnn_dynamic_shape_export_audit"]["evidence"]
    )
    assert (
        "AOTAutograd gradient-export persistence remains blocked"
        in rows["torch_bounded_qnn_dynamic_shape_export_audit"]["blocked_reasons"]
    )
    assert (
        "dynamic batch" in rows["torch_bounded_qnn_dynamic_shape_export_audit"]["claim_boundary"]
    )
    assert rows["torch_bounded_qnn_aot_autograd_export_audit"]["state"] == "diagnostic"
    assert rows["torch_bounded_qnn_aot_autograd_export_audit"]["fail_closed"] is True
    assert (
        rows["torch_bounded_qnn_aot_autograd_export_audit"]["backing_api"]
        == "run_torch_aot_autograd_export_audit"
    )
    assert (
        "PhaseTorchAOTAutogradExportResult"
        in rows["torch_bounded_qnn_aot_autograd_export_audit"]["evidence"]
    )
    assert (
        "persisted AOTAutograd forward/backward FX graphs"
        in rows["torch_bounded_qnn_aot_autograd_export_audit"]["evidence"]
    )
    assert (
        "cross-runtime AOTAutograd execution remains blocked"
        in rows["torch_bounded_qnn_aot_autograd_export_audit"]["blocked_reasons"]
    )
    assert (
        "AOTAutograd FX" in rows["torch_bounded_qnn_aot_autograd_export_audit"]["claim_boundary"]
    )
    assert rows["provider_and_hardware_gradients"]["state"] == "blocked"
    assert rows["gui_frontend"]["state"] == "planned"
    generated_from = cast(list[str], payload["generated_from"])
    assert "program_ad_effect_ir.v1" in generated_from
    assert "without upgrading" in str(payload["claim_boundary"])


def test_differentiable_dashboard_status_can_include_conformance_backing() -> None:
    _require_torch_backend()

    status = differentiable_dashboard_status(include_conformance=True)
    rows = {row.surface: row for row in status.rows}

    assert rows["benchmark_conformance"].state == "conformance_backed"
    assert rows["program_ad_ir_roundtrip_conformance"].state == "conformance_backed"
    assert rows["program_ad_ir_roundtrip_conformance"].fail_closed is False
    assert rows["program_ad_ir_roundtrip_conformance"].blocked_reasons == ()
    assert "program_ad_effect_ir.v1 parser" in (
        rows["program_ad_ir_roundtrip_conformance"].claim_boundary
    )
    assert rows["program_ad_control_phi_metadata"].state == "conformance_backed"
    assert rows["program_ad_control_phi_metadata"].fail_closed is False
    assert rows["program_ad_control_phi_metadata"].blocked_reasons == ()
    assert "control-join provenance" in rows["program_ad_control_phi_metadata"].claim_boundary
    assert rows["program_ad_python_semantics"].state == "conformance_backed"
    assert rows["program_ad_python_semantics"].fail_closed is False
    assert rows["program_ad_python_semantics"].blocked_reasons == ()
    assert rows["program_ad_reverse_adjoint_replay"].state == "conformance_backed"
    assert rows["program_ad_reverse_adjoint_replay"].fail_closed is False
    assert rows["program_ad_reverse_adjoint_replay"].blocked_reasons == ()
    assert "generated adjoint steps" in (rows["program_ad_reverse_adjoint_replay"].claim_boundary)
    assert "validated at WholeProgramADResult construction" in (
        rows["program_ad_reverse_adjoint_replay"].claim_boundary
    )
    assert "caller-visible replay" in rows["program_ad_reverse_adjoint_replay"].claim_boundary
    assert rows["program_ad_registry_dispatch_coverage"].state == "conformance_backed"
    assert rows["program_ad_registry_dispatch_coverage"].fail_closed is False
    assert rows["program_ad_registry_dispatch_coverage"].blocked_reasons == ()
    assert "not executable Rust, LLVM, JIT" in (
        rows["program_ad_registry_dispatch_coverage"].claim_boundary
    )
    assert "finite local pullback scales" in (
        rows["program_ad_reverse_adjoint_replay"].claim_boundary
    )
    assert "cotangent-flow rows" in rows["program_ad_reverse_adjoint_replay"].claim_boundary
    assert "reverse effect-order rows" in (
        rows["program_ad_reverse_adjoint_replay"].claim_boundary
    )
    assert "runtime control/phi row bindings" in (
        rows["program_ad_reverse_adjoint_replay"].claim_boundary
    )
    assert rows["program_ad_elementwise_primitives"].state == "conformance_backed"
    assert rows["program_ad_elementwise_primitives"].fail_closed is False
    assert rows["program_ad_elementwise_primitives"].blocked_reasons == ()
    assert "inverse-trig boundary contracts" in (
        rows["program_ad_elementwise_primitives"].claim_boundary
    )
    assert rows["program_ad_array_indexing"].state == "conformance_backed"
    assert rows["program_ad_array_indexing"].fail_closed is False
    assert rows["program_ad_array_indexing"].blocked_reasons == ()
    assert "constant-insert" in rows["program_ad_array_indexing"].claim_boundary
    assert rows["program_ad_linalg_primitives"].state == "conformance_backed"
    assert rows["program_ad_linalg_primitives"].fail_closed is False
    assert rows["program_ad_linalg_primitives"].blocked_reasons == ()
    assert "wider native LLVM/JIT kernels" in rows["program_ad_linalg_primitives"].claim_boundary
    assert rows["program_ad_structured_primitives"].state == "conformance_backed"
    assert rows["program_ad_structured_primitives"].fail_closed is False
    assert rows["program_ad_structured_primitives"].blocked_reasons == ()
    assert "structured numeric Program AD primitive conformance" in (
        rows["program_ad_structured_primitives"].claim_boundary
    )
    assert rows["program_ad_cumulative_primitives"].state == "conformance_backed"
    assert rows["program_ad_cumulative_primitives"].fail_closed is False
    assert rows["program_ad_cumulative_primitives"].blocked_reasons == ()
    assert "cumsum, cumprod, and diff" in rows["program_ad_cumulative_primitives"].claim_boundary
    assert rows["program_ad_assembly_primitives"].state == "conformance_backed"
    assert rows["program_ad_assembly_primitives"].fail_closed is False
    assert rows["program_ad_assembly_primitives"].blocked_reasons == ()
    assert (
        "like-constructor and stack assembly"
        in rows["program_ad_assembly_primitives"].claim_boundary
    )
    assert rows["program_ad_reduction_primitives"].state == "conformance_backed"
    assert rows["program_ad_reduction_primitives"].fail_closed is False
    assert rows["program_ad_reduction_primitives"].blocked_reasons == ()
    assert "scalar q order-statistics" in rows["program_ad_reduction_primitives"].claim_boundary
    assert rows["program_ad_shape_primitives"].state == "conformance_backed"
    assert rows["program_ad_shape_primitives"].fail_closed is False
    assert rows["program_ad_shape_primitives"].blocked_reasons == ()
    assert "rank promotion" in rows["program_ad_shape_primitives"].claim_boundary
    assert rows["program_ad_broadcast_primitives"].state == "conformance_backed"
    assert rows["program_ad_broadcast_primitives"].fail_closed is False
    assert rows["program_ad_broadcast_primitives"].blocked_reasons == ()
    assert "rank broadcasting" in rows["program_ad_broadcast_primitives"].claim_boundary
    assert rows["program_ad_selection_primitives"].state == "conformance_backed"
    assert rows["program_ad_selection_primitives"].fail_closed is False
    assert rows["program_ad_selection_primitives"].blocked_reasons == ()
    assert "static selection folds" in rows["program_ad_selection_primitives"].claim_boundary
    assert rows["higher_order_transform_algebra"].state == "conformance_backed"
    assert rows["higher_order_transform_algebra"].fail_closed is False
    assert rows["higher_order_transform_algebra"].blocked_reasons == ()
    assert "Hessian transforms" in rows["higher_order_transform_algebra"].claim_boundary
    assert "no compiler" in rows["higher_order_transform_algebra"].claim_boundary
