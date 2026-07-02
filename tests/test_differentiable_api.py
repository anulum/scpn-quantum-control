# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- tests for unified differentiable API facade
"""Tests for scpn_quantum_control.differentiable_api."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control as scpn
from scpn_quantum_control.differentiable_api import (
    DifferentiabilityDiagnosticReport,
    DifferentiableDashboardCapabilityRow,
    DifferentiableDashboardCapabilityState,
    DifferentiableDashboardStatus,
    UnifiedDifferentiableAPIResult,
    differentiable_api,
    differentiable_benchmark_report,
    differentiable_compile_report,
    differentiable_dashboard_status,
    differentiable_frontend_report,
    differentiable_gradient,
    differentiable_hessian,
    differentiable_jacobian,
    differentiable_qfi_fss_report,
    differentiable_support_report,
    differentiable_transform_algebra_report,
    differentiable_value,
    explain_differentiability,
)
from scpn_quantum_control.differentiable_transform_algebra import (
    REQUIRED_TRANSFORM_ALGEBRA_CATEGORIES,
    TransformAlgebraAudit,
    TransformAlgebraCase,
    assert_transform_algebra_audit_passes,
    run_transform_algebra_audit,
)

FloatArray = NDArray[np.float64]


def _scalar_objective(values: FloatArray) -> float:
    return float(values[0] ** 2 + 3.0 * values[1])


def _vector_objective(values: FloatArray) -> FloatArray:
    return np.array([values[0] + values[1], values[0] * values[1]], dtype=float)


def _require_torch_backend() -> None:
    pytest.importorskip("torch", reason="native Torch differentiable rows require PyTorch")


def test_unified_differentiable_value_and_gradient_share_schema() -> None:
    values = np.array([2.0, -1.0], dtype=float)

    value = differentiable_value(_scalar_objective, values, method="finite_difference")
    gradient = differentiable_gradient(_scalar_objective, values, method="finite_difference")

    assert isinstance(value, UnifiedDifferentiableAPIResult)
    assert value.operation == "value"
    assert value.supported
    assert value.gradient is None
    assert value.value == pytest.approx(1.0)
    assert value.to_dict()["fail_closed"] is False

    assert gradient.operation == "gradient"
    assert gradient.supported
    assert gradient.gradient is not None
    np.testing.assert_allclose(gradient.gradient, np.array([4.0, 3.0]), atol=1e-5)
    assert gradient.to_dict()["gradient"] == pytest.approx([4.0, 3.0], abs=1e-5)
    assert "no hardware execution" in gradient.claim_boundary


def test_unified_differentiable_jacobian_and_hessian_routes_are_explicit() -> None:
    values = np.array([2.0, -1.0], dtype=float)

    jacobian = differentiable_jacobian(_vector_objective, values)
    hessian = differentiable_hessian(_scalar_objective, values)

    assert jacobian.operation == "jacobian"
    assert jacobian.jacobian is not None
    np.testing.assert_allclose(
        jacobian.jacobian,
        np.array([[1.0, 1.0], [-1.0, 2.0]], dtype=float),
        atol=1e-5,
    )
    assert jacobian.payload["objective_value"] == pytest.approx([1.0, -2.0])

    assert hessian.operation == "hessian"
    assert hessian.hessian is not None
    np.testing.assert_allclose(
        hessian.hessian,
        np.array([[2.0, 0.0], [0.0, 0.0]], dtype=float),
        atol=1e-4,
    )
    assert hessian.value == pytest.approx(1.0)


def test_unified_differentiable_support_report_fails_closed_for_unsupported_route() -> None:
    report = differentiable_support_report(
        gate="unregistered_gate",
        observable="pauli_expectation",
    )

    assert report.operation == "support_report"
    assert report.fail_closed
    assert report.payload["supported"] is False
    assert "no registered parameter-shift generator" in report.payload["blocked_reasons"][0]
    payload = cast(dict[str, object], report.to_dict()["payload"])
    assert payload["requires_hardware_policy"] is False


def test_explain_differentiability_reports_reasons_and_matrices() -> None:
    report = explain_differentiability(
        gate="unregistered_gate",
        observable="pauli_expectation",
        backend="hardware",
        shots=1024,
    )

    assert isinstance(report, DifferentiabilityDiagnosticReport)
    assert report.fail_closed
    assert "no registered parameter-shift generator" in report.blocked_reasons[0]
    assert "statevector_simulator" in report.suggested_alternatives
    assert "finite_shot_simulator" in report.suggested_alternatives
    assert report.support_payload["requires_hardware_policy"] is True
    assert "do not execute objectives" in report.claim_boundary

    dependency_rows = {str(row["framework"]): row for row in report.dependency_matrix}
    assert dependency_rows["jax"]["optional_dependency"] == "jax"
    assert dependency_rows["pytorch"]["supported"] is True
    assert dependency_rows["provider_hardware_gradient"]["supported"] is False

    device_rows = {str(row["backend"]): row for row in report.device_matrix}
    assert device_rows["hardware_qpu"]["hardware"] is True
    assert device_rows["statevector_simulator"]["supports_parameter_shift"] is True

    backend_rows = {str(row["backend"]): row for row in report.backend_matrix}
    assert backend_rows["hardware_qpu"]["fail_closed"] is True
    assert backend_rows["finite_shot_simulator"]["method"] == "stochastic_parameter_shift"
    report_payload = report.to_dict()
    dependency_payload = cast(list[dict[str, object]], report_payload["dependency_matrix"])
    assert dependency_payload[0]["framework"] == "jax"


def test_unified_differentiable_compile_report_filters_registered_primitives() -> None:
    report = differentiable_compile_report(
        primitive_identities=("scpn.program_ad.array:getitem@1",)
    )

    assert report.operation == "compile_report"
    assert report.supported
    assert report.payload["primitive_count"] == 1
    assert report.payload["primitive_identities"] == ["scpn.program_ad.array:getitem@1"]
    assert "scpn_diff.primitive" in str(report.payload["mlir"])

    with pytest.raises(ValueError, match="unknown primitive identities"):
        differentiable_compile_report(primitive_identities=("missing:primitive@1",))


def test_unified_differentiable_transform_algebra_report_is_bounded() -> None:
    report = differentiable_transform_algebra_report()
    dispatched = differentiable_api("transform_algebra_report")

    assert report.operation == "transform_algebra_report"
    assert report.supported is True
    assert report.method == "differentiable_transform_algebra"
    assert report.payload["missing_categories"] == []
    categories = cast(list[str], report.payload["categories"])
    blocked_count = cast(int, report.payload["blocked_count"])
    assert set(categories) == set(REQUIRED_TRANSFORM_ALGEBRA_CATEGORIES)
    assert blocked_count >= 4
    assert "finite differences remain diagnostic" in report.claim_boundary
    assert dispatched.to_dict() == report.to_dict()


def test_unified_differentiable_qfi_fss_report_is_bounded() -> None:
    report = differentiable_qfi_fss_report(
        system_sizes=[2, 3],
        k_range=np.linspace(0.5, 3.0, 6),
    )
    dispatched = differentiable_api(
        "qfi_fss_report",
        system_sizes=[2, 3],
        k_range=np.linspace(0.5, 3.0, 6),
    )

    assert report.operation == "qfi_fss_report"
    assert report.supported is True
    assert report.method == "qfi_finite_size_scaling"
    assert report.payload["system_sizes"] == [2, 3]
    assert report.payload["bkt_fit"]["model"] == "bkt_log_correction"
    assert report.payload["power_fit"]["model"] == "power_law_nu_1"
    assert "no hardware" in report.claim_boundary
    assert dispatched.to_dict() == report.to_dict()
    assert scpn.differentiable_qfi_fss_report is differentiable_qfi_fss_report


def test_unified_differentiable_benchmark_report_is_non_performance_evidence() -> None:
    _require_torch_backend()

    report = differentiable_benchmark_report()

    assert report.operation == "benchmark_report"
    assert report.supported
    assert report.payload["program_ad_case_count"] > 0
    assert report.payload["quantum_gradient_case_count"] > 0
    assert report.payload["support_audit_passed"] is True
    assert "not isolated performance" in report.claim_boundary


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
    assert "generated adjoint steps" in rows["program_ad_reverse_adjoint_replay"]["claim_boundary"]
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


def test_differentiable_dashboard_status_validates_rows() -> None:
    row = DifferentiableDashboardCapabilityRow(
        surface="demo",
        state="unsupported",
        backing_api="demo_api",
        evidence=("contract",),
        blocked_reasons=("not implemented",),
        claim_boundary="bounded",
    )

    assert row.fail_closed is True
    assert row.to_dict()["state"] == "unsupported"
    with pytest.raises(ValueError, match="surface"):
        DifferentiableDashboardCapabilityRow(
            surface="",
            state="planned",
            backing_api="demo_api",
            evidence=("contract",),
            blocked_reasons=(),
            claim_boundary="bounded",
        )
    with pytest.raises(ValueError, match="rows"):
        DifferentiableDashboardStatus(
            rows=(),
            status_api_ready=True,
            generated_from=("demo",),
            claim_boundary="bounded",
        )


def test_unified_differentiable_dispatcher_and_root_exports() -> None:
    values = np.array([2.0, -1.0], dtype=float)
    calls = {"count": 0}

    def frontend_objective(inputs: FloatArray) -> object:
        calls["count"] += 1
        return np.sin(inputs[0]) + inputs[1]

    gradient = differentiable_api(
        "gradient",
        objective=_scalar_objective,
        values=values,
        method="finite_difference",
    )
    support = differentiable_api(
        "support_report",
        gate="ry",
        observable="pauli_expectation",
        n_params=2,
    )
    diagnostic = differentiable_api(
        "diagnostic_report",
        gate="unregistered_gate",
        observable="pauli_expectation",
    )
    dashboard = differentiable_api("dashboard_status")
    frontend_direct = differentiable_frontend_report(frontend_objective)
    frontend_dispatched = differentiable_api("frontend_report", objective=frontend_objective)

    assert gradient.gradient is not None
    np.testing.assert_allclose(gradient.gradient, np.array([4.0, 3.0]), atol=1e-5)
    assert calls == {"count": 0}
    assert support.supported
    assert diagnostic.fail_closed
    assert diagnostic.payload["blocked_reasons"]
    assert dashboard.supported
    assert dashboard.payload["status_api_ready"] is True
    assert dashboard.payload["rows"]
    assert frontend_direct.operation == "frontend_report"
    assert frontend_direct.supported is True
    assert frontend_direct.payload["frontend_ready"] is True
    assert frontend_direct.payload["bytecode_basic_block_count"] > 0
    assert frontend_direct.payload["source_region_count"] > 0
    assert frontend_direct.payload["source_bytecode_line_map_count"] > 0
    assert frontend_direct.payload["symbol_scope_entry_count"] > 0
    assert frontend_direct.payload["unsupported_semantic_diagnostic_count"] == 0
    assert frontend_direct.payload["unsupported_semantic_diagnostics"] == []
    assert int(frontend_direct.payload["source_start_line"]) > 0
    assert int(frontend_direct.payload["source_end_line"]) >= int(
        frontend_direct.payload["source_start_line"]
    )
    first_line_map = frontend_direct.payload["source_bytecode_line_map"][0]
    assert int(first_line_map["line_number"]) > 0
    assert first_line_map["absolute_line_number"] is None or int(
        first_line_map["absolute_line_number"]
    ) >= int(frontend_direct.payload["source_start_line"])
    assert first_line_map["region_ids"]
    assert len(str(frontend_direct.payload["frontend_digest"])) == 64
    assert frontend_dispatched.supported is True
    assert frontend_dispatched.payload["bytecode_instruction_count"] > 0
    assert scpn.DifferentiableDashboardStatus is DifferentiableDashboardStatus
    assert scpn.DifferentiableDashboardCapabilityRow is DifferentiableDashboardCapabilityRow
    assert scpn.DifferentiableDashboardCapabilityState is DifferentiableDashboardCapabilityState
    assert scpn.differentiable_dashboard_status is differentiable_dashboard_status
    assert scpn.differentiable_frontend_report is differentiable_frontend_report
    assert scpn.differentiable_transform_algebra_report is differentiable_transform_algebra_report
    assert scpn.explain_differentiability is explain_differentiability
    assert scpn.differentiable_api is differentiable_api
    assert scpn.differentiable_gradient is differentiable_gradient
    assert scpn.differentiable_value is differentiable_value
    assert scpn.TransformAlgebraAudit is TransformAlgebraAudit
    assert scpn.TransformAlgebraCase is TransformAlgebraCase
    assert scpn.run_transform_algebra_audit is run_transform_algebra_audit
    assert scpn.assert_transform_algebra_audit_passes is assert_transform_algebra_audit_passes

    with pytest.raises(ValueError, match="objective is required"):
        differentiable_api("gradient", values=values)
