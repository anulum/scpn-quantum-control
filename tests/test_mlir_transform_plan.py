# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR Transform Plan Tests
"""Contract tests for MLIR compiler AD transform planning and public exports."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

import scpn_quantum_control as scpn
from scpn_quantum_control.compiler.mlir import (
    CompilerADExecutableConfig,
    CompilerADKernelVerification,
    CompilerADTransformPlan,
    DifferentiableMLIRCompileConfig,
    ExecutableCompilerADKernel,
    ExecutableWholeProgramADBatchResult,
    ExecutableWholeProgramADKernel,
    MLIRCompileConfig,
    NativeWholeProgramADKernel,
    PrimitiveLoweringStatus,
    WholeProgramADNativeLoweringReport,
    analyse_whole_program_ad_native_lowering,
    build_compiler_ad_transform_plan,
    clear_native_whole_program_ad_compile_cache,
    compile_compiler_ad_transform_plan_to_mlir,
    compile_custom_derivative_rule_to_executable,
    compile_custom_derivative_rule_to_mlir,
    compile_kuramoto_to_mlir,
    compile_registered_primitive_to_executable,
    compile_whole_program_ad_trace_to_executable,
    compile_whole_program_ad_trace_to_native_llvm_jit,
    make_program_ad_linalg_matrix_power_executable_lowering_rule,
    make_program_ad_linalg_multi_dot_executable_lowering_rule,
    native_whole_program_ad_compile_cache_stats,
)
from scpn_quantum_control.differentiable import (
    CustomDerivativeRegistry,
    CustomDerivativeRule,
    PrimitiveIdentity,
    PrimitiveTransformRule,
    primitive_contract_for,
    program_ad_linalg_matrix_power_derivative_rule,
    program_ad_linalg_multi_dot_derivative_rule,
)


def test_compiler_ad_transform_plan_emits_dialect_ops_and_fail_closed_backends() -> None:
    """Compiler-backed AD planning should expose real dialect metadata without backend overclaim."""
    identity = PrimitiveIdentity("scpn.quantum", "rx_expectation", "1")
    rule = CustomDerivativeRule(
        name="rx_expectation_rule",
        value_fn=lambda values: np.array([np.cos(values[0])], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array(
            [-np.sin(values[0]) * tangent[0]], dtype=np.float64
        ),
        vjp_rule=lambda values, cotangent: np.array(
            [-np.sin(values[0]) * cotangent[0]], dtype=np.float64
        ),
        parameter_names=("theta",),
        trainable=(True,),
    )
    registry = CustomDerivativeRegistry()

    def shape_rule(args: tuple[object, ...]) -> tuple[int, ...]:
        del args
        return (1,)

    def dtype_rule(args: tuple[object, ...]) -> str:
        del args
        return "float64"

    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            shape_rule=shape_rule,
            dtype_rule=dtype_rule,
            nondifferentiable_policy="fail_closed_at_branch_points",
            effect="pure",
            lowering_metadata={
                "mlir_op": "scpn_diff.rx_expectation",
                "rust": "blocked: rust batching backend not linked",
                "llvm": "blocked: llvm lowering backend not linked",
            },
        )
    )

    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    repeat = compile_compiler_ad_transform_plan_to_mlir(plan)

    assert isinstance(plan, CompilerADTransformPlan)
    assert isinstance(plan.statuses[0], PrimitiveLoweringStatus)
    assert plan.statuses[0].has_batching_rule is False
    assert plan.statuses[0].has_shape_rule is True
    assert plan.statuses[0].has_dtype_rule is True
    assert plan.statuses[0].has_static_argument_rule is False
    assert plan.statuses[0].lowering_metadata["mlir_op"] == "scpn_diff.rx_expectation"
    assert plan.statuses[0].mlir_runtime_verification == "not_declared"
    assert plan.statuses[0].nondifferentiable_policy == "fail_closed_at_branch_points"
    assert plan.statuses[0].effect == "pure"
    assert module.text == repeat.text
    assert module.sha256 == repeat.sha256
    assert module.resource_counts["primitives"] == 1
    assert module.resource_counts["jvp_rules"] == 1
    assert module.resource_counts["vjp_rules"] == 1
    assert module.resource_counts["batching_rules"] == 0
    assert module.resource_counts["reverse_contracts"] == 0
    assert module.resource_counts["reverse_incomplete_primitives"] == 0
    assert module.resource_counts["adjoint_contracts"] == 0
    assert module.resource_counts["adjoint_incomplete_primitives"] == 1
    assert module.resource_counts["forward_contracts"] == 0
    assert module.resource_counts["forward_incomplete_primitives"] == 1
    assert module.resource_counts["transform_contracts"] == 0
    assert module.resource_counts["transform_incomplete_primitives"] == 1
    assert module.resource_counts["native_backend_contracts"] == 0
    assert module.resource_counts["native_backend_incomplete_primitives"] == 1
    assert module.resource_counts["rust_backend_contracts"] == 0
    assert module.resource_counts["rust_backend_incomplete_primitives"] == 1
    assert module.resource_counts["rust_backend_blockers"] == 1
    assert module.resource_counts["llvm_backend_contracts"] == 0
    assert module.resource_counts["llvm_backend_incomplete_primitives"] == 1
    assert module.resource_counts["llvm_backend_blockers"] == 1
    assert module.resource_counts["jit_backend_contracts"] == 0
    assert module.resource_counts["jit_backend_incomplete_primitives"] == 1
    assert module.resource_counts["jit_backend_blockers"] == 1
    assert module.resource_counts["mlir_runtime_contracts"] == 0
    assert module.resource_counts["mlir_runtime_incomplete_primitives"] == 1
    assert module.resource_counts["executable_backends"] == 0
    assert module.metadata["executable_backend"] == "none"
    assert module.metadata["jvp_rule_primitives"] == ["scpn.quantum:rx_expectation@1"]
    assert module.metadata["vjp_rule_primitives"] == ["scpn.quantum:rx_expectation@1"]
    assert module.metadata["batching_rule_primitives"] == []
    assert module.metadata["registry_contract_primitives"] == []
    assert module.metadata["reverse_contract_primitives"] == []
    assert module.metadata["reverse_incomplete_primitives"] == []
    assert module.metadata["adjoint_contract_primitives"] == []
    assert module.metadata["adjoint_incomplete_primitives"] == ["scpn.quantum:rx_expectation@1"]
    assert module.metadata["forward_contract_primitives"] == []
    assert module.metadata["forward_incomplete_primitives"] == ["scpn.quantum:rx_expectation@1"]
    assert module.metadata["transform_contract_primitives"] == []
    assert module.metadata["transform_incomplete_primitives"] == ["scpn.quantum:rx_expectation@1"]
    assert module.metadata["native_backend_contract_primitives"] == []
    assert module.metadata["native_backend_incomplete_primitives"] == [
        "scpn.quantum:rx_expectation@1"
    ]
    assert module.metadata["rust_backend_contract_primitives"] == []
    assert module.metadata["rust_backend_incomplete_primitives"] == [
        "scpn.quantum:rx_expectation@1"
    ]
    assert module.metadata["rust_backend_blockers"] == {
        "scpn.quantum:rx_expectation@1": "blocked: rust batching backend not linked"
    }
    assert module.metadata["llvm_backend_contract_primitives"] == []
    assert module.metadata["llvm_backend_incomplete_primitives"] == [
        "scpn.quantum:rx_expectation@1"
    ]
    assert module.metadata["llvm_backend_blockers"] == {
        "scpn.quantum:rx_expectation@1": "blocked: llvm lowering backend not linked"
    }
    assert module.metadata["jit_backend_contract_primitives"] == []
    assert module.metadata["jit_backend_incomplete_primitives"] == [
        "scpn.quantum:rx_expectation@1"
    ]
    assert module.metadata["jit_backend_blockers"] == {
        "scpn.quantum:rx_expectation@1": "blocked: no JIT differentiable primitive backend"
    }
    assert module.metadata["mlir_runtime_contract_primitives"] == []
    assert module.metadata["mlir_runtime_incomplete_primitives"] == [
        "scpn.quantum:rx_expectation@1"
    ]
    assert module.metadata["mlir_runtime_blockers"] == {
        "scpn.quantum:rx_expectation@1": "blocked: no MLIR-runtime lowering rule"
    }
    assert module.metadata["mlir_runtime_verification_primitives"] == {}
    assert module.metadata["primitive_readiness"]["scpn.quantum:rx_expectation@1"] == {
        "adjoint_contract": False,
        "forward_contract": False,
        "jit_backend_contract": False,
        "llvm_backend_contract": False,
        "mlir_runtime_contract": False,
        "native_backend_contract": False,
        "registry_contract": False,
        "reverse_contract": False,
        "rust_backend_contract": False,
        "transform_contract": False,
        "verdict": "registry_incomplete",
    }
    assert module.metadata["primitive_readiness_verdict_counts"] == {"registry_incomplete": 1}
    assert module.metadata["primitive_hard_gaps"]["scpn.quantum:rx_expectation@1"] == [
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
    ]
    assert module.metadata["primitive_next_hard_gap"] == {
        "scpn.quantum:rx_expectation@1": "registry_contract"
    }
    assert module.metadata["primitive_hard_gap_counts"]["registry_contract"] == 1
    assert module.metadata["primitive_hard_gap_primitives"]["registry_contract"] == [
        "scpn.quantum:rx_expectation@1"
    ]
    assert module.metadata["primitive_hard_gap_primitives"]["native_backend_contract"] == [
        "scpn.quantum:rx_expectation@1"
    ]
    assert module.metadata["primitive_hard_gap_priority"] == [
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
    ]
    assert module.metadata["primitive_hard_gap_frontier"]["registry_contract"] == {
        "count": 1,
        "next_primitive": "scpn.quantum:rx_expectation@1",
        "primitives": ["scpn.quantum:rx_expectation@1"],
    }
    assert module.metadata["primitive_hard_gap_frontier"]["native_backend_contract"] == {
        "count": 1,
        "next_primitive": "scpn.quantum:rx_expectation@1",
        "primitives": ["scpn.quantum:rx_expectation@1"],
    }
    assert module.resource_counts["mlir_runtime_blockers"] == 1
    assert module.resource_counts["mlir_runtime_verifications"] == 0
    assert module.resource_counts["primitive_readiness_verdicts"] == 1
    assert module.resource_counts["primitive_hard_gaps"] == 10
    assert module.resource_counts["primitive_next_hard_gaps"] == 1
    assert module.resource_counts["primitive_hard_gap_priority_classes"] == 10
    assert module.resource_counts["primitive_hard_gap_frontier_classes"] == 10
    assert module.resource_counts["primitive_readiness_registry_incomplete"] == 1
    assert module.resource_counts["primitive_readiness_forward_interchange_only"] == 0
    assert module.resource_counts["primitive_readiness_transform_interchange_only"] == 0
    assert module.resource_counts["primitive_readiness_mlir_runtime_verified"] == 0
    assert module.resource_counts["primitive_readiness_native_executable"] == 0
    assert "scpn_diff.primitive" in module.text
    assert "scpn_diff.lowering_status" in module.text
    assert "batching_rule = false" in module.text
    assert "shape_rule = true" in module.text
    assert "dtype_rule = true" in module.text
    assert "static_argument_rule = false" in module.text
    assert "scpn_diff.lowering_metadata" in module.text
    assert 'policy = "fail_closed_at_branch_points"' in module.text
    assert 'effect = "pure"' in module.text
    assert 'execution = "interchange_only"' in module.text
    assert "blocked: rust batching backend not linked" in module.text
    assert "blocked: llvm lowering backend not linked" in module.text
    assert "blocked: no JIT differentiable primitive backend" in module.text
    assert "scpn_diff.rx_expectation" in module.text


def test_compiler_ad_plan_does_not_count_uncontracted_policy_effect_coverage() -> None:
    """Compiler AD provenance should not overstate semantics for derivative-only rules."""
    identity = PrimitiveIdentity("scpn.quantum", "derivative_only", "1")
    rule = CustomDerivativeRule(
        name="derivative_only_rule",
        value_fn=lambda values: np.array([values[0]], dtype=np.float64),
        jvp_rule=lambda _values, tangent: np.array([tangent[0]], dtype=np.float64),
        parameter_names=("theta",),
        trainable=(True,),
    )
    registry = CustomDerivativeRegistry()
    registry.register(identity, rule)

    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)

    assert plan.statuses[0].identity == identity
    assert plan.statuses[0].nondifferentiable_policy == "not_declared"
    assert plan.statuses[0].effect == "pure"
    assert module.metadata["nondifferentiable_policies"] == {}
    assert module.metadata["effects"] == {}
    assert module.metadata["uncontracted_primitives"] == ["scpn.quantum:derivative_only@1"]
    assert module.resource_counts["nondifferentiable_policies"] == 0
    assert module.resource_counts["effects"] == 0
    assert module.resource_counts["uncontracted_primitives"] == 1


def test_compiler_ad_plan_marks_policy_only_primitives_uncontracted() -> None:
    """Policy-only primitives should remain uncontracted without boundary metadata."""
    identity = PrimitiveIdentity("scpn.quantum", "policy_only", "1")
    rule = CustomDerivativeRule(
        name="policy_only_rule",
        value_fn=lambda values: np.array([values[0]], dtype=np.float64),
        jvp_rule=lambda _values, tangent: np.array([tangent[0]], dtype=np.float64),
        parameter_names=("theta",),
        trainable=(True,),
    )
    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            lowering_metadata={"mlir_op": "scpn_diff.policy_only"},
            nondifferentiable_policy="fail_closed_at_branch_points",
            effect="pure",
        )
    )

    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)

    assert plan.statuses[0].nondifferentiable_policy == "fail_closed_at_branch_points"
    assert plan.statuses[0].nondifferentiable_boundary == "not_declared"
    assert module.metadata["effects"] == {}
    assert module.metadata["nondifferentiable_policies"] == {}
    assert module.metadata["nondifferentiable_boundaries"] == {}
    assert module.metadata["boundary_contract_primitives"] == []
    assert module.metadata["registry_contract_primitives"] == []
    assert module.metadata["jvp_rule_primitives"] == ["scpn.quantum:policy_only@1"]
    assert module.metadata["vjp_rule_primitives"] == []
    assert module.metadata["reverse_contract_primitives"] == []
    assert module.metadata["reverse_incomplete_primitives"] == ["scpn.quantum:policy_only@1"]
    assert module.metadata["adjoint_contract_primitives"] == []
    assert module.metadata["adjoint_incomplete_primitives"] == ["scpn.quantum:policy_only@1"]
    assert module.metadata["forward_contract_primitives"] == []
    assert module.metadata["forward_incomplete_primitives"] == ["scpn.quantum:policy_only@1"]
    assert module.metadata["transform_contract_primitives"] == []
    assert module.metadata["transform_incomplete_primitives"] == ["scpn.quantum:policy_only@1"]
    assert module.metadata["native_backend_contract_primitives"] == []
    assert module.metadata["native_backend_incomplete_primitives"] == [
        "scpn.quantum:policy_only@1"
    ]
    assert module.metadata["rust_backend_contract_primitives"] == []
    assert module.metadata["rust_backend_incomplete_primitives"] == ["scpn.quantum:policy_only@1"]
    assert module.metadata["rust_backend_blockers"] == {
        "scpn.quantum:policy_only@1": "blocked: no Rust differentiable primitive backend"
    }
    assert module.metadata["llvm_backend_contract_primitives"] == []
    assert module.metadata["llvm_backend_incomplete_primitives"] == ["scpn.quantum:policy_only@1"]
    assert module.metadata["llvm_backend_blockers"] == {
        "scpn.quantum:policy_only@1": "blocked: no LLVM/JIT differentiable primitive backend"
    }
    assert module.metadata["jit_backend_contract_primitives"] == []
    assert module.metadata["jit_backend_incomplete_primitives"] == ["scpn.quantum:policy_only@1"]
    assert module.metadata["jit_backend_blockers"] == {
        "scpn.quantum:policy_only@1": "blocked: no JIT differentiable primitive backend"
    }
    assert module.metadata["mlir_runtime_contract_primitives"] == []
    assert module.metadata["mlir_runtime_incomplete_primitives"] == ["scpn.quantum:policy_only@1"]
    assert module.metadata["mlir_runtime_blockers"] == {
        "scpn.quantum:policy_only@1": "blocked: no MLIR-runtime lowering rule"
    }
    assert module.metadata["mlir_runtime_verification_primitives"] == {}
    assert module.metadata["primitive_readiness"]["scpn.quantum:policy_only@1"]["verdict"] == (
        "registry_incomplete"
    )
    assert module.metadata["primitive_readiness_verdict_counts"] == {"registry_incomplete": 1}
    assert module.metadata["primitive_next_hard_gap"] == {
        "scpn.quantum:policy_only@1": "registry_contract"
    }
    assert module.metadata["primitive_hard_gap_counts"]["registry_contract"] == 1
    assert module.metadata["primitive_hard_gap_primitives"]["registry_contract"] == [
        "scpn.quantum:policy_only@1"
    ]
    assert module.metadata["primitive_hard_gap_priority"][0] == "registry_contract"
    assert module.metadata["primitive_hard_gap_frontier"]["registry_contract"] == {
        "count": 1,
        "next_primitive": "scpn.quantum:policy_only@1",
        "primitives": ["scpn.quantum:policy_only@1"],
    }
    assert module.metadata["uncontracted_primitives"] == ["scpn.quantum:policy_only@1"]
    assert module.resource_counts["boundary_contracts"] == 0
    assert module.resource_counts["registry_contracts"] == 0
    assert module.resource_counts["reverse_contracts"] == 0
    assert module.resource_counts["reverse_incomplete_primitives"] == 1
    assert module.resource_counts["adjoint_contracts"] == 0
    assert module.resource_counts["adjoint_incomplete_primitives"] == 1
    assert module.resource_counts["forward_contracts"] == 0
    assert module.resource_counts["forward_incomplete_primitives"] == 1
    assert module.resource_counts["transform_contracts"] == 0
    assert module.resource_counts["transform_incomplete_primitives"] == 1
    assert module.resource_counts["native_backend_contracts"] == 0
    assert module.resource_counts["native_backend_incomplete_primitives"] == 1
    assert module.resource_counts["rust_backend_contracts"] == 0
    assert module.resource_counts["rust_backend_incomplete_primitives"] == 1
    assert module.resource_counts["rust_backend_blockers"] == 1
    assert module.resource_counts["llvm_backend_contracts"] == 0
    assert module.resource_counts["llvm_backend_incomplete_primitives"] == 1
    assert module.resource_counts["llvm_backend_blockers"] == 1
    assert module.resource_counts["jit_backend_contracts"] == 0
    assert module.resource_counts["jit_backend_incomplete_primitives"] == 1
    assert module.resource_counts["jit_backend_blockers"] == 1
    assert module.resource_counts["mlir_runtime_contracts"] == 0
    assert module.resource_counts["mlir_runtime_incomplete_primitives"] == 1
    assert module.resource_counts["mlir_runtime_blockers"] == 1
    assert module.resource_counts["mlir_runtime_verifications"] == 0
    assert module.resource_counts["primitive_readiness_verdicts"] == 1
    assert module.resource_counts["primitive_hard_gaps"] == 10
    assert module.resource_counts["primitive_next_hard_gaps"] == 1
    assert module.resource_counts["primitive_readiness_registry_incomplete"] == 1
    assert module.resource_counts["primitive_readiness_forward_interchange_only"] == 0
    assert module.resource_counts["primitive_readiness_transform_interchange_only"] == 0
    assert module.resource_counts["primitive_readiness_mlir_runtime_verified"] == 0
    assert module.resource_counts["primitive_readiness_native_executable"] == 0
    assert module.resource_counts["effects"] == 0
    assert module.resource_counts["nondifferentiable_policies"] == 0
    assert module.resource_counts["nondifferentiable_boundaries"] == 0
    assert module.resource_counts["nondifferentiable_boundary_policies"] == 0
    assert module.resource_counts["uncontracted_primitives"] == 1


def test_compiler_ad_rust_backend_metadata_requires_static_signature_parity() -> None:
    """Rust backend claims should fail closed unless they bind the same static contract."""
    identity = PrimitiveIdentity("scpn.compiler_ad.native", "rust_signature_guard", "1")
    base_metadata: dict[str, str] = {
        "native_backend": "native_llvm_jit",
        "native_backend_verification": "verified: native rust signature guard value/JVP/VJP",
        "static_derivative_factory": "native_rust_signature_guard_llvm_jit",
        "static_signature": "primitive:eig;dimension:2;layout:row_major",
        "nondifferentiable_boundary": "real_simple_boundary",
        "nondifferentiable_boundary_policy": "fail_closed",
        "rust_backend": "rust_pyo3",
        "rust_backend_verification": "verified: Rust PyO3 rust signature guard parity",
        "rust_backend_functions": "rust_signature_guard_value,rust_signature_guard_jvp",
    }

    def status_for(lowering_metadata: Mapping[str, str]) -> PrimitiveLoweringStatus:
        return PrimitiveLoweringStatus(
            identity=identity,
            rule_name="rust_signature_guard_rule",
            has_jvp=True,
            has_vjp=True,
            mlir_op="scpn_diff.rust_signature_guard",
            has_batching_rule=True,
            has_shape_rule=True,
            has_dtype_rule=True,
            has_static_argument_rule=True,
            has_lowering_rule=True,
            static_derivative_factory="native_rust_signature_guard_llvm_jit",
            static_signature="primitive:eig;dimension:2;layout:row_major",
            nondifferentiable_policy="real_simple_domain",
            nondifferentiable_boundary="real_simple_boundary",
            nondifferentiable_boundary_policy="fail_closed",
            mlir_lowering="available: executable scpn_diff MLIR-runtime primitive kernel",
            mlir_runtime_verification="verified: rust signature guard JVP",
            rust_lowering="available: Rust PyO3 rust signature guard kernel",
            llvm_lowering="available: native LLVM MCJIT rust signature guard AD kernel",
            jit_lowering="available: native LLVM MCJIT rust signature guard AD kernel",
            lowering_metadata=dict(lowering_metadata),
        )

    with pytest.raises(ValueError, match="rust_backend_signature"):
        status_for(base_metadata)

    missing_functions_metadata = {
        **base_metadata,
        "rust_backend_signature": "primitive:eig;dimension:2;layout:row_major",
    }
    del missing_functions_metadata["rust_backend_functions"]
    with pytest.raises(ValueError, match="rust_backend_functions"):
        status_for(missing_functions_metadata)

    mismatched_metadata = {
        **base_metadata,
        "rust_backend_signature": "primitive:determinant;dimension:2;layout:row_major",
    }
    with pytest.raises(ValueError, match="rust_backend_signature must match static_signature"):
        status_for(mismatched_metadata)

    valid_metadata = {
        **base_metadata,
        "rust_backend_signature": "primitive:eig;dimension:2;layout:row_major",
    }
    status = status_for(valid_metadata)

    assert status.lowering_metadata["rust_backend"] == "rust_pyo3"
    assert status.lowering_metadata["rust_backend_signature"] == status.static_signature
    assert status.lowering_metadata["rust_backend_verification"].startswith("verified:")


def test_compiler_ad_plan_surfaces_static_linalg_lowering_metadata() -> None:
    """Compiler AD planning should expose static linalg signatures without native overclaim."""
    registry = CustomDerivativeRegistry()
    for name in ("matrix_power", "multi_dot"):
        contract = primitive_contract_for(PrimitiveIdentity("scpn.program_ad.linalg", name, "1"))
        registry.register_transform(
            PrimitiveTransformRule(
                identity=contract.identity,
                derivative_rule=contract.derivative_rule,
                batching_rule=contract.batching_rule,
                lowering_metadata=contract.lowering_metadata,
                shape_rule=contract.shape_rule,
                dtype_rule=contract.dtype_rule,
                static_argument_rule=contract.static_argument_rule,
                nondifferentiable_policy=contract.nondifferentiable_policy,
                effect=contract.effect,
            )
        )

    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)

    statuses = {status.identity.name: status for status in plan.statuses}
    expected_jvp_primitives = [status.identity.key for status in plan.statuses if status.has_jvp]
    expected_vjp_primitives = [status.identity.key for status in plan.statuses if status.has_vjp]
    expected_reverse_contracts = [
        status.identity.key
        for status in plan.statuses
        if status.has_vjp
        and status.identity.key in module.metadata["registry_contract_primitives"]
    ]
    expected_reverse_incomplete = [
        status.identity.key for status in plan.statuses if not status.has_vjp
    ]
    expected_adjoint_contracts = [
        status.identity.key
        for status in plan.statuses
        if status.has_vjp
        and status.effect == "pure"
        and status.identity.key in module.metadata["registry_contract_primitives"]
    ]
    expected_adjoint_incomplete = [
        status.identity.key
        for status in plan.statuses
        if status.identity.key not in expected_adjoint_contracts
    ]
    expected_forward_contracts = [
        status.identity.key
        for status in plan.statuses
        if status.has_jvp
        and status.identity.key in module.metadata["registry_contract_primitives"]
    ]
    expected_forward_incomplete = [
        status.identity.key
        for status in plan.statuses
        if status.identity.key not in expected_forward_contracts
    ]
    expected_transform_contracts = [
        status.identity.key
        for status in plan.statuses
        if status.identity.key in module.metadata["forward_contract_primitives"]
        and status.identity.key in module.metadata["reverse_contract_primitives"]
        and status.identity.key in module.metadata["adjoint_contract_primitives"]
    ]
    expected_transform_incomplete = [
        status.identity.key
        for status in plan.statuses
        if status.identity.key not in expected_transform_contracts
    ]
    expected_native_contracts = [
        status.identity.key
        for status in plan.statuses
        if "blocked" not in status.rust_lowering.lower()
        and "blocked" not in status.llvm_lowering.lower()
        and "blocked" not in status.jit_lowering.lower()
    ]
    expected_native_incomplete = [
        status.identity.key
        for status in plan.statuses
        if status.identity.key not in expected_native_contracts
    ]
    expected_rust_contracts = [
        status.identity.key
        for status in plan.statuses
        if "blocked" not in status.rust_lowering.lower()
    ]
    expected_rust_incomplete = [
        status.identity.key
        for status in plan.statuses
        if status.identity.key not in expected_rust_contracts
    ]
    expected_llvm_contracts = [
        status.identity.key
        for status in plan.statuses
        if "blocked" not in status.llvm_lowering.lower()
    ]
    expected_llvm_incomplete = [
        status.identity.key
        for status in plan.statuses
        if status.identity.key not in expected_llvm_contracts
    ]
    expected_jit_contracts = [
        status.identity.key
        for status in plan.statuses
        if "blocked" not in status.jit_lowering.lower()
    ]
    expected_jit_incomplete = [
        status.identity.key
        for status in plan.statuses
        if status.identity.key not in expected_jit_contracts
    ]
    expected_mlir_runtime_contracts = [
        status.identity.key
        for status in plan.statuses
        if status.has_lowering_rule and status.mlir_runtime_verification.startswith("verified:")
    ]
    expected_mlir_runtime_incomplete = [
        status.identity.key
        for status in plan.statuses
        if status.identity.key not in expected_mlir_runtime_contracts
    ]
    assert statuses["matrix_power"].has_batching_rule is True
    assert statuses["multi_dot"].has_batching_rule is True
    assert statuses["matrix_power"].has_static_argument_rule is True
    assert statuses["multi_dot"].has_static_argument_rule is True
    assert statuses["matrix_power"].nondifferentiable_boundary == "negative_power_singular_matrix"
    assert statuses["matrix_power"].nondifferentiable_boundary_policy == "fail_closed"
    assert statuses["multi_dot"].nondifferentiable_boundary == "static_shape_alignment"
    assert statuses["multi_dot"].nondifferentiable_boundary_policy == "fail_closed"
    assert (
        statuses["matrix_power"].lowering_metadata["static_derivative_factory"]
        == "program_ad_linalg_matrix_power_derivative_rule"
    )
    assert statuses["matrix_power"].lowering_metadata["static_signature"] == "power:i64"
    assert (
        statuses["multi_dot"].lowering_metadata["static_signature"]
        == "operand_shapes:ranked_tensor_shape_sequence"
    )
    assert module.resource_counts["executable_backends"] == 0
    assert module.metadata["static_argument_primitives"] == [
        "scpn.program_ad.linalg:matrix_power@1",
        "scpn.program_ad.linalg:multi_dot@1",
    ]
    assert module.metadata["batching_rule_primitives"] == [
        "scpn.program_ad.linalg:matrix_power@1",
        "scpn.program_ad.linalg:multi_dot@1",
    ]
    assert module.metadata["nondifferentiable_boundaries"] == {
        "scpn.program_ad.linalg:matrix_power@1": "negative_power_singular_matrix",
        "scpn.program_ad.linalg:multi_dot@1": "static_shape_alignment",
    }
    assert module.metadata["nondifferentiable_boundary_policies"] == {
        "scpn.program_ad.linalg:matrix_power@1": "fail_closed",
        "scpn.program_ad.linalg:multi_dot@1": "fail_closed",
    }
    assert module.metadata["boundary_contract_primitives"] == [
        "scpn.program_ad.linalg:matrix_power@1",
        "scpn.program_ad.linalg:multi_dot@1",
    ]
    assert module.metadata["registry_contract_primitives"] == [
        "scpn.program_ad.linalg:matrix_power@1",
        "scpn.program_ad.linalg:multi_dot@1",
    ]
    assert module.metadata["jvp_rule_primitives"] == expected_jvp_primitives
    assert module.metadata["vjp_rule_primitives"] == expected_vjp_primitives
    assert module.metadata["reverse_contract_primitives"] == expected_reverse_contracts
    assert module.metadata["reverse_incomplete_primitives"] == expected_reverse_incomplete
    assert module.metadata["adjoint_contract_primitives"] == expected_adjoint_contracts
    assert module.metadata["adjoint_incomplete_primitives"] == expected_adjoint_incomplete
    assert module.metadata["forward_contract_primitives"] == expected_forward_contracts
    assert module.metadata["forward_incomplete_primitives"] == expected_forward_incomplete
    assert module.metadata["transform_contract_primitives"] == expected_transform_contracts
    assert module.metadata["transform_incomplete_primitives"] == expected_transform_incomplete
    assert module.metadata["native_backend_contract_primitives"] == expected_native_contracts
    assert module.metadata["native_backend_incomplete_primitives"] == expected_native_incomplete
    assert module.metadata["rust_backend_contract_primitives"] == expected_rust_contracts
    assert module.metadata["rust_backend_incomplete_primitives"] == expected_rust_incomplete
    assert module.metadata["rust_backend_blockers"] == {
        status.identity.key: status.rust_lowering
        for status in plan.statuses
        if "blocked" in status.rust_lowering.lower()
    }
    assert module.metadata["llvm_backend_contract_primitives"] == expected_llvm_contracts
    assert module.metadata["llvm_backend_incomplete_primitives"] == expected_llvm_incomplete
    assert module.metadata["llvm_backend_blockers"] == {
        status.identity.key: status.llvm_lowering
        for status in plan.statuses
        if "blocked" in status.llvm_lowering.lower()
    }
    assert module.metadata["jit_backend_contract_primitives"] == expected_jit_contracts
    assert module.metadata["jit_backend_incomplete_primitives"] == expected_jit_incomplete
    assert module.metadata["jit_backend_blockers"] == {
        status.identity.key: status.jit_lowering
        for status in plan.statuses
        if "blocked" in status.jit_lowering.lower()
    }
    assert module.metadata["mlir_runtime_contract_primitives"] == expected_mlir_runtime_contracts
    assert (
        module.metadata["mlir_runtime_incomplete_primitives"] == expected_mlir_runtime_incomplete
    )
    assert module.metadata["mlir_runtime_blockers"] == {
        status.identity.key: "blocked: no MLIR-runtime lowering rule"
        for status in plan.statuses
        if status.identity.key in expected_mlir_runtime_incomplete
    }
    assert module.metadata["mlir_runtime_verification_primitives"] == {}
    assert module.metadata["primitive_readiness"] == {
        status.identity.key: {
            "adjoint_contract": status.identity.key
            in module.metadata["adjoint_contract_primitives"],
            "forward_contract": status.identity.key
            in module.metadata["forward_contract_primitives"],
            "jit_backend_contract": status.identity.key in expected_jit_contracts,
            "llvm_backend_contract": status.identity.key in expected_llvm_contracts,
            "mlir_runtime_contract": status.identity.key in expected_mlir_runtime_contracts,
            "native_backend_contract": status.identity.key in expected_native_contracts,
            "registry_contract": status.identity.key
            in module.metadata["registry_contract_primitives"],
            "reverse_contract": status.identity.key
            in module.metadata["reverse_contract_primitives"],
            "rust_backend_contract": status.identity.key in expected_rust_contracts,
            "transform_contract": status.identity.key in expected_transform_contracts,
            "verdict": "forward_interchange_only",
        }
        for status in plan.statuses
    }
    assert module.metadata["primitive_readiness_verdict_counts"] == {
        "forward_interchange_only": len(plan.statuses)
    }
    expected_linalg_hard_gaps = {
        status.identity.key: [
            "reverse_contract",
            "adjoint_contract",
            "transform_contract",
            "mlir_runtime_contract",
            "rust_backend_contract",
            "llvm_backend_contract",
            "jit_backend_contract",
            "native_backend_contract",
        ]
        for status in plan.statuses
    }
    assert module.metadata["primitive_hard_gaps"] == expected_linalg_hard_gaps
    assert module.metadata["primitive_next_hard_gap"] == {
        status.identity.key: "reverse_contract" for status in plan.statuses
    }
    assert module.metadata["primitive_hard_gap_counts"] == {
        "adjoint_contract": len(plan.statuses),
        "jit_backend_contract": len(plan.statuses),
        "llvm_backend_contract": len(plan.statuses),
        "mlir_runtime_contract": len(plan.statuses),
        "native_backend_contract": len(plan.statuses),
        "reverse_contract": len(plan.statuses),
        "rust_backend_contract": len(plan.statuses),
        "transform_contract": len(plan.statuses),
    }
    assert module.metadata["primitive_hard_gap_primitives"] == {
        "adjoint_contract": [status.identity.key for status in plan.statuses],
        "jit_backend_contract": [status.identity.key for status in plan.statuses],
        "llvm_backend_contract": [status.identity.key for status in plan.statuses],
        "mlir_runtime_contract": [status.identity.key for status in plan.statuses],
        "native_backend_contract": [status.identity.key for status in plan.statuses],
        "reverse_contract": [status.identity.key for status in plan.statuses],
        "rust_backend_contract": [status.identity.key for status in plan.statuses],
        "transform_contract": [status.identity.key for status in plan.statuses],
    }
    assert module.metadata["primitive_hard_gap_priority"] == [
        "reverse_contract",
        "adjoint_contract",
        "transform_contract",
        "mlir_runtime_contract",
        "rust_backend_contract",
        "llvm_backend_contract",
        "jit_backend_contract",
        "native_backend_contract",
    ]
    assert module.metadata["primitive_hard_gap_frontier"]["reverse_contract"] == {
        "count": len(plan.statuses),
        "next_primitive": plan.statuses[0].identity.key,
        "primitives": [status.identity.key for status in plan.statuses],
    }
    assert module.metadata["primitive_hard_gap_frontier"]["native_backend_contract"] == {
        "count": len(plan.statuses),
        "next_primitive": plan.statuses[0].identity.key,
        "primitives": [status.identity.key for status in plan.statuses],
    }
    assert module.resource_counts["batching_rules"] == 2
    assert module.resource_counts["boundary_contracts"] == 2
    assert module.resource_counts["registry_contracts"] == 2
    assert module.resource_counts["reverse_contracts"] == len(expected_reverse_contracts)
    assert module.resource_counts["reverse_incomplete_primitives"] == len(
        expected_reverse_incomplete
    )
    assert module.resource_counts["adjoint_contracts"] == len(expected_adjoint_contracts)
    assert module.resource_counts["adjoint_incomplete_primitives"] == len(
        expected_adjoint_incomplete
    )
    assert module.resource_counts["forward_contracts"] == len(expected_forward_contracts)
    assert module.resource_counts["forward_incomplete_primitives"] == len(
        expected_forward_incomplete
    )
    assert module.resource_counts["transform_contracts"] == len(expected_transform_contracts)
    assert module.resource_counts["transform_incomplete_primitives"] == len(
        expected_transform_incomplete
    )
    assert module.resource_counts["native_backend_contracts"] == len(expected_native_contracts)
    assert module.resource_counts["native_backend_incomplete_primitives"] == len(
        expected_native_incomplete
    )
    assert module.resource_counts["rust_backend_contracts"] == len(expected_rust_contracts)
    assert module.resource_counts["rust_backend_incomplete_primitives"] == len(
        expected_rust_incomplete
    )
    assert module.resource_counts["rust_backend_blockers"] == len(expected_rust_incomplete)
    assert module.resource_counts["llvm_backend_contracts"] == len(expected_llvm_contracts)
    assert module.resource_counts["llvm_backend_incomplete_primitives"] == len(
        expected_llvm_incomplete
    )
    assert module.resource_counts["llvm_backend_blockers"] == len(expected_llvm_incomplete)
    assert module.resource_counts["jit_backend_contracts"] == len(expected_jit_contracts)
    assert module.resource_counts["jit_backend_incomplete_primitives"] == len(
        expected_jit_incomplete
    )
    assert module.resource_counts["jit_backend_blockers"] == len(expected_jit_incomplete)
    assert module.resource_counts["mlir_runtime_contracts"] == len(expected_mlir_runtime_contracts)
    assert module.resource_counts["mlir_runtime_incomplete_primitives"] == len(
        expected_mlir_runtime_incomplete
    )
    assert module.resource_counts["mlir_runtime_blockers"] == len(expected_mlir_runtime_incomplete)
    assert module.resource_counts["mlir_runtime_verifications"] == 0
    assert module.resource_counts["primitive_readiness_verdicts"] == len(plan.statuses)
    assert module.resource_counts["primitive_hard_gaps"] == 8 * len(plan.statuses)
    assert module.resource_counts["primitive_next_hard_gaps"] == len(plan.statuses)
    assert module.resource_counts["primitive_hard_gap_priority_classes"] == 8
    assert module.resource_counts["primitive_hard_gap_frontier_classes"] == 8
    assert module.resource_counts["primitive_readiness_registry_incomplete"] == 0
    assert module.resource_counts["primitive_readiness_forward_interchange_only"] == len(
        plan.statuses
    )
    assert module.resource_counts["primitive_readiness_transform_interchange_only"] == 0
    assert module.resource_counts["primitive_readiness_mlir_runtime_verified"] == 0
    assert module.resource_counts["primitive_readiness_native_executable"] == 0
    assert module.resource_counts["nondifferentiable_boundaries"] == 2
    assert module.resource_counts["nondifferentiable_boundary_policies"] == 2
    assert "batching_rule = true" in module.text
    assert "static_argument_rule = true" in module.text
    assert 'boundary = "negative_power_singular_matrix"' in module.text
    assert 'boundary_policy = "fail_closed"' in module.text
    assert 'key = "static_signature", value = "power:i64"' in module.text
    assert (
        'key = "static_signature", value = "operand_shapes:ranked_tensor_shape_sequence"'
        in module.text
    )
    assert "blocked_until_executable_linalg_lowering" in module.text


def test_compiler_ad_plan_promotes_static_derivative_factory_contracts() -> None:
    """Compiler AD plans should expose static derivative factory contracts directly."""
    primitive_keys = (
        "scpn.program_ad.array:getitem",
        "scpn.program_ad.shape:reshape",
        "scpn.program_ad.elementwise:multiply",
        "scpn.program_ad.product:matmul",
        "scpn.program_ad.cumulative:diff",
        "scpn.program_ad.linalg:solve",
    )
    expected_factories = {
        "scpn.program_ad.array:getitem@1": "program_ad_array_getitem_derivative_rule",
        "scpn.program_ad.shape:reshape@1": "program_ad_shape_reshape_derivative_rule",
        "scpn.program_ad.elementwise:multiply@1": (
            "program_ad_elementwise_binary_derivative_rule"
        ),
        "scpn.program_ad.product:matmul@1": "program_ad_product_matmul_derivative_rule",
        "scpn.program_ad.cumulative:diff@1": "program_ad_cumulative_diff_derivative_rule",
        "scpn.program_ad.linalg:solve@1": "program_ad_linalg_solve_derivative_rule",
    }
    expected_signatures = {
        "scpn.program_ad.array:getitem@1": (
            "source_shape:ranked_tensor_shape;index:static_gather_index"
        ),
        "scpn.program_ad.shape:reshape@1": "source_shape:ranked_tensor_shape;target_shape",
        "scpn.program_ad.elementwise:multiply@1": (
            "left_shape:ranked_tensor_shape;right_shape:ranked_tensor_shape"
        ),
        "scpn.program_ad.product:matmul@1": (
            "left_shape:ranked_tensor_shape;right_shape:ranked_tensor_shape"
        ),
        "scpn.program_ad.cumulative:diff@1": "source_shape:ranked_tensor_shape;order_axis",
        "scpn.program_ad.linalg:solve@1": "matrix_shape:rank2_square;rhs_shape:rank1_or_rank2",
    }
    registry = CustomDerivativeRegistry()
    for key in primitive_keys:
        contract = primitive_contract_for(key)
        registry.register_transform(
            PrimitiveTransformRule(
                identity=contract.identity,
                derivative_rule=contract.derivative_rule,
                batching_rule=contract.batching_rule,
                lowering_metadata=contract.lowering_metadata,
                shape_rule=contract.shape_rule,
                dtype_rule=contract.dtype_rule,
                static_argument_rule=contract.static_argument_rule,
                nondifferentiable_policy=contract.nondifferentiable_policy,
                effect=contract.effect,
            )
        )

    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)

    statuses = {status.identity.key: status for status in plan.statuses}
    assert {
        key: status.static_derivative_factory for key, status in statuses.items()
    } == expected_factories
    assert {
        key: status.static_signature for key, status in statuses.items()
    } == expected_signatures
    assert module.metadata["static_derivative_factories"] == expected_factories
    assert module.metadata["static_derivative_signatures"] == expected_signatures
    assert module.resource_counts["static_derivative_factories"] == len(expected_factories)
    assert module.resource_counts["static_derivative_signatures"] == len(expected_signatures)
    assert module.resource_counts["shape_rules"] == len(expected_factories)
    assert module.resource_counts["dtype_rules"] == len(expected_factories)
    assert module.resource_counts["static_argument_rules"] == len(expected_factories)
    expected_rule_primitives = sorted(expected_factories)
    expected_policies = {
        key: statuses[key].nondifferentiable_policy for key in expected_rule_primitives
    }
    expected_effects = {key: statuses[key].effect for key in expected_rule_primitives}
    assert module.metadata["shape_rule_primitives"] == expected_rule_primitives
    assert module.metadata["dtype_rule_primitives"] == expected_rule_primitives
    assert module.metadata["static_argument_primitives"] == expected_rule_primitives
    assert module.metadata["nondifferentiable_policies"] == expected_policies
    assert module.metadata["effects"] == expected_effects
    assert module.resource_counts["nondifferentiable_policies"] == len(expected_policies)
    assert module.resource_counts["effects"] == len(expected_effects)
    assert 'static_derivative_factory = "program_ad_array_getitem_derivative_rule"' in module.text
    assert (
        'static_signature = "source_shape:ranked_tensor_shape;index:static_gather_index"'
        in module.text
    )
    assert 'static_derivative_factory = "program_ad_linalg_solve_derivative_rule"' in module.text
    assert 'static_signature = "matrix_shape:rank2_square;rhs_shape:rank1_or_rank2"' in module.text


def test_compiler_ad_plan_does_not_count_not_required_static_factories() -> None:
    """Compiler AD resource counts should count only actual static factories."""
    registry = CustomDerivativeRegistry()
    for key in (
        "scpn.program_ad.elementwise:sin",
        "scpn.program_ad.elementwise:multiply",
    ):
        contract = primitive_contract_for(key)
        registry.register_transform(
            PrimitiveTransformRule(
                identity=contract.identity,
                derivative_rule=contract.derivative_rule,
                batching_rule=contract.batching_rule,
                lowering_metadata=contract.lowering_metadata,
                shape_rule=contract.shape_rule,
                dtype_rule=contract.dtype_rule,
                static_argument_rule=contract.static_argument_rule,
                nondifferentiable_policy=contract.nondifferentiable_policy,
                effect=contract.effect,
            )
        )

    module = compile_compiler_ad_transform_plan_to_mlir(build_compiler_ad_transform_plan(registry))

    assert module.resource_counts["static_derivative_factories"] == 1
    assert module.resource_counts["static_derivative_signatures"] == 1
    assert module.metadata["static_derivative_factories"] == {
        "scpn.program_ad.elementwise:multiply@1": ("program_ad_elementwise_binary_derivative_rule")
    }
    assert module.metadata["static_derivative_signatures"] == {
        "scpn.program_ad.elementwise:multiply@1": (
            "left_shape:ranked_tensor_shape;right_shape:ranked_tensor_shape"
        )
    }


def test_compiler_ad_transform_plan_rejects_incomplete_static_factory_metadata() -> None:
    """Compiler AD planning should fail closed on unpaired static factory metadata."""
    rule = CustomDerivativeRule(
        name="incomplete_static_metadata_rule",
        value_fn=lambda values: np.array([values[0]], dtype=np.float64),
        jvp_rule=lambda _values, tangent: np.array([tangent[0]], dtype=np.float64),
    )
    cases = (
        (
            PrimitiveIdentity("scpn.test", "missing_static_signature", "1"),
            {
                "mlir_op": "scpn_diff.missing_static_signature",
                "static_derivative_factory": "program_ad_test_derivative_rule",
            },
            "static_signature",
        ),
        (
            PrimitiveIdentity("scpn.test", "missing_static_factory", "1"),
            {
                "mlir_op": "scpn_diff.missing_static_factory",
                "static_signature": "source_shape:ranked_tensor_shape",
            },
            "static_derivative_factory",
        ),
        (
            PrimitiveIdentity("scpn.test", "factory_not_required_with_signature", "1"),
            {
                "mlir_op": "scpn_diff.factory_not_required_with_signature",
                "static_derivative_factory": "not_required",
                "static_signature": "source_shape:ranked_tensor_shape",
            },
            "static_derivative_factory",
        ),
    )
    for identity, lowering_metadata, message in cases:
        registry = CustomDerivativeRegistry()
        registry.register_transform(
            PrimitiveTransformRule(
                identity=identity,
                derivative_rule=rule,
                lowering_metadata=lowering_metadata,
            )
        )

        with pytest.raises(ValueError, match=message):
            build_compiler_ad_transform_plan(registry)


def test_primitive_lowering_status_rejects_conflicting_static_metadata() -> None:
    """Primitive lowering status should reject contradictory static metadata fields."""
    identity = PrimitiveIdentity("scpn.test", "conflicting_static_metadata", "1")
    rule_name = "conflicting_static_metadata_rule"
    cases = (
        (
            {
                "mlir_op": "scpn_diff.conflicting_static_metadata",
                "static_derivative_factory": "metadata_factory_rule",
                "static_signature": "source_shape:ranked_tensor_shape",
            },
            "field_factory_rule",
            "source_shape:ranked_tensor_shape",
            "static_derivative_factory",
        ),
        (
            {
                "mlir_op": "scpn_diff.conflicting_static_metadata",
                "static_derivative_factory": "metadata_factory_rule",
                "static_signature": "source_shape:ranked_tensor_shape",
            },
            "metadata_factory_rule",
            "other_shape:ranked_tensor_shape",
            "static_signature",
        ),
    )
    for lowering_metadata, static_factory, static_signature, message in cases:
        with pytest.raises(ValueError, match=message):
            PrimitiveLoweringStatus(
                identity=identity,
                rule_name=rule_name,
                has_jvp=True,
                has_vjp=False,
                mlir_op="scpn_diff.conflicting_static_metadata",
                lowering_metadata=lowering_metadata,
                static_derivative_factory=static_factory,
                static_signature=static_signature,
            )


def test_primitive_lowering_status_rejects_inconsistent_lowering_rule_provenance() -> None:
    """Primitive lowering status should keep lowering-rule provenance consistent."""
    identity = PrimitiveIdentity("scpn.test", "lowering_provenance", "1")
    cases = (
        (
            False,
            "available: executable scpn_diff MLIR-runtime primitive kernel",
            "has_lowering_rule",
        ),
        (True, "available: scpn_diff dialect interchange", "mlir_lowering"),
    )
    for has_lowering_rule, mlir_lowering, message in cases:
        with pytest.raises(ValueError, match=message):
            PrimitiveLoweringStatus(
                identity=identity,
                rule_name="lowering_provenance_rule",
                has_jvp=True,
                has_vjp=False,
                mlir_op="scpn_diff.lowering_provenance",
                has_lowering_rule=has_lowering_rule,
                mlir_lowering=mlir_lowering,
            )
    with pytest.raises(ValueError, match="mlir_runtime_verification"):
        PrimitiveLoweringStatus(
            identity=identity,
            rule_name="lowering_provenance_rule",
            has_jvp=True,
            has_vjp=False,
            mlir_op="scpn_diff.lowering_provenance",
            has_lowering_rule=True,
            mlir_lowering="available: executable scpn_diff MLIR-runtime primitive kernel",
        )


def test_primitive_lowering_status_rejects_native_backend_overclaims() -> None:
    """Primitive lowering status should fail closed on native backend availability claims."""
    identity = PrimitiveIdentity("scpn.test", "native_backend_overclaim", "1")
    with pytest.raises(ValueError, match="rust_lowering"):
        PrimitiveLoweringStatus(
            identity=identity,
            rule_name="native_backend_overclaim_rule",
            has_jvp=True,
            has_vjp=False,
            mlir_op="scpn_diff.native_backend_overclaim",
            rust_lowering="available: Rust differentiable backend",
        )
    with pytest.raises(ValueError, match="llvm_lowering"):
        PrimitiveLoweringStatus(
            identity=identity,
            rule_name="native_backend_overclaim_rule",
            has_jvp=True,
            has_vjp=False,
            mlir_op="scpn_diff.native_backend_overclaim",
            llvm_lowering="available: LLVM/JIT differentiable backend",
        )
    with pytest.raises(ValueError, match="jit_lowering"):
        PrimitiveLoweringStatus(
            identity=identity,
            rule_name="native_backend_overclaim_rule",
            has_jvp=True,
            has_vjp=False,
            mlir_op="scpn_diff.native_backend_overclaim",
            jit_lowering="available: JIT differentiable backend",
        )


def test_static_linalg_lowering_factories_verify_executable_kernels() -> None:
    """Static linalg lowering factories should execute verified MLIR-runtime kernels."""
    matrix = np.array([[2.0, -0.5], [0.75, 1.5]], dtype=np.float64)
    tangent_matrix = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float64)
    left = np.array([0.75, -1.5], dtype=np.float64)
    right = np.array([1.25, 0.5], dtype=np.float64)
    tangent_left = np.array([0.2, -0.1], dtype=np.float64)
    tangent_right = np.array([0.4, -0.3], dtype=np.float64)

    power_rule = program_ad_linalg_matrix_power_derivative_rule(2)
    power_lowering = make_program_ad_linalg_matrix_power_executable_lowering_rule(
        2,
        matrix.reshape(-1),
        sample_tangent=tangent_matrix.reshape(-1),
    )
    power_kernel = power_lowering(power_rule)

    assert power_kernel.backend == "mlir_runtime"
    assert power_kernel.verification.passed is True
    assert power_kernel.verification.jvp_close is True
    assert "native LLVM/JIT code generation remains fail-closed" in power_kernel.claim_boundary
    np.testing.assert_allclose(
        power_kernel.value(matrix.reshape(-1)), (matrix @ matrix).reshape(-1)
    )
    np.testing.assert_allclose(
        power_kernel.jvp(matrix.reshape(-1), tangent_matrix.reshape(-1)),
        (tangent_matrix @ matrix + matrix @ tangent_matrix).reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    multi_values = np.concatenate((left, matrix.reshape(-1), right))
    multi_tangent = np.concatenate((tangent_left, tangent_matrix.reshape(-1), tangent_right))
    multi_rule = program_ad_linalg_multi_dot_derivative_rule(((2,), (2, 2), (2,)))
    multi_lowering = make_program_ad_linalg_multi_dot_executable_lowering_rule(
        ((2,), (2, 2), (2,)),
        multi_values,
        sample_tangent=multi_tangent,
    )
    multi_kernel = multi_lowering(multi_rule)

    expected_multi_jvp = (
        np.linalg.multi_dot((tangent_left, matrix, right))
        + np.linalg.multi_dot((left, tangent_matrix, right))
        + np.linalg.multi_dot((left, matrix, tangent_right))
    )
    assert multi_kernel.backend == "mlir_runtime"
    assert multi_kernel.verification.passed is True
    assert multi_kernel.verification.jvp_close is True
    np.testing.assert_allclose(
        multi_kernel.value(multi_values),
        np.asarray(np.linalg.multi_dot((left, matrix, right))).reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        multi_kernel.jvp(multi_values, multi_tangent),
        np.asarray(expected_multi_jvp).reshape(-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_static_linalg_lowering_rules_register_into_compiler_ad_plan() -> None:
    """Concrete static linalg lowering rules should mark only MLIR-runtime execution available."""
    identity = PrimitiveIdentity("scpn.program_ad.linalg", "matrix_power", "1")
    contract = primitive_contract_for(identity)
    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=contract.derivative_rule,
            batching_rule=contract.batching_rule,
            lowering_rule=make_program_ad_linalg_matrix_power_executable_lowering_rule(
                2, np.array([2.0, -0.5, 0.75, 1.5], dtype=np.float64)
            ),
            lowering_metadata={
                **contract.lowering_metadata,
                "mlir": "available: executable scpn_diff MLIR-runtime linalg kernel",
                "mlir_runtime_verification": "verified: deterministic matrix_power sample JVP",
            },
            shape_rule=contract.shape_rule,
            dtype_rule=contract.dtype_rule,
            static_argument_rule=contract.static_argument_rule,
            nondifferentiable_policy=contract.nondifferentiable_policy,
            effect=contract.effect,
        )
    )

    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    kernel = compile_registered_primitive_to_executable(registry, identity, np.array([0.0]))

    assert plan.statuses[0].has_lowering_rule is True
    assert plan.statuses[0].has_static_argument_rule is True
    assert (
        plan.statuses[0].mlir_lowering
        == "available: executable scpn_diff MLIR-runtime linalg kernel"
    )
    assert (
        plan.statuses[0].mlir_runtime_verification
        == "verified: deterministic matrix_power sample JVP"
    )
    assert module.metadata["mlir_runtime_lowering_primitives"] == [
        "scpn.program_ad.linalg:matrix_power@1"
    ]
    assert module.metadata["mlir_runtime_contract_primitives"] == [
        "scpn.program_ad.linalg:matrix_power@1"
    ]
    assert module.metadata["mlir_runtime_incomplete_primitives"] == []
    assert module.metadata["mlir_runtime_blockers"] == {}
    assert module.metadata["mlir_runtime_verification_primitives"] == {
        "scpn.program_ad.linalg:matrix_power@1": (
            "verified: deterministic matrix_power sample JVP"
        )
    }
    assert module.metadata["primitive_readiness"]["scpn.program_ad.linalg:matrix_power@1"] == {
        "adjoint_contract": False,
        "forward_contract": True,
        "jit_backend_contract": False,
        "llvm_backend_contract": False,
        "mlir_runtime_contract": True,
        "native_backend_contract": False,
        "registry_contract": True,
        "reverse_contract": False,
        "rust_backend_contract": False,
        "transform_contract": False,
        "verdict": "mlir_runtime_verified",
    }
    assert module.metadata["primitive_readiness_verdict_counts"] == {"mlir_runtime_verified": 1}
    assert module.metadata["primitive_hard_gaps"]["scpn.program_ad.linalg:matrix_power@1"] == [
        "reverse_contract",
        "adjoint_contract",
        "transform_contract",
        "rust_backend_contract",
        "llvm_backend_contract",
        "jit_backend_contract",
        "native_backend_contract",
    ]
    assert module.metadata["primitive_next_hard_gap"] == {
        "scpn.program_ad.linalg:matrix_power@1": "reverse_contract"
    }
    assert module.metadata["primitive_hard_gap_counts"] == {
        "adjoint_contract": 1,
        "jit_backend_contract": 1,
        "llvm_backend_contract": 1,
        "native_backend_contract": 1,
        "reverse_contract": 1,
        "rust_backend_contract": 1,
        "transform_contract": 1,
    }
    assert module.metadata["primitive_hard_gap_primitives"] == {
        "adjoint_contract": ["scpn.program_ad.linalg:matrix_power@1"],
        "jit_backend_contract": ["scpn.program_ad.linalg:matrix_power@1"],
        "llvm_backend_contract": ["scpn.program_ad.linalg:matrix_power@1"],
        "native_backend_contract": ["scpn.program_ad.linalg:matrix_power@1"],
        "reverse_contract": ["scpn.program_ad.linalg:matrix_power@1"],
        "rust_backend_contract": ["scpn.program_ad.linalg:matrix_power@1"],
        "transform_contract": ["scpn.program_ad.linalg:matrix_power@1"],
    }
    assert module.metadata["primitive_hard_gap_priority"] == [
        "reverse_contract",
        "adjoint_contract",
        "transform_contract",
        "rust_backend_contract",
        "llvm_backend_contract",
        "jit_backend_contract",
        "native_backend_contract",
    ]
    assert module.metadata["primitive_hard_gap_frontier"]["reverse_contract"] == {
        "count": 1,
        "next_primitive": "scpn.program_ad.linalg:matrix_power@1",
        "primitives": ["scpn.program_ad.linalg:matrix_power@1"],
    }
    assert module.metadata["primitive_hard_gap_frontier"]["native_backend_contract"] == {
        "count": 1,
        "next_primitive": "scpn.program_ad.linalg:matrix_power@1",
        "primitives": ["scpn.program_ad.linalg:matrix_power@1"],
    }
    assert module.metadata["rust_backend_contract_primitives"] == []
    assert module.metadata["rust_backend_incomplete_primitives"] == [
        "scpn.program_ad.linalg:matrix_power@1"
    ]
    assert module.metadata["rust_backend_blockers"] == {
        "scpn.program_ad.linalg:matrix_power@1": plan.statuses[0].rust_lowering
    }
    assert module.metadata["llvm_backend_contract_primitives"] == []
    assert module.metadata["llvm_backend_incomplete_primitives"] == [
        "scpn.program_ad.linalg:matrix_power@1"
    ]
    assert module.metadata["llvm_backend_blockers"] == {
        "scpn.program_ad.linalg:matrix_power@1": plan.statuses[0].llvm_lowering
    }
    assert module.metadata["jit_backend_contract_primitives"] == []
    assert module.metadata["jit_backend_incomplete_primitives"] == [
        "scpn.program_ad.linalg:matrix_power@1"
    ]
    assert module.metadata["jit_backend_blockers"] == {
        "scpn.program_ad.linalg:matrix_power@1": plan.statuses[0].jit_lowering
    }
    assert module.resource_counts["mlir_runtime_lowerings"] == 1
    assert module.resource_counts["mlir_runtime_contracts"] == 1
    assert module.resource_counts["mlir_runtime_incomplete_primitives"] == 0
    assert module.resource_counts["mlir_runtime_blockers"] == 0
    assert module.resource_counts["mlir_runtime_verifications"] == 1
    assert module.resource_counts["primitive_readiness_verdicts"] == 1
    assert module.resource_counts["primitive_hard_gaps"] == 7
    assert module.resource_counts["primitive_next_hard_gaps"] == 1
    assert module.resource_counts["primitive_hard_gap_priority_classes"] == 7
    assert module.resource_counts["primitive_hard_gap_frontier_classes"] == 7
    assert module.resource_counts["primitive_readiness_registry_incomplete"] == 0
    assert module.resource_counts["primitive_readiness_forward_interchange_only"] == 0
    assert module.resource_counts["primitive_readiness_transform_interchange_only"] == 0
    assert module.resource_counts["primitive_readiness_mlir_runtime_verified"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 0
    assert module.resource_counts["rust_backend_contracts"] == 0
    assert module.resource_counts["rust_backend_incomplete_primitives"] == 1
    assert module.resource_counts["rust_backend_blockers"] == 1
    assert module.resource_counts["llvm_backend_contracts"] == 0
    assert module.resource_counts["llvm_backend_incomplete_primitives"] == 1
    assert module.resource_counts["llvm_backend_blockers"] == 1
    assert module.resource_counts["jit_backend_contracts"] == 0
    assert module.resource_counts["jit_backend_incomplete_primitives"] == 1
    assert module.resource_counts["jit_backend_blockers"] == 1
    assert module.resource_counts["executable_backends"] == 0
    assert module.metadata["executable_backend"] == "none"
    assert "available: executable scpn_diff MLIR-runtime linalg kernel" in module.text
    assert "verified: deterministic matrix_power sample JVP" in module.text
    assert "blocked_until_executable_linalg_lowering" in module.text
    assert kernel.backend == "mlir_runtime"
    assert kernel.verification.passed is True


def test_static_linalg_lowering_rules_register_multi_shape_compiler_ad_plan() -> None:
    """Static linalg lowerings should plan and verify multiple concrete signatures."""
    registry = CustomDerivativeRegistry()
    matrix_power_identity = PrimitiveIdentity("scpn.program_ad.linalg", "matrix_power", "1")
    multi_dot_identity = PrimitiveIdentity("scpn.program_ad.linalg", "multi_dot", "1")

    matrix_power_contract = primitive_contract_for(matrix_power_identity)
    matrix_power_values = np.array(
        [[1.5, -0.25, 0.5], [0.0, 2.0, 0.75], [0.25, -0.5, 1.25]],
        dtype=np.float64,
    )
    registry.register_transform(
        PrimitiveTransformRule(
            identity=matrix_power_identity,
            derivative_rule=matrix_power_contract.derivative_rule,
            batching_rule=matrix_power_contract.batching_rule,
            lowering_rule=make_program_ad_linalg_matrix_power_executable_lowering_rule(
                3,
                matrix_power_values.reshape(-1),
            ),
            lowering_metadata={
                **matrix_power_contract.lowering_metadata,
                "mlir": "available: executable scpn_diff MLIR-runtime linalg kernel",
                "mlir_runtime_verification": "verified: 3x3 matrix_power power=3 sample JVP",
            },
            shape_rule=matrix_power_contract.shape_rule,
            dtype_rule=matrix_power_contract.dtype_rule,
            static_argument_rule=matrix_power_contract.static_argument_rule,
            nondifferentiable_policy=matrix_power_contract.nondifferentiable_policy,
            effect=matrix_power_contract.effect,
        )
    )

    multi_dot_contract = primitive_contract_for(multi_dot_identity)
    left = np.array([[1.0, -0.5, 0.75], [0.25, 1.5, -1.0]], dtype=np.float64)
    middle = np.array(
        [[1.25, -0.5, 0.0, 0.5], [0.75, 1.0, -0.25, 0.25], [0.0, 0.5, 1.5, -0.75]],
        dtype=np.float64,
    )
    right = np.array(
        [[1.0, -0.25], [0.5, 1.25], [-0.75, 0.0], [0.25, 0.5]],
        dtype=np.float64,
    )
    multi_dot_values = np.concatenate((left.reshape(-1), middle.reshape(-1), right.reshape(-1)))
    registry.register_transform(
        PrimitiveTransformRule(
            identity=multi_dot_identity,
            derivative_rule=multi_dot_contract.derivative_rule,
            batching_rule=multi_dot_contract.batching_rule,
            lowering_rule=make_program_ad_linalg_multi_dot_executable_lowering_rule(
                ((2, 3), (3, 4), (4, 2)),
                multi_dot_values,
            ),
            lowering_metadata={
                **multi_dot_contract.lowering_metadata,
                "mlir": "available: executable scpn_diff MLIR-runtime linalg kernel",
                "mlir_runtime_verification": "verified: 2x3__3x4__4x2 multi_dot sample JVP",
            },
            shape_rule=multi_dot_contract.shape_rule,
            dtype_rule=multi_dot_contract.dtype_rule,
            static_argument_rule=multi_dot_contract.static_argument_rule,
            nondifferentiable_policy=multi_dot_contract.nondifferentiable_policy,
            effect=multi_dot_contract.effect,
        )
    )

    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    matrix_power_kernel = compile_registered_primitive_to_executable(
        registry, matrix_power_identity, matrix_power_values.reshape(-1)
    )
    multi_dot_kernel = compile_registered_primitive_to_executable(
        registry, multi_dot_identity, multi_dot_values
    )

    assert [status.identity for status in plan.statuses] == [
        matrix_power_identity,
        multi_dot_identity,
    ]
    assert module.metadata["mlir_runtime_lowering_primitives"] == [
        "scpn.program_ad.linalg:matrix_power@1",
        "scpn.program_ad.linalg:multi_dot@1",
    ]
    assert module.metadata["mlir_runtime_incomplete_primitives"] == []
    assert module.resource_counts["mlir_runtime_contracts"] == 2
    assert module.resource_counts["mlir_runtime_verifications"] == 2
    assert module.metadata["primitive_readiness_verdict_counts"] == {"mlir_runtime_verified": 2}
    assert matrix_power_kernel.verification.passed is True
    assert multi_dot_kernel.verification.passed is True
    np.testing.assert_allclose(
        matrix_power_kernel.value(matrix_power_values.reshape(-1)),
        np.linalg.matrix_power(matrix_power_values, 3).reshape(-1),
    )
    np.testing.assert_allclose(
        multi_dot_kernel.value(multi_dot_values),
        np.linalg.multi_dot((left, middle, right)).reshape(-1),
    )


def test_compiler_ad_transform_plan_rejects_empty_and_executable_backend_claims() -> None:
    """Compiler AD planning must fail closed until executable backends exist."""
    with pytest.raises(ValueError, match="at least one primitive"):
        CompilerADTransformPlan(())
    status = PrimitiveLoweringStatus(
        identity=PrimitiveIdentity("scpn.test", "primitive", "1"),
        rule_name="rule",
        has_jvp=True,
        has_vjp=False,
        mlir_op="scpn_diff.scpn_test_primitive",
    )
    with pytest.raises(ValueError, match="executable_backend"):
        CompilerADTransformPlan((status,), executable_backend="llvm")
    with pytest.raises(ValueError, match="nondifferentiable_policy"):
        PrimitiveLoweringStatus(
            identity=PrimitiveIdentity("scpn.test", "primitive", "1"),
            rule_name="rule",
            has_jvp=True,
            has_vjp=False,
            mlir_op="scpn_diff.scpn_test_primitive",
            nondifferentiable_policy="",
        )
    with pytest.raises(ValueError, match="nondifferentiable_boundary"):
        PrimitiveLoweringStatus(
            identity=PrimitiveIdentity("scpn.test", "primitive", "1"),
            rule_name="rule",
            has_jvp=True,
            has_vjp=False,
            mlir_op="scpn_diff.scpn_test_primitive",
            nondifferentiable_boundary_policy="fail_closed",
        )
    with pytest.raises(ValueError, match="nondifferentiable_policy"):
        PrimitiveLoweringStatus(
            identity=PrimitiveIdentity("scpn.test", "primitive", "1"),
            rule_name="rule",
            has_jvp=True,
            has_vjp=False,
            mlir_op="scpn_diff.scpn_test_primitive",
            nondifferentiable_boundary="declared_test_boundary",
            nondifferentiable_boundary_policy="fail_closed",
        )
    with pytest.raises(ValueError, match="registry"):
        build_compiler_ad_transform_plan(object())  # type: ignore[arg-type]


def test_mlir_compiler_api_exported_from_package_root() -> None:
    """MLIR compiler surfaces should remain stable package-root imports."""
    assert scpn.CompilerADExecutableConfig is CompilerADExecutableConfig
    assert scpn.CompilerADKernelVerification is CompilerADKernelVerification
    assert scpn.CompilerADTransformPlan is CompilerADTransformPlan
    assert scpn.DifferentiableMLIRCompileConfig is DifferentiableMLIRCompileConfig
    assert scpn.ExecutableCompilerADKernel is ExecutableCompilerADKernel
    assert scpn.ExecutableWholeProgramADBatchResult is ExecutableWholeProgramADBatchResult
    assert scpn.ExecutableWholeProgramADKernel is ExecutableWholeProgramADKernel
    assert scpn.MLIRCompileConfig is MLIRCompileConfig
    assert scpn.NativeWholeProgramADKernel is NativeWholeProgramADKernel
    assert scpn.WholeProgramADNativeLoweringReport is WholeProgramADNativeLoweringReport
    assert (
        scpn.analyse_whole_program_ad_native_lowering is analyse_whole_program_ad_native_lowering
    )
    assert scpn.PrimitiveLoweringStatus is PrimitiveLoweringStatus
    assert scpn.build_compiler_ad_transform_plan is build_compiler_ad_transform_plan
    assert (
        scpn.compile_compiler_ad_transform_plan_to_mlir
        is compile_compiler_ad_transform_plan_to_mlir
    )
    assert (
        scpn.compile_custom_derivative_rule_to_executable
        is compile_custom_derivative_rule_to_executable
    )
    assert scpn.compile_custom_derivative_rule_to_mlir is compile_custom_derivative_rule_to_mlir
    assert (
        scpn.compile_registered_primitive_to_executable
        is compile_registered_primitive_to_executable
    )
    assert (
        scpn.compile_whole_program_ad_trace_to_executable
        is compile_whole_program_ad_trace_to_executable
    )
    assert (
        scpn.compile_whole_program_ad_trace_to_native_llvm_jit
        is compile_whole_program_ad_trace_to_native_llvm_jit
    )
    assert (
        scpn.native_whole_program_ad_compile_cache_stats
        is native_whole_program_ad_compile_cache_stats
    )
    assert (
        scpn.clear_native_whole_program_ad_compile_cache
        is clear_native_whole_program_ad_compile_cache
    )
    assert scpn.compile_kuramoto_to_mlir is compile_kuramoto_to_mlir
    assert (
        scpn.make_program_ad_linalg_matrix_power_executable_lowering_rule
        is make_program_ad_linalg_matrix_power_executable_lowering_rule
    )
    assert (
        scpn.make_program_ad_linalg_multi_dot_executable_lowering_rule
        is make_program_ad_linalg_multi_dot_executable_lowering_rule
    )
