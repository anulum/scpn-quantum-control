# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- tests for MLIR, realtime, and cloud-native surfaces
"""Tests for production-grade compiler/runtime/deployment surfaces."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_quantum_control as scpn
import scpn_quantum_control.compiler.mlir as compiler_mlir
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
    compile_whole_program_ad_trace_to_mlir,
    compile_whole_program_ad_trace_to_native_llvm_jit,
    make_program_ad_linalg_matrix_power_executable_lowering_rule,
    make_program_ad_linalg_multi_dot_executable_lowering_rule,
    native_whole_program_ad_compile_cache_stats,
    native_whole_program_ad_linalg_support,
)
from scpn_quantum_control.control.realtime_runtime import (
    RealtimeRuntimeConfig,
    RealtimeSLAConfig,
    VirtualRealtimeClock,
    enforce_realtime_sla,
    evaluate_realtime_sla,
    run_realtime_control_loop,
)
from scpn_quantum_control.deployment.cloud_native import (
    CloudDeploymentSpec,
    ContainerResources,
    generate_cloud_manifests,
)
from scpn_quantum_control.differentiable import (
    CustomDerivativeRegistry,
    CustomDerivativeRule,
    Parameter,
    PrimitiveIdentity,
    PrimitiveTransformRule,
    primitive_contract_for,
    program_ad_linalg_matrix_power_derivative_rule,
    program_ad_linalg_multi_dot_derivative_rule,
    program_adjoint_value_and_grad,
    vmap,
    whole_program_value_and_grad,
)
from scpn_quantum_control.kuramoto_core import build_kuramoto_problem


def _dense_determinant_offsets(size: int) -> np.ndarray:
    """Return a deterministic non-diagonal perturbation for native determinant tests."""

    rows = np.arange(size, dtype=np.float64).reshape(size, 1) + 1.0
    cols = np.arange(size, dtype=np.float64).reshape(1, size) + 1.0
    offsets = 0.011 * np.sin(rows * (cols + 0.5)) + 0.007 * np.cos(rows + 2.0 * cols)
    np.fill_diagonal(offsets, 0.0)
    return offsets


def _problem():
    return build_kuramoto_problem(
        np.array(
            [
                [0.0, 0.25, 0.0],
                [0.25, 0.0, -0.5],
                [0.0, -0.5, 0.0],
            ],
            dtype=np.float64,
        ),
        np.array([0.1, -0.2, 0.3], dtype=np.float64),
        metadata={"experiment": "compiler-smoke"},
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
    base_kwargs = {
        "identity": identity,
        "rule_name": "rust_signature_guard_rule",
        "has_jvp": True,
        "has_vjp": True,
        "mlir_op": "scpn_diff.rust_signature_guard",
        "has_batching_rule": True,
        "has_shape_rule": True,
        "has_dtype_rule": True,
        "has_static_argument_rule": True,
        "has_lowering_rule": True,
        "static_derivative_factory": "native_rust_signature_guard_llvm_jit",
        "static_signature": "primitive:eig;dimension:2;layout:row_major",
        "nondifferentiable_policy": "real_simple_domain",
        "nondifferentiable_boundary": "real_simple_boundary",
        "nondifferentiable_boundary_policy": "fail_closed",
        "mlir_lowering": "available: executable scpn_diff MLIR-runtime primitive kernel",
        "mlir_runtime_verification": "verified: rust signature guard JVP",
        "rust_lowering": "available: Rust PyO3 rust signature guard kernel",
        "llvm_lowering": "available: native LLVM MCJIT rust signature guard AD kernel",
        "jit_lowering": "available: native LLVM MCJIT rust signature guard AD kernel",
    }
    base_metadata = {
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

    with pytest.raises(ValueError, match="rust_backend_signature"):
        PrimitiveLoweringStatus(**base_kwargs, lowering_metadata=base_metadata)

    missing_functions_metadata = {
        **base_metadata,
        "rust_backend_signature": "primitive:eig;dimension:2;layout:row_major",
    }
    del missing_functions_metadata["rust_backend_functions"]
    with pytest.raises(ValueError, match="rust_backend_functions"):
        PrimitiveLoweringStatus(**base_kwargs, lowering_metadata=missing_functions_metadata)

    mismatched_metadata = {
        **base_metadata,
        "rust_backend_signature": "primitive:determinant;dimension:2;layout:row_major",
    }
    with pytest.raises(ValueError, match="rust_backend_signature must match static_signature"):
        PrimitiveLoweringStatus(**base_kwargs, lowering_metadata=mismatched_metadata)

    valid_metadata = {
        **base_metadata,
        "rust_backend_signature": "primitive:eig;dimension:2;layout:row_major",
    }
    status = PrimitiveLoweringStatus(**base_kwargs, lowering_metadata=valid_metadata)

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


def test_kuramoto_mlir_emits_deterministic_text_digest_and_resources() -> None:
    """MLIR export should be deterministic and explicit about compiler resources."""

    module = compile_kuramoto_to_mlir(
        _problem(),
        MLIRCompileConfig(time=0.4, trotter_steps=2, trotter_order=2),
    )
    repeat = compile_kuramoto_to_mlir(
        _problem(),
        MLIRCompileConfig(time=0.4, trotter_steps=2, trotter_order=2),
    )

    assert module.text == repeat.text
    assert module.sha256 == repeat.sha256
    assert module.resource_counts["n_oscillators"] == 3
    assert module.resource_counts["coupling_terms"] == 2
    assert module.resource_counts["trotter_steps"] == 2
    assert 'scpn.module = "kuramoto_xy"' in module.text
    assert "scpn.omega" in module.text
    assert "scpn.coupling" in module.text
    assert "scpn.trotter_evolve" in module.text


def test_kuramoto_mlir_rejects_invalid_compile_config() -> None:
    """MLIR config should fail closed before emitting misleading IR."""

    with pytest.raises(ValueError, match="trotter_steps"):
        MLIRCompileConfig(time=0.1, trotter_steps=0)
    with pytest.raises(ValueError, match="time"):
        MLIRCompileConfig(time=float("nan"))


def test_differentiable_mlir_lowers_custom_derivative_rule_deterministically() -> None:
    """Differentiable primitive lowering should be deterministic and auditable."""

    rule = CustomDerivativeRule(
        name="linear_residual",
        value_fn=lambda values: np.array(
            [values[0] + 2.0 * values[1], values[0] - values[1]],
            dtype=np.float64,
        ),
        jvp_rule=lambda values, tangent: np.array(
            [tangent[0] + 2.0 * tangent[1], tangent[0] - tangent[1]],
            dtype=np.float64,
        ),
        parameter_names=("theta", "phi"),
        trainable=(True, False),
    )

    module = compile_custom_derivative_rule_to_mlir(
        rule,
        np.array([1.5, -0.25], dtype=np.float64),
        DifferentiableMLIRCompileConfig(),
    )
    repeat = compile_custom_derivative_rule_to_mlir(
        rule,
        np.array([1.5, -0.25], dtype=np.float64),
        DifferentiableMLIRCompileConfig(),
    )

    assert module.text == repeat.text
    assert module.sha256 == repeat.sha256
    assert module.dialect == "scpn_diff"
    assert module.resource_counts["parameters"] == 2
    assert module.resource_counts["outputs"] == 2
    assert module.resource_counts["jacobian_nnz"] == 2
    assert module.resource_counts["trainable_parameters"] == 1
    assert 'scpn.module = "differentiable_primitive"' in module.text
    assert 'scpn.rule = "linear_residual"' in module.text
    assert "scpn_diff.parameter" in module.text
    assert "scpn_diff.value" in module.text
    assert "scpn_diff.jacobian" in module.text
    assert 'execution = "interchange_only"' in module.text
    assert module.metadata["target"] == "mlir"


def test_custom_derivative_rule_compiles_to_verified_executable_ad_kernel() -> None:
    """Compiler AD should execute differentiated primitive kernels with MLIR provenance."""

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
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.rx_expectation",
                "llvm": "blocked: native LLVM lowering backend not linked",
            },
        )
    )

    kernel = compile_registered_primitive_to_executable(
        registry,
        identity,
        np.array([0.25], dtype=np.float64),
        sample_tangent=np.array([0.5], dtype=np.float64),
        sample_cotangent=np.array([2.0], dtype=np.float64),
    )

    assert isinstance(kernel, ExecutableCompilerADKernel)
    assert isinstance(kernel.verification, CompilerADKernelVerification)
    assert kernel.backend == "mlir_runtime"
    assert kernel.verification.passed is True
    assert kernel.mlir_module.metadata["target"] == "mlir"
    assert "differentiable_primitive" in kernel.mlir_module.text
    np.testing.assert_allclose(kernel.value(np.array([0.25])), [np.cos(0.25)])
    np.testing.assert_allclose(
        kernel.jvp(np.array([0.25]), np.array([0.5])),
        [-np.sin(0.25) * 0.5],
    )
    np.testing.assert_allclose(
        kernel.vjp(np.array([0.25]), np.array([2.0])),
        [-np.sin(0.25) * 2.0],
    )


def test_executable_compiler_ad_kernel_verifies_scalar_gradient_output() -> None:
    """Executable compiler AD should expose verified scalar-output gradients."""

    rule = CustomDerivativeRule(
        name="quadratic_phase_rule",
        value_fn=lambda values: np.array([values[0] ** 2 + np.sin(values[1])], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array(
            [2.0 * values[0] * tangent[0] + np.cos(values[1]) * tangent[1]],
            dtype=np.float64,
        ),
        vjp_rule=lambda values, cotangent: np.array(
            [2.0 * values[0] * cotangent[0], np.cos(values[1]) * cotangent[0]],
            dtype=np.float64,
        ),
        parameter_names=("theta", "phase"),
        trainable=(True, True),
    )
    values = np.array([0.75, -0.3], dtype=np.float64)

    kernel = compile_custom_derivative_rule_to_executable(
        rule,
        values,
        sample_tangent=np.array([0.25, -0.5], dtype=np.float64),
        sample_cotangent=np.array([1.0], dtype=np.float64),
    )

    assert kernel.backend == "mlir_runtime"
    assert kernel.verification.gradient_close is True
    assert "verified executable MLIR-runtime" in kernel.claim_boundary
    assert "native LLVM/JIT code generation remains fail-closed" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert 'source = "verified_mlir_runtime_vjp_cotangent_one"' in kernel.llvm_gradient_ir
    assert "define void @quadratic_phase_rule_gradient" in kernel.llvm_gradient_ir
    assert "scpn_diff.custom_rule" in kernel.mlir_module.text
    np.testing.assert_allclose(
        kernel.gradient(values),
        [1.5, np.cos(-0.3)],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_native_llvm_jit_scalar_quadratic_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT scalar AD kernels should execute and update compiler contracts."""

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "scalar_quadratic", "1")
    rule = CustomDerivativeRule(
        name="native_scalar_quadratic_rule",
        value_fn=lambda values: np.array(
            [3.0 * values[0] ** 2 - 2.0 * values[0] + 0.5],
            dtype=np.float64,
        ),
        jvp_rule=lambda values, tangent: np.array(
            [(6.0 * values[0] - 2.0) * tangent[0]],
            dtype=np.float64,
        ),
        vjp_rule=lambda values, cotangent: np.array(
            [(6.0 * values[0] - 2.0) * cotangent[0]],
            dtype=np.float64,
        ),
        parameter_names=("x",),
        trainable=(True,),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([0.75], dtype=np.float64)
    tangent = np.array([-0.25], dtype=np.float64)
    cotangent = np.array([1.5], dtype=np.float64)

    kernel = compiler_mlir.compile_scalar_quadratic_ad_to_native_llvm_jit(
        rule,
        quadratic=3.0,
        linear=-2.0,
        constant=0.5,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert kernel.verification.gradient_close is True
    assert "verified native LLVM MCJIT" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "target triple" in kernel.llvm_gradient_ir
    assert "define void @native_scalar_quadratic_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_scalar_quadratic_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_scalar_quadratic_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_scalar_quadratic_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(kernel.value(values), [0.6875], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(kernel.jvp(values, tangent), [-0.625], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(kernel.vjp(values, cotangent), [3.75], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(kernel.gradient(values), [2.5], rtol=1.0e-12, atol=1.0e-12)

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=lambda batch, fn: np.asarray([fn(item) for item in batch]),
            lowering_rule=compiler_mlir.make_scalar_quadratic_native_llvm_jit_lowering_rule(
                quadratic=3.0,
                linear=-2.0,
                constant=0.5,
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_scalar_quadratic",
                "mlir_runtime_verification": (
                    "verified: native LLVM/JIT scalar quadratic sample JVP"
                ),
                "llvm": "available: native LLVM MCJIT scalar AD kernel",
                "jit": "available: native LLVM MCJIT scalar AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT value/JVP/VJP/gradient"
                ),
                "static_derivative_factory": "native_scalar_quadratic_llvm_jit",
                "static_signature": "quadratic:f64,linear:f64,constant:f64",
                "nondifferentiable_boundary": "none_scalar_polynomial",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (1,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="polynomial_is_smooth_over_reals",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["executable_backends"] == 1
    assert module.resource_counts["llvm_backend_contracts"] == 1
    assert module.resource_counts["jit_backend_contracts"] == 1
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["llvm_backend_contract_primitives"] == [identity.key]
    assert module.metadata["jit_backend_contract_primitives"] == [identity.key]
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["llvm_backend_contract"] is True
    assert module.metadata["primitive_readiness"][identity.key]["jit_backend_contract"] is True
    assert module.metadata["primitive_readiness"][identity.key]["native_backend_contract"] is True
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]


def test_native_llvm_jit_scalar_unary_sin_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT nonlinear scalar unary AD kernels should execute."""

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "scalar_unary_sin", "1")
    rule = CustomDerivativeRule(
        name="native_scalar_unary_sin_rule",
        value_fn=lambda values: np.array([np.sin(values[0])], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array(
            [np.cos(values[0]) * tangent[0]],
            dtype=np.float64,
        ),
        vjp_rule=lambda values, cotangent: np.array(
            [np.cos(values[0]) * cotangent[0]],
            dtype=np.float64,
        ),
        parameter_names=("x",),
        trainable=(True,),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([0.5], dtype=np.float64)
    tangent = np.array([-0.75], dtype=np.float64)
    cotangent = np.array([2.0], dtype=np.float64)

    kernel = compiler_mlir.compile_scalar_unary_elementwise_ad_to_native_llvm_jit(
        rule,
        primitive="sin",
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    expected_gradient = np.cos(values)
    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert kernel.verification.gradient_close is True
    assert "verified native LLVM MCJIT scalar unary" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "declare double @llvm.sin.f64(double)" in kernel.llvm_gradient_ir
    assert "declare double @llvm.cos.f64(double)" in kernel.llvm_gradient_ir
    assert "define void @native_scalar_unary_sin_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_scalar_unary_sin_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_scalar_unary_sin_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_scalar_unary_sin_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(kernel.value(values), np.sin(values), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        kernel.jvp(values, tangent), expected_gradient * tangent, rtol=1.0e-12, atol=1.0e-12
    )
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        expected_gradient * cotangent,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.gradient(values), expected_gradient, rtol=1.0e-12, atol=1.0e-12
    )

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=lambda batch, fn: np.asarray([fn(item) for item in batch]),
            lowering_rule=(
                compiler_mlir.make_scalar_unary_elementwise_native_llvm_jit_lowering_rule(
                    primitive="sin",
                    sample_values=values,
                    config=config,
                    sample_tangent=tangent,
                    sample_cotangent=cotangent,
                )
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_scalar_unary_sin",
                "mlir_runtime_verification": "verified: native LLVM/JIT scalar unary sin JVP",
                "llvm": "available: native LLVM MCJIT scalar unary AD kernel",
                "jit": "available: native LLVM MCJIT scalar unary AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT scalar unary value/JVP/VJP/gradient"
                ),
                "static_derivative_factory": "native_scalar_unary_sin_llvm_jit",
                "static_signature": "primitive:sin;input:f64",
                "nondifferentiable_boundary": "none_smooth_unary",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (1,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="smooth_unary_real_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]


def test_native_llvm_jit_scalar_binary_multiply_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT scalar binary elementwise AD kernels should execute."""

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "scalar_binary_multiply", "1")
    rule = CustomDerivativeRule(
        name="native_scalar_binary_multiply_rule",
        value_fn=lambda values: np.array([values[0] * values[1]], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array(
            [values[1] * tangent[0] + values[0] * tangent[1]],
            dtype=np.float64,
        ),
        vjp_rule=lambda values, cotangent: np.array(
            [values[1] * cotangent[0], values[0] * cotangent[0]],
            dtype=np.float64,
        ),
        parameter_names=("x", "y"),
        trainable=(True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([1.5, -2.0], dtype=np.float64)
    tangent = np.array([0.25, -0.5], dtype=np.float64)
    cotangent = np.array([1.25], dtype=np.float64)

    kernel = compiler_mlir.compile_scalar_binary_elementwise_ad_to_native_llvm_jit(
        rule,
        primitive="multiply",
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert kernel.verification.gradient_close is True
    assert "verified native LLVM MCJIT scalar binary" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "define void @native_scalar_binary_multiply_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_scalar_binary_multiply_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_scalar_binary_multiply_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_scalar_binary_multiply_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(kernel.value(values), [-3.0], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(kernel.jvp(values, tangent), [-1.25], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        [-2.5, 1.875],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.gradient(values),
        [-2.0, 1.5],
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=lambda batch, fn: np.asarray([fn(item) for item in batch]),
            lowering_rule=(
                compiler_mlir.make_scalar_binary_elementwise_native_llvm_jit_lowering_rule(
                    primitive="multiply",
                    sample_values=values,
                    config=config,
                    sample_tangent=tangent,
                    sample_cotangent=cotangent,
                )
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_scalar_binary_multiply",
                "mlir_runtime_verification": (
                    "verified: native LLVM/JIT scalar binary multiply JVP"
                ),
                "llvm": "available: native LLVM MCJIT scalar binary AD kernel",
                "jit": "available: native LLVM MCJIT scalar binary AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT scalar binary value/JVP/VJP/gradient"
                ),
                "static_derivative_factory": "native_scalar_binary_multiply_llvm_jit",
                "static_signature": "primitive:multiply;left:f64;right:f64",
                "nondifferentiable_boundary": "none_smooth_binary",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (1,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="smooth_binary_real_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]


def test_native_llvm_jit_vector_dot_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT vector dot-product AD kernels should execute."""

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "vector_dot", "1")
    rule = CustomDerivativeRule(
        name="native_vector_dot_rule",
        value_fn=lambda values: np.array([np.dot(values[:2], values[2:])], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array(
            [np.dot(values[2:], tangent[:2]) + np.dot(values[:2], tangent[2:])],
            dtype=np.float64,
        ),
        vjp_rule=lambda values, cotangent: np.concatenate(
            [values[2:] * cotangent[0], values[:2] * cotangent[0]]
        ).astype(np.float64),
        parameter_names=("x0", "x1", "y0", "y1"),
        trainable=(True, True, True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([1.0, 2.0, -3.0, 4.0], dtype=np.float64)
    tangent = np.array([0.5, -1.0, 2.0, -0.25], dtype=np.float64)
    cotangent = np.array([1.25], dtype=np.float64)

    kernel = compiler_mlir.compile_vector_dot_ad_to_native_llvm_jit(
        rule,
        dimension=2,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert kernel.verification.gradient_close is True
    assert "verified native LLVM MCJIT vector dot" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "define void @native_vector_dot_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_vector_dot_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_vector_dot_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_vector_dot_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(kernel.value(values), [5.0], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(kernel.jvp(values, tangent), [-4.0], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        [-3.75, 5.0, 1.25, 2.5],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.gradient(values),
        [-3.0, 4.0, 1.0, 2.0],
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=lambda batch, fn: np.asarray([fn(item) for item in batch]),
            lowering_rule=compiler_mlir.make_vector_dot_native_llvm_jit_lowering_rule(
                dimension=2,
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_vector_dot",
                "mlir_runtime_verification": "verified: native LLVM/JIT vector dot JVP",
                "llvm": "available: native LLVM MCJIT vector dot AD kernel",
                "jit": "available: native LLVM MCJIT vector dot AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT vector dot value/JVP/VJP/gradient"
                ),
                "static_derivative_factory": "native_vector_dot_llvm_jit",
                "static_signature": "primitive:dot;dimension:2;layout:x_then_y",
                "nondifferentiable_boundary": "none_bilinear_vector_dot",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (1,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="smooth_bilinear_real_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]

    rust_registry = CustomDerivativeRegistry()
    rust_registry.register_transform(
        compiler_mlir.make_vector_dot_native_llvm_jit_primitive_transform(
            identity,
            rule,
            dimension=2,
            sample_values=values,
            config=config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        )
    )
    rust_plan = build_compiler_ad_transform_plan(rust_registry)
    rust_module = compile_compiler_ad_transform_plan_to_mlir(rust_plan)
    rust_registered_kernel = compile_registered_primitive_to_executable(
        rust_registry,
        identity,
        values,
    )

    assert (
        scpn.make_vector_dot_native_llvm_jit_primitive_transform
        is compiler_mlir.make_vector_dot_native_llvm_jit_primitive_transform
    )
    assert rust_registered_kernel.backend == "native_llvm_jit"
    assert rust_module.resource_counts["native_backend_contracts"] == 1
    assert rust_module.resource_counts["rust_backend_contracts"] == 1
    assert rust_module.resource_counts["primitive_readiness_native_executable"] == 1
    assert rust_module.resource_counts["primitive_hard_gaps"] == 0
    assert rust_module.metadata["primitive_hard_gaps"][identity.key] == []
    assert rust_module.metadata["rust_backend_contract_primitives"] == [identity.key]
    assert rust_module.metadata["rust_backend_signatures"] == {
        identity.key: "primitive:dot;dimension:2;layout:x_then_y"
    }
    assert rust_module.metadata["rust_backend_functions"] == {
        identity.key: "vector_dot_value,vector_dot_jvp,vector_dot_vjp,vector_dot_gradient"
    }
    assert rust_module.metadata["rust_backend_verification_primitives"] == {
        identity.key: "verified: scpn_quantum_engine vector_dot value/JVP/VJP/gradient parity"
    }
    np.testing.assert_allclose(
        rust_registered_kernel.gradient(values),
        [-3.0, 4.0, 1.0, 2.0],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_native_llvm_jit_matrix_quadratic_form_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT matrix quadratic-form AD kernels should execute."""

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "matrix_quadratic_form", "1")

    def unpack(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        matrix = values[:4].reshape(2, 2)
        vector = values[4:]
        return matrix, vector

    def value_fn(values: np.ndarray) -> np.ndarray:
        matrix, vector = unpack(values)
        return np.array([vector @ matrix @ vector], dtype=np.float64)

    def jvp_rule(values: np.ndarray, tangent: np.ndarray) -> np.ndarray:
        matrix, vector = unpack(values)
        matrix_tangent, vector_tangent = unpack(tangent)
        return np.array(
            [
                vector @ matrix_tangent @ vector
                + vector_tangent @ matrix @ vector
                + vector @ matrix @ vector_tangent
            ],
            dtype=np.float64,
        )

    def vjp_rule(values: np.ndarray, cotangent: np.ndarray) -> np.ndarray:
        matrix, vector = unpack(values)
        cotangent_value = cotangent[0]
        matrix_gradient = np.outer(vector, vector).reshape(-1)
        vector_gradient = (matrix + matrix.T) @ vector
        return np.concatenate([matrix_gradient, vector_gradient]) * cotangent_value

    rule = CustomDerivativeRule(
        name="native_matrix_quadratic_form_rule",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
        parameter_names=("a00", "a01", "a10", "a11", "x0", "x1"),
        trainable=(True, True, True, True, True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([2.0, -1.0, 0.5, 3.0, 1.5, -2.0], dtype=np.float64)
    tangent = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.25], dtype=np.float64)
    cotangent = np.array([1.25], dtype=np.float64)

    kernel = compiler_mlir.compile_matrix_quadratic_form_ad_to_native_llvm_jit(
        rule,
        dimension=2,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert kernel.verification.gradient_close is True
    assert "verified native LLVM MCJIT matrix quadratic form" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "define void @native_matrix_quadratic_form_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_quadratic_form_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_quadratic_form_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_quadratic_form_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(kernel.value(values), [18.0], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(kernel.jvp(values, tangent), [-5.1625], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        [2.8125, -3.75, -3.75, 5.0, 8.75, -15.9375],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.gradient(values),
        [2.25, -3.0, -3.0, 4.0, 7.0, -12.75],
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=lambda batch, fn: np.asarray([fn(item) for item in batch]),
            lowering_rule=compiler_mlir.make_matrix_quadratic_form_native_llvm_jit_lowering_rule(
                dimension=2,
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_matrix_quadratic_form",
                "mlir_runtime_verification": (
                    "verified: native LLVM/JIT matrix quadratic form JVP"
                ),
                "llvm": "available: native LLVM MCJIT matrix quadratic form AD kernel",
                "jit": "available: native LLVM MCJIT matrix quadratic form AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT matrix quadratic form value/JVP/VJP/gradient"
                ),
                "static_derivative_factory": "native_matrix_quadratic_form_llvm_jit",
                "static_signature": "primitive:quadratic_form;dimension:2;layout:matrix_then_vector",
                "nondifferentiable_boundary": "none_smooth_matrix_quadratic_form",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (1,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="smooth_matrix_quadratic_form_real_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]

    rust_registry = CustomDerivativeRegistry()
    rust_registry.register_transform(
        compiler_mlir.make_matrix_quadratic_form_native_llvm_jit_primitive_transform(
            identity,
            rule,
            dimension=2,
            sample_values=values,
            config=config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        )
    )
    rust_plan = build_compiler_ad_transform_plan(rust_registry)
    rust_module = compile_compiler_ad_transform_plan_to_mlir(rust_plan)
    rust_registered_kernel = compile_registered_primitive_to_executable(
        rust_registry,
        identity,
        values,
    )

    assert rust_registered_kernel.backend == "native_llvm_jit"
    assert rust_plan.executable_backend == "native_llvm_jit"
    assert rust_module.metadata["executable_backend"] == "native_llvm_jit"
    assert rust_module.resource_counts["rust_backend_contracts"] == 1
    assert rust_module.resource_counts["rust_backend_incomplete_primitives"] == 0
    assert rust_module.resource_counts["rust_backend_blockers"] == 0
    assert rust_module.resource_counts["primitive_hard_gaps"] == 0
    assert rust_module.metadata["rust_backend_contract_primitives"] == [identity.key]
    assert rust_module.metadata["rust_backend_incomplete_primitives"] == []
    assert rust_module.metadata["rust_backend_blockers"] == {}
    assert rust_module.metadata["rust_backend_signatures"] == {
        identity.key: "primitive:quadratic_form;dimension:2;layout:matrix_then_vector"
    }
    assert rust_module.metadata["rust_backend_functions"] == {
        identity.key: (
            "matrix_quadratic_form_value,matrix_quadratic_form_jvp,"
            "matrix_quadratic_form_vjp,matrix_quadratic_form_gradient"
        )
    }
    assert rust_module.metadata["rust_backend_verification_primitives"] == {
        identity.key: (
            "verified: scpn_quantum_engine matrix_quadratic_form value/JVP/VJP/gradient parity"
        )
    }
    assert rust_module.resource_counts["rust_backend_verifications"] == 1
    assert rust_module.metadata["primitive_hard_gaps"][identity.key] == []
    assert identity.key not in rust_module.metadata["primitive_next_hard_gap"]
    assert rust_module.metadata["primitive_readiness"][identity.key] == {
        "adjoint_contract": True,
        "forward_contract": True,
        "jit_backend_contract": True,
        "llvm_backend_contract": True,
        "mlir_runtime_contract": True,
        "native_backend_contract": True,
        "registry_contract": True,
        "reverse_contract": True,
        "rust_backend_contract": True,
        "transform_contract": True,
        "verdict": "native_executable",
    }
    np.testing.assert_allclose(
        rust_registered_kernel.gradient(values),
        [2.25, -3.0, -3.0, 4.0, 7.0, -12.75],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert scpn.make_matrix_quadratic_form_native_llvm_jit_primitive_transform is (
        compiler_mlir.make_matrix_quadratic_form_native_llvm_jit_primitive_transform
    )


def test_native_llvm_jit_vector_squared_norm_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT vector squared-norm AD kernels should execute."""

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "vector_squared_norm", "1")
    rule = CustomDerivativeRule(
        name="native_vector_squared_norm_rule",
        value_fn=lambda values: np.array([np.dot(values, values)], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array(
            [2.0 * np.dot(values, tangent)],
            dtype=np.float64,
        ),
        vjp_rule=lambda values, cotangent: (2.0 * values * cotangent[0]).astype(np.float64),
        parameter_names=("x0", "x1", "x2"),
        trainable=(True, True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([1.5, -2.0, 0.25], dtype=np.float64)
    tangent = np.array([-0.5, 0.75, 2.0], dtype=np.float64)
    cotangent = np.array([1.25], dtype=np.float64)

    kernel = compiler_mlir.compile_vector_squared_norm_ad_to_native_llvm_jit(
        rule,
        dimension=3,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert kernel.verification.gradient_close is True
    assert "verified native LLVM MCJIT vector squared norm" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "define void @native_vector_squared_norm_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_vector_squared_norm_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_vector_squared_norm_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_vector_squared_norm_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(kernel.value(values), [6.3125], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(kernel.jvp(values, tangent), [-3.5], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        [3.75, -5.0, 0.625],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.gradient(values),
        [3.0, -4.0, 0.5],
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=lambda batch, fn: np.asarray([fn(item) for item in batch]),
            lowering_rule=compiler_mlir.make_vector_squared_norm_native_llvm_jit_lowering_rule(
                dimension=3,
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_vector_squared_norm",
                "mlir_runtime_verification": "verified: native LLVM/JIT vector squared norm JVP",
                "llvm": "available: native LLVM MCJIT vector squared norm AD kernel",
                "jit": "available: native LLVM MCJIT vector squared norm AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT vector squared norm value/JVP/VJP/gradient"
                ),
                "static_derivative_factory": "native_vector_squared_norm_llvm_jit",
                "static_signature": "primitive:squared_norm;dimension:3",
                "nondifferentiable_boundary": "none_smooth_vector_squared_norm",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (1,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="smooth_vector_squared_norm_real_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]

    rust_registry = CustomDerivativeRegistry()
    rust_registry.register_transform(
        compiler_mlir.make_vector_squared_norm_native_llvm_jit_primitive_transform(
            identity,
            rule,
            dimension=3,
            sample_values=values,
            config=config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        )
    )
    rust_plan = build_compiler_ad_transform_plan(rust_registry)
    rust_module = compile_compiler_ad_transform_plan_to_mlir(rust_plan)

    assert (
        scpn.make_vector_squared_norm_native_llvm_jit_primitive_transform
        is compiler_mlir.make_vector_squared_norm_native_llvm_jit_primitive_transform
    )
    assert rust_module.resource_counts["native_backend_contracts"] == 1
    assert rust_module.resource_counts["rust_backend_contracts"] == 1
    assert rust_module.resource_counts["primitive_readiness_native_executable"] == 1
    assert rust_module.metadata["primitive_readiness"][identity.key]["verdict"] == (
        "native_executable"
    )
    assert rust_module.metadata["primitive_hard_gaps"][identity.key] == []
    assert rust_module.metadata["rust_backend_contract_primitives"] == [identity.key]
    assert rust_module.metadata["rust_backend_signatures"] == {
        identity.key: "primitive:squared_norm;dimension:3"
    }
    assert rust_module.metadata["rust_backend_functions"] == {
        identity.key: (
            "vector_squared_norm_value,vector_squared_norm_jvp,"
            "vector_squared_norm_vjp,vector_squared_norm_gradient"
        )
    }


def test_native_llvm_jit_matrix_vector_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT matrix-vector AD kernels should execute vector-output AD."""

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "matrix_vector_product", "1")

    def unpack(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        matrix = values[:4].reshape(2, 2)
        vector = values[4:]
        return matrix, vector

    def value_fn(values: np.ndarray) -> np.ndarray:
        matrix, vector = unpack(values)
        return matrix @ vector

    def jvp_rule(values: np.ndarray, tangent: np.ndarray) -> np.ndarray:
        matrix, vector = unpack(values)
        matrix_tangent, vector_tangent = unpack(tangent)
        return matrix_tangent @ vector + matrix @ vector_tangent

    def vjp_rule(values: np.ndarray, cotangent: np.ndarray) -> np.ndarray:
        matrix, vector = unpack(values)
        matrix_gradient = np.outer(cotangent, vector).reshape(-1)
        vector_gradient = matrix.T @ cotangent
        return np.concatenate([matrix_gradient, vector_gradient])

    rule = CustomDerivativeRule(
        name="native_matrix_vector_product_rule",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
        parameter_names=("a00", "a01", "a10", "a11", "x0", "x1"),
        trainable=(True, True, True, True, True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([2.0, -1.0, 0.5, 3.0, 1.5, -2.0], dtype=np.float64)
    tangent = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.25], dtype=np.float64)
    cotangent = np.array([1.25, -0.5], dtype=np.float64)

    kernel = compiler_mlir.compile_matrix_vector_product_ad_to_native_llvm_jit(
        rule,
        dimension=2,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert kernel.verification.gradient_close is None
    assert "verified native LLVM MCJIT matrix-vector product" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "define void @native_matrix_vector_product_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_vector_product_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_vector_product_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_vector_product_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(kernel.value(values), [5.0, -5.25], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        kernel.jvp(values, tangent), [-0.7, 0.15], rtol=1.0e-12, atol=1.0e-12
    )
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        [1.875, -2.5, -0.75, 1.0, 2.25, -2.75],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    with pytest.raises(ValueError, match="requires scalar output"):
        kernel.gradient(values)

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=lambda batch, fn: np.asarray([fn(item) for item in batch]),
            lowering_rule=compiler_mlir.make_matrix_vector_product_native_llvm_jit_lowering_rule(
                dimension=2,
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_matrix_vector_product",
                "mlir_runtime_verification": "verified: native LLVM/JIT matrix-vector JVP",
                "llvm": "available: native LLVM MCJIT matrix-vector AD kernel",
                "jit": "available: native LLVM MCJIT matrix-vector AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT matrix-vector value/JVP/VJP"
                ),
                "static_derivative_factory": "native_matrix_vector_product_llvm_jit",
                "static_signature": "primitive:matvec;dimension:2;layout:matrix_then_vector",
                "nondifferentiable_boundary": "none_smooth_matrix_vector_product",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (2,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="smooth_matrix_vector_product_real_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]

    rust_registry = CustomDerivativeRegistry()
    rust_registry.register_transform(
        compiler_mlir.make_matrix_vector_product_native_llvm_jit_primitive_transform(
            identity,
            rule,
            dimension=2,
            sample_values=values,
            config=config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        )
    )
    rust_plan = build_compiler_ad_transform_plan(rust_registry)
    rust_module = compile_compiler_ad_transform_plan_to_mlir(rust_plan)
    rust_registered_kernel = compile_registered_primitive_to_executable(
        rust_registry,
        identity,
        values,
    )

    assert rust_registered_kernel.backend == "native_llvm_jit"
    assert rust_plan.executable_backend == "native_llvm_jit"
    assert rust_module.metadata["executable_backend"] == "native_llvm_jit"
    assert rust_module.resource_counts["rust_backend_contracts"] == 1
    assert rust_module.resource_counts["rust_backend_incomplete_primitives"] == 0
    assert rust_module.resource_counts["rust_backend_blockers"] == 0
    assert rust_module.resource_counts["primitive_hard_gaps"] == 0
    assert rust_module.metadata["rust_backend_contract_primitives"] == [identity.key]
    assert rust_module.metadata["rust_backend_incomplete_primitives"] == []
    assert rust_module.metadata["rust_backend_blockers"] == {}
    assert rust_module.metadata["rust_backend_signatures"] == {
        identity.key: "primitive:matvec;dimension:2;layout:matrix_then_vector"
    }
    assert rust_module.metadata["rust_backend_functions"] == {
        identity.key: (
            "matrix_vector_product_value,matrix_vector_product_jvp,"
            "matrix_vector_product_vjp,matrix_vector_product_sum_gradient"
        )
    }
    assert rust_module.metadata["rust_backend_verification_primitives"] == {
        identity.key: (
            "verified: scpn_quantum_engine matrix_vector_product value/JVP/VJP/sum-gradient parity"
        )
    }
    assert rust_module.resource_counts["rust_backend_verifications"] == 1
    assert rust_module.metadata["primitive_hard_gaps"][identity.key] == []
    assert identity.key not in rust_module.metadata["primitive_next_hard_gap"]
    assert rust_module.metadata["primitive_readiness"][identity.key] == {
        "adjoint_contract": True,
        "forward_contract": True,
        "jit_backend_contract": True,
        "llvm_backend_contract": True,
        "mlir_runtime_contract": True,
        "native_backend_contract": True,
        "registry_contract": True,
        "reverse_contract": True,
        "rust_backend_contract": True,
        "transform_contract": True,
        "verdict": "native_executable",
    }
    batched_value = vmap(
        lambda row: row,
        primitive_identity=identity,
        registry=rust_registry,
    )(np.vstack([values, values + np.array([0.25, 0.1, 0.05, -0.2, 0.3, -0.4])]))
    np.testing.assert_allclose(
        batched_value[0],
        rust_registered_kernel.value(values),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert scpn.make_matrix_vector_product_native_llvm_jit_primitive_transform is (
        compiler_mlir.make_matrix_vector_product_native_llvm_jit_primitive_transform
    )


def test_native_llvm_jit_matrix_matrix_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT matrix-matrix AD kernels should execute matrix-output AD."""

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "matrix_matrix_product", "1")

    def unpack(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left = values[:4].reshape(2, 2)
        right = values[4:].reshape(2, 2)
        return left, right

    def value_fn(values: np.ndarray) -> np.ndarray:
        left, right = unpack(values)
        return (left @ right).reshape(-1)

    def jvp_rule(values: np.ndarray, tangent: np.ndarray) -> np.ndarray:
        left, right = unpack(values)
        left_tangent, right_tangent = unpack(tangent)
        return (left_tangent @ right + left @ right_tangent).reshape(-1)

    def vjp_rule(values: np.ndarray, cotangent: np.ndarray) -> np.ndarray:
        left, right = unpack(values)
        cotangent_matrix = cotangent.reshape(2, 2)
        left_gradient = cotangent_matrix @ right.T
        right_gradient = left.T @ cotangent_matrix
        return np.concatenate([left_gradient.reshape(-1), right_gradient.reshape(-1)])

    rule = CustomDerivativeRule(
        name="native_matrix_matrix_product_rule",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
        parameter_names=("a00", "a01", "a10", "a11", "b00", "b01", "b10", "b11"),
        trainable=(True, True, True, True, True, True, True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([1.0, -2.0, 0.5, 3.0, 4.0, -1.0, 2.0, 0.25], dtype=np.float64)
    tangent = np.array([0.2, -0.1, 0.3, 0.4, -0.5, 0.75, 0.25, -0.2], dtype=np.float64)
    cotangent = np.array([1.25, -0.5, 0.75, 2.0], dtype=np.float64)

    kernel = compiler_mlir.compile_matrix_matrix_product_ad_to_native_llvm_jit(
        rule,
        dimension=2,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert kernel.verification.gradient_close is None
    assert "verified native LLVM MCJIT matrix-matrix product" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "define void @native_matrix_matrix_product_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_matrix_product_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_matrix_product_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_matrix_product_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(
        kernel.value(values),
        [0.0, -1.5, 8.0, 0.25],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.jvp(values, tangent),
        [-0.4, 0.925, 2.5, -0.425],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        [5.5, 2.375, 1.0, 2.0, 1.625, 0.5, -0.25, 7.0],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    with pytest.raises(ValueError, match="requires scalar output"):
        kernel.gradient(values)

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=lambda batch, fn: np.asarray([fn(item) for item in batch]),
            lowering_rule=compiler_mlir.make_matrix_matrix_product_native_llvm_jit_lowering_rule(
                dimension=2,
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_matrix_matrix_product",
                "mlir_runtime_verification": "verified: native LLVM/JIT matrix-matrix JVP",
                "llvm": "available: native LLVM MCJIT matrix-matrix AD kernel",
                "jit": "available: native LLVM MCJIT matrix-matrix AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT matrix-matrix value/JVP/VJP"
                ),
                "static_derivative_factory": "native_matrix_matrix_product_llvm_jit",
                "static_signature": "primitive:matmul;dimension:2;layout:left_then_right",
                "nondifferentiable_boundary": "none_smooth_matrix_matrix_product",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (2, 2),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="smooth_matrix_matrix_product_real_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]

    rust_registry = CustomDerivativeRegistry()
    rust_registry.register_transform(
        compiler_mlir.make_matrix_matrix_product_native_llvm_jit_primitive_transform(
            identity,
            rule,
            dimension=2,
            sample_values=values,
            config=config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        )
    )
    rust_plan = build_compiler_ad_transform_plan(rust_registry)
    rust_module = compile_compiler_ad_transform_plan_to_mlir(rust_plan)
    rust_registered_kernel = compile_registered_primitive_to_executable(
        rust_registry,
        identity,
        values,
    )

    assert rust_registered_kernel.backend == "native_llvm_jit"
    assert rust_plan.executable_backend == "native_llvm_jit"
    assert rust_module.metadata["executable_backend"] == "native_llvm_jit"
    assert rust_module.resource_counts["rust_backend_contracts"] == 1
    assert rust_module.resource_counts["rust_backend_incomplete_primitives"] == 0
    assert rust_module.resource_counts["rust_backend_blockers"] == 0
    assert rust_module.resource_counts["primitive_hard_gaps"] == 0
    assert rust_module.metadata["rust_backend_contract_primitives"] == [identity.key]
    assert rust_module.metadata["rust_backend_incomplete_primitives"] == []
    assert rust_module.metadata["rust_backend_blockers"] == {}
    assert rust_module.metadata["rust_backend_signatures"] == {
        identity.key: "primitive:matmul;dimension:2;layout:left_then_right"
    }
    assert rust_module.metadata["rust_backend_functions"] == {
        identity.key: (
            "matrix_matrix_product_value,matrix_matrix_product_jvp,"
            "matrix_matrix_product_vjp,matrix_matrix_product_sum_gradient"
        )
    }
    assert rust_module.metadata["rust_backend_verification_primitives"] == {
        identity.key: (
            "verified: scpn_quantum_engine matrix_matrix_product value/JVP/VJP/sum-gradient parity"
        )
    }
    assert rust_module.resource_counts["rust_backend_verifications"] == 1
    assert rust_module.metadata["primitive_hard_gaps"][identity.key] == []
    assert identity.key not in rust_module.metadata["primitive_next_hard_gap"]
    assert rust_module.metadata["primitive_readiness"][identity.key] == {
        "adjoint_contract": True,
        "forward_contract": True,
        "jit_backend_contract": True,
        "llvm_backend_contract": True,
        "mlir_runtime_contract": True,
        "native_backend_contract": True,
        "registry_contract": True,
        "reverse_contract": True,
        "rust_backend_contract": True,
        "transform_contract": True,
        "verdict": "native_executable",
    }
    batched_value = vmap(
        lambda row: row,
        primitive_identity=identity,
        registry=rust_registry,
    )(np.vstack([values, values + np.array([0.25, 0.1, 0.05, -0.2, 0.3, -0.4, 0.2, 0.1])]))
    np.testing.assert_allclose(
        batched_value[0],
        rust_registered_kernel.value(values),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert scpn.make_matrix_matrix_product_native_llvm_jit_primitive_transform is (
        compiler_mlir.make_matrix_matrix_product_native_llvm_jit_primitive_transform
    )


def test_native_llvm_jit_matrix_trace_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT matrix trace AD kernels should execute scalar-output AD."""

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "matrix_trace", "1")
    rule = CustomDerivativeRule(
        name="native_matrix_trace_rule",
        value_fn=lambda values: np.array([np.trace(values.reshape(2, 2))], dtype=np.float64),
        jvp_rule=lambda _values, tangent: np.array(
            [np.trace(tangent.reshape(2, 2))],
            dtype=np.float64,
        ),
        vjp_rule=lambda _values, cotangent: np.array(
            [cotangent[0], 0.0, 0.0, cotangent[0]],
            dtype=np.float64,
        ),
        parameter_names=("a00", "a01", "a10", "a11"),
        trainable=(True, True, True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float64)
    tangent = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float64)
    cotangent = np.array([1.25], dtype=np.float64)

    kernel = compiler_mlir.compile_matrix_trace_ad_to_native_llvm_jit(
        rule,
        dimension=2,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert kernel.verification.gradient_close is True
    assert "verified native LLVM MCJIT matrix trace" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "define void @native_matrix_trace_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_trace_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_trace_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_trace_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(kernel.value(values), [5.0], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(kernel.jvp(values, tangent), [0.5], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        [1.25, 0.0, 0.0, 1.25],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.gradient(values),
        [1.0, 0.0, 0.0, 1.0],
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=lambda batch, fn: np.asarray([fn(item) for item in batch]),
            lowering_rule=compiler_mlir.make_matrix_trace_native_llvm_jit_lowering_rule(
                dimension=2,
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_matrix_trace",
                "mlir_runtime_verification": "verified: native LLVM/JIT matrix trace JVP",
                "llvm": "available: native LLVM MCJIT matrix trace AD kernel",
                "jit": "available: native LLVM MCJIT matrix trace AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT matrix trace value/JVP/VJP/gradient"
                ),
                "static_derivative_factory": "native_matrix_trace_llvm_jit",
                "static_signature": "primitive:trace;dimension:2;layout:row_major",
                "nondifferentiable_boundary": "none_smooth_matrix_trace",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (1,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="smooth_matrix_trace_real_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]

    rust_registry = CustomDerivativeRegistry()
    rust_registry.register_transform(
        compiler_mlir.make_matrix_trace_native_llvm_jit_primitive_transform(
            identity,
            rule,
            dimension=2,
            sample_values=values,
            config=config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        )
    )
    rust_plan = build_compiler_ad_transform_plan(rust_registry)
    rust_module = compile_compiler_ad_transform_plan_to_mlir(rust_plan)
    rust_registered_kernel = compile_registered_primitive_to_executable(
        rust_registry,
        identity,
        values,
    )

    assert (
        scpn.make_matrix_trace_native_llvm_jit_primitive_transform
        is compiler_mlir.make_matrix_trace_native_llvm_jit_primitive_transform
    )
    assert rust_registered_kernel.backend == "native_llvm_jit"
    assert rust_module.resource_counts["native_backend_contracts"] == 1
    assert rust_module.resource_counts["rust_backend_contracts"] == 1
    assert rust_module.resource_counts["primitive_readiness_native_executable"] == 1
    assert rust_module.resource_counts["primitive_hard_gaps"] == 0
    assert rust_module.metadata["primitive_hard_gaps"][identity.key] == []
    assert rust_module.metadata["rust_backend_contract_primitives"] == [identity.key]
    assert rust_module.metadata["rust_backend_signatures"] == {
        identity.key: "primitive:trace;dimension:2;layout:row_major"
    }
    assert rust_module.metadata["rust_backend_functions"] == {
        identity.key: (
            "matrix_trace_value,matrix_trace_jvp,matrix_trace_vjp,matrix_trace_gradient"
        )
    }
    assert rust_module.metadata["rust_backend_verification_primitives"] == {
        identity.key: ("verified: scpn_quantum_engine matrix_trace value/JVP/VJP/gradient parity")
    }
    np.testing.assert_allclose(
        rust_registered_kernel.gradient(values),
        [1.0, 0.0, 0.0, 1.0],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_native_llvm_jit_matrix_frobenius_norm_squared_kernel_executes_and_marks_plan_native() -> (
    None
):
    """Native LLVM/JIT Frobenius-squared AD kernels should execute scalar matrix reductions."""

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "matrix_frobenius_norm_squared", "1")
    rule = CustomDerivativeRule(
        name="native_matrix_frobenius_norm_squared_rule",
        value_fn=lambda values: np.array(
            [np.sum(np.square(values.reshape(2, 2)))],
            dtype=np.float64,
        ),
        jvp_rule=lambda values, tangent: np.array(
            [2.0 * np.sum(values.reshape(2, 2) * tangent.reshape(2, 2))],
            dtype=np.float64,
        ),
        vjp_rule=lambda values, cotangent: 2.0 * cotangent[0] * values,
        parameter_names=("a00", "a01", "a10", "a11"),
        trainable=(True, True, True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float64)
    tangent = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float64)
    cotangent = np.array([1.25], dtype=np.float64)

    kernel = compiler_mlir.compile_matrix_frobenius_norm_squared_ad_to_native_llvm_jit(
        rule,
        dimension=2,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert kernel.verification.gradient_close is True
    assert "verified native LLVM MCJIT matrix Frobenius-squared" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert (
        "define void @native_matrix_frobenius_norm_squared_rule_value" in kernel.llvm_gradient_ir
    )
    assert "define void @native_matrix_frobenius_norm_squared_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_frobenius_norm_squared_rule_vjp" in kernel.llvm_gradient_ir
    assert (
        "define void @native_matrix_frobenius_norm_squared_rule_gradient"
        in kernel.llvm_gradient_ir
    )
    np.testing.assert_allclose(kernel.value(values), [14.25], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(kernel.jvp(values, tangent), [3.5], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        [5.0, -2.5, 1.25, 7.5],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.gradient(values),
        [4.0, -2.0, 1.0, 6.0],
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=lambda batch, fn: np.asarray([fn(item) for item in batch]),
            lowering_rule=compiler_mlir.make_matrix_frobenius_norm_squared_native_llvm_jit_lowering_rule(
                dimension=2,
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_matrix_frobenius_norm_squared",
                "mlir_runtime_verification": (
                    "verified: native LLVM/JIT matrix Frobenius-squared JVP"
                ),
                "llvm": "available: native LLVM MCJIT matrix Frobenius-squared AD kernel",
                "jit": "available: native LLVM MCJIT matrix Frobenius-squared AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT matrix Frobenius-squared value/JVP/VJP/gradient"
                ),
                "static_derivative_factory": "native_matrix_frobenius_norm_squared_llvm_jit",
                "static_signature": "primitive:frobenius_norm_squared;dimension:2;layout:row_major",
                "nondifferentiable_boundary": "none_smooth_matrix_frobenius_norm_squared",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (1,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="smooth_matrix_frobenius_norm_squared_real_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]

    rust_registry = CustomDerivativeRegistry()
    rust_registry.register_transform(
        compiler_mlir.make_matrix_frobenius_norm_squared_native_llvm_jit_primitive_transform(
            identity,
            rule,
            dimension=2,
            sample_values=values,
            config=config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        )
    )
    rust_plan = build_compiler_ad_transform_plan(rust_registry)
    rust_module = compile_compiler_ad_transform_plan_to_mlir(rust_plan)
    rust_registered_kernel = compile_registered_primitive_to_executable(
        rust_registry,
        identity,
        values,
    )

    assert (
        scpn.make_matrix_frobenius_norm_squared_native_llvm_jit_primitive_transform
        is compiler_mlir.make_matrix_frobenius_norm_squared_native_llvm_jit_primitive_transform
    )
    assert rust_registered_kernel.backend == "native_llvm_jit"
    assert rust_module.resource_counts["native_backend_contracts"] == 1
    assert rust_module.resource_counts["rust_backend_contracts"] == 1
    assert rust_module.resource_counts["primitive_readiness_native_executable"] == 1
    assert rust_module.resource_counts["primitive_hard_gaps"] == 0
    assert rust_module.metadata["primitive_hard_gaps"][identity.key] == []
    assert rust_module.metadata["rust_backend_contract_primitives"] == [identity.key]
    assert rust_module.metadata["rust_backend_signatures"] == {
        identity.key: "primitive:frobenius_norm_squared;dimension:2;layout:row_major"
    }
    assert rust_module.metadata["rust_backend_functions"] == {
        identity.key: (
            "matrix_frobenius_norm_squared_value,matrix_frobenius_norm_squared_jvp,"
            "matrix_frobenius_norm_squared_vjp,matrix_frobenius_norm_squared_gradient"
        )
    }
    assert rust_module.metadata["rust_backend_verification_primitives"] == {
        identity.key: (
            "verified: scpn_quantum_engine matrix_frobenius_norm_squared "
            "value/JVP/VJP/gradient parity"
        )
    }
    np.testing.assert_allclose(
        rust_registered_kernel.gradient(values),
        [4.0, -2.0, 1.0, 6.0],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_native_llvm_jit_matrix_2x2_determinant_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT determinant AD kernels should execute exact scalar matrix AD."""

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "matrix_2x2_determinant", "1")
    rule = CustomDerivativeRule(
        name="native_matrix_2x2_determinant_rule",
        value_fn=lambda values: np.array(
            [values[0] * values[3] - values[1] * values[2]],
            dtype=np.float64,
        ),
        jvp_rule=lambda values, tangent: np.array(
            [
                tangent[0] * values[3]
                + values[0] * tangent[3]
                - tangent[1] * values[2]
                - values[1] * tangent[2]
            ],
            dtype=np.float64,
        ),
        vjp_rule=lambda values, cotangent: (
            cotangent[0]
            * np.array([values[3], -values[2], -values[1], values[0]], dtype=np.float64)
        ),
        parameter_names=("a00", "a01", "a10", "a11"),
        trainable=(True, True, True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float64)
    tangent = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float64)
    cotangent = np.array([1.25], dtype=np.float64)

    kernel = compiler_mlir.compile_matrix_2x2_determinant_ad_to_native_llvm_jit(
        rule,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert kernel.verification.gradient_close is True
    assert "verified native LLVM MCJIT 2x2 determinant" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "define void @native_matrix_2x2_determinant_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_2x2_determinant_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_2x2_determinant_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_2x2_determinant_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(kernel.value(values), [6.5], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(kernel.jvp(values, tangent), [1.5], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        [3.75, -0.625, 1.25, 2.5],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.gradient(values),
        [3.0, -0.5, 1.0, 2.0],
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=lambda batch, fn: np.asarray([fn(item) for item in batch]),
            lowering_rule=compiler_mlir.make_matrix_2x2_determinant_native_llvm_jit_lowering_rule(
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_matrix_2x2_determinant",
                "mlir_runtime_verification": "verified: native LLVM/JIT 2x2 determinant JVP",
                "llvm": "available: native LLVM MCJIT 2x2 determinant AD kernel",
                "jit": "available: native LLVM MCJIT 2x2 determinant AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT 2x2 determinant value/JVP/VJP/gradient"
                ),
                "static_derivative_factory": "native_matrix_2x2_determinant_llvm_jit",
                "static_signature": "primitive:determinant;dimension:2;layout:row_major",
                "nondifferentiable_boundary": "none_polynomial_matrix_2x2_determinant",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (1,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="polynomial_matrix_2x2_determinant_real_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]

    rust_registry = CustomDerivativeRegistry()
    rust_registry.register_transform(
        compiler_mlir.make_matrix_2x2_determinant_native_llvm_jit_primitive_transform(
            identity,
            rule,
            sample_values=values,
            config=config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        )
    )
    rust_plan = build_compiler_ad_transform_plan(rust_registry)
    rust_module = compile_compiler_ad_transform_plan_to_mlir(rust_plan)
    rust_registered_kernel = compile_registered_primitive_to_executable(
        rust_registry,
        identity,
        values,
    )

    assert (
        scpn.make_matrix_2x2_determinant_native_llvm_jit_primitive_transform
        is compiler_mlir.make_matrix_2x2_determinant_native_llvm_jit_primitive_transform
    )
    assert rust_registered_kernel.backend == "native_llvm_jit"
    assert rust_module.resource_counts["native_backend_contracts"] == 1
    assert rust_module.resource_counts["rust_backend_contracts"] == 1
    assert rust_module.resource_counts["primitive_readiness_native_executable"] == 1
    assert rust_module.resource_counts["primitive_hard_gaps"] == 0
    assert rust_module.metadata["primitive_hard_gaps"][identity.key] == []
    assert rust_module.metadata["rust_backend_contract_primitives"] == [identity.key]
    assert rust_module.metadata["rust_backend_signatures"] == {
        identity.key: "primitive:determinant;dimension:2;layout:row_major"
    }
    assert rust_module.metadata["rust_backend_functions"] == {
        identity.key: (
            "matrix_2x2_determinant_value,matrix_2x2_determinant_jvp,"
            "matrix_2x2_determinant_vjp,matrix_2x2_determinant_gradient"
        )
    }
    assert rust_module.metadata["rust_backend_verification_primitives"] == {
        identity.key: (
            "verified: scpn_quantum_engine matrix_2x2_determinant value/JVP/VJP/gradient parity"
        )
    }
    np.testing.assert_allclose(
        rust_registered_kernel.gradient(values),
        [3.0, -0.5, 1.0, 2.0],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_executable_ad_kernel_batching_rule_dispatches_native_value_jvp_and_vjp() -> None:
    """Primitive vmap batching should dispatch native executable kernels without fallback calls."""

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "batched_matrix_2x2_determinant", "1")
    rule = CustomDerivativeRule(
        name="native_batched_matrix_2x2_determinant_rule",
        value_fn=lambda values: np.array(
            [values[0] * values[3] - values[1] * values[2]],
            dtype=np.float64,
        ),
        jvp_rule=lambda values, tangent: np.array(
            [
                tangent[0] * values[3]
                + values[0] * tangent[3]
                - tangent[1] * values[2]
                - values[1] * tangent[2]
            ],
            dtype=np.float64,
        ),
        vjp_rule=lambda values, cotangent: (
            cotangent[0]
            * np.array([values[3], -values[2], -values[1], values[0]], dtype=np.float64)
        ),
        parameter_names=("a00", "a01", "a10", "a11"),
        trainable=(True, True, True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    sample_values = np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float64)
    sample_tangent = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float64)
    sample_cotangent = np.array([1.25], dtype=np.float64)
    kernel = compiler_mlir.compile_matrix_2x2_determinant_ad_to_native_llvm_jit(
        rule,
        sample_values=sample_values,
        config=config,
        sample_tangent=sample_tangent,
        sample_cotangent=sample_cotangent,
    )
    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=compiler_mlir.make_executable_ad_kernel_batching_rule(kernel),
            lowering_rule=compiler_mlir.make_matrix_2x2_determinant_native_llvm_jit_lowering_rule(
                sample_values=sample_values,
                config=config,
                sample_tangent=sample_tangent,
                sample_cotangent=sample_cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_matrix_2x2_determinant",
                "llvm": "available: native LLVM MCJIT 2x2 determinant AD kernel",
                "jit": "available: native LLVM MCJIT 2x2 determinant AD kernel",
                "native_backend": "native_llvm_jit",
                "static_derivative_factory": "native_matrix_2x2_determinant_llvm_jit",
                "static_signature": "primitive:determinant;dimension:2;layout:row_major",
                "nondifferentiable_boundary": "none_polynomial_matrix_2x2_determinant",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (1,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="polynomial_matrix_2x2_determinant_real_domain",
            effect="pure",
        )
    )

    values = np.array(
        [
            [2.0, -1.0, 0.5, 3.0],
            [1.5, 0.25, -2.0, 4.0],
        ],
        dtype=np.float64,
    )
    tangents = np.array(
        [
            [0.1, -0.2, 0.3, 0.4],
            [-0.5, 0.75, 0.25, -0.2],
        ],
        dtype=np.float64,
    )
    cotangents = np.array([[1.25], [-0.5]], dtype=np.float64)

    def unreachable(*_args: object) -> np.ndarray:
        raise AssertionError("primitive-specific executable batching rule was not used")

    batched_value = vmap(unreachable, primitive_identity=identity, registry=registry)(values)
    batched_jvp = vmap(
        unreachable,
        in_axes=(0, 0),
        primitive_identity=identity,
        registry=registry,
    )(values, tangents)
    batched_vjp = vmap(
        unreachable,
        in_axes=(0, 0),
        primitive_identity=identity,
        registry=registry,
    )(values, cotangents)

    expected_values = np.asarray([kernel.value(row) for row in values])
    expected_jvps = np.asarray(
        [kernel.jvp(row, tangent) for row, tangent in zip(values, tangents)]
    )
    expected_vjps = np.asarray(
        [kernel.vjp(row, cotangent) for row, cotangent in zip(values, cotangents)]
    )
    np.testing.assert_allclose(batched_value, expected_values, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(batched_jvp, expected_jvps, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(batched_vjp, expected_vjps, rtol=1.0e-12, atol=1.0e-12)
    assert registry.require_complete_contract(identity).batching_rule is not None

    with pytest.raises(ValueError, match="ambiguous"):
        ambiguous_rule = CustomDerivativeRule(
            name="ambiguous_rule",
            value_fn=lambda row: np.asarray(row, dtype=np.float64),
            jvp_rule=lambda row, tangent: np.asarray(tangent, dtype=np.float64),
            vjp_rule=lambda row, cotangent: np.asarray(cotangent, dtype=np.float64),
            parameter_names=("x0", "x1"),
            trainable=(True, True),
        )
        ambiguous_kernel = compile_custom_derivative_rule_to_executable(
            ambiguous_rule,
            np.array([1.0, 2.0], dtype=np.float64),
            CompilerADExecutableConfig(),
        )
        compiler_mlir.make_executable_ad_kernel_batching_rule(ambiguous_kernel)(
            unreachable,
            (
                np.array([[1.0, 2.0]], dtype=np.float64),
                np.array([[0.5, -0.25]], dtype=np.float64),
            ),
            (0, 0),
            0,
        )


def test_native_llvm_jit_matrix_2x2_inverse_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT inverse AD kernels should execute exact nonsingular matrix AD."""

    def inverse(values: np.ndarray) -> np.ndarray:
        return np.linalg.inv(values.reshape(2, 2)).reshape(4)

    def inverse_jvp(values: np.ndarray, tangent: np.ndarray) -> np.ndarray:
        inverse_matrix = np.linalg.inv(values.reshape(2, 2))
        tangent_matrix = tangent.reshape(2, 2)
        return (-inverse_matrix @ tangent_matrix @ inverse_matrix).reshape(4)

    def inverse_vjp(values: np.ndarray, cotangent: np.ndarray) -> np.ndarray:
        inverse_matrix = np.linalg.inv(values.reshape(2, 2))
        cotangent_matrix = cotangent.reshape(2, 2)
        return (-(inverse_matrix.T @ cotangent_matrix @ inverse_matrix.T)).reshape(4)

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "matrix_2x2_inverse", "1")
    rule = CustomDerivativeRule(
        name="native_matrix_2x2_inverse_rule",
        value_fn=inverse,
        jvp_rule=inverse_jvp,
        vjp_rule=inverse_vjp,
        parameter_names=("a00", "a01", "a10", "a11"),
        trainable=(True, True, True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float64)
    tangent = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float64)
    cotangent = np.array([0.75, -1.25, 0.5, 2.0], dtype=np.float64)

    kernel = compiler_mlir.compile_matrix_2x2_inverse_ad_to_native_llvm_jit(
        rule,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert "verified native LLVM MCJIT 2x2 inverse" in kernel.claim_boundary
    assert "public gradient remains scalar-output fail-closed" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "define void @native_matrix_2x2_inverse_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_2x2_inverse_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_2x2_inverse_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_2x2_inverse_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(kernel.value(values), inverse(values), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        kernel.jvp(values, tangent),
        inverse_jvp(values, tangent),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        inverse_vjp(values, cotangent),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    with pytest.raises(ValueError):
        kernel.gradient(values)

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=lambda batch, fn: np.asarray([fn(item) for item in batch]),
            lowering_rule=compiler_mlir.make_matrix_2x2_inverse_native_llvm_jit_lowering_rule(
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_matrix_2x2_inverse",
                "mlir_runtime_verification": "verified: native LLVM/JIT 2x2 inverse JVP",
                "llvm": "available: native LLVM MCJIT 2x2 inverse AD kernel",
                "jit": "available: native LLVM MCJIT 2x2 inverse AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT 2x2 inverse value/JVP/VJP"
                ),
                "static_derivative_factory": "native_matrix_2x2_inverse_llvm_jit",
                "static_signature": "primitive:inverse;dimension:2;layout:row_major",
                "nondifferentiable_boundary": "singular_matrix_2x2_inverse",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (4,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="nonsingular_matrix_2x2_inverse_real_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]

    rust_registry = CustomDerivativeRegistry()
    rust_registry.register_transform(
        compiler_mlir.make_matrix_2x2_inverse_native_llvm_jit_primitive_transform(
            identity,
            rule,
            sample_values=values,
            config=config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        )
    )
    rust_plan = build_compiler_ad_transform_plan(rust_registry)
    rust_module = compile_compiler_ad_transform_plan_to_mlir(rust_plan)
    rust_registered_kernel = compile_registered_primitive_to_executable(
        rust_registry,
        identity,
        values,
    )

    assert (
        scpn.make_matrix_2x2_inverse_native_llvm_jit_primitive_transform
        is compiler_mlir.make_matrix_2x2_inverse_native_llvm_jit_primitive_transform
    )
    assert rust_registered_kernel.backend == "native_llvm_jit"
    assert rust_module.resource_counts["native_backend_contracts"] == 1
    assert rust_module.resource_counts["rust_backend_contracts"] == 1
    assert rust_module.resource_counts["primitive_readiness_native_executable"] == 1
    assert rust_module.resource_counts["primitive_hard_gaps"] == 0
    assert rust_module.metadata["primitive_hard_gaps"][identity.key] == []
    assert rust_module.metadata["rust_backend_contract_primitives"] == [identity.key]
    assert rust_module.metadata["rust_backend_signatures"] == {
        identity.key: "primitive:inverse;dimension:2;layout:row_major"
    }
    assert rust_module.metadata["rust_backend_functions"] == {
        identity.key: (
            "matrix_2x2_inverse_value,matrix_2x2_inverse_jvp,"
            "matrix_2x2_inverse_vjp,matrix_2x2_inverse_sum_gradient"
        )
    }
    assert rust_module.metadata["rust_backend_verification_primitives"] == {
        identity.key: (
            "verified: scpn_quantum_engine matrix_2x2_inverse value/JVP/VJP/sum-gradient parity"
        )
    }


def test_native_llvm_jit_matrix_2x2_solve_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT solve AD kernels should execute exact nonsingular linear solves."""

    def solve(values: np.ndarray) -> np.ndarray:
        matrix = values[:4].reshape(2, 2)
        vector = values[4:]
        return np.linalg.solve(matrix, vector)

    def solve_jvp(values: np.ndarray, tangent: np.ndarray) -> np.ndarray:
        matrix = values[:4].reshape(2, 2)
        primal_solution = np.linalg.solve(matrix, values[4:])
        tangent_matrix = tangent[:4].reshape(2, 2)
        tangent_vector = tangent[4:]
        return np.linalg.solve(matrix, tangent_vector - tangent_matrix @ primal_solution)

    def solve_vjp(values: np.ndarray, cotangent: np.ndarray) -> np.ndarray:
        matrix = values[:4].reshape(2, 2)
        primal_solution = np.linalg.solve(matrix, values[4:])
        adjoint_vector = np.linalg.solve(matrix.T, cotangent)
        return np.concatenate(
            [
                (-np.outer(adjoint_vector, primal_solution)).reshape(4),
                adjoint_vector,
            ]
        )

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "matrix_2x2_solve", "1")
    rule = CustomDerivativeRule(
        name="native_matrix_2x2_solve_rule",
        value_fn=solve,
        jvp_rule=solve_jvp,
        vjp_rule=solve_vjp,
        parameter_names=("a00", "a01", "a10", "a11", "b0", "b1"),
        trainable=(True, True, True, True, True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([2.0, -1.0, 0.5, 3.0, 1.5, -2.0], dtype=np.float64)
    tangent = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.75], dtype=np.float64)
    cotangent = np.array([1.25, -0.75], dtype=np.float64)

    kernel = compiler_mlir.compile_matrix_2x2_solve_ad_to_native_llvm_jit(
        rule,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert "verified native LLVM MCJIT 2x2 solve" in kernel.claim_boundary
    assert "public gradient remains scalar-output fail-closed" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "define void @native_matrix_2x2_solve_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_2x2_solve_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_2x2_solve_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_2x2_solve_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(kernel.value(values), solve(values), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        kernel.jvp(values, tangent),
        solve_jvp(values, tangent),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        solve_vjp(values, cotangent),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    with pytest.raises(ValueError):
        kernel.gradient(values)
    with pytest.raises(ValueError, match="nonsingular"):
        compiler_mlir.compile_matrix_2x2_solve_ad_to_native_llvm_jit(
            rule,
            sample_values=np.array([1.0, 2.0, 2.0, 4.0, 1.0, -1.0], dtype=np.float64),
            config=config,
        )

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=lambda batch, fn: np.asarray([fn(item) for item in batch]),
            lowering_rule=compiler_mlir.make_matrix_2x2_solve_native_llvm_jit_lowering_rule(
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_matrix_2x2_solve",
                "mlir_runtime_verification": "verified: native LLVM/JIT 2x2 solve JVP",
                "llvm": "available: native LLVM MCJIT 2x2 solve AD kernel",
                "jit": "available: native LLVM MCJIT 2x2 solve AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT 2x2 solve value/JVP/VJP"
                ),
                "static_derivative_factory": "native_matrix_2x2_solve_llvm_jit",
                "static_signature": "primitive:solve;dimension:2;layout:row_major",
                "nondifferentiable_boundary": "singular_matrix_2x2_solve",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (2,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="nonsingular_matrix_2x2_solve_real_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]

    rust_registry = CustomDerivativeRegistry()
    rust_registry.register_transform(
        compiler_mlir.make_matrix_2x2_solve_native_llvm_jit_primitive_transform(
            identity,
            rule,
            sample_values=values,
            config=config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        )
    )
    rust_plan = build_compiler_ad_transform_plan(rust_registry)
    rust_module = compile_compiler_ad_transform_plan_to_mlir(rust_plan)
    rust_registered_kernel = compile_registered_primitive_to_executable(
        rust_registry,
        identity,
        values,
    )

    assert (
        scpn.make_matrix_2x2_solve_native_llvm_jit_primitive_transform
        is compiler_mlir.make_matrix_2x2_solve_native_llvm_jit_primitive_transform
    )
    assert rust_registered_kernel.backend == "native_llvm_jit"
    assert rust_module.resource_counts["native_backend_contracts"] == 1
    assert rust_module.resource_counts["rust_backend_contracts"] == 1
    assert rust_module.resource_counts["primitive_readiness_native_executable"] == 1
    assert rust_module.resource_counts["primitive_hard_gaps"] == 0
    assert rust_module.metadata["primitive_hard_gaps"][identity.key] == []
    assert rust_module.metadata["rust_backend_contract_primitives"] == [identity.key]
    assert rust_module.metadata["rust_backend_signatures"] == {
        identity.key: "primitive:solve;dimension:2;layout:row_major"
    }
    assert rust_module.metadata["rust_backend_functions"] == {
        identity.key: (
            "matrix_2x2_solve_value,matrix_2x2_solve_jvp,"
            "matrix_2x2_solve_vjp,matrix_2x2_solve_sum_gradient"
        )
    }
    assert rust_module.metadata["rust_backend_verification_primitives"] == {
        identity.key: (
            "verified: scpn_quantum_engine matrix_2x2_solve value/JVP/VJP/sum-gradient parity"
        )
    }


def test_native_llvm_jit_symmetric_2x2_cholesky_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT Cholesky AD kernels should execute exact SPD factorisation AD."""

    def cholesky(values: np.ndarray) -> np.ndarray:
        a00, a01, a11 = values
        l00 = np.sqrt(a00)
        l10 = a01 / l00
        l11 = np.sqrt(a11 - l10 * l10)
        return np.array([l00, l10, l11], dtype=np.float64)

    def cholesky_jvp(values: np.ndarray, tangent: np.ndarray) -> np.ndarray:
        a00, a01, a11 = values
        t00, t01, t11 = tangent
        l00 = np.sqrt(a00)
        l10 = a01 / l00
        l11 = np.sqrt(a11 - l10 * l10)
        tangent_l00 = t00 / (2.0 * l00)
        tangent_l10 = t01 / l00 - l10 * tangent_l00 / l00
        tangent_l11 = (t11 - 2.0 * l10 * tangent_l10) / (2.0 * l11)
        return np.array([tangent_l00, tangent_l10, tangent_l11], dtype=np.float64)

    def cholesky_vjp(values: np.ndarray, cotangent: np.ndarray) -> np.ndarray:
        a00, a01, a11 = values
        cotangent_l00, cotangent_l10, cotangent_l11 = cotangent
        l00 = np.sqrt(a00)
        l10 = a01 / l00
        l11 = np.sqrt(a11 - l10 * l10)
        adjoint_schur = cotangent_l11 / (2.0 * l11)
        adjoint_l10 = cotangent_l10 - 2.0 * l10 * adjoint_schur
        adjoint_l00 = cotangent_l00 - adjoint_l10 * a01 / (l00 * l00)
        return np.array(
            [
                adjoint_l00 / (2.0 * l00),
                adjoint_l10 / l00,
                adjoint_schur,
            ],
            dtype=np.float64,
        )

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "symmetric_2x2_cholesky", "1")
    rule = CustomDerivativeRule(
        name="native_symmetric_2x2_cholesky_rule",
        value_fn=cholesky,
        jvp_rule=cholesky_jvp,
        vjp_rule=cholesky_vjp,
        parameter_names=("a00", "a01", "a11"),
        trainable=(True, True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([4.0, 1.0, 3.0], dtype=np.float64)
    tangent = np.array([0.2, -0.3, 0.4], dtype=np.float64)
    cotangent = np.array([1.25, -0.75, 0.5], dtype=np.float64)

    kernel = compiler_mlir.compile_symmetric_2x2_cholesky_ad_to_native_llvm_jit(
        rule,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert "verified native LLVM MCJIT SPD symmetric 2x2 Cholesky" in kernel.claim_boundary
    assert "non-SPD matrices remain fail-closed" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "define void @native_symmetric_2x2_cholesky_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_symmetric_2x2_cholesky_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_symmetric_2x2_cholesky_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_symmetric_2x2_cholesky_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(
        kernel.value(values),
        cholesky(values),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.jvp(values, tangent),
        cholesky_jvp(values, tangent),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        cholesky_vjp(values, cotangent),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    with pytest.raises(ValueError):
        kernel.gradient(values)
    with pytest.raises(ValueError, match="positive definite"):
        compiler_mlir.compile_symmetric_2x2_cholesky_ad_to_native_llvm_jit(
            rule,
            sample_values=np.array([1.0, 2.0, 1.0], dtype=np.float64),
            config=config,
        )

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=lambda batch, fn: np.asarray([fn(item) for item in batch]),
            lowering_rule=compiler_mlir.make_symmetric_2x2_cholesky_native_llvm_jit_lowering_rule(
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_symmetric_2x2_cholesky",
                "mlir_runtime_verification": (
                    "verified: native LLVM/JIT SPD symmetric 2x2 Cholesky JVP"
                ),
                "llvm": "available: native LLVM MCJIT SPD symmetric 2x2 Cholesky AD kernel",
                "jit": "available: native LLVM MCJIT SPD symmetric 2x2 Cholesky AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT SPD symmetric 2x2 Cholesky value/JVP/VJP"
                ),
                "static_derivative_factory": "native_symmetric_2x2_cholesky_llvm_jit",
                "static_signature": "primitive:cholesky;dimension:2;layout:upper_triangle",
                "nondifferentiable_boundary": "non_spd_symmetric_2x2_cholesky",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (3,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="symmetric_2x2_spd_cholesky_real_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]

    rust_registry = CustomDerivativeRegistry()
    rust_registry.register_transform(
        compiler_mlir.make_symmetric_2x2_cholesky_native_llvm_jit_primitive_transform(
            identity,
            rule,
            sample_values=values,
            config=config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        )
    )
    rust_plan = build_compiler_ad_transform_plan(rust_registry)
    rust_module = compile_compiler_ad_transform_plan_to_mlir(rust_plan)
    rust_registered_kernel = compile_registered_primitive_to_executable(
        rust_registry,
        identity,
        values,
    )

    assert (
        scpn.make_symmetric_2x2_cholesky_native_llvm_jit_primitive_transform
        is compiler_mlir.make_symmetric_2x2_cholesky_native_llvm_jit_primitive_transform
    )
    assert rust_registered_kernel.backend == "native_llvm_jit"
    assert rust_module.resource_counts["native_backend_contracts"] == 1
    assert rust_module.resource_counts["rust_backend_contracts"] == 1
    assert rust_module.resource_counts["primitive_readiness_native_executable"] == 1
    assert rust_module.resource_counts["primitive_hard_gaps"] == 0
    assert rust_module.metadata["primitive_hard_gaps"][identity.key] == []
    assert rust_module.metadata["rust_backend_contract_primitives"] == [identity.key]
    assert rust_module.metadata["rust_backend_signatures"] == {
        identity.key: "primitive:cholesky;dimension:2;layout:upper_triangle"
    }
    assert rust_module.metadata["rust_backend_functions"] == {
        identity.key: (
            "symmetric_2x2_cholesky_value,symmetric_2x2_cholesky_jvp,"
            "symmetric_2x2_cholesky_vjp,symmetric_2x2_cholesky_sum_gradient"
        )
    }
    assert rust_module.metadata["rust_backend_verification_primitives"] == {
        identity.key: (
            "verified: scpn_quantum_engine symmetric_2x2_cholesky "
            "value/JVP/VJP/sum-gradient parity"
        )
    }


def test_native_llvm_jit_symmetric_2x2_eigenvalues_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT eigenspectrum AD kernels should execute distinct symmetric 2x2 AD."""

    def eigenvalues(values: np.ndarray) -> np.ndarray:
        a00, a01, a11 = values
        centre = 0.5 * (a00 + a11)
        half_delta = 0.5 * (a00 - a11)
        radius = np.sqrt(half_delta * half_delta + a01 * a01)
        return np.array([centre - radius, centre + radius], dtype=np.float64)

    def eigenvalues_jvp(values: np.ndarray, tangent: np.ndarray) -> np.ndarray:
        a00, a01, a11 = values
        t00, t01, t11 = tangent
        half_delta = 0.5 * (a00 - a11)
        tangent_half_delta = 0.5 * (t00 - t11)
        tangent_centre = 0.5 * (t00 + t11)
        radius = np.sqrt(half_delta * half_delta + a01 * a01)
        tangent_radius = (half_delta * tangent_half_delta + a01 * t01) / radius
        return np.array(
            [tangent_centre - tangent_radius, tangent_centre + tangent_radius],
            dtype=np.float64,
        )

    def eigenvalues_vjp(values: np.ndarray, cotangent: np.ndarray) -> np.ndarray:
        a00, a01, a11 = values
        lower_cotangent, upper_cotangent = cotangent
        half_delta = 0.5 * (a00 - a11)
        radius = np.sqrt(half_delta * half_delta + a01 * a01)
        half_term = half_delta / (2.0 * radius)
        offdiag_term = a01 / radius
        return np.array(
            [
                lower_cotangent * (0.5 - half_term) + upper_cotangent * (0.5 + half_term),
                (upper_cotangent - lower_cotangent) * offdiag_term,
                lower_cotangent * (0.5 + half_term) + upper_cotangent * (0.5 - half_term),
            ],
            dtype=np.float64,
        )

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "symmetric_2x2_eigenvalues", "1")
    rule = CustomDerivativeRule(
        name="native_symmetric_2x2_eigenvalues_rule",
        value_fn=eigenvalues,
        jvp_rule=eigenvalues_jvp,
        vjp_rule=eigenvalues_vjp,
        parameter_names=("a00", "a01", "a11"),
        trainable=(True, True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([2.0, 0.5, 3.0], dtype=np.float64)
    tangent = np.array([0.1, -0.2, 0.4], dtype=np.float64)
    cotangent = np.array([1.25, -0.75], dtype=np.float64)

    kernel = compiler_mlir.compile_symmetric_2x2_eigenvalues_ad_to_native_llvm_jit(
        rule,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert "verified native LLVM MCJIT distinct symmetric 2x2 eigenvalues" in kernel.claim_boundary
    assert "repeated eigenvalues remain fail-closed" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "define void @native_symmetric_2x2_eigenvalues_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_symmetric_2x2_eigenvalues_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_symmetric_2x2_eigenvalues_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_symmetric_2x2_eigenvalues_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(
        kernel.value(values),
        eigenvalues(values),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.jvp(values, tangent),
        eigenvalues_jvp(values, tangent),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        eigenvalues_vjp(values, cotangent),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    with pytest.raises(ValueError):
        kernel.gradient(values)
    with pytest.raises(ValueError, match="distinct"):
        compiler_mlir.compile_symmetric_2x2_eigenvalues_ad_to_native_llvm_jit(
            rule,
            sample_values=np.array([1.0, 0.0, 1.0], dtype=np.float64),
            config=config,
        )

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=lambda batch, fn: np.asarray([fn(item) for item in batch]),
            lowering_rule=compiler_mlir.make_symmetric_2x2_eigenvalues_native_llvm_jit_lowering_rule(
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_symmetric_2x2_eigenvalues",
                "mlir_runtime_verification": (
                    "verified: native LLVM/JIT distinct symmetric 2x2 eigenvalue JVP"
                ),
                "llvm": "available: native LLVM MCJIT distinct symmetric 2x2 eigenvalue AD kernel",
                "jit": "available: native LLVM MCJIT distinct symmetric 2x2 eigenvalue AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT distinct symmetric 2x2 eigenvalue value/JVP/VJP"
                ),
                "static_derivative_factory": "native_symmetric_2x2_eigenvalues_llvm_jit",
                "static_signature": "primitive:eigvalsh;dimension:2;layout:upper_triangle",
                "nondifferentiable_boundary": "repeated_symmetric_2x2_eigenvalue",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (2,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="distinct_symmetric_2x2_eigenvalues_real_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]

    rust_registry = CustomDerivativeRegistry()
    rust_registry.register_transform(
        compiler_mlir.make_symmetric_2x2_eigenvalues_native_llvm_jit_primitive_transform(
            identity,
            rule,
            sample_values=values,
            config=config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        )
    )
    rust_plan = build_compiler_ad_transform_plan(rust_registry)
    rust_module = compile_compiler_ad_transform_plan_to_mlir(rust_plan)
    rust_registered_kernel = compile_registered_primitive_to_executable(
        rust_registry,
        identity,
        values,
    )

    assert (
        scpn.make_symmetric_2x2_eigenvalues_native_llvm_jit_primitive_transform
        is compiler_mlir.make_symmetric_2x2_eigenvalues_native_llvm_jit_primitive_transform
    )
    assert rust_registered_kernel.backend == "native_llvm_jit"
    assert rust_module.resource_counts["native_backend_contracts"] == 1
    assert rust_module.resource_counts["rust_backend_contracts"] == 1
    assert rust_module.resource_counts["primitive_readiness_native_executable"] == 1
    assert rust_module.resource_counts["primitive_hard_gaps"] == 0
    assert rust_module.metadata["primitive_hard_gaps"][identity.key] == []
    assert rust_module.metadata["rust_backend_contract_primitives"] == [identity.key]
    assert rust_module.metadata["rust_backend_signatures"] == {
        identity.key: "primitive:eigvalsh;dimension:2;layout:upper_triangle"
    }
    assert rust_module.metadata["rust_backend_functions"] == {
        identity.key: (
            "symmetric_2x2_eigenvalues_value,symmetric_2x2_eigenvalues_jvp,"
            "symmetric_2x2_eigenvalues_vjp,symmetric_2x2_eigenvalues_sum_gradient"
        )
    }
    assert rust_module.metadata["rust_backend_verification_primitives"] == {
        identity.key: (
            "verified: scpn_quantum_engine symmetric_2x2_eigenvalues "
            "value/JVP/VJP/sum-gradient parity"
        )
    }


def test_native_llvm_jit_matrix_2x2_eigenvalues_kernel_executes_and_marks_plan_native() -> None:
    """Native LLVM/JIT eigenspectrum AD should cover real-simple nonsymmetric 2x2 matrices."""

    def eigenvalues(values: np.ndarray) -> np.ndarray:
        a00, a01, a10, a11 = values
        trace = a00 + a11
        discriminant = (a00 - a11) * (a00 - a11) + 4.0 * a01 * a10
        root = np.sqrt(discriminant)
        return np.array([0.5 * (trace - root), 0.5 * (trace + root)], dtype=np.float64)

    def eigenvalues_jvp(values: np.ndarray, tangent: np.ndarray) -> np.ndarray:
        a00, a01, a10, a11 = values
        t00, t01, t10, t11 = tangent
        delta = a00 - a11
        trace_tangent = t00 + t11
        discriminant = delta * delta + 4.0 * a01 * a10
        root = np.sqrt(discriminant)
        discriminant_tangent = 2.0 * delta * (t00 - t11) + 4.0 * (t01 * a10 + a01 * t10)
        root_tangent = discriminant_tangent / (2.0 * root)
        return np.array(
            [0.5 * (trace_tangent - root_tangent), 0.5 * (trace_tangent + root_tangent)],
            dtype=np.float64,
        )

    def eigenvalues_vjp(values: np.ndarray, cotangent: np.ndarray) -> np.ndarray:
        a00, a01, a10, a11 = values
        lower_cotangent, upper_cotangent = cotangent
        delta = a00 - a11
        discriminant = delta * delta + 4.0 * a01 * a10
        root = np.sqrt(discriminant)
        alpha = 0.5 * (lower_cotangent + upper_cotangent)
        beta = (upper_cotangent - lower_cotangent) / (4.0 * root)
        delta_term = 2.0 * delta * beta
        return np.array(
            [
                alpha + delta_term,
                4.0 * a10 * beta,
                4.0 * a01 * beta,
                alpha - delta_term,
            ],
            dtype=np.float64,
        )

    identity = PrimitiveIdentity("scpn.compiler_ad.native", "matrix_2x2_eigenvalues", "1")
    rule = CustomDerivativeRule(
        name="native_matrix_2x2_eigenvalues_rule",
        value_fn=eigenvalues,
        jvp_rule=eigenvalues_jvp,
        vjp_rule=eigenvalues_vjp,
        parameter_names=("a00", "a01", "a10", "a11"),
        trainable=(True, True, True, True),
    )
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    values = np.array([2.0, 0.25, 0.75, 1.0], dtype=np.float64)
    tangent = np.array([0.1, -0.2, 0.4, -0.3], dtype=np.float64)
    cotangent = np.array([1.25, -0.75], dtype=np.float64)

    kernel = compiler_mlir.compile_matrix_2x2_eigenvalues_ad_to_native_llvm_jit(
        rule,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.passed is True
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert "verified native LLVM MCJIT real-simple nonsymmetric 2x2 eigenvalues" in (
        kernel.claim_boundary
    )
    assert "complex or repeated eigenvalues remain fail-closed" in kernel.claim_boundary
    assert kernel.llvm_gradient_ir is not None
    assert "define void @native_matrix_2x2_eigenvalues_rule_value" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_2x2_eigenvalues_rule_jvp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_2x2_eigenvalues_rule_vjp" in kernel.llvm_gradient_ir
    assert "define void @native_matrix_2x2_eigenvalues_rule_gradient" in kernel.llvm_gradient_ir
    np.testing.assert_allclose(
        kernel.value(values), eigenvalues(values), rtol=1.0e-12, atol=1.0e-12
    )
    np.testing.assert_allclose(
        kernel.jvp(values, tangent),
        eigenvalues_jvp(values, tangent),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        eigenvalues_vjp(values, cotangent),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    with pytest.raises(ValueError):
        kernel.gradient(values)
    with pytest.raises(ValueError, match="real distinct eigenvalues"):
        compiler_mlir.compile_matrix_2x2_eigenvalues_ad_to_native_llvm_jit(
            rule,
            sample_values=np.array([0.0, -1.0, 1.0, 0.0], dtype=np.float64),
            config=config,
        )
    with pytest.raises(ValueError, match="real distinct eigenvalues"):
        compiler_mlir.compile_matrix_2x2_eigenvalues_ad_to_native_llvm_jit(
            rule,
            sample_values=np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64),
            config=config,
        )

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=lambda batch, fn: np.asarray([fn(item) for item in batch]),
            lowering_rule=compiler_mlir.make_matrix_2x2_eigenvalues_native_llvm_jit_lowering_rule(
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_matrix_2x2_eigenvalues",
                "mlir_runtime_verification": (
                    "verified: native LLVM/JIT real-simple nonsymmetric 2x2 eigenvalue JVP"
                ),
                "llvm": "available: native LLVM MCJIT real-simple nonsymmetric 2x2 eigenvalue AD kernel",
                "jit": "available: native LLVM MCJIT real-simple nonsymmetric 2x2 eigenvalue AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT real-simple nonsymmetric 2x2 eigenvalue value/JVP/VJP"
                ),
                "static_derivative_factory": "native_matrix_2x2_eigenvalues_llvm_jit",
                "static_signature": "primitive:eigvals;dimension:2;layout:row_major;domain:real_simple",
                "nondifferentiable_boundary": "nonreal_or_repeated_matrix_2x2_eigenvalue",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (2,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="real_simple_matrix_2x2_eigenvalues_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert scpn.compile_matrix_2x2_eigenvalues_ad_to_native_llvm_jit is (
        compiler_mlir.compile_matrix_2x2_eigenvalues_ad_to_native_llvm_jit
    )
    assert scpn.make_matrix_2x2_eigenvalues_native_llvm_jit_lowering_rule is (
        compiler_mlir.make_matrix_2x2_eigenvalues_native_llvm_jit_lowering_rule
    )
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]

    rust_registry = CustomDerivativeRegistry()
    rust_registry.register_transform(
        compiler_mlir.make_matrix_2x2_eigenvalues_native_llvm_jit_primitive_transform(
            identity,
            rule,
            sample_values=values,
            config=config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        )
    )
    rust_plan = build_compiler_ad_transform_plan(rust_registry)
    rust_module = compile_compiler_ad_transform_plan_to_mlir(rust_plan)
    rust_registered_kernel = compile_registered_primitive_to_executable(
        rust_registry,
        identity,
        values,
    )

    assert rust_registered_kernel.backend == "native_llvm_jit"
    assert rust_plan.executable_backend == "native_llvm_jit"
    assert rust_module.metadata["executable_backend"] == "native_llvm_jit"
    assert rust_module.resource_counts["rust_backend_contracts"] == 1
    assert rust_module.resource_counts["rust_backend_incomplete_primitives"] == 0
    assert rust_module.resource_counts["rust_backend_blockers"] == 0
    assert rust_module.resource_counts["primitive_hard_gaps"] == 0
    assert rust_module.metadata["rust_backend_contract_primitives"] == [identity.key]
    assert rust_module.metadata["rust_backend_incomplete_primitives"] == []
    assert rust_module.metadata["rust_backend_blockers"] == {}
    assert rust_module.metadata["rust_backend_signatures"] == {
        identity.key: "primitive:eigvals;dimension:2;layout:row_major;domain:real_simple"
    }
    assert rust_module.metadata["rust_backend_functions"] == {
        identity.key: (
            "matrix_2x2_eigenvalues_value,matrix_2x2_eigenvalues_jvp,"
            "matrix_2x2_eigenvalues_vjp,matrix_2x2_eigenvalues_sum_gradient"
        )
    }
    assert rust_module.metadata["rust_backend_verification_primitives"] == {
        identity.key: (
            "verified: scpn_quantum_engine matrix_2x2_eigenvalues "
            "value/JVP/VJP/sum-gradient parity"
        )
    }
    assert rust_module.resource_counts["rust_backend_verifications"] == 1
    assert rust_module.metadata["primitive_hard_gaps"][identity.key] == []
    assert identity.key not in rust_module.metadata["primitive_next_hard_gap"]
    assert rust_module.metadata["primitive_readiness"][identity.key] == {
        "adjoint_contract": True,
        "forward_contract": True,
        "jit_backend_contract": True,
        "llvm_backend_contract": True,
        "mlir_runtime_contract": True,
        "native_backend_contract": True,
        "registry_contract": True,
        "reverse_contract": True,
        "rust_backend_contract": True,
        "transform_contract": True,
        "verdict": "native_executable",
    }
    batched_value = vmap(
        lambda row: row,
        primitive_identity=identity,
        registry=rust_registry,
    )(np.vstack([values, values + np.array([0.25, 0.1, 0.05, -0.2], dtype=np.float64)]))
    np.testing.assert_allclose(
        batched_value[0],
        rust_registered_kernel.value(values),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert scpn.make_matrix_2x2_eigenvalues_native_llvm_jit_primitive_transform is (
        compiler_mlir.make_matrix_2x2_eigenvalues_native_llvm_jit_primitive_transform
    )


def test_native_llvm_jit_matrix_2x2_eigensystem_kernel_executes_and_marks_plan_native() -> None:
    """Bounded 2x2 nonsymmetric eigensystem AD must execute as native LLVM/JIT."""

    def eigensystem(values: np.ndarray) -> np.ndarray:
        a00, a01, a10, a11 = values
        trace = a00 + a11
        delta = a00 - a11
        discriminant = delta * delta + 4.0 * a01 * a10
        root = float(np.sqrt(discriminant))
        lower = 0.5 * (trace - root)
        upper = 0.5 * (trace + root)
        q_lower = 0.5 * (-delta - root)
        q_upper = 0.5 * (-delta + root)
        lower_vec = np.array([a01, q_lower], dtype=np.float64)
        upper_vec = np.array([a01, q_upper], dtype=np.float64)
        lower_vec = lower_vec / np.linalg.norm(lower_vec)
        upper_vec = upper_vec / np.linalg.norm(upper_vec)
        return np.array(
            [lower, upper, lower_vec[0], upper_vec[0], lower_vec[1], upper_vec[1]],
            dtype=np.float64,
        )

    def eigensystem_jvp(values: np.ndarray, tangent: np.ndarray) -> np.ndarray:
        a00, a01, a10, a11 = values
        t00, t01, t10, t11 = tangent
        trace_t = t00 + t11
        delta = a00 - a11
        delta_t = t00 - t11
        discriminant = delta * delta + 4.0 * a01 * a10
        root = float(np.sqrt(discriminant))
        discriminant_t = 2.0 * delta * delta_t + 4.0 * (t01 * a10 + a01 * t10)
        root_t = discriminant_t / (2.0 * root)
        lower_t = 0.5 * (trace_t - root_t)
        upper_t = 0.5 * (trace_t + root_t)
        q_lower = 0.5 * (-(a00 - a11) - root)
        q_upper = 0.5 * (-(a00 - a11) + root)
        q_lower_t = lower_t - t00
        q_upper_t = upper_t - t00

        def vector_tangent(q_value: float, q_tangent: float) -> np.ndarray:
            raw = np.array([a01, q_value], dtype=np.float64)
            raw_tangent = np.array([t01, q_tangent], dtype=np.float64)
            norm = np.linalg.norm(raw)
            vector = raw / norm
            return (raw_tangent - vector * float(vector @ raw_tangent)) / norm

        lower_vec_t = vector_tangent(float(q_lower), float(q_lower_t))
        upper_vec_t = vector_tangent(float(q_upper), float(q_upper_t))
        return np.array(
            [
                lower_t,
                upper_t,
                lower_vec_t[0],
                upper_vec_t[0],
                lower_vec_t[1],
                upper_vec_t[1],
            ],
            dtype=np.float64,
        )

    def eigensystem_vjp(values: np.ndarray, cotangent: np.ndarray) -> np.ndarray:
        adjoint = np.zeros(4, dtype=np.float64)
        for index in range(4):
            basis = np.zeros(4, dtype=np.float64)
            basis[index] = 1.0
            adjoint[index] = float(cotangent @ eigensystem_jvp(values, basis))
        return adjoint

    import scpn_quantum_control as scpn

    identity = PrimitiveIdentity(
        "scpn.compiler_ad.native",
        "matrix_2x2_eigensystem",
        "1",
    )
    rule = CustomDerivativeRule(
        "native_matrix_2x2_eigensystem_rule",
        eigensystem,
        eigensystem_jvp,
        eigensystem_vjp,
        parameter_names=("a00", "a01", "a10", "a11"),
        trainable=(True, True, True, True),
    )
    values = np.array([2.0, 0.25, 0.75, 1.0], dtype=np.float64)
    tangent = np.array([0.1, -0.2, 0.4, -0.3], dtype=np.float64)
    cotangent = np.array([1.25, -0.75, 0.5, -0.25, 0.3, -0.6], dtype=np.float64)
    config = CompilerADExecutableConfig(backend="native_llvm_jit")

    kernel = compiler_mlir.compile_matrix_2x2_eigensystem_ad_to_native_llvm_jit(
        rule,
        sample_values=values,
        config=config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    assert kernel.backend == "native_llvm_jit"
    assert kernel.verification.value_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert kernel.verification.gradient_close is None
    assert kernel.claim_boundary == (
        "verified native LLVM MCJIT real-simple nonsymmetric 2x2 eigensystem value/JVP/VJP "
        "kernel with sum-output gradient provenance; complex spectra, repeated eigenvalues, "
        "and zero upper off-diagonal eigenvector charts remain fail-closed"
    )
    assert "matrix_2x2_eigensystem" in (kernel.llvm_gradient_ir or "")
    np.testing.assert_allclose(kernel.value(values), eigensystem(values), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        kernel.jvp(values, tangent),
        eigensystem_jvp(values, tangent),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        kernel.vjp(values, cotangent),
        eigensystem_vjp(values, cotangent),
        rtol=1e-12,
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="scalar output"):
        kernel.gradient(values)
    with pytest.raises(ValueError, match="real distinct eigenvalues"):
        kernel.value(np.array([0.0, -1.0, 1.0, 0.0], dtype=np.float64))
    with pytest.raises(ValueError, match="real distinct eigenvalues"):
        kernel.value(np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64))
    with pytest.raises(ValueError, match="upper off-diagonal eigenvector chart"):
        kernel.value(np.array([2.0, 0.0, 1.0, 1.0], dtype=np.float64))

    registry = CustomDerivativeRegistry()
    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            batching_rule=lambda batch, fn: np.asarray([fn(item) for item in batch]),
            lowering_rule=compiler_mlir.make_matrix_2x2_eigensystem_native_llvm_jit_lowering_rule(
                sample_values=values,
                config=config,
                sample_tangent=tangent,
                sample_cotangent=cotangent,
            ),
            lowering_metadata={
                "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
                "mlir_op": "scpn_diff.native_matrix_2x2_eigensystem",
                "mlir_runtime_verification": (
                    "verified: native LLVM/JIT real-simple nonsymmetric 2x2 eigensystem JVP"
                ),
                "llvm": "available: native LLVM MCJIT real-simple nonsymmetric 2x2 eigensystem AD kernel",
                "jit": "available: native LLVM MCJIT real-simple nonsymmetric 2x2 eigensystem AD kernel",
                "native_backend": "native_llvm_jit",
                "native_backend_verification": (
                    "verified: native LLVM MCJIT real-simple nonsymmetric 2x2 eigensystem value/JVP/VJP"
                ),
                "static_derivative_factory": "native_matrix_2x2_eigensystem_llvm_jit",
                "static_signature": (
                    "primitive:eig;dimension:2;layout:row_major;domain:real_simple_upper_chart"
                ),
                "nondifferentiable_boundary": (
                    "nonreal_repeated_or_zero_upper_chart_matrix_2x2_eigensystem"
                ),
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            shape_rule=lambda _args: (6,),
            dtype_rule=lambda _args: "float64",
            static_argument_rule=lambda args: args,
            nondifferentiable_policy="real_simple_upper_chart_matrix_2x2_eigensystem_domain",
            effect="pure",
        )
    )
    plan = build_compiler_ad_transform_plan(registry)
    module = compile_compiler_ad_transform_plan_to_mlir(plan)
    registered_kernel = compile_registered_primitive_to_executable(registry, identity, values)

    assert registered_kernel.backend == "native_llvm_jit"
    assert plan.executable_backend == "native_llvm_jit"
    assert module.metadata["executable_backend"] == "native_llvm_jit"
    assert module.resource_counts["native_backend_contracts"] == 1
    assert module.resource_counts["primitive_readiness_native_executable"] == 1
    assert scpn.compile_matrix_2x2_eigensystem_ad_to_native_llvm_jit is (
        compiler_mlir.compile_matrix_2x2_eigensystem_ad_to_native_llvm_jit
    )
    assert scpn.make_matrix_2x2_eigensystem_native_llvm_jit_lowering_rule is (
        compiler_mlir.make_matrix_2x2_eigensystem_native_llvm_jit_lowering_rule
    )
    assert module.metadata["native_backend_contract_primitives"] == [identity.key]
    assert module.metadata["primitive_readiness"][identity.key]["verdict"] == "native_executable"
    assert module.metadata["primitive_hard_gaps"][identity.key] == ["rust_backend_contract"]

    rust_registry = CustomDerivativeRegistry()
    rust_registry.register_transform(
        compiler_mlir.make_matrix_2x2_eigensystem_native_llvm_jit_primitive_transform(
            identity,
            rule,
            sample_values=values,
            config=config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        )
    )
    rust_plan = build_compiler_ad_transform_plan(rust_registry)
    rust_module = compile_compiler_ad_transform_plan_to_mlir(rust_plan)
    rust_registered_kernel = compile_registered_primitive_to_executable(
        rust_registry,
        identity,
        values,
    )

    assert rust_registered_kernel.backend == "native_llvm_jit"
    assert rust_plan.executable_backend == "native_llvm_jit"
    assert rust_module.metadata["executable_backend"] == "native_llvm_jit"
    assert rust_module.resource_counts["rust_backend_contracts"] == 1
    assert rust_module.resource_counts["rust_backend_incomplete_primitives"] == 0
    assert rust_module.resource_counts["rust_backend_blockers"] == 0
    assert rust_module.resource_counts["primitive_hard_gaps"] == 0
    assert rust_module.metadata["rust_backend_contract_primitives"] == [identity.key]
    assert rust_module.metadata["rust_backend_incomplete_primitives"] == []
    assert rust_module.metadata["rust_backend_blockers"] == {}
    assert rust_module.metadata["rust_backend_signatures"] == {
        identity.key: "primitive:eig;dimension:2;layout:row_major;domain:real_simple_upper_chart"
    }
    assert rust_module.metadata["rust_backend_functions"] == {
        identity.key: (
            "matrix_2x2_eigensystem_value,matrix_2x2_eigensystem_jvp,"
            "matrix_2x2_eigensystem_vjp,matrix_2x2_eigensystem_sum_gradient"
        )
    }
    assert rust_module.metadata["rust_backend_verification_primitives"] == {
        identity.key: (
            "verified: scpn_quantum_engine matrix_2x2_eigensystem "
            "value/JVP/VJP/sum-gradient parity"
        )
    }
    assert rust_module.resource_counts["rust_backend_verifications"] == 1
    assert rust_module.metadata["primitive_hard_gaps"][identity.key] == []
    assert identity.key not in rust_module.metadata["primitive_next_hard_gap"]
    assert rust_module.metadata["primitive_readiness"][identity.key] == {
        "adjoint_contract": True,
        "forward_contract": True,
        "jit_backend_contract": True,
        "llvm_backend_contract": True,
        "mlir_runtime_contract": True,
        "native_backend_contract": True,
        "registry_contract": True,
        "reverse_contract": True,
        "rust_backend_contract": True,
        "transform_contract": True,
        "verdict": "native_executable",
    }
    batched_value = vmap(
        lambda row: row,
        primitive_identity=identity,
        registry=rust_registry,
    )(np.vstack([values, values + np.array([0.25, 0.1, 0.05, -0.2], dtype=np.float64)]))
    np.testing.assert_allclose(
        batched_value[0],
        rust_registered_kernel.value(values),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert scpn.make_matrix_2x2_eigensystem_native_llvm_jit_primitive_transform is (
        compiler_mlir.make_matrix_2x2_eigensystem_native_llvm_jit_primitive_transform
    )


def test_differentiable_mlir_rejects_executable_target_claims() -> None:
    """LLVM/JIT target names must fail until backed by a real executable backend."""

    with pytest.raises(ValueError, match="target"):
        DifferentiableMLIRCompileConfig(target="llvm")
    with pytest.raises(ValueError, match="backend"):
        CompilerADExecutableConfig(backend="llvm")
    with pytest.raises(ValueError, match="boolean"):
        DifferentiableMLIRCompileConfig(include_numeric_payload=1)  # type: ignore[arg-type]


def test_whole_program_ad_mlir_exports_trace_and_polyglot_status() -> None:
    """Whole-program AD trace lowering should be deterministic and honest."""

    def objective(values: np.ndarray) -> object:
        if values[0] > 0.0:
            return np.sin(values[0]) + values[1] ** 2
        return values[1]

    result = whole_program_value_and_grad(
        objective,
        np.array([0.25, -0.5], dtype=np.float64),
        parameters=(Parameter("theta"), Parameter("bias")),
    )
    module = compile_whole_program_ad_trace_to_mlir(result, DifferentiableMLIRCompileConfig())
    repeat = compile_whole_program_ad_trace_to_mlir(result, DifferentiableMLIRCompileConfig())

    assert module.text == repeat.text
    assert module.sha256 == repeat.sha256
    assert module.resource_counts["parameters"] == 2
    assert module.resource_counts["trace_events"] == len(result.trace_events)
    assert module.metadata["polyglot_targets"]["llvm"].startswith("blocked")
    assert 'scpn.module = "whole_program_ad"' in module.text
    assert "scpn_diff.trace_event" in module.text
    assert 'execution = "python_whole_program_ad_interchange"' in module.text


def test_whole_program_ad_trace_executable_replays_supported_scalar_ir() -> None:
    """Executable program AD trace kernels should replay gradients fail-closed."""

    def objective(values: np.ndarray) -> object:
        branch = values[0] if values[0] > values[1] else values[1]
        return np.sin(values[0] * values[1]) + np.log(values[2] + 4.0) + branch * values[2]

    values = np.array([1.25, -0.25, 0.5], dtype=np.float64)
    parameters = (Parameter("x"), Parameter("y"), Parameter("z"))

    kernel = compile_whole_program_ad_trace_to_executable(objective, values, parameters)
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        values,
        parameters,
    )
    value, gradient = kernel.value_and_grad(values)

    assert isinstance(kernel, ExecutableWholeProgramADKernel)
    assert kernel.backend == "program_ad_trace_replay"
    assert kernel.parameter_names == ("x", "y", "z")
    assert kernel.parameter_shape == (3,)
    assert kernel.mlir_module.resource_counts["parameters"] == 3
    assert "whole_program_ad" in kernel.mlir_module.text
    assert "branch/signature changes fail closed" in kernel.claim_boundary
    assert kernel.mlir_module.metadata["polyglot_targets"]["llvm"].startswith("blocked")
    assert value == pytest.approx(reference_value)
    assert kernel.value(values) == pytest.approx(reference_value)
    np.testing.assert_allclose(gradient, reference_gradient, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        kernel.gradient(values),
        reference_gradient,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    with pytest.raises(ValueError, match="branch signature"):
        kernel.value_and_grad(np.array([-0.25, 1.25, 0.5], dtype=np.float64))
    with pytest.raises(ValueError, match="one-dimensional"):
        kernel.value_and_grad(np.array([[1.25, -0.25, 0.5]], dtype=np.float64))
    with pytest.raises(ValueError, match="finite"):
        kernel.value_and_grad(np.array([1.25, np.nan, 0.5], dtype=np.float64))


def test_whole_program_ad_trace_executable_batches_same_branch_rows() -> None:
    """Executable program AD trace kernels should batch same-signature rows."""

    def objective(values: np.ndarray) -> object:
        branch = values[0] if values[0] > values[1] else values[1]
        return np.sin(values[0] * values[1]) + np.log(values[2] + 4.0) + branch * values[2]

    sample = np.array([1.25, -0.25, 0.5], dtype=np.float64)
    batch = np.array(
        [
            [1.25, -0.25, 0.5],
            [1.1, -0.4, 0.75],
            [2.0, 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    parameters = (Parameter("x"), Parameter("y"), Parameter("z"))
    kernel = compile_whole_program_ad_trace_to_executable(objective, sample, parameters)

    result = kernel.batch_value_and_grad(batch)
    reference = [program_adjoint_value_and_grad(objective, row, parameters) for row in batch]

    assert isinstance(result, ExecutableWholeProgramADBatchResult)
    assert result.backend == "program_ad_trace_replay"
    assert result.parameter_names == ("x", "y", "z")
    assert result.mlir_sha256 == kernel.mlir_module.sha256
    assert result.row_signatures == (kernel.branch_signature,) * batch.shape[0]
    np.testing.assert_allclose(
        result.values,
        np.array([item[0] for item in reference], dtype=np.float64),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.gradients,
        np.vstack([item[1] for item in reference]),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(kernel.batch_value(batch), result.values)
    np.testing.assert_allclose(kernel.batch_gradient(batch), result.gradients)

    branch_drift_batch = batch.copy()
    branch_drift_batch[1] = np.array([-0.25, 1.25, 0.5], dtype=np.float64)
    with pytest.raises(ValueError, match="branch signature"):
        kernel.batch_value_and_grad(branch_drift_batch)
    with pytest.raises(ValueError, match="two-dimensional"):
        kernel.batch_value_and_grad(sample)
    with pytest.raises(ValueError, match="at least one row"):
        kernel.batch_value_and_grad(np.empty((0, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="parameter shape"):
        kernel.batch_value_and_grad(np.ones((2, 2), dtype=np.float64))
    with pytest.raises(ValueError, match="finite"):
        kernel.batch_value_and_grad(
            np.array([[1.25, -0.25, 0.5], [1.0, np.nan, 0.5]], dtype=np.float64)
        )


def test_whole_program_ad_trace_native_llvm_jit_executes_branchless_scalar_ir() -> None:
    """Native whole-program AD lowering should execute supported branchless traces."""

    def objective(values: np.ndarray) -> object:
        return (
            np.sin(values[0] * values[1])
            + np.log(values[2] + 4.0)
            + values[0] ** 2
            - values[1] / 2.0
        )

    sample = np.array([1.25, -0.25, 0.5], dtype=np.float64)
    batch = np.array(
        [
            [1.25, -0.25, 0.5],
            [1.1, -0.4, 0.75],
            [2.0, 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    tangent = np.array([0.5, -1.0, 2.0], dtype=np.float64)
    parameters = (Parameter("x"), Parameter("y"), Parameter("z"))

    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        sample,
        parameters,
    )
    value, gradient = kernel.value_and_grad(sample)
    batch_reference = [program_adjoint_value_and_grad(objective, row, parameters) for row in batch]
    batch_result = kernel.batch_value_and_grad(batch)

    assert isinstance(kernel, NativeWholeProgramADKernel)
    assert kernel.backend == "native_llvm_jit"
    assert callable(kernel.native_functions["batch_value_gradient"])
    assert callable(kernel.native_functions["batch_jvp"])
    assert callable(kernel.native_functions["batch_vjp"])
    assert kernel.verification.passed
    assert kernel.verification.gradient_close is True
    assert kernel.verification.jvp_close is True
    assert kernel.verification.vjp_close is True
    assert kernel.mlir_module.resource_counts["native_whole_program_kernels"] == 1
    assert kernel.mlir_module.resource_counts["native_whole_program_batch_kernels"] == 1
    assert kernel.mlir_module.resource_counts["native_whole_program_batch_transform_kernels"] == 2
    assert kernel.mlir_module.resource_counts["native_lowerable_ops"] == len(
        kernel.lowering_report.lowerable_ops
    )
    assert kernel.mlir_module.resource_counts["native_unsupported_ops"] == 0
    assert kernel.mlir_module.metadata["native_backend"] == "native_llvm_jit"
    assert kernel.mlir_module.metadata["native_lowering_report"]["supported"] is True
    assert kernel.mlir_module.metadata["native_lowering_report"]["unsupported_ops"] == ()
    assert isinstance(kernel.lowering_report, WholeProgramADNativeLoweringReport)
    assert kernel.lowering_report.supported is True
    assert kernel.lowering_report.unsupported_ops == ()
    assert kernel.lowering_report.operation_count == len(kernel.source_result.ir_nodes)
    assert set(kernel.lowering_report.lowerable_ops) >= {
        "parameter",
        "mul",
        "sin",
        "log",
        "pow",
        "sub",
    }
    assert kernel.mlir_module.metadata["polyglot_targets"]["llvm"].startswith("available")
    assert "scpn_diff.native_llvm_jit" in kernel.mlir_module.text
    assert "_batch_value_gradient" in kernel.llvm_ir
    assert "_batch_jvp" in kernel.llvm_ir
    assert "_batch_vjp" in kernel.llvm_ir
    assert "native LLVM/JIT execution" in kernel.claim_boundary
    assert value == pytest.approx(reference_value)
    assert kernel.value(sample) == pytest.approx(reference_value)
    np.testing.assert_allclose(gradient, reference_gradient, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(
        kernel.gradient(sample),
        reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    assert kernel.jvp(sample, tangent) == pytest.approx(float(np.dot(reference_gradient, tangent)))
    np.testing.assert_allclose(
        kernel.vjp(sample, np.array([2.0], dtype=np.float64)),
        2.0 * reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        batch_result.values,
        np.array([item[0] for item in batch_reference], dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        batch_result.gradients,
        np.vstack([item[1] for item in batch_reference]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    assert "compiled batched native LLVM/JIT" in batch_result.claim_boundary
    batch_tangents = np.vstack([tangent, 0.5 * tangent, -tangent])
    batch_cotangents = np.array([1.0, 2.0, -0.5], dtype=np.float64)
    np.testing.assert_allclose(
        kernel.batch_jvp(batch, batch_tangents),
        np.array(
            [float(np.dot(item[1], row)) for item, row in zip(batch_reference, batch_tangents)],
            dtype=np.float64,
        ),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        kernel.batch_vjp(batch, batch_cotangents),
        np.vstack([scale * item[1] for scale, item in zip(batch_cotangents, batch_reference)]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(kernel.batch_value(batch), batch_result.values)
    np.testing.assert_allclose(kernel.batch_gradient(batch), batch_result.gradients)

    with pytest.raises(ValueError, match="tangent shape"):
        kernel.jvp(sample, np.ones(2, dtype=np.float64))
    with pytest.raises(ValueError, match="cotangent"):
        kernel.vjp(sample, np.ones(2, dtype=np.float64))
    with pytest.raises(ValueError, match="two-dimensional"):
        kernel.batch_value_and_grad(sample)
    with pytest.raises(ValueError, match="parameter shape"):
        kernel.batch_value_and_grad(np.ones((2, 2), dtype=np.float64))


def test_whole_program_ad_trace_native_llvm_jit_reuses_verified_cache() -> None:
    """Native program AD should reuse verified compile artefacts deterministically."""

    clear_native_whole_program_ad_compile_cache()
    assert native_whole_program_ad_compile_cache_stats()["entries"] == 0

    def objective(values: np.ndarray) -> object:
        return np.log1p(values[0] * values[0]) + np.tanh(values[1]) + values[2] ** 3

    sample = np.array([0.125, -0.375, 0.75], dtype=np.float64)
    replay = np.array([0.2, -0.25, 0.5], dtype=np.float64)
    parameters = (Parameter("cache_x"), Parameter("cache_y"), Parameter("cache_z"))

    first = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    first_stats = native_whole_program_ad_compile_cache_stats()
    second = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    second_stats = native_whole_program_ad_compile_cache_stats()
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        replay,
        parameters,
    )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.cache_key == second.cache_key
    assert first.mlir_module.metadata["native_compile_cache_key"] == first.cache_key
    assert second.mlir_module.metadata["native_compile_cache_key"] == second.cache_key
    assert first.mlir_module.metadata["native_compile_cache_hit"] is False
    assert second.mlir_module.metadata["native_compile_cache_hit"] is True
    assert first_stats["entries"] == 1
    assert first_stats["max_size"] >= 1
    assert first_stats["keys"] == (first.cache_key,)
    assert second_stats["entries"] == 1
    assert second_stats["keys"] == (first.cache_key,)
    assert first.mlir_module.resource_counts["native_compile_cache_hit"] == 0
    assert second.mlir_module.resource_counts["native_compile_cache_hit"] == 1
    assert first.native_functions["engine"] is second.native_functions["engine"]
    assert first.native_functions["value"] is second.native_functions["value"]
    assert first.native_functions["batch_jvp"] is second.native_functions["batch_jvp"]

    value, gradient = second.value_and_grad(replay)
    assert value == pytest.approx(reference_value)
    np.testing.assert_allclose(gradient, reference_gradient, rtol=1.0e-10, atol=1.0e-10)

    assert clear_native_whole_program_ad_compile_cache() == 1
    assert native_whole_program_ad_compile_cache_stats()["entries"] == 0
    third = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    assert third.cache_hit is False
    assert third.cache_key == first.cache_key
    assert third.native_functions["engine"] is not first.native_functions["engine"]
    assert clear_native_whole_program_ad_compile_cache() == 1


def test_whole_program_ad_native_lowering_report_blocks_unsupported_ops() -> None:
    """Native program AD lowering should report replay-supported ops that lack LLVM lowering."""

    def objective(values: np.ndarray) -> object:
        matrix = np.diag(values[:17])
        return np.linalg.det(matrix) + np.sin(values[17])

    sample = np.linspace(1.1, 1.7, 18, dtype=np.float64)
    parameters = tuple(Parameter(f"x{index}") for index in range(18))

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)

    assert isinstance(report, WholeProgramADNativeLoweringReport)
    assert report.supported is False
    assert report.unsupported_ops == ("linalg:det:17x17",)
    assert report.unsupported_operation_count == 1
    assert report.lowerable_operation_count == len(result.ir_nodes) - 1
    assert "unsupported native ops: linalg:det:17x17" in report.fail_closed_reason
    assert report.as_metadata()["unsupported_ops"] == report.unsupported_ops

    with pytest.raises(ValueError, match="unsupported native ops: linalg:det:17x17"):
        compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)


def test_whole_program_ad_trace_native_llvm_jit_lowers_wide_determinants() -> None:
    """Native program AD should lower helper-backed 6x6 through 16x16 determinants."""

    for size in range(6, 17):

        def objective(
            values: np.ndarray,
            *,
            matrix_size: int = size,
        ) -> object:
            matrix = np.diag(values[:matrix_size])
            return (
                np.linalg.det(matrix)
                + 0.01 * values[matrix_size] * values[0]
                - np.sin(values[matrix_size - 1])
            )

        sample = np.linspace(1.1, 1.6, size + 1, dtype=np.float64)
        replay = np.linspace(1.7, 1.2, size + 1, dtype=np.float64)
        parameters = tuple(Parameter(f"x{index}") for index in range(size + 1))

        result = whole_program_value_and_grad(objective, sample, parameters)
        report = analyse_whole_program_ad_native_lowering(result)
        kernel = compile_whole_program_ad_trace_to_native_llvm_jit(
            objective,
            sample,
            parameters,
        )
        reference_value, reference_gradient = program_adjoint_value_and_grad(
            objective,
            replay,
            parameters,
        )
        det_op = f"linalg:det:{size}x{size}"

        assert report.supported is True
        assert report.unsupported_ops == ()
        assert det_op in report.lowerable_ops
        assert det_op in kernel.supported_ops
        assert f"scpn_det{size}_fl_value_partials" in kernel.llvm_ir
        assert f"%det{size}_helper_matrix_" in kernel.llvm_ir
        assert kernel.mlir_module.metadata["native_lowering_report"]["supported"] is True
        assert kernel.mlir_module.metadata["native_lowering_report"]["unsupported_ops"] == ()
        assert kernel.value(replay) == pytest.approx(reference_value, rel=1.0e-9, abs=1.0e-9)
        np.testing.assert_allclose(
            kernel.gradient(replay),
            reference_gradient,
            rtol=1.0e-8,
            atol=1.0e-8,
        )
        batch = np.vstack(
            [
                sample,
                replay,
                np.linspace(1.25, 1.75, size + 1, dtype=np.float64),
            ]
        )
        batch_reference = [
            program_adjoint_value_and_grad(objective, row, parameters) for row in batch
        ]
        batch_result = kernel.batch_value_and_grad(batch)
        np.testing.assert_allclose(
            batch_result.values,
            np.array([item[0] for item in batch_reference], dtype=np.float64),
            rtol=1.0e-8,
            atol=1.0e-8,
        )
        np.testing.assert_allclose(
            batch_result.gradients,
            np.vstack([item[1] for item in batch_reference]),
            rtol=1.0e-8,
            atol=1.0e-8,
        )


def test_whole_program_ad_native_linalg_support_contract_reports_dense_det_boundary() -> None:
    """Native linalg support contracts should expose exact fail-closed determinant limits."""

    support = native_whole_program_ad_linalg_support()

    assert scpn.native_whole_program_ad_linalg_support is native_whole_program_ad_linalg_support
    assert compiler_mlir.native_whole_program_ad_linalg_support is (
        native_whole_program_ad_linalg_support
    )
    assert support["determinant_expression_sizes"] == (2, 3, 4, 5)
    assert support["determinant_helper_sizes"] == tuple(range(6, 17))
    assert support["determinant_static_dense_sizes"] == tuple(range(2, 17))
    assert support["determinant_fail_closed_from"] == 17
    assert support["determinant_derivative"] == "exact_forward_partials"
    assert support["determinant_policy"] == "static_dense_native_or_fail_closed"
    assert support["unsupported_policy"] == "fail_closed_report_before_compile"


def test_whole_program_ad_trace_native_llvm_jit_lowers_dense_wide_determinants() -> None:
    """Native wide determinant helpers should match replay AD on non-diagonal matrices."""

    for size in (7, 9, 11, 13, 15, 16):
        offsets = _dense_determinant_offsets(size)

        def objective(
            values: np.ndarray,
            *,
            matrix_size: int = size,
            dense_offsets: np.ndarray = offsets,
        ) -> object:
            matrix = np.diag(values[:matrix_size]) + dense_offsets
            return np.linalg.det(matrix) + 0.005 * values[0] * values[matrix_size - 1]

        sample = np.linspace(1.25, 1.75, size, dtype=np.float64)
        replay = np.linspace(1.8, 1.3, size, dtype=np.float64)
        parameters = tuple(Parameter(f"dense_x{index}") for index in range(size))
        det_op = f"linalg:det:{size}x{size}"

        result = whole_program_value_and_grad(objective, sample, parameters)
        report = analyse_whole_program_ad_native_lowering(result)
        kernel = compile_whole_program_ad_trace_to_native_llvm_jit(
            objective,
            sample,
            parameters,
        )
        reference_value, reference_gradient = program_adjoint_value_and_grad(
            objective,
            replay,
            parameters,
        )

        assert report.supported is True
        assert det_op in report.lowerable_ops
        assert report.unsupported_ops == ()
        assert f"scpn_det{size}_fl_value_partials" in kernel.llvm_ir
        assert kernel.value(replay) == pytest.approx(reference_value, rel=1.0e-9, abs=1.0e-9)
        np.testing.assert_allclose(
            kernel.gradient(replay),
            reference_gradient,
            rtol=1.0e-8,
            atol=1.0e-8,
        )


def test_whole_program_ad_trace_native_llvm_jit_lowers_5x5_determinant() -> None:
    """Native program AD should lower scalar 5x5 determinant nodes."""

    def objective(values: np.ndarray) -> object:
        matrix = values[0:25].reshape((5, 5))
        return np.linalg.det(matrix) + 0.0625 * values[25] * values[0] - np.cos(values[24])

    sample = np.array(
        [
            1.2,
            0.1,
            -0.2,
            0.3,
            0.0,
            0.2,
            1.4,
            0.5,
            -0.1,
            0.3,
            0.2,
            -0.3,
            1.1,
            0.4,
            -0.2,
            -0.1,
            0.2,
            0.3,
            1.3,
            0.4,
            0.1,
            -0.2,
            0.5,
            -0.3,
            1.5,
            0.75,
        ],
        dtype=np.float64,
    )
    replay = np.array(
        [
            1.5,
            -0.2,
            0.4,
            -0.1,
            0.2,
            0.3,
            1.1,
            -0.5,
            0.2,
            -0.4,
            -0.4,
            0.6,
            1.6,
            -0.3,
            0.5,
            0.2,
            -0.1,
            0.5,
            1.25,
            -0.2,
            0.4,
            0.1,
            -0.3,
            0.6,
            1.4,
            -0.4,
        ],
        dtype=np.float64,
    )
    parameters = tuple(Parameter(f"a{index}") for index in range(25)) + (Parameter("scale"),)

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        replay,
        parameters,
    )

    assert report.supported is True
    assert report.unsupported_ops == ()
    assert "linalg:det:5x5" in report.lowerable_ops
    assert "linalg:det:5x5" in kernel.supported_ops
    assert "det5_value" in kernel.llvm_ir
    assert "det5_cofactor_44" in kernel.llvm_ir
    assert kernel.mlir_module.metadata["native_lowering_report"]["supported"] is True
    assert kernel.mlir_module.metadata["native_lowering_report"]["unsupported_ops"] == ()
    assert kernel.value(replay) == pytest.approx(reference_value)
    np.testing.assert_allclose(
        kernel.gradient(replay),
        reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    batch = np.array(
        [
            sample,
            replay,
            [
                1.1,
                0.2,
                -0.3,
                0.4,
                -0.2,
                -0.2,
                1.3,
                0.1,
                -0.5,
                0.3,
                0.3,
                -0.4,
                1.7,
                0.2,
                -0.1,
                0.5,
                -0.1,
                0.2,
                1.2,
                0.4,
                -0.3,
                0.6,
                0.1,
                -0.2,
                1.6,
                0.25,
            ],
        ],
        dtype=np.float64,
    )
    batch_reference = [program_adjoint_value_and_grad(objective, row, parameters) for row in batch]
    batch_result = kernel.batch_value_and_grad(batch)
    np.testing.assert_allclose(
        batch_result.values,
        np.array([item[0] for item in batch_reference], dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        batch_result.gradients,
        np.vstack([item[1] for item in batch_reference]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_whole_program_ad_trace_native_llvm_jit_lowers_4x4_determinant() -> None:
    """Native program AD should lower scalar 4x4 determinant nodes."""

    def objective(values: np.ndarray) -> object:
        matrix = values[0:16].reshape((4, 4))
        return np.linalg.det(matrix) + 0.125 * values[16] * values[0] - np.sin(values[15])

    sample = np.array(
        [
            1.2,
            0.1,
            -0.2,
            0.3,
            0.0,
            1.4,
            0.5,
            -0.1,
            0.2,
            -0.3,
            1.1,
            0.4,
            -0.1,
            0.2,
            0.3,
            1.3,
            0.75,
        ],
        dtype=np.float64,
    )
    replay = np.array(
        [
            1.5,
            -0.2,
            0.4,
            -0.1,
            0.3,
            1.1,
            -0.5,
            0.2,
            -0.4,
            0.6,
            1.6,
            -0.3,
            0.2,
            -0.1,
            0.5,
            1.25,
            -0.4,
        ],
        dtype=np.float64,
    )
    parameters = tuple(Parameter(f"a{index}") for index in range(16)) + (Parameter("scale"),)

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        replay,
        parameters,
    )

    assert report.supported is True
    assert report.unsupported_ops == ()
    assert "linalg:det:4x4" in report.lowerable_ops
    assert "linalg:det:4x4" in kernel.supported_ops
    assert "det4_value" in kernel.llvm_ir
    assert "det4_cofactor_33" in kernel.llvm_ir
    assert kernel.mlir_module.metadata["native_lowering_report"]["supported"] is True
    assert kernel.mlir_module.metadata["native_lowering_report"]["unsupported_ops"] == ()
    assert kernel.value(replay) == pytest.approx(reference_value)
    np.testing.assert_allclose(
        kernel.gradient(replay),
        reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    batch = np.array(
        [
            sample,
            replay,
            [
                1.1,
                0.2,
                -0.3,
                0.4,
                -0.2,
                1.3,
                0.1,
                -0.5,
                0.3,
                -0.4,
                1.7,
                0.2,
                0.5,
                -0.1,
                0.2,
                1.2,
                0.25,
            ],
        ],
        dtype=np.float64,
    )
    batch_reference = [program_adjoint_value_and_grad(objective, row, parameters) for row in batch]
    batch_result = kernel.batch_value_and_grad(batch)
    np.testing.assert_allclose(
        batch_result.values,
        np.array([item[0] for item in batch_reference], dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        batch_result.gradients,
        np.vstack([item[1] for item in batch_reference]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_whole_program_ad_trace_native_llvm_jit_lowers_3x3_determinant() -> None:
    """Native program AD should lower scalar 3x3 determinant nodes."""

    def objective(values: np.ndarray) -> object:
        matrix = values[0:9].reshape((3, 3))
        return np.linalg.det(matrix) + 0.25 * values[9] * values[0] - np.cos(values[8])

    sample = np.array(
        [1.2, 0.1, -0.2, 0.3, 1.5, 0.4, -0.1, 0.2, 1.1, 0.75],
        dtype=np.float64,
    )
    replay = np.array(
        [1.0, -0.3, 0.4, 0.2, 1.25, -0.5, 0.6, 0.1, 1.4, -0.2],
        dtype=np.float64,
    )
    parameters = (
        Parameter("a00"),
        Parameter("a01"),
        Parameter("a02"),
        Parameter("a10"),
        Parameter("a11"),
        Parameter("a12"),
        Parameter("a20"),
        Parameter("a21"),
        Parameter("a22"),
        Parameter("scale"),
    )

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        replay,
        parameters,
    )

    assert report.supported is True
    assert report.unsupported_ops == ()
    assert "linalg:det:3x3" in report.lowerable_ops
    assert "linalg:det:3x3" in kernel.supported_ops
    assert "det3_cofactor_00" in kernel.llvm_ir
    assert "det3_cofactor_22" in kernel.llvm_ir
    assert kernel.mlir_module.metadata["native_lowering_report"]["supported"] is True
    assert kernel.mlir_module.metadata["native_lowering_report"]["unsupported_ops"] == ()
    assert kernel.value(replay) == pytest.approx(reference_value)
    np.testing.assert_allclose(
        kernel.gradient(replay),
        reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    batch = np.array(
        [
            sample,
            replay,
            [1.4, 0.2, -0.1, -0.3, 1.1, 0.5, 0.4, -0.2, 1.6, 0.3],
        ],
        dtype=np.float64,
    )
    batch_reference = [program_adjoint_value_and_grad(objective, row, parameters) for row in batch]
    batch_result = kernel.batch_value_and_grad(batch)
    np.testing.assert_allclose(
        batch_result.values,
        np.array([item[0] for item in batch_reference], dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        batch_result.gradients,
        np.vstack([item[1] for item in batch_reference]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_whole_program_ad_trace_native_llvm_jit_lowers_2x2_inverse_and_solve() -> None:
    """Native program AD should lower scalar 2x2 inverse and solve nodes."""

    def objective(values: np.ndarray) -> object:
        matrix = values[0:4].reshape((2, 2))
        rhs = values[4:6]
        return (
            np.linalg.inv(matrix).sum()
            + 0.5 * np.linalg.solve(matrix, rhs).sum()
            + values[6] * values[0]
            - np.cos(values[5])
        )

    sample = np.array([1.0, 0.2, 0.3, 1.5, 0.4, -0.2, 0.75], dtype=np.float64)
    replay = np.array([1.5, -0.4, 0.6, 2.0, -0.25, 0.7, -0.5], dtype=np.float64)
    singular = np.array([1.0, 2.0, 0.5, 1.0, 0.4, -0.2, 0.75], dtype=np.float64)
    parameters = (
        Parameter("a00"),
        Parameter("a01"),
        Parameter("a10"),
        Parameter("a11"),
        Parameter("rhs0"),
        Parameter("rhs1"),
        Parameter("scale"),
    )

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        replay,
        parameters,
    )

    assert report.supported is True
    assert report.unsupported_ops == ()
    assert {
        "linalg:inv:2x2:0:0",
        "linalg:inv:2x2:0:1",
        "linalg:inv:2x2:1:0",
        "linalg:inv:2x2:1:1",
        "linalg:solve:2x2:rhs:2:0",
        "linalg:solve:2x2:rhs:2:1",
    }.issubset(report.lowerable_ops)
    assert "inv2_det" in kernel.llvm_ir
    assert "solve2_det" in kernel.llvm_ir
    assert kernel.mlir_module.metadata["native_lowering_report"]["supported"] is True
    assert kernel.mlir_module.metadata["native_lowering_report"]["unsupported_ops"] == ()
    assert kernel.value(replay) == pytest.approx(reference_value)
    np.testing.assert_allclose(
        kernel.gradient(replay),
        reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    batch = np.array(
        [
            sample,
            replay,
            [2.0, 0.1, -0.2, 1.25, 0.5, -0.35, 0.2],
        ],
        dtype=np.float64,
    )
    batch_reference = [program_adjoint_value_and_grad(objective, row, parameters) for row in batch]
    batch_result = kernel.batch_value_and_grad(batch)
    np.testing.assert_allclose(
        batch_result.values,
        np.array([item[0] for item in batch_reference], dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        batch_result.gradients,
        np.vstack([item[1] for item in batch_reference]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        kernel.gradient(singular)


def test_whole_program_ad_trace_native_llvm_jit_lowers_2x2_product_linalg_ops() -> None:
    """Native program AD should lower scalar 2x2 matrix_power and multi_dot nodes."""

    def objective(values: np.ndarray) -> object:
        left = values[0:4].reshape((2, 2))
        right = values[4:8].reshape((2, 2))
        return (
            np.linalg.matrix_power(left, 2).sum()
            + 0.5 * np.linalg.multi_dot((left, right)).sum()
            + values[8] * values[0]
            - np.sin(values[7])
        )

    sample = np.array([1.0, 0.2, 0.3, 1.5, 0.4, -0.2, 0.6, 0.9, 0.75], dtype=np.float64)
    replay = np.array([1.5, -0.4, 0.6, 2.0, -0.25, 0.7, -0.5, 1.25, -0.2], dtype=np.float64)
    parameters = (
        Parameter("left00"),
        Parameter("left01"),
        Parameter("left10"),
        Parameter("left11"),
        Parameter("right00"),
        Parameter("right01"),
        Parameter("right10"),
        Parameter("right11"),
        Parameter("scale"),
    )

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        replay,
        parameters,
    )

    assert report.supported is True
    assert report.unsupported_ops == ()
    assert {
        "linalg:matrix_power:2x2:power:2:0:0",
        "linalg:matrix_power:2x2:power:2:0:1",
        "linalg:matrix_power:2x2:power:2:1:0",
        "linalg:matrix_power:2x2:power:2:1:1",
        "linalg:multi_dot:2x2__2x2:out:2x2:0",
        "linalg:multi_dot:2x2__2x2:out:2x2:1",
        "linalg:multi_dot:2x2__2x2:out:2x2:2",
        "linalg:multi_dot:2x2__2x2:out:2x2:3",
    }.issubset(report.lowerable_ops)
    assert "matrix_power2_first" in kernel.llvm_ir
    assert "multi_dot2_first" in kernel.llvm_ir
    assert kernel.mlir_module.metadata["native_lowering_report"]["supported"] is True
    assert kernel.mlir_module.metadata["native_lowering_report"]["unsupported_ops"] == ()
    assert kernel.value(replay) == pytest.approx(reference_value)
    np.testing.assert_allclose(
        kernel.gradient(replay),
        reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    batch = np.array(
        [
            sample,
            replay,
            [2.0, 0.1, -0.2, 1.25, 0.5, -0.35, 0.2, 1.1, 0.4],
        ],
        dtype=np.float64,
    )
    batch_reference = [program_adjoint_value_and_grad(objective, row, parameters) for row in batch]
    batch_result = kernel.batch_value_and_grad(batch)
    np.testing.assert_allclose(
        batch_result.values,
        np.array([item[0] for item in batch_reference], dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        batch_result.gradients,
        np.vstack([item[1] for item in batch_reference]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_whole_program_ad_trace_native_llvm_jit_lowers_2x2_linalg_scalar_ops() -> None:
    """Native program AD should lower scalar 2x2 det and trace linalg nodes."""

    def objective(values: np.ndarray) -> object:
        matrix = values[0:4].reshape((2, 2))
        return (
            np.linalg.det(matrix)
            + 0.25 * np.trace(matrix)
            + values[4] * values[0]
            - np.sin(values[3])
        )

    sample = np.array([1.0, 0.2, 0.3, 1.5, 0.75], dtype=np.float64)
    replay = np.array([1.5, -0.4, 0.6, 2.0, -0.25], dtype=np.float64)
    parameters = (
        Parameter("a00"),
        Parameter("a01"),
        Parameter("a10"),
        Parameter("a11"),
        Parameter("scale"),
    )

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        replay,
        parameters,
    )

    assert report.supported is True
    assert report.unsupported_ops == ()
    assert "linalg:det:2x2" in report.lowerable_ops
    assert "linalg:trace:2x2:offset:0" in report.lowerable_ops
    assert "linalg:det:2x2" in kernel.supported_ops
    assert "linalg:trace:2x2:offset:0" in kernel.supported_ops
    assert kernel.mlir_module.metadata["native_lowering_report"]["supported"] is True
    assert kernel.mlir_module.metadata["native_lowering_report"]["unsupported_ops"] == ()
    assert "det2_diag" in kernel.llvm_ir
    assert "det2_offdiag" in kernel.llvm_ir
    assert kernel.value(replay) == pytest.approx(reference_value)
    np.testing.assert_allclose(
        kernel.gradient(replay),
        reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    batch = np.array(
        [
            sample,
            replay,
            [2.0, 0.1, -0.2, 1.25, 0.5],
        ],
        dtype=np.float64,
    )
    batch_reference = [program_adjoint_value_and_grad(objective, row, parameters) for row in batch]
    batch_result = kernel.batch_value_and_grad(batch)
    np.testing.assert_allclose(
        batch_result.values,
        np.array([item[0] for item in batch_reference], dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        batch_result.gradients,
        np.vstack([item[1] for item in batch_reference]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_whole_program_ad_trace_native_llvm_jit_lowers_static_trace_ops() -> None:
    """Native program AD should lower static square and rectangular trace nodes."""

    def objective(values: np.ndarray) -> object:
        rectangle = values[0:20].reshape((4, 5))
        square = values[20:45].reshape((5, 5))
        return (
            np.trace(rectangle, offset=1)
            + 0.5 * np.trace(rectangle, offset=-1)
            + 0.25 * np.trace(square)
            + values[45] * values[0]
            - np.cos(values[44])
        )

    sample = np.linspace(-0.6, 0.9, 46, dtype=np.float64)
    replay = np.linspace(0.8, -0.7, 46, dtype=np.float64)
    parameters = tuple(Parameter(f"x{index}") for index in range(46))

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        replay,
        parameters,
    )

    assert report.supported is True
    assert report.unsupported_ops == ()
    assert {
        "linalg:trace:4x5:offset:1",
        "linalg:trace:4x5:offset:-1",
        "linalg:trace:5x5:offset:0",
    }.issubset(report.lowerable_ops)
    assert {
        "linalg:trace:4x5:offset:1",
        "linalg:trace:4x5:offset:-1",
        "linalg:trace:5x5:offset:0",
    }.issubset(kernel.supported_ops)
    assert "trace_" in kernel.llvm_ir
    assert kernel.mlir_module.metadata["native_lowering_report"]["supported"] is True
    assert kernel.mlir_module.metadata["native_lowering_report"]["unsupported_ops"] == ()
    assert kernel.value(replay) == pytest.approx(reference_value)
    np.testing.assert_allclose(
        kernel.gradient(replay),
        reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    batch = np.vstack(
        [
            sample,
            replay,
            np.linspace(-0.25, 1.25, 46, dtype=np.float64),
        ]
    )
    batch_reference = [program_adjoint_value_and_grad(objective, row, parameters) for row in batch]
    batch_result = kernel.batch_value_and_grad(batch)
    np.testing.assert_allclose(
        batch_result.values,
        np.array([item[0] for item in batch_reference], dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        batch_result.gradients,
        np.vstack([item[1] for item in batch_reference]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_whole_program_ad_trace_native_llvm_jit_lowers_static_diagonal_ops() -> None:
    """Native program AD should lower static diagonal gather/scatter nodes."""

    def objective(values: np.ndarray) -> object:
        rectangle = values[0:12].reshape((3, 4))
        diagonal = values[12:16]
        flattened = values[16:20]
        return (
            np.diag(rectangle, k=1).sum()
            + 0.5 * np.diag(diagonal, k=-1).sum()
            + 0.25 * np.diagflat(flattened, k=2).sum()
            + values[20] * values[0]
            - np.sin(values[19])
        )

    sample = np.linspace(-0.8, 0.6, 21, dtype=np.float64)
    replay = np.linspace(0.7, -0.9, 21, dtype=np.float64)
    parameters = tuple(Parameter(f"x{index}") for index in range(21))

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        replay,
        parameters,
    )

    expected_ops = {
        "linalg:diag:3x4:offset:1:extract:0",
        "linalg:diag:3x4:offset:1:extract:1",
        "linalg:diag:3x4:offset:1:extract:2",
        "linalg:diag:4:offset:-1:construct:0",
        "linalg:diag:4:offset:-1:construct:1",
        "linalg:diag:4:offset:-1:construct:2",
        "linalg:diag:4:offset:-1:construct:3",
        "linalg:diagflat:4:offset:2:construct:0",
        "linalg:diagflat:4:offset:2:construct:1",
        "linalg:diagflat:4:offset:2:construct:2",
        "linalg:diagflat:4:offset:2:construct:3",
    }
    assert report.supported is True
    assert report.unsupported_ops == ()
    assert expected_ops.issubset(report.lowerable_ops)
    assert expected_ops.issubset(kernel.supported_ops)
    assert "diag_" in kernel.llvm_ir
    assert kernel.mlir_module.metadata["native_lowering_report"]["supported"] is True
    assert kernel.mlir_module.metadata["native_lowering_report"]["unsupported_ops"] == ()
    assert kernel.value(replay) == pytest.approx(reference_value)
    np.testing.assert_allclose(
        kernel.gradient(replay),
        reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    batch = np.vstack(
        [
            sample,
            replay,
            np.linspace(-0.25, 1.25, 21, dtype=np.float64),
        ]
    )
    batch_reference = [program_adjoint_value_and_grad(objective, row, parameters) for row in batch]
    batch_result = kernel.batch_value_and_grad(batch)
    np.testing.assert_allclose(
        batch_result.values,
        np.array([item[0] for item in batch_reference], dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        batch_result.gradients,
        np.vstack([item[1] for item in batch_reference]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_whole_program_ad_trace_native_llvm_jit_lowers_scalar_where() -> None:
    """Native program AD should lower scalar np.where selection traces."""

    def objective(values: np.ndarray) -> object:
        selected = np.where(values[0] > values[1], values[0:1] ** 2, values[1:2] ** 2)
        return selected.sum() + values[2] * values[0]

    sample = np.array([1.25, -0.25, 0.5], dtype=np.float64)
    replay = np.array([-0.5, 1.0, 2.0], dtype=np.float64)
    parameters = (Parameter("left"), Parameter("right"), Parameter("scale"))

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        replay,
        parameters,
    )

    assert report.supported is True
    assert report.unsupported_ops == ()
    assert "where" in report.lowerable_ops
    assert "where" in kernel.supported_ops
    assert kernel.mlir_module.metadata["native_lowering_report"]["supported"] is True
    assert kernel.mlir_module.metadata["native_lowering_report"]["unsupported_ops"] == ()
    assert "select i1" in kernel.llvm_ir
    assert kernel.value(replay) == pytest.approx(reference_value)
    np.testing.assert_allclose(
        kernel.gradient(replay),
        reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    batch = np.array(
        [
            [1.25, -0.25, 0.5],
            [-0.5, 1.0, 2.0],
            [2.0, -1.0, -0.25],
        ],
        dtype=np.float64,
    )
    batch_reference = [program_adjoint_value_and_grad(objective, row, parameters) for row in batch]
    batch_result = kernel.batch_value_and_grad(batch)
    np.testing.assert_allclose(
        batch_result.values,
        np.array([item[0] for item in batch_reference], dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        batch_result.gradients,
        np.vstack([item[1] for item in batch_reference]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    with pytest.raises(ValueError, match="non-differentiable at equality"):
        kernel.gradient(np.array([1.0, 1.0, 0.5], dtype=np.float64))


def test_whole_program_ad_trace_native_llvm_jit_lowers_scalar_clip() -> None:
    """Native program AD should lower strict scalar np.clip selection traces."""

    def objective(values: np.ndarray) -> object:
        clipped = np.clip(values[0:1], values[1:2], values[2:3])
        return clipped.sum() + values[3] * values[0]

    sample = np.array([0.25, -0.5, 1.5, 2.0], dtype=np.float64)
    lower_replay = np.array([-1.0, -0.5, 1.5, 2.0], dtype=np.float64)
    upper_replay = np.array([2.0, -0.5, 1.5, -0.25], dtype=np.float64)
    parameters = (
        Parameter("value"),
        Parameter("lower"),
        Parameter("upper"),
        Parameter("scale"),
    )

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    lower_value, lower_gradient = program_adjoint_value_and_grad(
        objective,
        lower_replay,
        parameters,
    )
    upper_value, upper_gradient = program_adjoint_value_and_grad(
        objective,
        upper_replay,
        parameters,
    )

    assert report.supported is True
    assert report.unsupported_ops == ()
    assert "clip" in report.lowerable_ops
    assert "clip" in kernel.supported_ops
    assert kernel.mlir_module.metadata["native_lowering_report"]["supported"] is True
    assert kernel.mlir_module.metadata["native_lowering_report"]["unsupported_ops"] == ()
    assert "fcmp olt" in kernel.llvm_ir
    assert "fcmp ogt" in kernel.llvm_ir
    assert "select i1" in kernel.llvm_ir
    assert kernel.value(lower_replay) == pytest.approx(lower_value)
    np.testing.assert_allclose(
        kernel.gradient(lower_replay),
        lower_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    assert kernel.value(upper_replay) == pytest.approx(upper_value)
    np.testing.assert_allclose(
        kernel.gradient(upper_replay),
        upper_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    batch = np.vstack([sample, lower_replay, upper_replay])
    batch_reference = [program_adjoint_value_and_grad(objective, row, parameters) for row in batch]
    batch_result = kernel.batch_value_and_grad(batch)
    np.testing.assert_allclose(
        batch_result.values,
        np.array([item[0] for item in batch_reference], dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        batch_result.gradients,
        np.vstack([item[1] for item in batch_reference]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    with pytest.raises(ValueError, match="clipping boundary"):
        kernel.gradient(np.array([-0.5, -0.5, 1.5, 2.0], dtype=np.float64))
    with pytest.raises(ValueError, match="lower bound"):
        kernel.value(np.array([0.0, 2.0, 1.0, 2.0], dtype=np.float64))


def test_whole_program_ad_trace_native_llvm_jit_lowers_strict_selection_ops() -> None:
    """Native program AD should lower strict no-tie maximum/minimum selection ops."""

    def objective(values: np.ndarray) -> object:
        return (
            np.maximum(values[0], values[1])
            + np.minimum(values[2], values[3])
            + values[4] * values[0]
        )

    sample = np.array([1.25, -0.25, 0.5, 1.5, 2.0], dtype=np.float64)
    replay = np.array([-0.5, 1.0, 2.0, -1.0, 0.25], dtype=np.float64)
    parameters = (
        Parameter("left"),
        Parameter("right"),
        Parameter("lower"),
        Parameter("upper"),
        Parameter("scale"),
    )

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        replay,
        parameters,
    )

    assert report.supported is True
    assert report.unsupported_ops == ()
    assert {"maximum", "minimum"}.issubset(report.lowerable_ops)
    assert kernel.lowering_report.lowerable_ops == report.lowerable_ops
    assert kernel.mlir_module.metadata["native_lowering_report"]["supported"] is True
    assert kernel.mlir_module.metadata["native_lowering_report"]["unsupported_ops"] == ()
    assert "maximum" in kernel.supported_ops
    assert "minimum" in kernel.supported_ops
    assert "fcmp ogt" in kernel.llvm_ir
    assert "fcmp olt" in kernel.llvm_ir
    assert "select i1" in kernel.llvm_ir
    assert kernel.value(replay) == pytest.approx(reference_value)
    np.testing.assert_allclose(
        kernel.gradient(replay),
        reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    with pytest.raises(ValueError, match="non-differentiable at equal inputs"):
        kernel.gradient(np.array([1.0, 1.0, 0.5, 1.5, 2.0], dtype=np.float64))
    with pytest.raises(ValueError, match="non-differentiable at equal inputs"):
        kernel.batch_value_and_grad(
            np.array(
                [
                    [1.25, -0.25, 0.5, 1.5, 2.0],
                    [1.0, 1.0, 0.5, 1.5, 2.0],
                ],
                dtype=np.float64,
            )
        )


def test_whole_program_ad_trace_native_llvm_jit_executes_stable_branch_path() -> None:
    """Native LLVM/JIT program AD should execute stable branch traces and reject drift."""

    def objective(values: np.ndarray) -> object:
        if values[0] > values[1]:
            return np.sin(values[0] * values[1]) + values[2]
        return np.cos(values[0] - values[1]) - values[2]

    sample = np.array([1.25, -0.25, 0.5], dtype=np.float64)
    same_branch = np.array(
        [
            [1.25, -0.25, 0.5],
            [1.1, -0.4, 0.75],
            [2.0, 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    parameters = (Parameter("x"), Parameter("y"), Parameter("z"))

    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(
        objective,
        sample,
        parameters,
    )
    ref_value, ref_gradient = program_adjoint_value_and_grad(
        objective,
        sample,
        parameters,
    )
    value, gradient = kernel.value_and_grad(sample)

    assert kernel.backend == "native_llvm_jit"
    assert any(item.startswith("branch:") for item in kernel.supported_ops)
    assert "stable executed branch signatures" in kernel.claim_boundary
    assert "stable executed branch signatures" in kernel.mlir_module.metadata["claim_boundary"]
    assert value == pytest.approx(ref_value)
    assert kernel.value(sample) == pytest.approx(ref_value)
    np.testing.assert_allclose(gradient, ref_gradient, rtol=1e-10, atol=1e-10)

    tangent = np.array([0.25, -0.5, 1.5], dtype=np.float64)
    assert kernel.jvp(sample, tangent) == pytest.approx(float(np.dot(ref_gradient, tangent)))
    np.testing.assert_allclose(
        kernel.vjp(sample, np.array([2.0], dtype=np.float64)),
        2.0 * ref_gradient,
        rtol=1e-10,
        atol=1e-10,
    )

    batch_result = kernel.batch_value_and_grad(same_branch)
    batch_reference = [
        program_adjoint_value_and_grad(objective, row, parameters) for row in same_branch
    ]
    np.testing.assert_allclose(
        batch_result.values,
        np.array([item[0] for item in batch_reference], dtype=np.float64),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        batch_result.gradients,
        np.vstack([item[1] for item in batch_reference]),
        rtol=1e-10,
        atol=1e-10,
    )

    branch_drift = np.array([-0.25, 1.25, 0.5], dtype=np.float64)
    with pytest.raises(ValueError, match="branch signature"):
        kernel.value(branch_drift)
    with pytest.raises(ValueError, match="branch signature"):
        kernel.gradient(branch_drift)
    with pytest.raises(ValueError, match="branch signature"):
        kernel.jvp(branch_drift, tangent)
    with pytest.raises(ValueError, match="branch signature"):
        kernel.vjp(branch_drift, np.array([1.0], dtype=np.float64))

    drift_batch = same_branch.copy()
    drift_batch[1] = branch_drift
    with pytest.raises(ValueError, match="branch signature"):
        kernel.batch_value_and_grad(drift_batch)
    with pytest.raises(ValueError, match="branch signature"):
        kernel.batch_jvp(drift_batch, np.vstack([tangent, tangent, tangent]))
    with pytest.raises(ValueError, match="branch signature"):
        kernel.batch_vjp(drift_batch, np.ones(3, dtype=np.float64))


def test_whole_program_ad_trace_native_llvm_jit_lowers_elementary_ops() -> None:
    """Native LLVM/JIT program AD should lower adjoint-supported scalar primitives."""

    def objective(values: np.ndarray) -> object:
        return (
            np.tan(values[0])
            + np.tanh(values[1])
            + np.expm1(values[0] * values[1])
            + np.log1p(values[2])
            + np.arcsin(values[0])
            - np.arccos(values[1])
            + np.reciprocal(values[3])
            + np.square(values[2])
            + np.abs(values[1])
        )

    sample = np.array([0.2, -0.3, 0.4, 1.5], dtype=np.float64)
    parameters = (
        Parameter("angle"),
        Parameter("offset"),
        Parameter("positive"),
        Parameter("denominator"),
    )
    tangent = np.array([0.5, -1.0, 1.5, -0.25], dtype=np.float64)

    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(
        objective,
        sample,
        parameters,
    )
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        sample,
        parameters,
    )
    value, gradient = kernel.value_and_grad(sample)

    assert kernel.backend == "native_llvm_jit"
    assert kernel.mlir_module.resource_counts["native_supported_elementary_ops"] >= 9
    assert "expanded elementary ops" in kernel.mlir_module.metadata["claim_boundary"]
    for op in (
        "tan",
        "tanh",
        "expm1",
        "log1p",
        "arcsin",
        "arccos",
        "reciprocal",
        "square",
        "abs",
    ):
        assert op in kernel.supported_ops
    assert value == pytest.approx(reference_value)
    np.testing.assert_allclose(gradient, reference_gradient, rtol=1.0e-10, atol=1.0e-10)
    assert kernel.jvp(sample, tangent) == pytest.approx(float(np.dot(reference_gradient, tangent)))
    np.testing.assert_allclose(
        kernel.vjp(sample, np.array([1.75], dtype=np.float64)),
        1.75 * reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )

    abs_boundary = sample.copy()
    abs_boundary[1] = 0.0
    with pytest.raises(ValueError, match="output must be finite"):
        kernel.gradient(abs_boundary)


def test_realtime_control_loop_records_deadline_jitter_and_misses() -> None:
    """Realtime runtime should account for deterministic deadline misses."""

    clock = VirtualRealtimeClock()
    config = RealtimeRuntimeConfig(
        sample_period_s=0.01,
        deadline_s=0.005,
        jitter_budget_s=0.002,
        max_missed_deadlines=2,
    )

    durations = [0.001, 0.006, 0.004]

    def step(index: int):
        clock.advance(durations[index])
        return {"index": float(index)}

    result = run_realtime_control_loop(3, step, config=config, clock=clock)

    assert result.completed
    assert result.missed_deadlines == 1
    assert result.max_latency_s == pytest.approx(0.006)
    assert result.records[1].deadline_missed
    assert result.records[2].jitter_s == pytest.approx(0.0)


def test_realtime_control_loop_fails_when_deadline_budget_is_exceeded() -> None:
    """Realtime runtime should abort instead of hiding repeated deadline misses."""

    clock = VirtualRealtimeClock()
    config = RealtimeRuntimeConfig(
        sample_period_s=0.01,
        deadline_s=0.002,
        max_missed_deadlines=0,
    )

    def step(_index: int):
        clock.advance(0.003)
        return {}

    with pytest.raises(RuntimeError, match="deadline"):
        run_realtime_control_loop(1, step, config=config, clock=clock)


def test_realtime_sla_accepts_sub_millisecond_loop() -> None:
    """SLA gate should pass when all observed latency stays within <1 ms budget."""

    clock = VirtualRealtimeClock()
    config = RealtimeRuntimeConfig(
        sample_period_s=0.0012,
        deadline_s=0.00095,
        jitter_budget_s=0.00015,
        max_missed_deadlines=0,
    )
    durations = [0.00052, 0.00061, 0.00074, 0.00068, 0.00057]

    def step(index: int):
        clock.advance(durations[index])
        return {"index": float(index)}

    result = run_realtime_control_loop(len(durations), step, config=config, clock=clock)
    sla = RealtimeSLAConfig(
        max_latency_s=0.001,
        max_jitter_s=0.00025,
        p95_latency_s=0.001,
        p99_latency_s=0.001,
        max_deadline_miss_rate=0.0,
    )
    report = evaluate_realtime_sla(result, sla=sla)

    assert report.compliant is True
    assert report.breach_reasons == ()
    assert report.observed_max_latency_s <= 0.001
    assert enforce_realtime_sla(result, sla=sla) == report


def test_realtime_sla_rejects_latency_breach() -> None:
    """SLA gate should fail closed on millisecond-class latency overshoot."""

    clock = VirtualRealtimeClock()
    config = RealtimeRuntimeConfig(
        sample_period_s=0.0014,
        deadline_s=0.0012,
        jitter_budget_s=0.00020,
        max_missed_deadlines=5,
    )
    durations = [0.00055, 0.00145, 0.00066]

    def step(index: int):
        clock.advance(durations[index])
        return {"index": float(index)}

    result = run_realtime_control_loop(len(durations), step, config=config, clock=clock)
    sla = RealtimeSLAConfig(
        max_latency_s=0.001,
        max_jitter_s=0.00030,
        p95_latency_s=0.001,
        p99_latency_s=0.001,
        max_deadline_miss_rate=0.5,
    )
    report = evaluate_realtime_sla(result, sla=sla)

    assert report.compliant is False
    assert any("max latency" in item for item in report.breach_reasons)
    with pytest.raises(RuntimeError, match="max latency"):
        enforce_realtime_sla(result, sla=sla)


def test_cloud_native_manifests_are_deterministic_and_secret_free() -> None:
    """Cloud deployment export should generate usable Kubernetes and Compose specs."""

    spec = CloudDeploymentSpec(
        name="scpn-qc",
        image="registry.example/scpn-quantum-control:0.9.7",
        command=("scpn-bench", "stable-core-contract-gate"),
        replicas=2,
        resources=ContainerResources(cpu="1000m", memory="1Gi"),
        env={"SCPN_EXECUTION_MODE": "offline"},
    )

    bundle = generate_cloud_manifests(spec)

    assert bundle.sha256 == generate_cloud_manifests(spec).sha256
    assert "deployment.yaml" in bundle.files
    assert "service.yaml" in bundle.files
    assert "docker-compose.yaml" in bundle.files
    assert "replicas: 2" in bundle.files["deployment.yaml"]
    assert "readOnlyRootFilesystem: true" in bundle.files["deployment.yaml"]
    assert "stable-core-contract-gate" in bundle.files["docker-compose.yaml"]


def test_cloud_native_manifests_reject_secret_like_env_and_bad_resources() -> None:
    """Deployment specs must not turn credentials into public manifests."""

    with pytest.raises(ValueError, match="secret"):
        CloudDeploymentSpec(
            name="bad",
            image="repo/scpn:latest",
            env={"IBM_TOKEN": "leak"},
        )
    with pytest.raises(ValueError, match="memory"):
        ContainerResources(cpu="500m", memory="1TB")


def test_compiler_realtime_and_deployment_api_exported_from_package_root() -> None:
    """The new production surfaces should be stable package-root imports."""

    import scpn_quantum_control as scpn

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
    assert scpn.RealtimeRuntimeConfig is RealtimeRuntimeConfig
    assert scpn.RealtimeSLAConfig is RealtimeSLAConfig
    assert scpn.run_realtime_control_loop is run_realtime_control_loop
    assert scpn.evaluate_realtime_sla is evaluate_realtime_sla
    assert scpn.enforce_realtime_sla is enforce_realtime_sla
    assert scpn.CloudDeploymentSpec is CloudDeploymentSpec
    assert scpn.generate_cloud_manifests is generate_cloud_manifests
