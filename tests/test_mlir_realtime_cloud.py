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

from scpn_quantum_control.compiler.mlir import (
    CompilerADExecutableConfig,
    CompilerADKernelVerification,
    CompilerADTransformPlan,
    DifferentiableMLIRCompileConfig,
    ExecutableCompilerADKernel,
    MLIRCompileConfig,
    PrimitiveLoweringStatus,
    build_compiler_ad_transform_plan,
    compile_compiler_ad_transform_plan_to_mlir,
    compile_custom_derivative_rule_to_executable,
    compile_custom_derivative_rule_to_mlir,
    compile_kuramoto_to_mlir,
    compile_registered_primitive_to_executable,
    compile_whole_program_ad_trace_to_mlir,
    make_program_ad_linalg_matrix_power_executable_lowering_rule,
    make_program_ad_linalg_multi_dot_executable_lowering_rule,
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
    whole_program_value_and_grad,
)
from scpn_quantum_control.kuramoto_core import build_kuramoto_problem


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
    assert module.resource_counts["executable_backends"] == 0
    assert module.metadata["executable_backend"] == "none"
    assert module.metadata["jvp_rule_primitives"] == ["scpn.quantum:rx_expectation@1"]
    assert module.metadata["vjp_rule_primitives"] == ["scpn.quantum:rx_expectation@1"]
    assert module.metadata["batching_rule_primitives"] == []
    assert module.metadata["registry_contract_primitives"] == []
    assert module.metadata["reverse_contract_primitives"] == []
    assert module.metadata["reverse_incomplete_primitives"] == []
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
    assert module.metadata["uncontracted_primitives"] == ["scpn.quantum:policy_only@1"]
    assert module.resource_counts["boundary_contracts"] == 0
    assert module.resource_counts["registry_contracts"] == 0
    assert module.resource_counts["reverse_contracts"] == 0
    assert module.resource_counts["reverse_incomplete_primitives"] == 1
    assert module.resource_counts["effects"] == 0
    assert module.resource_counts["nondifferentiable_policies"] == 0
    assert module.resource_counts["nondifferentiable_boundaries"] == 0
    assert module.resource_counts["nondifferentiable_boundary_policies"] == 0
    assert module.resource_counts["uncontracted_primitives"] == 1


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
    assert module.resource_counts["batching_rules"] == 2
    assert module.resource_counts["boundary_contracts"] == 2
    assert module.resource_counts["registry_contracts"] == 2
    assert module.resource_counts["reverse_contracts"] == len(expected_reverse_contracts)
    assert module.resource_counts["reverse_incomplete_primitives"] == len(
        expected_reverse_incomplete
    )
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
        "scpn.program_ad.array:getitem@1": "source_shape:ranked_tensor_shape;index:basic_index",
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
    assert 'static_signature = "source_shape:ranked_tensor_shape;index:basic_index"' in module.text
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
    assert module.metadata["mlir_runtime_lowering_primitives"] == [
        "scpn.program_ad.linalg:matrix_power@1"
    ]
    assert module.resource_counts["mlir_runtime_lowerings"] == 1
    assert module.resource_counts["executable_backends"] == 0
    assert module.metadata["executable_backend"] == "none"
    assert "available: executable scpn_diff MLIR-runtime linalg kernel" in module.text
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
    assert scpn.MLIRCompileConfig is MLIRCompileConfig
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
