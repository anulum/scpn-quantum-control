# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- differentiable result contract tests
"""Tests for differentiable result containers and primitive metadata contracts."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control.differentiable import (
    ArmijoLineSearchResult,
    CustomDerivativeCheckResult,
    CustomDerivativeRegistry,
    CustomDerivativeRule,
    FixedPointSensitivityResult,
    GradientCheckResult,
    GradientResult,
    HessianResult,
    HVPResult,
    ImplicitSensitivityResult,
    JacobianResult,
    JVPResult,
    NaturalGradientOptimizationResult,
    NaturalGradientResult,
    OptimizationResult,
    PrimitiveContract,
    PrimitiveIdentity,
    PrimitiveTransformRule,
    ShotAllocationResult,
    SparseMatrixResult,
    StochasticGradientResult,
    VJPResult,
)


def test_differentiable_result_contracts_cover_fail_closed_metadata_boundaries() -> None:
    """Derivative result containers should reject malformed numerical metadata."""

    gradient = GradientResult(
        value=1.0,
        gradient=np.array([1.0, -2.0]),
        method="exact",
        shift=None,
        coefficient=1.0,
        evaluations=1,
        parameter_names=("x", "y"),
        trainable=(True, True),
    )
    other_gradient = GradientResult(
        value=1.1,
        gradient=np.array([1.0, -2.0]),
        method="exact",
        shift=None,
        coefficient=1.0,
        evaluations=1,
        parameter_names=("x", "y"),
        trainable=(True, True),
    )
    jvp_result = JVPResult(
        value=np.array([1.0, 2.0]),
        jvp=np.array([0.5, -0.25]),
        tangent=np.array([0.1, 0.2]),
        method="exact",
        step=0.0,
        evaluations=1,
        parameter_names=("x", "y"),
        trainable=(True, True),
    )
    vjp_result = VJPResult(
        value=np.array([1.0, 2.0]),
        cotangent=np.array([0.5, -0.25]),
        vjp=np.array([0.1, 0.2]),
        method="exact",
        step=0.0,
        evaluations=1,
        parameter_names=("x", "y"),
        trainable=(True, True),
    )
    sparse_result = SparseMatrixResult(
        row_indices=np.array([0, 1]),
        column_indices=np.array([1, 0]),
        values=np.array([2.0, 3.0]),
        shape=(2, 2),
        method="coo",
        parameter_names=("x", "y"),
        trainable=(True, True),
    )
    natural_gradient_result = NaturalGradientResult(
        base_gradient=gradient,
        metric=np.eye(2),
        natural_gradient=np.array([1.0, -2.0]),
        damping=0.0,
        condition_number=1.0,
    )

    valid_cases = [
        StochasticGradientResult(
            value=1.0,
            gradient=np.array([1.0, 2.0]),
            standard_error=np.array([0.1, 0.2]),
            covariance=np.eye(2),
            confidence_radius=np.array([0.2, 0.4]),
            shots=np.array([[16.0, 32.0], [16.0, 32.0]]),
            confidence_level=0.95,
            method="shot_noise_parameter_shift",
            shift=math.pi / 2.0,
            coefficient=0.5,
            evaluations=4,
            parameter_names=("x", "y"),
            trainable=(True, False),
        ),
        ShotAllocationResult(
            shots=np.array([[8.0, 10.0], [8.0, 10.0]]),
            predicted_standard_error=np.array([0.25, 0.2]),
            covariance=np.eye(2),
            target_standard_error=0.3,
            total_shots=36,
            method="variance_weighted",
            parameter_names=("x", "y"),
            trainable=(True, True),
        ),
        OptimizationResult(
            values=np.array([0.0, 1.0]),
            final_gradient=gradient,
            value_history=(3.0, 2.0, 1.0),
            steps=2,
            converged=True,
            reason="gradient_tolerance",
        ),
        ArmijoLineSearchResult(
            values=np.array([0.8, 1.2]),
            value=0.5,
            step_size=0.25,
            direction=np.array([-1.0, 0.5]),
            directional_derivative=-0.75,
            accepted=True,
            evaluations=2,
            value_history=(1.0, 0.5),
            reason="accepted",
            parameter_names=("x", "y"),
            trainable=(True, True),
        ),
        GradientCheckResult(
            reference=gradient,
            candidate=other_gradient,
            max_abs_error=0.0,
            l2_error=0.0,
            value_delta=0.1,
            tolerance=1.0e-6,
            passed=True,
        ),
        CustomDerivativeCheckResult(
            custom_jvp=jvp_result,
            custom_vjp=vjp_result,
            reference_jvp=jvp_result,
            reference_vjp=vjp_result,
            adjoint_inner_error=0.0,
            jvp_l2_error=0.0,
            vjp_l2_error=0.0,
            tolerance=1.0e-9,
            passed=True,
        ),
        JacobianResult(
            value=np.array([1.0, 2.0]),
            jacobian=np.eye(2),
            method="exact",
            step=0.0,
            evaluations=1,
            parameter_names=("x", "y"),
            trainable=(True, True),
        ),
        HessianResult(
            value=1.0,
            hessian=np.eye(2),
            method="exact",
            step=1.0e-3,
            evaluations=4,
            parameter_names=("x", "y"),
            trainable=(True, True),
        ),
        sparse_result,
        HVPResult(
            value=1.0,
            hvp=np.array([1.0, 2.0]),
            tangent=np.array([0.5, -0.5]),
            method="exact",
            step=1.0e-3,
            evaluations=4,
            parameter_names=("x", "y"),
            trainable=(True, True),
        ),
        natural_gradient_result,
        ImplicitSensitivityResult(
            sensitivity=np.eye(2),
            hessian=np.eye(2),
            cross_derivative=np.eye(2),
            damping=0.0,
            condition_number=1.0,
            method="implicit",
            parameter_names=("x", "y"),
            trainable=(True, True),
            hyperparameter_names=("a", "b"),
        ),
        FixedPointSensitivityResult(
            sensitivity=np.eye(2),
            state_jacobian=0.5 * np.eye(2),
            parameter_jacobian=np.eye(2),
            system_matrix=0.5 * np.eye(2),
            damping=0.0,
            condition_number=1.0,
            method="fixed_point",
            parameter_names=("x", "y"),
            trainable=(True, True),
            hyperparameter_names=("a", "b"),
        ),
    ]
    assert len(valid_cases) == 13
    assert sparse_result.nnz == 2
    np.testing.assert_allclose(sparse_result.to_dense(), np.array([[0.0, 2.0], [3.0, 0.0]]))

    invalid_factories: tuple[tuple[Callable[[], object], str], ...] = (
        (
            lambda: StochasticGradientResult(
                value=1.0,
                gradient=np.array([1.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(1),
                confidence_radius=np.array([0.1]),
                shots=np.array([[1.0], [1.0]]),
                confidence_level=0.95,
                method="x",
                shift=1.0,
                coefficient=1.0,
                evaluations=1,
                parameter_names=("x",),
                trainable=(True,),
            ),
            "standard_error shape",
        ),
        (
            lambda: ShotAllocationResult(
                shots=np.array([[1.5], [2.0]]),
                predicted_standard_error=np.array([0.1]),
                covariance=np.eye(1),
                target_standard_error=0.1,
                total_shots=3,
                method="x",
                parameter_names=("x",),
                trainable=(True,),
            ),
            "positive integer",
        ),
        (
            lambda: OptimizationResult(
                values=np.array([1.0]),
                final_gradient=gradient,
                value_history=(1.0,),
                steps=0,
                converged=True,
                reason="ok",
            ),
            "values length",
        ),
        (
            lambda: ArmijoLineSearchResult(
                values=np.array([1.0]),
                value=1.0,
                step_size=-1.0,
                direction=np.array([1.0]),
                directional_derivative=-1.0,
                accepted=True,
                evaluations=1,
                value_history=(1.0,),
                reason="accepted",
                parameter_names=("x",),
                trainable=(True,),
            ),
            "step_size",
        ),
        (
            lambda: GradientCheckResult(
                reference=gradient,
                candidate=GradientResult(
                    value=1.0,
                    gradient=np.array([1.0]),
                    method="exact",
                    shift=None,
                    coefficient=1.0,
                    evaluations=1,
                    parameter_names=("x",),
                    trainable=(True,),
                ),
                max_abs_error=0.0,
                l2_error=0.0,
                value_delta=0.0,
                tolerance=0.0,
                passed=True,
            ),
            "matching shapes",
        ),
        (
            lambda: GradientResult(
                value=1.0,
                gradient=np.array([1.0, 0.5]),
                method="exact",
                shift=None,
                coefficient=1.0,
                evaluations=1,
                parameter_names=("x", "frozen"),
                trainable=(True, False),
            ),
            "gradient must be zero for non-trainable parameters",
        ),
        (
            lambda: CustomDerivativeCheckResult(
                custom_jvp=cast(Any, object()),
                custom_vjp=vjp_result,
                reference_jvp=jvp_result,
                reference_vjp=vjp_result,
                adjoint_inner_error=0.0,
                jvp_l2_error=0.0,
                vjp_l2_error=0.0,
                tolerance=0.0,
                passed=True,
            ),
            "custom_jvp",
        ),
        (
            lambda: JacobianResult(
                value=np.array([[1.0]]),
                jacobian=np.eye(1),
                method="exact",
                step=0.0,
                evaluations=1,
                parameter_names=("x",),
                trainable=(True,),
            ),
            "one-dimensional",
        ),
        (
            lambda: JacobianResult(
                value=np.array([1.0, 2.0]),
                jacobian=np.array([[1.0, 0.5], [2.0, -0.25]]),
                method="exact",
                step=0.0,
                evaluations=1,
                parameter_names=("x", "frozen"),
                trainable=(True, False),
            ),
            "jacobian must be zero for non-trainable parameters",
        ),
        (
            lambda: JVPResult(
                value=np.array([1.0]),
                jvp=np.array([1.0, 2.0]),
                tangent=np.array([1.0]),
                method="exact",
                step=0.0,
                evaluations=1,
                parameter_names=("x",),
                trainable=(True,),
            ),
            "shape must match",
        ),
        (
            lambda: JVPResult(
                value=np.array([1.0]),
                jvp=np.array([1.0]),
                tangent=np.array([1.0, 0.5]),
                method="exact",
                step=0.0,
                evaluations=1,
                parameter_names=("x", "frozen"),
                trainable=(True, False),
            ),
            "JVP tangent must be zero for non-trainable parameters",
        ),
        (
            lambda: VJPResult(
                value=np.array([1.0]),
                cotangent=np.array([1.0, 2.0]),
                vjp=np.array([1.0]),
                method="exact",
                step=0.0,
                evaluations=1,
                parameter_names=("x",),
                trainable=(True,),
            ),
            "cotangent shape",
        ),
        (
            lambda: VJPResult(
                value=np.array([1.0]),
                cotangent=np.array([1.0]),
                vjp=np.array([1.0, 0.25]),
                method="exact",
                step=0.0,
                evaluations=1,
                parameter_names=("x", "frozen"),
                trainable=(True, False),
            ),
            "VJP must be zero for non-trainable parameters",
        ),
        (
            lambda: HessianResult(
                value=1.0,
                hessian=np.array([[1.0, 2.0]]),
                method="exact",
                step=1.0e-3,
                evaluations=1,
                parameter_names=("x",),
                trainable=(True,),
            ),
            "square",
        ),
        (
            lambda: HessianResult(
                value=1.0,
                hessian=np.array([[1.0, 0.5], [0.5, 0.25]]),
                method="exact",
                step=1.0e-3,
                evaluations=1,
                parameter_names=("x", "frozen"),
                trainable=(True, False),
            ),
            "hessian columns must be zero for non-trainable parameters",
        ),
        (
            lambda: SparseMatrixResult(
                row_indices=np.array([0, 0]),
                column_indices=np.array([0, 0]),
                values=np.array([1.0, 2.0]),
                shape=(1, 1),
                method="coo",
                parameter_names=("x",),
                trainable=(True,),
            ),
            "duplicate",
        ),
        (
            lambda: SparseMatrixResult(
                row_indices=np.array([0]),
                column_indices=np.array([1]),
                values=np.array([0.5]),
                shape=(1, 2),
                method="coo",
                parameter_names=("x", "frozen"),
                trainable=(True, False),
            ),
            "sparse values must be zero for non-trainable parameters",
        ),
        (
            lambda: HVPResult(
                value=1.0,
                hvp=np.array([1.0]),
                tangent=np.array([1.0, 2.0]),
                method="exact",
                step=1.0e-3,
                evaluations=1,
                parameter_names=("x",),
                trainable=(True,),
            ),
            "tangent shape",
        ),
        (
            lambda: HVPResult(
                value=1.0,
                hvp=np.array([1.0, 0.25]),
                tangent=np.array([1.0, 0.0]),
                method="exact",
                step=1.0e-3,
                evaluations=1,
                parameter_names=("x", "frozen"),
                trainable=(True, False),
            ),
            "HVP must be zero for non-trainable parameters",
        ),
        (
            lambda: HVPResult(
                value=1.0,
                hvp=np.array([1.0, 0.0]),
                tangent=np.array([1.0, 0.25]),
                method="exact",
                step=1.0e-3,
                evaluations=1,
                parameter_names=("x", "frozen"),
                trainable=(True, False),
            ),
            "HVP tangent must be zero for non-trainable parameters",
        ),
        (
            lambda: NaturalGradientResult(
                base_gradient=gradient,
                metric=np.array([[1.0, 2.0], [0.0, 1.0]]),
                natural_gradient=np.array([1.0, -2.0]),
                damping=0.0,
                condition_number=1.0,
            ),
            "symmetric",
        ),
        (
            lambda: NaturalGradientOptimizationResult(
                values=np.array([0.0, 1.0]),
                final_gradient=gradient,
                final_natural_gradient=natural_gradient_result,
                value_history=(1.0,),
                gradient_norm_history=(1.0,),
                natural_step_norm_history=(0.5,),
                steps=0,
                converged=False,
                reason="max_steps",
                best_values=np.array([0.0, 1.0]),
                best_value=1.0,
            ),
            "one value per update step",
        ),
        (
            lambda: ImplicitSensitivityResult(
                sensitivity=np.ones((2, 1)),
                hessian=np.eye(2),
                cross_derivative=np.ones((1, 2)),
                damping=0.0,
                condition_number=1.0,
                method="implicit",
                parameter_names=("x", "y"),
                trainable=(True, True),
                hyperparameter_names=("a", "b"),
            ),
            "shape must match",
        ),
        (
            lambda: FixedPointSensitivityResult(
                sensitivity=np.eye(2),
                state_jacobian=np.ones((2, 1)),
                parameter_jacobian=np.eye(2),
                system_matrix=np.eye(2),
                damping=0.0,
                condition_number=1.0,
                method="fixed_point",
                parameter_names=("x", "y"),
                trainable=(True, True),
                hyperparameter_names=("a", "b"),
            ),
            "state_jacobian must be square",
        ),
    )
    for factory, message in invalid_factories:
        with pytest.raises(ValueError, match=message):
            factory()


def test_primitive_identity_and_contracts_fail_closed_on_malformed_metadata() -> None:
    """Primitive registry contracts should preserve typed identity and metadata safety."""

    def batching_rule(
        _function: Callable[..., object],
        values: tuple[object, ...],
        in_axes: tuple[int | None, ...],
        _axis: int,
    ) -> object:
        return values, in_axes

    rule = CustomDerivativeRule(
        name="identity_contract_rule",
        value_fn=lambda values: np.asarray(values, dtype=np.float64),
        jvp_rule=lambda _values, tangent: np.asarray(tangent, dtype=np.float64),
        parameter_names=("theta",),
        trainable=(True,),
    )
    identity = PrimitiveIdentity("quantum", "rx", "2")
    assert identity.key == "quantum:rx@2"
    assert PrimitiveIdentity.parse(identity) is identity
    assert PrimitiveIdentity.parse("quantum:rx@2") == identity
    assert PrimitiveIdentity.parse("quantum:rx") == PrimitiveIdentity("quantum", "rx", "1")

    transform = PrimitiveTransformRule(
        identity=identity,
        derivative_rule=rule,
        batching_rule=batching_rule,
        lowering_rule=lambda lowered_rule: {"name": lowered_rule.name},
        lowering_metadata={
            "mlir_op": "scpn_diff.rx",
            "llvm": "blocked",
            "nondifferentiable_boundary": "declared_test_boundary",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (1,),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args[:1],
        nondifferentiable_policy="none",
        effect="pure",
    )
    contract = PrimitiveContract.from_transform(transform)
    assert contract.identity == identity
    assert contract.lowering_metadata["mlir_op"] == "scpn_diff.rx"

    registry = CustomDerivativeRegistry()
    registry.register_transform(transform)
    assert registry.require_complete_contract(identity) == contract
    assert registry.require_effect(identity) == "pure"
    assert registry.require_nondifferentiable_policy(identity) == "none"
    assert registry.require_shape_rule(identity)((np.array([1.0]),)) == (1,)
    assert registry.require_dtype_rule(identity)((np.array([1.0]),)) == "float64"
    assert registry.require_static_argument_rule(identity)(("theta", "phi")) == ("theta",)

    malformed_cases: tuple[tuple[Callable[[], object], str], ...] = (
        (lambda: PrimitiveIdentity("bad namespace", "rx"), "whitespace"),
        (lambda: PrimitiveIdentity.parse("missing_namespace"), "namespace:name"),
        (
            lambda: CustomDerivativeRule(
                name="",
                value_fn=lambda values: values,
                jvp_rule=lambda _values, tangent: tangent,
            ),
            "name",
        ),
        (
            lambda: PrimitiveTransformRule(identity=cast(Any, "bad"), derivative_rule=rule),
            "PrimitiveIdentity",
        ),
        (
            lambda: PrimitiveTransformRule(
                identity=identity,
                derivative_rule=rule,
                lowering_metadata={"mlir_op": ""},
            ),
            "metadata values",
        ),
        (
            lambda: PrimitiveContract(
                identity=identity,
                derivative_rule=rule,
                batching_rule=None,
                lowering_rule=None,
                lowering_metadata={"": "bad"},
                shape_rule=None,
                dtype_rule=None,
                static_argument_rule=None,
                nondifferentiable_policy="none",
                effect="pure",
            ),
            "metadata keys",
        ),
        (lambda: CustomDerivativeRegistry().require_complete_contract(identity), "no primitive"),
    )
    for factory, message in malformed_cases:
        with pytest.raises(ValueError, match=message):
            factory()
