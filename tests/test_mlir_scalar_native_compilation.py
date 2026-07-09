# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR Scalar Native Compilation Tests
"""Real public-contract tests for scalar native LLVM/JIT compiler lowering."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.compiler import (
    CompilerADExecutableConfig,
    compile_scalar_binary_elementwise_ad_to_native_llvm_jit,
    compile_scalar_quadratic_ad_to_native_llvm_jit,
    compile_scalar_unary_elementwise_ad_to_native_llvm_jit,
    make_scalar_binary_elementwise_native_llvm_jit_lowering_rule,
    make_scalar_quadratic_native_llvm_jit_lowering_rule,
    make_scalar_unary_elementwise_native_llvm_jit_lowering_rule,
)
from scpn_quantum_control.compiler import mlir_scalar_native_compilation as scalar_native
from scpn_quantum_control.differentiable import CustomDerivativeRule

FloatArray = NDArray[np.float64]
NativeFunction = Callable[..., None]


def _quadratic_rule(name: str = "scalar_quadratic_contract_rule") -> CustomDerivativeRule:
    """Return the quadratic rule used by native scalar compiler tests."""

    def value(values: FloatArray) -> FloatArray:
        return np.array([2.0 * values[0] ** 2 - 3.0 * values[0] + 0.25], dtype=np.float64)

    def jvp(values: FloatArray, tangent: FloatArray) -> FloatArray:
        return np.array([(4.0 * values[0] - 3.0) * tangent[0]], dtype=np.float64)

    def vjp(values: FloatArray, cotangent: FloatArray) -> FloatArray:
        return np.array([(4.0 * values[0] - 3.0) * cotangent[0]], dtype=np.float64)

    return CustomDerivativeRule(
        name=name,
        value_fn=value,
        jvp_rule=jvp,
        vjp_rule=vjp,
        parameter_names=("x",),
        trainable=(True,),
    )


def _unary_rule(primitive: str) -> CustomDerivativeRule:
    """Return a unary rule matching the requested native primitive."""

    def value(values: FloatArray) -> FloatArray:
        if primitive == "sin":
            return np.array([np.sin(values[0])], dtype=np.float64)
        if primitive == "cos":
            return np.array([np.cos(values[0])], dtype=np.float64)
        if primitive == "exp":
            return np.array([np.exp(values[0])], dtype=np.float64)
        raise ValueError("unsupported test primitive")

    def derivative(values: FloatArray) -> float:
        if primitive == "sin":
            return float(np.cos(values[0]))
        if primitive == "cos":
            return float(-np.sin(values[0]))
        if primitive == "exp":
            return float(np.exp(values[0]))
        raise ValueError("unsupported test primitive")

    def jvp(values: FloatArray, tangent: FloatArray) -> FloatArray:
        return np.array([derivative(values) * tangent[0]], dtype=np.float64)

    def vjp(values: FloatArray, cotangent: FloatArray) -> FloatArray:
        return np.array([derivative(values) * cotangent[0]], dtype=np.float64)

    return CustomDerivativeRule(
        name=f"scalar_{primitive}_contract_rule",
        value_fn=value,
        jvp_rule=jvp,
        vjp_rule=vjp,
        parameter_names=("x",),
        trainable=(True,),
    )


def _binary_rule(primitive: str) -> CustomDerivativeRule:
    """Return a binary rule matching the requested native primitive."""

    def value(values: FloatArray) -> FloatArray:
        if primitive == "add":
            return np.array([values[0] + values[1]], dtype=np.float64)
        if primitive == "subtract":
            return np.array([values[0] - values[1]], dtype=np.float64)
        if primitive == "multiply":
            return np.array([values[0] * values[1]], dtype=np.float64)
        raise ValueError("unsupported test primitive")

    def gradient(values: FloatArray) -> FloatArray:
        if primitive == "add":
            return np.array([1.0, 1.0], dtype=np.float64)
        if primitive == "subtract":
            return np.array([1.0, -1.0], dtype=np.float64)
        if primitive == "multiply":
            return np.array([values[1], values[0]], dtype=np.float64)
        raise ValueError("unsupported test primitive")

    def jvp(values: FloatArray, tangent: FloatArray) -> FloatArray:
        return np.array([float(np.dot(gradient(values), tangent))], dtype=np.float64)

    def vjp(values: FloatArray, cotangent: FloatArray) -> FloatArray:
        return cast(FloatArray, (gradient(values) * cotangent[0]).astype(np.float64))

    return CustomDerivativeRule(
        name=f"scalar_{primitive}_contract_rule",
        value_fn=value,
        jvp_rule=jvp,
        vjp_rule=vjp,
        parameter_names=("x", "y"),
        trainable=(True, True),
    )


@pytest.mark.parametrize(
    ("primitive", "sample"),
    (
        ("sin", np.array([0.25], dtype=np.float64)),
        ("cos", np.array([0.35], dtype=np.float64)),
        ("exp", np.array([-0.2], dtype=np.float64)),
    ),
)
def test_unary_native_primitives_execute_public_contract(
    primitive: str,
    sample: FloatArray,
) -> None:
    """Unary scalar native kernels should match NumPy derivatives."""

    tangent = np.array([0.6], dtype=np.float64)
    cotangent = np.array([-1.75], dtype=np.float64)
    kernel = compile_scalar_unary_elementwise_ad_to_native_llvm_jit(
        _unary_rule(primitive),
        primitive=primitive.upper(),
        sample_values=sample,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    if primitive == "sin":
        expected_value = np.sin(sample)
        expected_gradient = np.cos(sample)
    elif primitive == "cos":
        expected_value = np.cos(sample)
        expected_gradient = -np.sin(sample)
    else:
        expected_value = np.exp(sample)
        expected_gradient = np.exp(sample)
    np.testing.assert_allclose(kernel.value(sample), expected_value, rtol=1.0e-12)
    np.testing.assert_allclose(kernel.jvp(sample, tangent), expected_gradient * tangent)
    np.testing.assert_allclose(kernel.vjp(sample, cotangent), expected_gradient * cotangent)
    np.testing.assert_allclose(kernel.gradient(sample), expected_gradient)
    assert kernel.rule_name == f"scalar_{primitive}_contract_rule"
    assert kernel.verification.passed is True


@pytest.mark.parametrize("primitive", ("add", "subtract", "multiply"))
def test_binary_native_primitives_execute_public_contract(primitive: str) -> None:
    """Binary scalar native kernels should expose correct JVP/VJP contracts."""

    values = np.array([1.25, -0.75], dtype=np.float64)
    tangent = np.array([0.5, -2.0], dtype=np.float64)
    cotangent = np.array([1.5], dtype=np.float64)
    kernel = compile_scalar_binary_elementwise_ad_to_native_llvm_jit(
        _binary_rule(primitive),
        primitive=primitive.upper(),
        sample_values=values,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )

    if primitive == "add":
        expected_value = np.array([0.5], dtype=np.float64)
        expected_gradient = np.array([1.0, 1.0], dtype=np.float64)
    elif primitive == "subtract":
        expected_value = np.array([2.0], dtype=np.float64)
        expected_gradient = np.array([1.0, -1.0], dtype=np.float64)
    else:
        expected_value = np.array([-0.9375], dtype=np.float64)
        expected_gradient = np.array([-0.75, 1.25], dtype=np.float64)
    np.testing.assert_allclose(kernel.value(values), expected_value, rtol=1.0e-12)
    np.testing.assert_allclose(kernel.jvp(values, tangent), [np.dot(expected_gradient, tangent)])
    np.testing.assert_allclose(kernel.vjp(values, cotangent), expected_gradient * cotangent[0])
    np.testing.assert_allclose(kernel.gradient(values), expected_gradient)
    assert kernel.verification.passed is True


def test_scalar_native_entrypoints_reject_invalid_public_contracts() -> None:
    """Scalar native factories should reject invalid rules, backends, samples, and primitives."""

    rule = _quadratic_rule()
    bad_rule = cast(CustomDerivativeRule, object())
    wrong_backend = CompilerADExecutableConfig(backend="mlir_runtime")
    with pytest.raises(ValueError, match="rule must be a CustomDerivativeRule"):
        compile_scalar_quadratic_ad_to_native_llvm_jit(
            bad_rule,
            quadratic=1.0,
            linear=0.0,
            constant=0.0,
            sample_values=[0.0],
        )
    with pytest.raises(ValueError, match="backend='native_llvm_jit'"):
        compile_scalar_quadratic_ad_to_native_llvm_jit(
            rule,
            quadratic=1.0,
            linear=0.0,
            constant=0.0,
            sample_values=[0.0],
            config=wrong_backend,
        )
    with pytest.raises(ValueError, match="coefficients must be finite"):
        compile_scalar_quadratic_ad_to_native_llvm_jit(
            rule,
            quadratic=np.inf,
            linear=0.0,
            constant=0.0,
            sample_values=[0.0],
        )
    with pytest.raises(ValueError, match="exactly one sample value"):
        compile_scalar_quadratic_ad_to_native_llvm_jit(
            rule,
            quadratic=1.0,
            linear=0.0,
            constant=0.0,
            sample_values=[0.0, 1.0],
        )
    with pytest.raises(ValueError, match="one of sin, cos, exp"):
        compile_scalar_unary_elementwise_ad_to_native_llvm_jit(
            _unary_rule("cos"),
            primitive="log",
            sample_values=[0.0],
        )
    with pytest.raises(ValueError, match="rule must be a CustomDerivativeRule"):
        compile_scalar_unary_elementwise_ad_to_native_llvm_jit(
            bad_rule,
            primitive="cos",
            sample_values=[0.0],
        )
    with pytest.raises(ValueError, match="backend='native_llvm_jit'"):
        compile_scalar_unary_elementwise_ad_to_native_llvm_jit(
            _unary_rule("cos"),
            primitive="cos",
            sample_values=[0.0],
            config=wrong_backend,
        )
    with pytest.raises(ValueError, match="exactly one sample value"):
        compile_scalar_unary_elementwise_ad_to_native_llvm_jit(
            _unary_rule("cos"),
            primitive="cos",
            sample_values=[0.0, 1.0],
        )
    with pytest.raises(ValueError, match="one of add, subtract, multiply"):
        compile_scalar_binary_elementwise_ad_to_native_llvm_jit(
            _binary_rule("add"),
            primitive="divide",
            sample_values=[0.0, 1.0],
        )
    with pytest.raises(ValueError, match="rule must be a CustomDerivativeRule"):
        compile_scalar_binary_elementwise_ad_to_native_llvm_jit(
            bad_rule,
            primitive="add",
            sample_values=[0.0, 1.0],
        )
    with pytest.raises(ValueError, match="backend='native_llvm_jit'"):
        compile_scalar_binary_elementwise_ad_to_native_llvm_jit(
            _binary_rule("add"),
            primitive="add",
            sample_values=[0.0, 1.0],
            config=wrong_backend,
        )
    with pytest.raises(ValueError, match="exactly two sample values"):
        compile_scalar_binary_elementwise_ad_to_native_llvm_jit(
            _binary_rule("add"),
            primitive="add",
            sample_values=[0.0],
        )


def test_scalar_native_lowering_rules_capture_and_require_samples() -> None:
    """Lowering-rule factories should use captured samples and fail without samples."""

    rule = _quadratic_rule()
    config = CompilerADExecutableConfig(backend="native_llvm_jit")
    quadratic_lowering = make_scalar_quadratic_native_llvm_jit_lowering_rule(
        quadratic=2.0,
        linear=-3.0,
        constant=0.25,
        sample_values=[0.4],
        config=config,
        sample_tangent=[0.2],
        sample_cotangent=[1.0],
    )
    unary_lowering = make_scalar_unary_elementwise_native_llvm_jit_lowering_rule(
        primitive="cos",
        sample_values=[0.4],
        config=config,
        sample_tangent=[0.2],
        sample_cotangent=[1.0],
    )
    binary_lowering = make_scalar_binary_elementwise_native_llvm_jit_lowering_rule(
        primitive="add",
        sample_values=[0.4, -0.1],
        config=config,
        sample_tangent=[0.2, 0.3],
        sample_cotangent=[1.0],
    )

    assert quadratic_lowering(rule).verification.passed is True
    assert unary_lowering(_unary_rule("cos")).verification.passed is True
    assert binary_lowering(_binary_rule("add")).verification.passed is True
    assert quadratic_lowering(rule, [0.5], config).verification.passed is True
    assert (
        unary_lowering(
            _unary_rule("cos"),
            [0.5],
            config,
            sample_tangent=[0.1],
            sample_cotangent=[1.0],
        ).verification.passed
        is True
    )
    assert (
        binary_lowering(
            _binary_rule("add"),
            [0.5, 0.25],
            config,
            sample_tangent=[0.1, -0.2],
            sample_cotangent=[1.0],
        ).verification.passed
        is True
    )

    with pytest.raises(ValueError, match="coefficients must be finite"):
        make_scalar_quadratic_native_llvm_jit_lowering_rule(
            quadratic=np.nan,
            linear=0.0,
            constant=0.0,
        )
    with pytest.raises(ValueError, match="quadratic lowering requires sample_values"):
        make_scalar_quadratic_native_llvm_jit_lowering_rule(
            quadratic=2.0,
            linear=-3.0,
            constant=0.25,
        )(rule)
    with pytest.raises(ValueError, match="unary lowering requires sample_values"):
        make_scalar_unary_elementwise_native_llvm_jit_lowering_rule(primitive="cos")(
            _unary_rule("cos")
        )
    with pytest.raises(ValueError, match="binary lowering requires sample_values"):
        make_scalar_binary_elementwise_native_llvm_jit_lowering_rule(primitive="add")(
            _binary_rule("add")
        )


def test_compiled_scalar_kernels_reject_wrong_runtime_shapes() -> None:
    """Compiled kernels should validate runtime vector arity before native calls."""

    quadratic = compile_scalar_quadratic_ad_to_native_llvm_jit(
        _quadratic_rule(),
        quadratic=2.0,
        linear=-3.0,
        constant=0.25,
        sample_values=[0.5],
        sample_tangent=[0.2],
        sample_cotangent=[1.0],
    )
    binary = compile_scalar_binary_elementwise_ad_to_native_llvm_jit(
        _binary_rule("add"),
        primitive="add",
        sample_values=[0.5, 0.25],
        sample_tangent=[0.2, -0.1],
        sample_cotangent=[1.0],
    )

    with pytest.raises(ValueError, match="requires one value"):
        quadratic.value(np.array([0.5, 0.25], dtype=np.float64))
    with pytest.raises(ValueError, match="requires one value"):
        quadratic.jvp(np.array([0.5, 0.25], dtype=np.float64), np.array([0.2], dtype=np.float64))
    with pytest.raises(ValueError, match="requires one tangent value"):
        quadratic.jvp(np.array([0.5], dtype=np.float64), np.array([0.2, 0.3], dtype=np.float64))
    with pytest.raises(ValueError, match="requires two values"):
        binary.value(np.array([0.5], dtype=np.float64))
    with pytest.raises(ValueError, match="requires two values"):
        binary.jvp(np.array([0.5], dtype=np.float64), np.array([0.2, -0.1], dtype=np.float64))
    with pytest.raises(ValueError, match="requires 2 tangent value"):
        binary.jvp(
            np.array([0.5, 0.25], dtype=np.float64),
            np.array([0.2], dtype=np.float64),
        )


def test_scalar_native_internal_guards_reject_invalid_helper_inputs() -> None:
    """Scalar native helper guards should reject unsupported internal contracts."""

    def unused_unary(_values: Any, _out: Any) -> None:
        raise AssertionError("guard should reject before native call")

    def unused_binary(_values: Any, _tangent: Any, _out: Any) -> None:
        raise AssertionError("guard should reject before native call")

    with pytest.raises(ValueError, match="one of sin, cos, exp"):
        scalar_native._scalar_unary_native_value_lines("log")
    with pytest.raises(ValueError, match="one of sin, cos, exp"):
        scalar_native._scalar_unary_native_gradient_lines("log")
    with pytest.raises(ValueError, match="one of add, subtract, multiply"):
        scalar_native._scalar_binary_native_gradient_lines("divide")
    with pytest.raises(ValueError, match="output_size must be one or two"):
        scalar_native._call_native_scalar_pair_unary(
            unused_unary,
            np.array([0.5, 0.25], dtype=np.float64),
            3,
        )
    with pytest.raises(ValueError, match="output_size must be one or two"):
        scalar_native._call_native_scalar_pair_binary(
            unused_binary,
            np.array([0.5, 0.25], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            "cotangent",
            3,
        )


@pytest.mark.parametrize("family", ("quadratic", "unary", "binary"))
def test_scalar_native_gradient_verification_rejects_mismatched_native_gradient(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
) -> None:
    """Gradient verification should reject native gradient kernels that disagree with VJP."""

    def fake_compile(_llvm_ir: str, _base_symbol: str) -> dict[str, NativeFunction]:
        if family == "binary":
            return _fake_binary_functions()
        if family == "unary":
            return _fake_cos_functions()
        return _fake_unary_functions()

    monkeypatch.setattr(scalar_native, "_compile_native_llvm_jit_functions", fake_compile)
    if family == "quadratic":
        with pytest.raises(ValueError, match="gradient kernel verification failed"):
            compile_scalar_quadratic_ad_to_native_llvm_jit(
                _quadratic_rule("fake_quadratic_gradient_rule"),
                quadratic=2.0,
                linear=-3.0,
                constant=0.25,
                sample_values=[0.5],
                sample_tangent=[0.2],
                sample_cotangent=[1.0],
            )
    elif family == "unary":
        with pytest.raises(ValueError, match="scalar unary gradient verification failed"):
            compile_scalar_unary_elementwise_ad_to_native_llvm_jit(
                _unary_rule("cos"),
                primitive="cos",
                sample_values=[0.5],
                sample_tangent=[0.2],
                sample_cotangent=[1.0],
            )
    else:
        with pytest.raises(ValueError, match="scalar binary gradient verification failed"):
            compile_scalar_binary_elementwise_ad_to_native_llvm_jit(
                _binary_rule("add"),
                primitive="add",
                sample_values=[0.5, 0.25],
                sample_tangent=[0.2, -0.1],
                sample_cotangent=[1.0],
            )


def _fake_unary_functions() -> dict[str, NativeFunction]:
    """Return fake scalar native callables with a bad gradient slot."""

    def value(values: Any, out: Any) -> None:
        x = float(values[0])
        out[0] = 2.0 * x * x - 3.0 * x + 0.25

    def jvp(values: Any, tangent: Any, out: Any) -> None:
        x = float(values[0])
        out[0] = (4.0 * x - 3.0) * float(tangent[0])

    def vjp(values: Any, cotangent: Any, out: Any) -> None:
        x = float(values[0])
        out[0] = (4.0 * x - 3.0) * float(cotangent[0])

    def gradient(_values: Any, out: Any) -> None:
        out[0] = 999.0

    return {"value": value, "jvp": jvp, "vjp": vjp, "gradient": gradient}


def _fake_cos_functions() -> dict[str, NativeFunction]:
    """Return fake cosine native callables with a bad gradient slot."""

    def value(values: Any, out: Any) -> None:
        out[0] = float(np.cos(float(values[0])))

    def jvp(values: Any, tangent: Any, out: Any) -> None:
        out[0] = float(-np.sin(float(values[0])) * float(tangent[0]))

    def vjp(values: Any, cotangent: Any, out: Any) -> None:
        out[0] = float(-np.sin(float(values[0])) * float(cotangent[0]))

    def gradient(_values: Any, out: Any) -> None:
        out[0] = 999.0

    return {"value": value, "jvp": jvp, "vjp": vjp, "gradient": gradient}


def _fake_binary_functions() -> dict[str, NativeFunction]:
    """Return fake binary native callables with a bad gradient slot."""

    def value(values: Any, out: Any) -> None:
        out[0] = float(values[0]) + float(values[1])

    def jvp(_values: Any, tangent: Any, out: Any) -> None:
        out[0] = float(tangent[0]) + float(tangent[1])

    def vjp(_values: Any, cotangent: Any, out: Any) -> None:
        out[0] = float(cotangent[0])
        out[1] = float(cotangent[0])

    def gradient(_values: Any, out: Any) -> None:
        out[0] = 999.0
        out[1] = 999.0

    return {"value": value, "jvp": jvp, "vjp": vjp, "gradient": gradient}
