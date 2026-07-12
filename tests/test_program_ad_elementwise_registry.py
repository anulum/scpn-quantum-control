# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD elementwise registry tests
# scpn-quantum-control -- Program AD elementwise registry tests
"""Tests for Program AD elementwise registry contracts and direct rules."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.program_ad_elementwise_primitives as elementwise_primitives
from scpn_quantum_control.differentiable import (
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    PrimitiveContract,
    PrimitiveIdentity,
    PrimitiveTransformRule,
    custom_derivative_rule_for,
    primitive_complete_contract_for,
    primitive_contract_for,
    whole_program_value_and_grad,
)


def _assert_allclose(
    actual: object, expected: object, *, rtol: float = 1.0e-7, atol: float = 0.0
) -> None:
    """Assert NumPy closeness across dynamically typed Program AD payloads."""

    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def _transform_rule_from_contract(contract: PrimitiveContract) -> PrimitiveTransformRule:
    """Return a mutable registry transform that exactly mirrors a contract."""

    return PrimitiveTransformRule(
        identity=contract.identity,
        derivative_rule=contract.derivative_rule,
        batching_rule=contract.batching_rule,
        lowering_rule=contract.lowering_rule,
        lowering_metadata=contract.lowering_metadata,
        shape_rule=contract.shape_rule,
        dtype_rule=contract.dtype_rule,
        static_argument_rule=contract.static_argument_rule,
        nondifferentiable_policy=contract.nondifferentiable_policy,
        effect=contract.effect,
    )


def test_program_ad_elementwise_registry_uses_extracted_contract_module() -> None:
    """The compatibility facade should expose the extracted elementwise contracts."""

    contract = primitive_contract_for("scpn.program_ad.elementwise:sin")

    assert contract.shape_rule is elementwise_primitives._program_ad_elementwise_shape
    assert contract.dtype_rule is elementwise_primitives._program_ad_elementwise_dtype_rule
    assert (
        contract.static_argument_rule
        is elementwise_primitives._program_ad_elementwise_static_arguments
    )
    assert contract.batching_rule is elementwise_primitives._program_ad_elementwise_batching_rule


def test_program_ad_elementwise_extracted_helpers_cover_fail_closed_boundaries() -> None:
    """Extracted elementwise helpers should validate helper inputs and fail closed."""

    class ShapeSequence:
        shape = [2, 1]

    class TraceLike:
        _items = (object(),)
        context = object()

    _assert_allclose(
        elementwise_primitives._as_real_numeric_array("operand", [True, 2]),
        np.array([1.0, 2.0]),
    )
    with pytest.raises(ValueError, match="real numeric"):
        elementwise_primitives._as_real_numeric_array("operand", np.array(["x"]))
    assert elementwise_primitives._normalise_axis("axis", -1, 2) == 1
    with pytest.raises(ValueError, match="scalar"):
        elementwise_primitives._normalise_axis("axis", 0, 0)
    with pytest.raises(ValueError, match="out of bounds"):
        elementwise_primitives._normalise_axis("axis", 2, 2)
    assert elementwise_primitives._program_ad_shape_signature(()) == "scalar"
    assert elementwise_primitives._program_ad_array_shape_of(ShapeSequence()) == (2, 1)
    assert elementwise_primitives._program_ad_array_dtype_of(TraceLike()) == "float64"
    with pytest.raises(ValueError, match="real numeric"):
        elementwise_primitives._program_ad_array_dtype_of(["x"])
    with pytest.raises(ValueError, match="broadcasting"):
        elementwise_primitives._broadcast_shape((2,), (3,))
    _assert_allclose(
        elementwise_primitives._program_ad_elementwise_unbroadcast(
            np.ones((2, 3), dtype=np.float64), target_shape=()
        ),
        np.array([6.0]),
    )
    _assert_allclose(
        elementwise_primitives._program_ad_elementwise_unbroadcast(
            np.arange(6.0, dtype=np.float64).reshape(2, 3), target_shape=(1, 3)
        ),
        np.array([3.0, 5.0, 7.0]),
    )


def test_program_ad_elementwise_extracted_unary_direct_rules_cover_all_branches() -> None:
    """Unary direct factories should cover every extracted direct-rule branch."""

    base_values = {
        "sin": np.array([0.2, 0.4]),
        "cos": np.array([0.2, 0.4]),
        "exp": np.array([0.2, 0.4]),
        "expm1": np.array([0.2, 0.4]),
        "log": np.array([1.2, 2.4]),
        "log1p": np.array([0.2, 0.4]),
        "sqrt": np.array([1.2, 2.4]),
        "tan": np.array([0.2, 0.4]),
        "tanh": np.array([0.2, 0.4]),
        "arcsin": np.array([0.2, 0.4]),
        "arccos": np.array([0.2, 0.4]),
        "reciprocal": np.array([1.2, 2.4]),
        "square": np.array([0.2, 0.4]),
        "abs": np.array([0.2, 0.4]),
        "negative": np.array([0.2, 0.4]),
    }
    tangent = np.array([0.5, -0.25], dtype=np.float64)
    expected_values: dict[str, Callable[[NDArray[np.float64]], NDArray[np.float64]]] = {
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
        "expm1": np.expm1,
        "log": np.log,
        "log1p": np.log1p,
        "sqrt": np.sqrt,
        "tan": np.tan,
        "tanh": np.tanh,
        "arcsin": np.arcsin,
        "arccos": np.arccos,
        "reciprocal": np.reciprocal,
        "square": np.square,
        "abs": np.abs,
        "negative": np.negative,
    }
    expected_jvps: dict[str, Callable[[NDArray[np.float64]], NDArray[np.float64]]] = {
        "sin": lambda value: np.cos(value) * tangent,
        "cos": lambda value: -np.sin(value) * tangent,
        "exp": lambda value: np.exp(value) * tangent,
        "expm1": lambda value: np.exp(value) * tangent,
        "log": lambda value: tangent / value,
        "log1p": lambda value: tangent / (1.0 + value),
        "sqrt": lambda value: tangent / (2.0 * np.sqrt(value)),
        "tan": lambda value: tangent / np.cos(value) ** 2,
        "tanh": lambda value: tangent * (1.0 - np.tanh(value) ** 2),
        "arcsin": lambda value: tangent / np.sqrt(1.0 - value**2),
        "arccos": lambda value: -tangent / np.sqrt(1.0 - value**2),
        "reciprocal": lambda value: -tangent / value**2,
        "square": lambda value: 2.0 * value * tangent,
        "abs": lambda value: np.sign(value) * tangent,
        "negative": lambda _value: np.negative(tangent),
    }

    for name, values in base_values.items():
        rule = elementwise_primitives._program_ad_elementwise_derivative_rule(name)
        assert rule.jvp_rule is not None
        assert rule.vjp_rule is not None
        _assert_allclose(rule.value_fn(values), expected_values[name](values))
        _assert_allclose(rule.jvp_rule(values, tangent), expected_jvps[name](values))
        _assert_allclose(rule.vjp_rule(values, tangent), expected_jvps[name](values))

    with pytest.raises(ValueError, match="greater than zero"):
        elementwise_primitives._program_ad_elementwise_unary_value("log", np.array([0.0]))
    with pytest.raises(ValueError, match="greater than -1"):
        elementwise_primitives._program_ad_elementwise_unary_value("log1p", np.array([-1.0]))
    with pytest.raises(ValueError, match="non-negative"):
        elementwise_primitives._program_ad_elementwise_unary_value("sqrt", np.array([-1.0]))
    with pytest.raises(ValueError, match="singular"):
        elementwise_primitives._program_ad_elementwise_unary_jvp(
            "sqrt", np.array([0.0]), np.array([1.0])
        )
    with pytest.raises(ValueError, match="values in"):
        elementwise_primitives._program_ad_elementwise_unary_value("arcsin", np.array([2.0]))
    with pytest.raises(ValueError, match="boundary"):
        elementwise_primitives._program_ad_elementwise_unary_jvp(
            "arccos", np.array([1.0]), np.array([1.0])
        )
    with pytest.raises(ValueError, match="non-zero"):
        elementwise_primitives._program_ad_elementwise_unary_value("reciprocal", np.array([0.0]))
    with pytest.raises(ValueError, match="undefined at zero"):
        elementwise_primitives._program_ad_elementwise_unary_jvp(
            "abs", np.array([0.0]), np.array([1.0])
        )
    with pytest.raises(ValueError, match="tangent shape"):
        elementwise_primitives._program_ad_elementwise_unary_jvp(
            "sin", np.array([1.0]), np.array([1.0, 2.0])
        )
    with pytest.raises(ValueError, match="unsupported"):
        elementwise_primitives._program_ad_elementwise_unary_value("unsupported", np.array([1.0]))
    with pytest.raises(ValueError, match="unsupported"):
        elementwise_primitives._program_ad_elementwise_unary_jvp(
            "unsupported", np.array([1.0]), np.array([1.0])
        )


def test_program_ad_elementwise_extracted_unary_singular_tan_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The extracted tan derivative guard should fail closed at exact singularities."""

    def zero_cos(_value: object) -> NDArray[np.float64]:
        return np.array([0.0], dtype=np.float64)

    monkeypatch.setattr(cast(Any, elementwise_primitives).np, "cos", zero_cos)
    with pytest.raises(ValueError, match="tan derivative"):
        elementwise_primitives._program_ad_elementwise_unary_jvp(
            "tan", np.array([1.0]), np.array([1.0])
        )


def test_program_ad_elementwise_extracted_binary_direct_rules_cover_all_branches() -> None:
    """Binary direct factories should cover flat direct value, JVP, and VJP branches."""

    left = np.array([3.0, 4.0], dtype=np.float64)
    right = np.array([1.5, 2.0], dtype=np.float64)
    values = np.concatenate((left, right))
    tangent_left = np.array([0.25, -0.5], dtype=np.float64)
    tangent_right = np.array([0.75, 1.25], dtype=np.float64)
    tangent = np.concatenate((tangent_left, tangent_right))
    cotangent = np.array([1.0, -2.0], dtype=np.float64)
    expected_values = {
        "add": left + right,
        "subtract": left - right,
        "multiply": left * right,
        "divide": left / right,
        "power": left**right,
        "maximum": np.maximum(left, right),
        "minimum": np.minimum(left, right),
    }
    expected_jvps = {
        "add": tangent_left + tangent_right,
        "subtract": tangent_left - tangent_right,
        "multiply": tangent_left * right + left * tangent_right,
        "divide": (tangent_left * right - left * tangent_right) / right**2,
        "power": left**right * (tangent_right * np.log(left) + right * tangent_left / left),
        "maximum": np.where(left > right, tangent_left, tangent_right),
        "minimum": np.where(left < right, tangent_left, tangent_right),
    }
    expected_vjps = {
        "add": np.concatenate((cotangent, cotangent)),
        "subtract": np.concatenate((cotangent, -cotangent)),
        "multiply": np.concatenate((cotangent * right, cotangent * left)),
        "divide": np.concatenate((cotangent / right, -cotangent * left / right**2)),
        "power": np.concatenate(
            (cotangent * right * left ** (right - 1.0), cotangent * left**right * np.log(left))
        ),
        "maximum": np.concatenate((np.where(left > right, cotangent, 0.0), np.zeros_like(right))),
        "minimum": np.concatenate((np.zeros_like(left), np.where(left < right, 0.0, cotangent))),
    }

    for name in expected_values:
        rule = elementwise_primitives._program_ad_elementwise_derivative_rule(name)
        assert rule.jvp_rule is not None
        assert rule.vjp_rule is not None
        _assert_allclose(rule.value_fn(values), expected_values[name])
        _assert_allclose(rule.jvp_rule(values, tangent), expected_jvps[name])
        _assert_allclose(rule.vjp_rule(values, cotangent), expected_vjps[name])

    with pytest.raises(ValueError, match="two equal flat operands"):
        elementwise_primitives._program_ad_elementwise_binary_value("add", np.array([1.0]))
    with pytest.raises(ValueError, match="tangent shape"):
        elementwise_primitives._program_ad_elementwise_binary_jvp(
            "add", np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0, 4.0])
        )
    with pytest.raises(ValueError, match="cotangent shape"):
        elementwise_primitives._program_ad_elementwise_binary_vjp(
            "add", np.array([1.0, 2.0]), np.array([1.0, 2.0])
        )
    with pytest.raises(ValueError, match="non-zero right operand"):
        elementwise_primitives._program_ad_elementwise_binary_value("divide", np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="positive left operand"):
        elementwise_primitives._program_ad_elementwise_binary_value("power", np.array([-1.0, 2.0]))
    with pytest.raises(ValueError, match="undefined at equal operands"):
        elementwise_primitives._program_ad_elementwise_binary_jvp(
            "maximum", np.array([1.0, 1.0]), np.array([1.0, 1.0])
        )
    with pytest.raises(ValueError, match="unsupported"):
        elementwise_primitives._program_ad_elementwise_binary_value("unsupported", values)
    with pytest.raises(ValueError, match="unsupported"):
        elementwise_primitives._program_ad_elementwise_binary_jvp("unsupported", values, tangent)
    with pytest.raises(ValueError, match="unsupported"):
        elementwise_primitives._program_ad_elementwise_binary_vjp("unsupported", values, cotangent)


def test_program_ad_elementwise_extracted_static_binary_factories_cover_boundaries() -> None:
    """Static binary factories should cover broadcast and malformed-static branches."""

    left = np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    right = np.array([[1.5, 2.0], [2.5, 3.0]], dtype=np.float64)
    tangent_left = np.full((2, 2), 0.5, dtype=np.float64)
    tangent_right = np.full((2, 2), -0.25, dtype=np.float64)
    cotangent = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float64)
    values = np.concatenate((left.reshape(-1), right.reshape(-1)))
    tangent = np.concatenate((tangent_left.reshape(-1), tangent_right.reshape(-1)))

    for name in ("add", "subtract", "multiply", "divide", "power", "maximum", "minimum"):
        rule = elementwise_primitives.program_ad_elementwise_binary_derivative_rule(
            name, left.shape, right.shape
        )
        assert rule.jvp_rule is not None
        assert rule.vjp_rule is not None
        _assert_allclose(
            rule.value_fn(values),
            elementwise_primitives._program_ad_elementwise_binary_static_value_array(
                name, left, right
            ),
        )
        _assert_allclose(
            rule.jvp_rule(values, tangent),
            elementwise_primitives._program_ad_elementwise_binary_static_jvp_array(
                name, left, right, tangent_left, tangent_right
            ),
        )
        left_adjoint, right_adjoint = (
            elementwise_primitives._program_ad_elementwise_binary_static_adjoint_arrays(
                name, left, right, cotangent
            )
        )
        _assert_allclose(
            rule.vjp_rule(values, cotangent.reshape(-1)),
            np.concatenate((left_adjoint.reshape(-1), right_adjoint.reshape(-1))),
        )

    with pytest.raises(ValueError, match="unsupported"):
        elementwise_primitives._program_ad_elementwise_normalise_binary_static_shapes(
            "unsupported", (2,), (2,)
        )
    with pytest.raises(ValueError, match="non-negative"):
        elementwise_primitives._program_ad_elementwise_normalise_binary_static_shapes(
            "add", (-1,), (2,)
        )
    with pytest.raises(ValueError, match="broadcast-compatible"):
        elementwise_primitives._program_ad_elementwise_normalise_binary_static_shapes(
            "add", (2,), (3,)
        )
    with pytest.raises(ValueError, match="flattened left operand"):
        elementwise_primitives._program_ad_elementwise_binary_static_split(
            "add", "values", np.array([1.0]), left_shape=(2,), right_shape=(2,)
        )
    with pytest.raises(ValueError, match="unsupported"):
        elementwise_primitives._program_ad_elementwise_binary_static_value_array(
            "unsupported", left, right
        )
    with pytest.raises(ValueError, match="unsupported"):
        elementwise_primitives._program_ad_elementwise_binary_static_jvp_array(
            "unsupported", left, right, tangent_left, tangent_right
        )
    with pytest.raises(ValueError, match="unsupported"):
        elementwise_primitives._program_ad_elementwise_binary_static_adjoint_arrays(
            "unsupported", left, right, cotangent
        )
    rule = elementwise_primitives.program_ad_elementwise_binary_derivative_rule("add", (2,), (2,))
    assert rule.vjp_rule is not None
    with pytest.raises(ValueError, match="cotangent shape"):
        rule.vjp_rule(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0]))


def test_program_ad_elementwise_extracted_contract_rules_cover_validation_branches() -> None:
    """Elementwise registry helpers should cover shape, dtype, static, and require branches."""

    assert elementwise_primitives._program_ad_elementwise_name(np.absolute) == "abs"
    assert elementwise_primitives._program_ad_elementwise_name(np.sin) == "sin"
    assert elementwise_primitives._program_ad_elementwise_shape((np.ones((2, 1)),)) == (2, 1)
    assert elementwise_primitives._program_ad_elementwise_shape(
        (np.ones((2, 1)), np.ones((3,)))
    ) == (2, 3)
    assert (
        elementwise_primitives._program_ad_elementwise_dtype_rule(
            (np.ones(1, dtype=np.float32), np.ones(1, dtype=np.int64))
        )
        == "float64"
    )
    assert elementwise_primitives._program_ad_elementwise_static_arguments((np.ones(1),)) == ()
    with pytest.raises(ValueError, match="shape rule"):
        elementwise_primitives._program_ad_elementwise_shape(())
    with pytest.raises(ValueError, match="dtype rule"):
        elementwise_primitives._program_ad_elementwise_dtype_rule(())
    with pytest.raises(ValueError, match="static rule"):
        elementwise_primitives._program_ad_elementwise_static_arguments(())
    assert (
        elementwise_primitives._program_ad_elementwise_lowering_metadata("divide")[
            "static_derivative_factory"
        ]
        == "program_ad_elementwise_binary_derivative_rule"
    )
    assert (
        elementwise_primitives._program_ad_elementwise_lowering_metadata("heaviside")[
            "static_signature"
        ]
        == "source_shape:ranked_tensor_shape;step_value"
    )
    assert (
        elementwise_primitives._program_ad_elementwise_lowering_metadata("sign")[
            "static_derivative_factory"
        ]
        == "blocked_derivative_losing"
    )
    assert (
        elementwise_primitives._program_ad_elementwise_lowering_metadata("sin")[
            "nondifferentiable_boundary"
        ]
        == "none"
    )
    unknown_rule = elementwise_primitives._program_ad_elementwise_derivative_rule("unknown")
    assert unknown_rule.jvp_rule is not None
    with pytest.raises(ValueError, match="operator-intercepted trace dispatch"):
        unknown_rule.value_fn(np.array([1.0]))
    with pytest.raises(ValueError, match="operator-intercepted trace dispatch"):
        unknown_rule.jvp_rule(np.array([1.0]), np.array([1.0]))

    discontinuous_rule = elementwise_primitives._program_ad_elementwise_derivative_rule("sign")
    assert discontinuous_rule.jvp_rule is not None
    assert discontinuous_rule.vjp_rule is not None
    with pytest.raises(ValueError, match="derivative-losing"):
        discontinuous_rule.value_fn(np.array([1.0]))
    with pytest.raises(ValueError, match="derivative-losing"):
        discontinuous_rule.jvp_rule(np.array([1.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="derivative-losing"):
        discontinuous_rule.vjp_rule(np.array([1.0]), np.array([1.0]))


def test_program_ad_elementwise_extracted_batching_rule_covers_branch_contracts() -> None:
    """Elementwise batching rules should validate axes and batch concrete operands."""

    def add(left: object, right: object) -> object:
        return np.asarray(left, dtype=np.float64) + np.asarray(right, dtype=np.float64)

    def unary(value: object) -> object:
        return np.asarray(value, dtype=np.float64) + 1.0

    assert np.asarray(
        elementwise_primitives._program_ad_elementwise_batching_rule(
            unary, (np.array([1.0, 2.0]),), (None,), 0
        )
    ).tolist() == [2.0, 3.0]
    batched = elementwise_primitives._program_ad_elementwise_batching_rule(
        add,
        (np.arange(6.0, dtype=np.float64).reshape(2, 3), np.array([10.0, 20.0, 30.0])),
        (0, None),
        -1,
    )
    _assert_allclose(batched, np.array([[10.0, 13.0], [21.0, 24.0], [32.0, 35.0]]))
    with pytest.raises(ValueError, match="axes must match"):
        elementwise_primitives._program_ad_elementwise_batching_rule(unary, (np.ones(1),), (), 0)
    with pytest.raises(ValueError, match="one or two operands"):
        elementwise_primitives._program_ad_elementwise_batching_rule(
            unary, (np.ones(1), np.ones(1), np.ones(1)), (0, 0, 0), 0
        )
    with pytest.raises(ValueError, match="real numeric"):
        elementwise_primitives._program_ad_elementwise_batching_rule(
            unary, (np.array(["x"]),), (None,), 0
        )
    with pytest.raises(ValueError, match="share one batch size"):
        elementwise_primitives._program_ad_elementwise_batching_rule(
            add, (np.ones((2, 1)), np.ones((3, 1))), (0, 0), 0
        )
    with pytest.raises(ValueError, match="out of bounds"):
        elementwise_primitives._program_ad_elementwise_batching_rule(
            unary, (np.ones((2, 1)),), (2,), 0
        )


def test_program_ad_elementwise_extracted_require_contract_validates_registry_drift() -> None:
    """Runtime contract validation should reject missing and drifted elementwise contracts."""

    identity = PrimitiveIdentity("scpn.program_ad.elementwise", "sin", "1")
    original_contract = primitive_contract_for(identity)
    original_transform = _transform_rule_from_contract(original_contract)
    malformed_contracts = (
        (
            PrimitiveContract(
                identity=identity,
                derivative_rule=original_contract.derivative_rule,
                batching_rule=original_contract.batching_rule,
                lowering_rule=original_contract.lowering_rule,
                lowering_metadata=original_contract.lowering_metadata,
                shape_rule=original_contract.shape_rule,
                dtype_rule=original_contract.dtype_rule,
                static_argument_rule=None,
                nondifferentiable_policy=original_contract.nondifferentiable_policy,
                effect=original_contract.effect,
            ),
            "missing static argument rule",
        ),
        (
            PrimitiveContract(
                identity=identity,
                derivative_rule=original_contract.derivative_rule,
                batching_rule=original_contract.batching_rule,
                lowering_rule=original_contract.lowering_rule,
                lowering_metadata=original_contract.lowering_metadata,
                shape_rule=None,
                dtype_rule=original_contract.dtype_rule,
                static_argument_rule=original_contract.static_argument_rule,
                nondifferentiable_policy=original_contract.nondifferentiable_policy,
                effect=original_contract.effect,
            ),
            "missing shape rule",
        ),
        (
            PrimitiveContract(
                identity=identity,
                derivative_rule=original_contract.derivative_rule,
                batching_rule=original_contract.batching_rule,
                lowering_rule=original_contract.lowering_rule,
                lowering_metadata=original_contract.lowering_metadata,
                shape_rule=original_contract.shape_rule,
                dtype_rule=None,
                static_argument_rule=original_contract.static_argument_rule,
                nondifferentiable_policy=original_contract.nondifferentiable_policy,
                effect=original_contract.effect,
            ),
            "missing dtype rule",
        ),
    )

    try:
        for malformed_contract, message in malformed_contracts:
            with pytest.raises(ValueError, match=message):
                elementwise_primitives._validate_program_ad_elementwise_contract_dispatch(
                    malformed_contract, (np.ones(1),)
                )
        for malformed_contract in (
            PrimitiveContract(
                identity=identity,
                derivative_rule=original_contract.derivative_rule,
                batching_rule=original_contract.batching_rule,
                lowering_rule=original_contract.lowering_rule,
                lowering_metadata=original_contract.lowering_metadata,
                shape_rule=original_contract.shape_rule,
                dtype_rule=original_contract.dtype_rule,
                static_argument_rule=cast(Any, lambda _args: "not-a-tuple"),
                nondifferentiable_policy=original_contract.nondifferentiable_policy,
                effect=original_contract.effect,
            ),
            PrimitiveContract(
                identity=identity,
                derivative_rule=original_contract.derivative_rule,
                batching_rule=original_contract.batching_rule,
                lowering_rule=original_contract.lowering_rule,
                lowering_metadata=original_contract.lowering_metadata,
                shape_rule=lambda _args: (-1,),
                dtype_rule=original_contract.dtype_rule,
                static_argument_rule=original_contract.static_argument_rule,
                nondifferentiable_policy=original_contract.nondifferentiable_policy,
                effect=original_contract.effect,
            ),
            PrimitiveContract(
                identity=identity,
                derivative_rule=original_contract.derivative_rule,
                batching_rule=original_contract.batching_rule,
                lowering_rule=original_contract.lowering_rule,
                lowering_metadata=original_contract.lowering_metadata,
                shape_rule=original_contract.shape_rule,
                dtype_rule=lambda _args: "",
                static_argument_rule=original_contract.static_argument_rule,
                nondifferentiable_policy=original_contract.nondifferentiable_policy,
                effect=original_contract.effect,
            ),
        ):
            with pytest.raises(ValueError):
                elementwise_primitives._validate_program_ad_elementwise_contract_dispatch(
                    malformed_contract, (np.ones(1),)
                )
        with pytest.raises(ValueError, match="identity registered"):
            elementwise_primitives._require_program_ad_elementwise_contract("missing")
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=identity,
                derivative_rule=original_contract.derivative_rule,
                batching_rule=original_contract.batching_rule,
                lowering_metadata=original_contract.lowering_metadata,
                shape_rule=original_contract.shape_rule,
                dtype_rule=original_contract.dtype_rule,
                static_argument_rule=original_contract.static_argument_rule,
                nondifferentiable_policy="wrong",
                effect=original_contract.effect,
            ),
            overwrite=True,
        )
        with pytest.raises(ValueError, match="invalid program AD elementwise primitive policy"):
            elementwise_primitives._require_program_ad_elementwise_contract("sin")
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=identity,
                derivative_rule=original_contract.derivative_rule,
                batching_rule=original_contract.batching_rule,
                lowering_metadata=original_contract.lowering_metadata,
                shape_rule=original_contract.shape_rule,
                dtype_rule=original_contract.dtype_rule,
                static_argument_rule=original_contract.static_argument_rule,
                nondifferentiable_policy=original_contract.nondifferentiable_policy,
                effect="stateful",
            ),
            overwrite=True,
        )
        with pytest.raises(ValueError, match="invalid program AD elementwise primitive effect"):
            elementwise_primitives._require_program_ad_elementwise_contract("sin")
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=identity,
                derivative_rule=original_contract.derivative_rule,
                batching_rule=None,
                lowering_metadata={},
                shape_rule=None,
                dtype_rule=None,
                static_argument_rule=None,
                nondifferentiable_policy=original_contract.nondifferentiable_policy,
                effect=original_contract.effect,
            ),
            overwrite=True,
        )
        with pytest.raises(ValueError, match="missing batching_rule"):
            elementwise_primitives._require_program_ad_elementwise_contract("sin")
    finally:
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(original_transform, overwrite=True)

    elementwise_primitives._register_program_ad_elementwise_primitive_contracts()
    assert (
        elementwise_primitives._require_program_ad_elementwise_contract("sin").identity == identity
    )
    assert (
        elementwise_primitives._require_program_ad_elementwise_contract(
            "sin", (np.ones(1),)
        ).identity
        == identity
    )


def test_program_ad_elementwise_primitives_are_registry_policy_gated() -> None:
    """Unary elementwise math should expose primitive registry contracts."""

    vector = np.array([0.25, 0.5, 0.75], dtype=np.float64)
    for name in (
        "sin",
        "cos",
        "exp",
        "expm1",
        "log",
        "log1p",
        "sqrt",
        "tan",
        "tanh",
        "arcsin",
        "arccos",
        "reciprocal",
        "square",
        "abs",
        "negative",
    ):
        contract = primitive_contract_for(f"scpn.program_ad.elementwise:{name}")
        assert contract.identity == PrimitiveIdentity("scpn.program_ad.elementwise", name, "1")
        assert contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
        assert contract.effect == "pure"
        assert contract.lowering_metadata["mlir_op"] == f"scpn_diff.elementwise.{name}"
        assert contract.lowering_metadata["static_derivative_factory"] == "not_required"
        assert contract.lowering_metadata["static_signature"] == "none"
        assert contract.shape_rule is not None
        assert contract.shape_rule((vector,)) == (3,)
        assert contract.dtype_rule is not None
        assert contract.dtype_rule((vector,)) == "float64"
        assert contract.static_argument_rule is not None
        assert contract.static_argument_rule((vector,)) == ()
        with pytest.raises(ValueError, match="incomplete primitive contract"):
            primitive_complete_contract_for(contract.identity)


def test_program_ad_unary_elementwise_primitives_expose_direct_value_jvp_kernels() -> None:
    """Unary elementwise primitive contracts should expose exact direct value/JVP rules."""

    regular_values = np.array([0.25, 0.5, 0.75], dtype=np.float64)
    tangent = np.array([1.5, -0.25, 0.75], dtype=np.float64)
    positive_values = np.array([1.25, 2.0, 3.5], dtype=np.float64)
    bounded_values = np.array([-0.5, 0.0, 0.5], dtype=np.float64)

    cases = {
        "sin": (regular_values, np.sin(regular_values), np.cos(regular_values) * tangent),
        "cos": (regular_values, np.cos(regular_values), -np.sin(regular_values) * tangent),
        "exp": (regular_values, np.exp(regular_values), np.exp(regular_values) * tangent),
        "expm1": (regular_values, np.expm1(regular_values), np.exp(regular_values) * tangent),
        "log": (positive_values, np.log(positive_values), tangent / positive_values),
        "log1p": (regular_values, np.log1p(regular_values), tangent / (1.0 + regular_values)),
        "sqrt": (
            positive_values,
            np.sqrt(positive_values),
            tangent / (2.0 * np.sqrt(positive_values)),
        ),
        "tan": (regular_values, np.tan(regular_values), tangent / np.cos(regular_values) ** 2),
        "tanh": (
            regular_values,
            np.tanh(regular_values),
            tangent * (1.0 - np.tanh(regular_values) ** 2),
        ),
        "arcsin": (
            bounded_values,
            np.arcsin(bounded_values),
            tangent / np.sqrt(1.0 - bounded_values**2),
        ),
        "arccos": (
            bounded_values,
            np.arccos(bounded_values),
            -tangent / np.sqrt(1.0 - bounded_values**2),
        ),
        "reciprocal": (
            positive_values,
            np.reciprocal(positive_values),
            -tangent / positive_values**2,
        ),
        "square": (regular_values, np.square(regular_values), 2.0 * regular_values * tangent),
        "abs": (positive_values, np.abs(positive_values), np.sign(positive_values) * tangent),
        "negative": (regular_values, np.negative(regular_values), -tangent),
    }

    for name, (values, expected_value, expected_jvp) in cases.items():
        rule = custom_derivative_rule_for(
            PrimitiveIdentity("scpn.program_ad.elementwise", name, "1")
        )
        assert rule.name == f"program_ad_elementwise_{name}_direct_rule"
        assert rule.jvp_rule is not None
        assert rule.vjp_rule is not None
        _assert_allclose(rule.value_fn(values), expected_value)
        _assert_allclose(rule.jvp_rule(values, tangent), expected_jvp)
        _assert_allclose(rule.vjp_rule(values, tangent), expected_jvp)

    with pytest.raises(ValueError, match="undefined at zero"):
        abs_rule = custom_derivative_rule_for(
            PrimitiveIdentity("scpn.program_ad.elementwise", "abs", "1")
        )
        assert abs_rule.jvp_rule is not None
        abs_rule.jvp_rule(np.array([-1.0, 0.0, 1.0]), tangent)


def test_program_ad_elementwise_boundary_metadata_is_explicit() -> None:
    """Elementwise contracts should expose fail-closed mathematical boundaries."""

    expected_boundaries = {
        "log": "positive_domain",
        "log1p": "greater_than_minus_one_domain",
        "sqrt": "nonnegative_domain_with_singular_zero_derivative",
        "arcsin": "closed_unit_interval_with_singular_endpoints",
        "arccos": "closed_unit_interval_with_singular_endpoints",
        "reciprocal": "nonzero_domain",
        "abs": "zero_cusp",
        "divide": "nonzero_denominator",
        "power": "positive_base_for_variable_exponent",
        "maximum": "equal_operand_tie",
        "minimum": "equal_operand_tie",
    }
    for name, boundary in expected_boundaries.items():
        metadata = primitive_contract_for(
            PrimitiveIdentity("scpn.program_ad.elementwise", name, "1")
        ).lowering_metadata
        assert metadata["nondifferentiable_boundary"] == boundary
        assert metadata["nondifferentiable_boundary_policy"] == "fail_closed"


def test_program_ad_discontinuous_elementwise_primitives_fail_closed_by_policy() -> None:
    """Discontinuous elementwise primitives should expose explicit fail-closed policies."""

    expected_boundaries = {
        "sign": "sign_step_derivative_losing_boundary",
        "heaviside": "heaviside_step_derivative_losing_boundary",
    }
    for name, boundary in expected_boundaries.items():
        contract = primitive_contract_for(
            PrimitiveIdentity("scpn.program_ad.elementwise", name, "1")
        )
        assert contract.lowering_metadata["nondifferentiable_boundary"] == boundary
        assert contract.lowering_metadata["nondifferentiable_boundary_policy"] == "fail_closed"
        assert (
            contract.lowering_metadata["static_derivative_factory"] == "blocked_derivative_losing"
        )

    with pytest.raises(ValueError, match="program AD sign is derivative-losing"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.sign(values)),
            np.array([-1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="program AD heaviside is derivative-losing"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.heaviside(values, 0.5)),
            np.array([-1.0, 2.0], dtype=np.float64),
        )


def test_program_ad_elementwise_primitives_validate_registry_rules_at_dispatch() -> None:
    """Supported unary elementwise primitives must execute through registry validation rules."""

    originals = {
        name: primitive_contract_for(f"scpn.program_ad.elementwise:{name}")
        for name in ("sin", "log1p", "sqrt", "reciprocal", "negative")
    }
    calls: dict[str, set[str]] = {name: set() for name in originals}

    for name, original in originals.items():
        assert original.shape_rule is not None
        assert original.dtype_rule is not None
        assert original.static_argument_rule is not None
        original_shape_rule = original.shape_rule
        original_dtype_rule = original.dtype_rule
        original_static_argument_rule = original.static_argument_rule

        def shape_rule(
            args: tuple[object, ...],
            *,
            primitive_name: str = name,
            wrapped_rule: Callable[[tuple[object, ...]], tuple[int, ...]] = original_shape_rule,
        ) -> tuple[int, ...]:
            calls[primitive_name].add("shape")
            return wrapped_rule(args)

        def dtype_rule(
            args: tuple[object, ...],
            *,
            primitive_name: str = name,
            wrapped_rule: Callable[[tuple[object, ...]], str] = original_dtype_rule,
        ) -> str:
            calls[primitive_name].add("dtype")
            return wrapped_rule(args)

        def static_argument_rule(
            args: tuple[object, ...],
            *,
            primitive_name: str = name,
            wrapped_rule: Callable[
                [tuple[object, ...]], tuple[object, ...]
            ] = original_static_argument_rule,
        ) -> tuple[object, ...]:
            calls[primitive_name].add("static")
            return wrapped_rule(args)

        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=original.identity,
                derivative_rule=original.derivative_rule,
                batching_rule=original.batching_rule,
                lowering_rule=original.lowering_rule,
                lowering_metadata=original.lowering_metadata,
                shape_rule=shape_rule,
                dtype_rule=dtype_rule,
                static_argument_rule=static_argument_rule,
                nondifferentiable_policy=original.nondifferentiable_policy,
                effect=original.effect,
            ),
            overwrite=True,
        )
    try:
        result = whole_program_value_and_grad(
            lambda values: np.sum(
                -np.sin(values)
                + np.log1p(values)
                + np.sqrt(values + 4.0)
                + np.reciprocal(values + 2.0)
            ),
            np.array([0.25, 0.5, 0.75], dtype=np.float64),
        )
    finally:
        for original in originals.values():
            DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
                _transform_rule_from_contract(original), overwrite=True
            )

    expected_value = float(
        np.sum(
            -np.sin(np.array([0.25, 0.5, 0.75]))
            + np.log1p(np.array([0.25, 0.5, 0.75]))
            + np.sqrt(np.array([4.25, 4.5, 4.75]))
            + np.reciprocal(np.array([2.25, 2.5, 2.75]))
        )
    )
    assert result.value == pytest.approx(expected_value)
    assert calls == {
        "sin": {"shape", "dtype", "static"},
        "log1p": {"shape", "dtype", "static"},
        "sqrt": {"shape", "dtype", "static"},
        "reciprocal": {"shape", "dtype", "static"},
        "negative": {"shape", "dtype", "static"},
    }


def test_program_ad_elementwise_runtime_dispatch_rejects_boundary_metadata_drift() -> None:
    """Runtime dispatch should enforce the same boundary metadata as coverage."""

    original = primitive_contract_for("scpn.program_ad.elementwise:sin")
    lowering_metadata = {
        key: value
        for key, value in original.lowering_metadata.items()
        if key not in {"nondifferentiable_boundary", "nondifferentiable_boundary_policy"}
    }
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
        PrimitiveTransformRule(
            identity=original.identity,
            derivative_rule=original.derivative_rule,
            batching_rule=original.batching_rule,
            lowering_rule=original.lowering_rule,
            lowering_metadata=lowering_metadata,
            shape_rule=original.shape_rule,
            dtype_rule=original.dtype_rule,
            static_argument_rule=original.static_argument_rule,
            nondifferentiable_policy=original.nondifferentiable_policy,
            effect=original.effect,
        ),
        overwrite=True,
    )
    try:
        with pytest.raises(ValueError, match="missing nondifferentiable_boundary"):
            whole_program_value_and_grad(
                lambda values: np.sum(np.sin(values)),
                np.array([0.25, 0.5], dtype=np.float64),
            )
    finally:
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            _transform_rule_from_contract(original), overwrite=True
        )
