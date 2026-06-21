# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD binary elementwise registry tests
"""Tests for Program AD binary elementwise contracts and direct rules."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control.program_ad_elementwise_primitives as elementwise_primitives
from scpn_quantum_control.differentiable import (
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    PrimitiveContract,
    PrimitiveIdentity,
    PrimitiveTransformRule,
    custom_derivative_rule_for,
    primitive_complete_contract_for,
    primitive_contract_for,
    program_ad_elementwise_binary_derivative_rule,
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


def test_program_ad_elementwise_binary_derivative_rule_is_extracted_identity() -> None:
    """The facade should re-export the extracted binary elementwise factory."""

    assert (
        program_ad_elementwise_binary_derivative_rule
        is elementwise_primitives.program_ad_elementwise_binary_derivative_rule
    )


def test_program_ad_binary_elementwise_primitives_are_registry_policy_gated() -> None:
    """Binary elementwise math should expose broadcast-aware primitive contracts."""

    left = np.arange(6.0, dtype=np.float64).reshape(2, 3)
    right = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    for name in ("add", "subtract", "multiply", "divide", "power", "maximum", "minimum"):
        contract = primitive_contract_for(f"scpn.program_ad.elementwise:{name}")
        assert contract.identity == PrimitiveIdentity("scpn.program_ad.elementwise", name, "1")
        assert contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
        assert contract.effect == "pure"
        assert contract.lowering_metadata["mlir_op"] == f"scpn_diff.elementwise.{name}"
        assert contract.shape_rule is not None
        assert contract.shape_rule((left, right)) == (2, 3)
        assert contract.dtype_rule is not None
        assert contract.dtype_rule((left, right)) == "float64"
        assert contract.static_argument_rule is not None
        assert contract.static_argument_rule((left, right)) == ()
        with pytest.raises(ValueError, match="incomplete primitive contract"):
            primitive_complete_contract_for(contract.identity)


def test_program_ad_binary_elementwise_primitives_expose_direct_value_jvp_kernels() -> None:
    """Binary elementwise primitive contracts should expose exact direct value/JVP rules."""

    left = np.array([3.0, 1.5, 4.0], dtype=np.float64)
    right = np.array([2.0, 5.0, 0.5], dtype=np.float64)
    tangent_left = np.array([0.25, -0.5, 1.25], dtype=np.float64)
    tangent_right = np.array([-1.0, 0.75, 0.5], dtype=np.float64)
    values = np.concatenate([left, right])
    tangent = np.concatenate([tangent_left, tangent_right])

    cases = {
        "add": (left + right, tangent_left + tangent_right),
        "subtract": (left - right, tangent_left - tangent_right),
        "multiply": (left * right, tangent_left * right + left * tangent_right),
        "divide": (left / right, (tangent_left * right - left * tangent_right) / right**2),
        "power": (
            left**right,
            left**right * (tangent_right * np.log(left) + right * tangent_left / left),
        ),
        "maximum": (np.maximum(left, right), np.where(left > right, tangent_left, tangent_right)),
        "minimum": (np.minimum(left, right), np.where(left < right, tangent_left, tangent_right)),
    }

    for name, (expected_value, expected_jvp) in cases.items():
        rule = custom_derivative_rule_for(
            PrimitiveIdentity("scpn.program_ad.elementwise", name, "1")
        )
        assert rule.name == f"program_ad_elementwise_{name}_direct_rule"
        assert rule.jvp_rule is not None
        assert rule.vjp_rule is not None
        _assert_allclose(rule.value_fn(values), expected_value)
        _assert_allclose(rule.jvp_rule(values, tangent), expected_jvp)
        cotangent = np.array([1.25, -0.5, 0.75], dtype=np.float64)
        if name == "add":
            expected_vjp = np.concatenate([cotangent, cotangent])
        elif name == "subtract":
            expected_vjp = np.concatenate([cotangent, -cotangent])
        elif name == "multiply":
            expected_vjp = np.concatenate([cotangent * right, cotangent * left])
        elif name == "divide":
            expected_vjp = np.concatenate([cotangent / right, -cotangent * left / right**2])
        elif name == "power":
            expected_vjp = np.concatenate(
                [
                    cotangent * right * left ** (right - 1.0),
                    cotangent * left**right * np.log(left),
                ]
            )
        elif name == "maximum":
            expected_vjp = np.concatenate(
                [np.where(left > right, cotangent, 0.0), np.where(left > right, 0.0, cotangent)]
            )
        else:
            expected_vjp = np.concatenate(
                [np.where(left < right, cotangent, 0.0), np.where(left < right, 0.0, cotangent)]
            )
        _assert_allclose(rule.vjp_rule(values, cotangent), expected_vjp)

    maximum_rule = custom_derivative_rule_for(
        PrimitiveIdentity("scpn.program_ad.elementwise", "maximum", "1")
    )
    assert maximum_rule.jvp_rule is not None
    with pytest.raises(ValueError, match="undefined at equal operands"):
        maximum_rule.jvp_rule(np.concatenate([left, left]), tangent)

    divide_rule = custom_derivative_rule_for(
        PrimitiveIdentity("scpn.program_ad.elementwise", "divide", "1")
    )
    assert divide_rule.jvp_rule is not None
    with pytest.raises(ValueError, match="non-zero right operand"):
        divide_rule.jvp_rule(np.concatenate([left, np.array([1.0, 0.0, 2.0])]), tangent)

    power_rule = custom_derivative_rule_for(
        PrimitiveIdentity("scpn.program_ad.elementwise", "power", "1")
    )
    assert power_rule.jvp_rule is not None
    with pytest.raises(ValueError, match="positive left operand"):
        power_rule.jvp_rule(np.concatenate([np.array([1.0, 0.0, 2.0]), right]), tangent)


def test_program_ad_binary_elementwise_primitives_validate_registry_rules_at_dispatch() -> None:
    """Supported binary elementwise primitives must execute through registry validation rules."""

    originals = {
        name: primitive_contract_for(f"scpn.program_ad.elementwise:{name}")
        for name in ("add", "multiply", "divide", "power", "maximum", "minimum")
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
                (values + np.array([2.0, 3.0, 4.0]))
                * np.array([1.5, 2.0, 2.5])
                / np.array([2.0, 4.0, 5.0])
                + np.power(values + 2.0, np.array([2.0, 1.5, 1.25]))
                + np.maximum(values, np.array([-1.0, 0.0, 0.5]))
                - np.minimum(values, np.array([-2.0, -1.0, 0.25]))
            ),
            np.array([0.25, 0.5, 0.75], dtype=np.float64),
        )
    finally:
        for original in originals.values():
            DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
                _transform_rule_from_contract(original), overwrite=True
            )

    values = np.array([0.25, 0.5, 0.75], dtype=np.float64)
    expected_value = float(
        np.sum(
            (values + np.array([2.0, 3.0, 4.0]))
            * np.array([1.5, 2.0, 2.5])
            / np.array([2.0, 4.0, 5.0])
            + np.power(values + 2.0, np.array([2.0, 1.5, 1.25]))
            + np.maximum(values, np.array([-1.0, 0.0, 0.5]))
            - np.minimum(values, np.array([-2.0, -1.0, 0.25]))
        )
    )
    assert result.value == pytest.approx(expected_value)
    assert calls == {
        "add": {"shape", "dtype", "static"},
        "multiply": {"shape", "dtype", "static"},
        "divide": {"shape", "dtype", "static"},
        "power": {"shape", "dtype", "static"},
        "maximum": {"shape", "dtype", "static"},
        "minimum": {"shape", "dtype", "static"},
    }


def test_program_ad_elementwise_binary_static_factory_supports_broadcasting() -> None:
    """Static binary elementwise factories should expose exact broadcast-aware VJPs."""

    left = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    right = np.array([0.5, -1.5, 2.0], dtype=np.float64)
    tangent_left = np.array([[0.25, -0.5, 1.0], [1.5, -0.75, 0.5]], dtype=np.float64)
    tangent_right = np.array([-0.25, 0.75, 1.25], dtype=np.float64)
    values = np.concatenate((left.reshape(-1), right.reshape(-1)))
    tangent = np.concatenate((tangent_left.reshape(-1), tangent_right.reshape(-1)))
    cotangent = np.array([[1.0, -0.5, 2.0], [0.25, 1.5, -1.0]], dtype=np.float64)

    multiply_rule = program_ad_elementwise_binary_derivative_rule("multiply", (2, 3), (3,))
    assert multiply_rule.name == "program_ad_elementwise_multiply_2x3_by_3_broadcast_direct_rule"
    assert multiply_rule.jvp_rule is not None
    assert multiply_rule.vjp_rule is not None
    _assert_allclose(multiply_rule.value_fn(values), (left * right).reshape(-1))
    _assert_allclose(
        multiply_rule.jvp_rule(values, tangent),
        (tangent_left * right + left * tangent_right).reshape(-1),
    )
    _assert_allclose(
        multiply_rule.vjp_rule(values, cotangent.reshape(-1)),
        np.concatenate(((cotangent * right).reshape(-1), np.sum(cotangent * left, axis=0))),
    )

    divide_rule = program_ad_elementwise_binary_derivative_rule("divide", (2, 3), (3,))
    assert divide_rule.vjp_rule is not None
    _assert_allclose(divide_rule.value_fn(values), (left / right).reshape(-1))
    _assert_allclose(
        divide_rule.vjp_rule(values, cotangent.reshape(-1)),
        np.concatenate(
            (
                (cotangent / right).reshape(-1),
                np.sum(-cotangent * left / right**2, axis=0),
            )
        ),
    )

    power_left = left + 2.0
    power_right = np.array([1.25, 2.0, 0.5], dtype=np.float64)
    power_values = np.concatenate((power_left.reshape(-1), power_right))
    power_cotangent = cotangent.reshape(-1)
    power_rule = program_ad_elementwise_binary_derivative_rule("power", (2, 3), (3,))
    assert power_rule.vjp_rule is not None
    _assert_allclose(power_rule.value_fn(power_values), (power_left**power_right).reshape(-1))
    _assert_allclose(
        power_rule.vjp_rule(power_values, power_cotangent),
        np.concatenate(
            (
                (cotangent * power_right * power_left ** (power_right - 1.0)).reshape(-1),
                np.sum(cotangent * power_left**power_right * np.log(power_left), axis=0),
            )
        ),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    scalar_add_rule = program_ad_elementwise_binary_derivative_rule("add", (2, 3), ())
    assert scalar_add_rule.vjp_rule is not None
    scalar_values = np.concatenate((left.reshape(-1), np.array([2.5], dtype=np.float64)))
    _assert_allclose(scalar_add_rule.value_fn(scalar_values), (left + 2.5).reshape(-1))
    _assert_allclose(
        scalar_add_rule.vjp_rule(scalar_values, cotangent.reshape(-1)),
        np.concatenate((cotangent.reshape(-1), np.array([np.sum(cotangent)], dtype=np.float64))),
    )

    with pytest.raises(ValueError, match="broadcast"):
        program_ad_elementwise_binary_derivative_rule("multiply", (2, 3), (2,))
    with pytest.raises(ValueError, match="positive left operand"):
        program_ad_elementwise_binary_derivative_rule("power", (2, 3), (3,)).value_fn(
            np.concatenate(((-np.abs(left)).reshape(-1), power_right))
        )
    with pytest.raises(ValueError, match="undefined at equal operands"):
        maximum_rule = program_ad_elementwise_binary_derivative_rule("maximum", (2, 3), (3,))
        assert maximum_rule.vjp_rule is not None
        maximum_rule.vjp_rule(
            np.concatenate((left.reshape(-1), left[0].reshape(-1))),
            cotangent.reshape(-1),
        )
