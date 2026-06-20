# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD elementwise registry tests
"""Tests for Program AD elementwise registry contracts and direct rules."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

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
