# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD signal primitive tests
"""Tests for Program AD convolution and correlation primitive semantics."""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable import (
    Parameter,
    PrimitiveIdentity,
    primitive_complete_contract_for,
    primitive_contract_for,
    program_ad_signal_convolve_derivative_rule,
    program_ad_signal_correlate_derivative_rule,
    program_adjoint_gradient,
    whole_program_value_and_grad,
)

FloatArray = NDArray[np.float64]
SignalMode = Literal["valid", "same", "full"]


def _assert_allclose(
    actual: object, expected: object, *, rtol: float = 1.0e-7, atol: float = 0.0
) -> None:
    """Assert NumPy closeness across dynamically typed Program AD result payloads."""

    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def test_program_ad_signal_direct_factories_have_dedicated_module_path() -> None:
    """Signal direct-rule factories should be available outside the facade."""

    from scpn_quantum_control import differentiable as differentiable_facade
    from scpn_quantum_control import program_ad_signal_primitives

    assert (
        program_ad_signal_primitives.program_ad_signal_convolve_derivative_rule
        is program_ad_signal_convolve_derivative_rule
    )
    assert (
        differentiable_facade.program_ad_signal_convolve_derivative_rule
        is program_ad_signal_convolve_derivative_rule
    )
    assert (
        program_ad_signal_primitives.program_ad_signal_correlate_derivative_rule
        is program_ad_signal_correlate_derivative_rule
    )
    assert (
        differentiable_facade.program_ad_signal_correlate_derivative_rule
        is program_ad_signal_correlate_derivative_rule
    )
    facade_exports = vars(differentiable_facade)
    assert (
        facade_exports["_require_program_ad_signal_contract"]
        is program_ad_signal_primitives._require_program_ad_signal_contract
    )
    assert (
        facade_exports["_register_program_ad_signal_primitive_contracts"]
        is program_ad_signal_primitives._register_program_ad_signal_primitive_contracts
    )


def test_program_ad_convolve_matches_signal_kernel_adjoint() -> None:
    """Program AD np.convolve should replay exact signal and kernel adjoints."""

    static_kernel = np.array([0.75, -1.25, 0.5], dtype=np.float64)
    static_signal = np.array([1.5, -0.5, 2.0, 0.25], dtype=np.float64)
    full_weights = np.array([0.2, -0.4, 0.6, -0.8, 1.0, -1.2], dtype=np.float64)
    same_weights = np.array([-0.5, 0.25, 1.25, -0.75], dtype=np.float64)
    valid_weights = np.array([1.4, -0.9], dtype=np.float64)

    def objective(values: Any) -> object:
        signal = values[:4]
        kernel = values[4:]
        return (
            np.sum(np.convolve(signal, kernel, mode="full") * full_weights)
            + 0.31 * np.sum(np.convolve(signal, static_kernel, mode="same") * same_weights)
            - 0.17 * np.sum(np.convolve(static_signal, kernel, mode="valid") * valid_weights)
        )

    values = np.array([1.0, -2.0, 0.5, 3.0, -1.5, 2.0, 0.75], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )

    expected = np.zeros_like(values)
    signal = values[:4]
    kernel = values[4:]
    for source_index in range(4):
        basis_signal = np.zeros(4, dtype=np.float64)
        basis_signal[source_index] = 1.0
        expected[source_index] = np.sum(
            np.convolve(basis_signal, kernel, mode="full") * full_weights
        ) + 0.31 * np.sum(np.convolve(basis_signal, static_kernel, mode="same") * same_weights)
    for kernel_index in range(3):
        basis_kernel = np.zeros(3, dtype=np.float64)
        basis_kernel[kernel_index] = 1.0
        expected[4 + kernel_index] = np.sum(
            np.convolve(signal, basis_kernel, mode="full") * full_weights
        ) - 0.17 * np.sum(np.convolve(static_signal, basis_kernel, mode="valid") * valid_weights)

    assert result.value == pytest.approx(float(cast(Any, objective(values))))
    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, atol=1.0e-12)


def test_program_ad_signal_convolve_contract_and_direct_rule() -> None:
    """np.convolve should expose a fail-closed signal primitive direct rule."""

    left = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float64)
    right = np.array([0.25, -0.75, 1.5], dtype=np.float64)
    tangent_left = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float64)
    tangent_right = np.array([-0.5, 0.25, 0.75], dtype=np.float64)
    cotangent = np.array([0.2, -0.4, 0.6, -0.8], dtype=np.float64)
    values = np.concatenate([left, right])
    tangent = np.concatenate([tangent_left, tangent_right])

    contract = primitive_contract_for("scpn.program_ad.signal:convolve")
    assert contract.identity == PrimitiveIdentity("scpn.program_ad.signal", "convolve", "1")
    assert contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert contract.effect == "pure"
    assert contract.lowering_metadata["mlir_op"] == "scpn_diff.signal.convolve"
    assert (
        contract.lowering_metadata["static_derivative_factory"]
        == "program_ad_signal_convolve_derivative_rule"
    )
    assert (
        contract.lowering_metadata["static_signature"] == "left_shape:rank1;right_shape:rank1;mode"
    )
    assert contract.lowering_metadata["nondifferentiable_boundary"] == (
        "rank1_nonempty_static_mode_window"
    )
    assert contract.lowering_metadata["nondifferentiable_boundary_policy"] == "fail_closed"
    assert contract.shape_rule is not None
    assert (
        contract.shape_rule((left, right, "same")) == np.convolve(left, right, mode="same").shape
    )
    assert contract.dtype_rule is not None
    assert contract.dtype_rule((left, right, "same")) == "float64"
    assert contract.static_argument_rule is not None
    assert contract.static_argument_rule((left, right, "same")) == ((4,), (3,), "same")
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(contract.identity)

    rule = program_ad_signal_convolve_derivative_rule(left.shape, right.shape, mode="same")
    assert rule.name == "program_ad_signal_convolve_left4_right3_mode_same_direct_rule"
    _assert_allclose(rule.value_fn(values), np.convolve(left, right, mode="same"))
    assert rule.jvp_rule is not None
    jvp_rule = rule.jvp_rule
    _assert_allclose(
        jvp_rule(values, tangent),
        np.convolve(tangent_left, right, mode="same")
        + np.convolve(left, tangent_right, mode="same"),
    )

    expected_vjp = np.zeros(values.size, dtype=np.float64)
    for source_index in range(values.size):
        basis_left = np.zeros(left.size, dtype=np.float64)
        basis_right = np.zeros(right.size, dtype=np.float64)
        if source_index < left.size:
            basis_left[source_index] = 1.0
        else:
            basis_right[source_index - left.size] = 1.0
        expected_vjp[source_index] = np.sum(
            (
                np.convolve(basis_left, right, mode="same")
                + np.convolve(left, basis_right, mode="same")
            )
            * cotangent
        )
    assert rule.vjp_rule is not None
    vjp_rule = rule.vjp_rule
    _assert_allclose(vjp_rule(values, cotangent), expected_vjp)


def test_program_ad_signal_convolve_batching_rule_maps_outer_axis() -> None:
    """Signal convolve batching should map left operands with static kernels."""

    contract = primitive_contract_for("scpn.program_ad.signal:convolve")
    assert contract.batching_rule is not None

    def convolve_fn(left: FloatArray, right: FloatArray, mode: SignalMode) -> FloatArray:
        return cast(FloatArray, cast(Any, np.convolve)(left, right, mode=mode))

    left_batch = np.array(
        [[1.0, -2.0, 0.5, 3.0], [-1.5, 0.25, 2.0, -0.75]],
        dtype=np.float64,
    )
    right = np.array([0.25, -0.75, 1.5], dtype=np.float64)
    expected = np.stack(
        [np.convolve(row, right, mode="valid") for row in left_batch],
        axis=0,
    )

    _assert_allclose(
        contract.batching_rule(
            convolve_fn,
            (left_batch, right, "valid"),
            (0, None, None),
            0,
        ),
        expected,
    )
    _assert_allclose(
        contract.batching_rule(
            convolve_fn,
            (left_batch, right, "valid"),
            (0, None, None),
            1,
        ),
        expected.T,
    )

    with pytest.raises(ValueError, match="keeps right operand and mode static"):
        contract.batching_rule(
            convolve_fn,
            (left_batch, right, "valid"),
            (0, 0, None),
            0,
        )


def test_program_ad_convolve_fails_closed_invalid_contracts() -> None:
    """Program AD np.convolve should reject invalid rank, mode, and empty operands."""

    with pytest.raises(ValueError, match="mode must be"):
        whole_program_value_and_grad(
            lambda values: np.sum(cast(Any, np.convolve)(values, np.array([1.0]), mode=True)),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="rank-1 operands"):
        whole_program_value_and_grad(
            lambda values: np.sum(
                np.convolve(np.reshape(values[:4], (2, 2)), np.array([1.0, -0.5]))
            ),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="non-empty"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.convolve(values[:0], np.array([1.0]))),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )


def test_program_ad_signal_convolve_direct_rule_fails_closed_invalid_static_contracts() -> None:
    """Convolve direct rules should reject invalid static signatures and payloads."""

    with pytest.raises(ValueError, match="rank-1 operands"):
        program_ad_signal_convolve_derivative_rule((2, 2), (3,), mode="full")
    with pytest.raises(ValueError, match="non-empty operands"):
        program_ad_signal_convolve_derivative_rule((0,), (3,), mode="full")

    rule = program_ad_signal_convolve_derivative_rule((4,), (3,), mode="same")
    with pytest.raises(ValueError, match="requires 7 values"):
        rule.value_fn(np.array([1.0, 2.0], dtype=np.float64))
    assert rule.vjp_rule is not None
    with pytest.raises(ValueError, match="cotangent matching output size"):
        rule.vjp_rule(
            np.arange(7.0, dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
        )


def test_program_ad_correlate_matches_signal_reference_adjoint() -> None:
    """Program AD np.correlate should replay exact signal and reference adjoints."""

    static_reference = np.array([0.5, -1.0, 0.25], dtype=np.float64)
    static_signal = np.array([1.25, -0.75, 2.5, 0.5], dtype=np.float64)
    full_weights = np.array([-0.3, 0.55, -0.8, 1.05, -1.3, 1.55], dtype=np.float64)
    same_weights = np.array([0.4, -0.2, 1.1, -0.9], dtype=np.float64)
    valid_weights = np.array([-1.2, 0.65], dtype=np.float64)

    def objective(values: Any) -> object:
        signal = values[:4]
        reference = values[4:]
        return (
            np.sum(np.correlate(signal, reference, mode="full") * full_weights)
            - 0.23 * np.sum(np.correlate(signal, static_reference, mode="same") * same_weights)
            + 0.19 * np.sum(np.correlate(static_signal, reference, mode="valid") * valid_weights)
        )

    values = np.array([0.5, -1.5, 2.0, -0.25, 1.25, -2.0, 0.75], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )

    expected = np.zeros_like(values)
    signal = values[:4]
    reference = values[4:]
    for source_index in range(4):
        basis_signal = np.zeros(4, dtype=np.float64)
        basis_signal[source_index] = 1.0
        expected[source_index] = np.sum(
            np.correlate(basis_signal, reference, mode="full") * full_weights
        ) - 0.23 * np.sum(np.correlate(basis_signal, static_reference, mode="same") * same_weights)
    for reference_index in range(3):
        basis_reference = np.zeros(3, dtype=np.float64)
        basis_reference[reference_index] = 1.0
        expected[4 + reference_index] = np.sum(
            np.correlate(signal, basis_reference, mode="full") * full_weights
        ) + 0.19 * np.sum(
            np.correlate(static_signal, basis_reference, mode="valid") * valid_weights
        )

    assert result.value == pytest.approx(float(cast(Any, objective(values))))
    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, atol=1.0e-12)


def test_program_ad_signal_correlate_contract_and_direct_rule() -> None:
    """np.correlate should expose a fail-closed signal primitive direct rule."""

    left = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float64)
    right = np.array([0.25, -0.75, 1.5], dtype=np.float64)
    tangent_left = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float64)
    tangent_right = np.array([-0.5, 0.25, 0.75], dtype=np.float64)
    cotangent = np.array([0.2, -0.4, 0.6, -0.8], dtype=np.float64)
    values = np.concatenate([left, right])
    tangent = np.concatenate([tangent_left, tangent_right])

    contract = primitive_contract_for("scpn.program_ad.signal:correlate")
    assert contract.identity == PrimitiveIdentity("scpn.program_ad.signal", "correlate", "1")
    assert contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert contract.effect == "pure"
    assert contract.lowering_metadata["mlir_op"] == "scpn_diff.signal.correlate"
    assert (
        contract.lowering_metadata["static_derivative_factory"]
        == "program_ad_signal_correlate_derivative_rule"
    )
    assert (
        contract.lowering_metadata["static_signature"] == "left_shape:rank1;right_shape:rank1;mode"
    )
    assert contract.lowering_metadata["nondifferentiable_boundary"] == (
        "rank1_nonempty_static_mode_window"
    )
    assert contract.lowering_metadata["nondifferentiable_boundary_policy"] == "fail_closed"
    assert contract.shape_rule is not None
    assert (
        contract.shape_rule((left, right, "same")) == np.correlate(left, right, mode="same").shape
    )
    assert contract.dtype_rule is not None
    assert contract.dtype_rule((left, right, "same")) == "float64"
    assert contract.static_argument_rule is not None
    assert contract.static_argument_rule((left, right, "same")) == ((4,), (3,), "same")
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(contract.identity)

    rule = program_ad_signal_correlate_derivative_rule(left.shape, right.shape, mode="same")
    assert rule.name == "program_ad_signal_correlate_left4_right3_mode_same_direct_rule"
    _assert_allclose(rule.value_fn(values), np.correlate(left, right, mode="same"))
    assert rule.jvp_rule is not None
    jvp_rule = rule.jvp_rule
    _assert_allclose(
        jvp_rule(values, tangent),
        np.correlate(tangent_left, right, mode="same")
        + np.correlate(left, tangent_right, mode="same"),
    )

    expected_vjp = np.zeros(values.size, dtype=np.float64)
    for source_index in range(values.size):
        basis_left = np.zeros(left.size, dtype=np.float64)
        basis_right = np.zeros(right.size, dtype=np.float64)
        if source_index < left.size:
            basis_left[source_index] = 1.0
        else:
            basis_right[source_index - left.size] = 1.0
        expected_vjp[source_index] = np.sum(
            (
                np.correlate(basis_left, right, mode="same")
                + np.correlate(left, basis_right, mode="same")
            )
            * cotangent
        )
    assert rule.vjp_rule is not None
    vjp_rule = rule.vjp_rule
    _assert_allclose(vjp_rule(values, cotangent), expected_vjp)


def test_program_ad_signal_correlate_batching_rule_maps_outer_axis() -> None:
    """Signal correlate batching should map left operands with static references."""

    contract = primitive_contract_for("scpn.program_ad.signal:correlate")
    assert contract.batching_rule is not None

    def correlate_fn(left: FloatArray, right: FloatArray, mode: SignalMode) -> FloatArray:
        return cast(FloatArray, cast(Any, np.correlate)(left, right, mode=mode))

    left_batch = np.array(
        [[1.0, -2.0, 0.5, 3.0], [-1.5, 0.25, 2.0, -0.75]],
        dtype=np.float64,
    )
    right = np.array([0.25, -0.75, 1.5], dtype=np.float64)
    expected = np.stack(
        [np.correlate(row, right, mode="valid") for row in left_batch],
        axis=0,
    )

    _assert_allclose(
        contract.batching_rule(
            correlate_fn,
            (left_batch, right, "valid"),
            (0, None, None),
            0,
        ),
        expected,
    )
    _assert_allclose(
        contract.batching_rule(
            correlate_fn,
            (left_batch, right, "valid"),
            (0, None, None),
            1,
        ),
        expected.T,
    )

    with pytest.raises(ValueError, match="keeps right operand and mode static"):
        contract.batching_rule(
            correlate_fn,
            (left_batch, right, "valid"),
            (0, 0, None),
            0,
        )


def test_program_ad_correlate_fails_closed_invalid_contracts() -> None:
    """Program AD np.correlate should reject invalid rank, mode, and empty operands."""

    with pytest.raises(ValueError, match="mode must be"):
        whole_program_value_and_grad(
            lambda values: np.sum(cast(Any, np.correlate)(values, np.array([1.0]), mode=True)),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="rank-1 operands"):
        whole_program_value_and_grad(
            lambda values: np.sum(
                np.correlate(np.reshape(values[:4], (2, 2)), np.array([1.0, -0.5]))
            ),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="non-empty"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.correlate(values[:0], np.array([1.0]))),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )


def test_program_ad_signal_correlate_direct_rule_fails_closed_invalid_static_contracts() -> None:
    """Correlate direct rules should reject invalid static signatures and payloads."""

    with pytest.raises(ValueError, match="rank-1 operands"):
        program_ad_signal_correlate_derivative_rule((2, 2), (3,), mode="full")
    with pytest.raises(ValueError, match="non-empty operands"):
        program_ad_signal_correlate_derivative_rule((0,), (3,), mode="full")

    rule = program_ad_signal_correlate_derivative_rule((4,), (3,), mode="same")
    with pytest.raises(ValueError, match="requires 7 values"):
        rule.value_fn(np.array([1.0, 2.0], dtype=np.float64))
    assert rule.vjp_rule is not None
    with pytest.raises(ValueError, match="cotangent matching output size"):
        rule.vjp_rule(
            np.arange(7.0, dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
        )
