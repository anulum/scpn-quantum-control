# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX Gradient Integration Tests
"""Integration tests for JAX gradient routes through the public bridge."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _phase_jax_bridge_test_helpers import (
    _FakeJAX,
    _Float32JNP,
    _objective,
)
from numpy.typing import NDArray

import scpn_quantum_control.phase.jax_bridge as jax_bridge
from scpn_quantum_control.phase import (
    PhaseJAXCustomVJPQNNGradientResult,
    PhaseJAXGradientAgreementResult,
    PhaseJAXNativeQNNGradientResult,
    PhaseJAXParameterShiftResult,
    check_jax_parameter_shift_agreement,
    is_phase_jax_available,
    jax_custom_vjp_qnn_value_and_grad,
    jax_native_qnn_value_and_grad,
    jax_parameter_shift_value_and_grad,
    multi_frequency_parameter_shift_rule,
    parameter_shift_qnn_classifier_gradient,
)


def test_phase_jax_bridge_parameter_shift_matches_closed_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jax = _FakeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    result = jax_parameter_shift_value_and_grad(
        _objective,
        np.array([0.2, -0.4], dtype=float),
    )

    assert isinstance(result, PhaseJAXParameterShiftResult)
    assert is_phase_jax_available()
    assert result.method == "parameter_shift"
    assert result.evaluations == 5
    assert not result.jitted
    assert not result.host_callback
    np.testing.assert_allclose(
        result.gradient,
        np.array([-np.sin(0.2), 0.25 * np.cos(-0.4)], dtype=float),
        atol=1e-12,
    )


def test_phase_jax_bridge_jit_uses_pure_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_jax = _FakeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    result = jax_parameter_shift_value_and_grad(
        _objective,
        np.array([0.2, -0.4], dtype=float),
        jit=True,
    )

    assert result.jit_requested
    assert result.jitted
    assert result.host_callback
    assert fake_jax.jit_calls == 1
    assert fake_jax.callback_calls == 1
    np.testing.assert_allclose(
        result.gradient,
        np.array([-np.sin(0.2), 0.25 * np.cos(-0.4)], dtype=float),
        atol=1e-12,
    )


def test_phase_jax_bridge_jit_uses_active_jax_callback_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jax = _FakeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, _Float32JNP))

    result = jax_parameter_shift_value_and_grad(
        _objective,
        np.array([0.2, -0.4], dtype=np.float64),
        jit=True,
    )

    value_shape, gradient_shape = fake_jax.callback_shape_dtypes
    assert result.jitted
    assert value_shape.dtype == np.dtype(np.float32)
    assert gradient_shape.dtype == np.dtype(np.float32)


def test_phase_jax_bridge_supports_multi_frequency_parameter_shift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jax = _FakeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])

    def objective(values: NDArray[np.float64]) -> float:
        return float(np.sin(values[0]) + 0.1 * np.cos(2.0 * values[0]))

    result = jax_parameter_shift_value_and_grad(
        objective,
        np.array([0.4], dtype=float),
        rule=rule,
        jit=True,
    )

    expected = np.array([np.cos(0.4) - 0.2 * np.sin(0.8)], dtype=float)
    assert result.method == "multi_frequency_parameter_shift"
    assert result.shift_terms == len(rule.terms)
    assert result.evaluations == 1 + 2 * len(rule.terms)
    assert result.host_callback
    np.testing.assert_allclose(result.gradient, expected, atol=1e-12)
    assert result.to_dict()["shift_terms"] == len(rule.terms)


def test_phase_jax_bridge_reports_gradient_agreement(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_jax = _FakeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    def jax_gradient(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([-np.sin(values[0]), 0.25 * np.cos(values[1])], dtype=float)

    result = check_jax_parameter_shift_agreement(
        _objective,
        jax_gradient,
        np.array([0.2, -0.4], dtype=float),
        tolerance=1e-12,
    )

    assert isinstance(result, PhaseJAXGradientAgreementResult)
    assert result.passed
    assert result.max_abs_error <= 1e-12
    assert result.evaluations == 5
    np.testing.assert_allclose(result.scpn_gradient, result.jax_gradient, atol=1e-12)
    assert result.to_dict()["passed"] is True


def test_phase_jax_bridge_reports_multi_frequency_gradient_agreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jax = _FakeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])

    def objective(values: NDArray[np.float64]) -> float:
        return float(np.sin(values[0]) + 0.1 * np.cos(2.0 * values[0]))

    def jax_gradient(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([np.cos(values[0]) - 0.2 * np.sin(2.0 * values[0])], dtype=float)

    result = check_jax_parameter_shift_agreement(
        objective,
        jax_gradient,
        np.array([0.4], dtype=float),
        tolerance=1e-12,
        rule=rule,
    )

    assert result.passed
    assert result.method == "multi_frequency_parameter_shift"
    assert result.shift_terms == len(rule.terms)
    assert result.evaluations == 1 + 2 * len(rule.terms)
    np.testing.assert_allclose(result.scpn_gradient, result.jax_gradient, atol=1e-12)


def test_phase_jax_bridge_reports_gradient_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_jax = _FakeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    def shifted_gradient(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([-np.sin(values[0]) + 0.01, 0.25 * np.cos(values[1])], dtype=float)

    result = check_jax_parameter_shift_agreement(
        _objective,
        shifted_gradient,
        np.array([0.2, -0.4], dtype=float),
        tolerance=1e-4,
    )

    assert not result.passed
    assert result.max_abs_error > result.tolerance
    assert result.l2_error > 0.0


def test_phase_jax_native_qnn_autodiff_agrees_with_parameter_shift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jax = _FakeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params = np.array([0.45], dtype=float)

    result = jax_native_qnn_value_and_grad(
        features,
        labels,
        params,
        tolerance=1e-5,
    )

    expected = parameter_shift_qnn_classifier_gradient(features, labels, params)
    assert isinstance(result, PhaseJAXNativeQNNGradientResult)
    assert result.passed
    assert result.native_framework_autodiff
    assert not result.host_callback
    assert not result.jitted
    np.testing.assert_allclose(result.gradient, expected, atol=1e-5)
    np.testing.assert_allclose(result.parameter_shift_gradient, expected, atol=1e-12)


def test_phase_jax_native_qnn_jit_records_native_no_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jax = _FakeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    result = jax_native_qnn_value_and_grad(
        np.array([[0.0], [np.pi]], dtype=float),
        np.array([0.0, 1.0], dtype=float),
        np.array([0.45], dtype=float),
        tolerance=1e-5,
        jit=True,
    )

    assert result.jit_requested
    assert result.jitted
    assert result.native_framework_autodiff
    assert not result.host_callback
    assert fake_jax.jit_calls == 1
    assert fake_jax.callback_calls == 0


def test_phase_jax_custom_vjp_qnn_uses_parameter_shift_backward_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jax = _FakeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params = np.array([0.45], dtype=float)

    result = jax_custom_vjp_qnn_value_and_grad(
        features,
        labels,
        params,
        tolerance=1e-12,
    )

    expected = parameter_shift_qnn_classifier_gradient(features, labels, params)
    assert isinstance(result, PhaseJAXCustomVJPQNNGradientResult)
    assert result.passed
    assert result.custom_vjp
    assert result.native_framework_autodiff
    assert not result.host_callback
    assert not result.jitted
    assert result.method == "jax_custom_vjp_bounded_phase_qnn_value_and_grad"
    assert fake_jax.custom_vjp_calls == 1
    assert fake_jax.custom_vjp_defvjp_calls == 1
    np.testing.assert_allclose(result.gradient, expected, atol=1e-12)
    np.testing.assert_allclose(result.parameter_shift_gradient, expected, atol=1e-12)
    assert result.to_dict()["custom_vjp"] is True


def test_phase_jax_custom_vjp_qnn_jit_keeps_native_no_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jax = _FakeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    result = jax_custom_vjp_qnn_value_and_grad(
        np.array([[0.0], [np.pi]], dtype=float),
        np.array([0.0, 1.0], dtype=float),
        np.array([0.45], dtype=float),
        tolerance=1e-12,
        jit=True,
    )

    assert result.jit_requested
    assert result.jitted
    assert result.custom_vjp
    assert not result.host_callback
    assert fake_jax.jit_calls == 1
    assert fake_jax.callback_calls == 0


def test_phase_jax_custom_vjp_qnn_fails_closed_without_custom_vjp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NoCustomVJPJAX(_FakeJAX):
        custom_vjp: Any = None

    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (_NoCustomVJPJAX(), np))

    with pytest.raises(RuntimeError, match="custom_vjp"):
        jax_custom_vjp_qnn_value_and_grad(
            np.array([[0.0], [np.pi]], dtype=float),
            np.array([0.0, 1.0], dtype=float),
            np.array([0.45], dtype=float),
        )


def test_phase_jax_native_qnn_fails_closed_on_shape_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jax = _FakeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    with pytest.raises(ValueError, match="params must have shape"):
        jax_native_qnn_value_and_grad(
            np.array([[0.0, 1.0]], dtype=float),
            np.array([0.0], dtype=float),
            np.array([0.45], dtype=float),
        )
