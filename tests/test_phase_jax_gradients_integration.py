# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX Gradient Integration Tests
"""Integration tests for JAX gradient routes through the public bridge."""

from __future__ import annotations

import sys
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
    """The JAX parameter-shift route matches the closed-form gradient of the objective."""
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
    """Under jit the parameter-shift route runs as a pure host callback."""
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
    """The callback shapes use the dtype of the active JAX arrays."""
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
    """A multi-frequency shift rule is honoured by the JAX route."""
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
    """A gradient callable that matches parameter shift is reported as agreeing."""
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
    """Agreement is also checked under a multi-frequency rule."""
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
    """A gradient callable that differs from parameter shift is reported as a mismatch."""
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
    """Native JAX autodiff of the bounded QNN loss matches parameter shift."""
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
    """The jitted native QNN route records no host callback."""
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
    """The custom-VJP backward rule reproduces the parameter-shift gradient without a host callback."""
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
    """The jitted custom-VJP route stays native, with no host callback."""
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
    """A JAX without custom_vjp is refused by the custom-VJP route."""

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
    """Parameters whose width differs from the feature width are refused by the native route."""
    fake_jax = _FakeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    with pytest.raises(ValueError, match="params must have shape"):
        jax_native_qnn_value_and_grad(
            np.array([[0.0, 1.0]], dtype=float),
            np.array([0.0], dtype=float),
            np.array([0.45], dtype=float),
        )


_FEATURES = np.array([[0.0], [np.pi]], dtype=float)
_LABELS = np.array([0.0, 1.0], dtype=float)
_PARAMS = np.array([0.45], dtype=float)


@pytest.mark.parametrize(
    ("features", "labels", "message"),
    [
        pytest.param(np.array([0.0, np.pi]), _LABELS, "two-dimensional array", id="1d"),
        pytest.param(np.empty((0, 1)), _LABELS[:0], "at least one sample", id="no-sample"),
        pytest.param(np.empty((2, 0)), _LABELS, "at least one feature column", id="no-feature"),
        pytest.param(np.array([[0.0], [np.nan]]), _LABELS, "features must contain only", id="nan"),
        pytest.param(_FEATURES, np.ones((2, 2)), "single-column matrix", id="label-matrix"),
        pytest.param(_FEATURES, np.array([0.0]), "same sample count", id="label-count"),
        pytest.param(_FEATURES, np.array([0.0, np.inf]), "labels must contain only", id="inf"),
        pytest.param(_FEATURES, np.array([0.0, 2.0]), r"closed interval \[0, 1\]", id="range"),
    ],
)
def test_phase_jax_qnn_routes_refuse_malformed_data(
    monkeypatch: pytest.MonkeyPatch, features: Any, labels: Any, message: str
) -> None:
    """Malformed features or labels are refused before any JAX transform is built."""
    fake_jax = _FakeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    with pytest.raises(ValueError, match=message):
        jax_native_qnn_value_and_grad(
            features, labels, np.zeros(max(1, features.shape[-1])), jit=True
        )
    assert fake_jax.jit_calls == 0


def test_phase_jax_qnn_routes_accept_a_single_column_label_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single-column label matrix gives the same native gradient as the label vector."""
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (_FakeJAX(), np))

    column = jax_native_qnn_value_and_grad(_FEATURES, _LABELS[:, None], _PARAMS)
    vector = jax_native_qnn_value_and_grad(_FEATURES, _LABELS, _PARAMS)

    np.testing.assert_array_equal(column.gradient, vector.gradient)


def test_phase_jax_default_loader_reports_a_missing_installation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With JAX unimportable, availability is false and the routes name the optional extra."""
    monkeypatch.setitem(sys.modules, "jax", None)

    assert is_phase_jax_available() is False
    with pytest.raises(ImportError, match=r"install scpn-quantum-control\[jax\]$"):
        jax_parameter_shift_value_and_grad(_objective, np.array([0.2, -0.4], dtype=float))


def test_phase_jax_parameter_shift_refuses_a_callback_that_never_ran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A JIT backend that answers the callback shape without calling it is refused."""

    class _SkippingCallbackJAX(_FakeJAX):
        """Fake JAX whose pure_callback returns zeros without running the host function."""

        def pure_callback(self, callback: Any, _shape_dtypes: Any, values: Any) -> Any:
            """Return zero outputs of the requested shapes."""
            del callback
            self.callback_calls += 1
            return 0.0, np.zeros_like(np.asarray(values, dtype=float))

    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (_SkippingCallbackJAX(), np))

    with pytest.raises(RuntimeError, match="^JAX parameter-shift callback did not execute$"):
        jax_parameter_shift_value_and_grad(
            _objective, np.array([0.2, -0.4], dtype=float), jit=True
        )


def test_phase_jax_custom_vjp_route_refuses_missing_jit_and_a_drifted_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The custom-VJP route needs jit when asked for it and a loss that matches parameter shift."""

    class _NoJITJAX(_FakeJAX):
        """Fake JAX without jit."""

        jit: Any = None

    class _DriftedLossJAX(_FakeJAX):
        """Fake JAX whose value_and_grad reports a loss one unit too high."""

        def value_and_grad(self, fn: Any) -> Any:
            """Shift the loss of the wrapped value-and-gradient."""
            wrapped = super().value_and_grad(fn)

            def shifted(values: Any) -> Any:
                """Return the drifted loss with the true gradient."""
                value, gradient = wrapped(values)
                return float(value) + 1.0, gradient

            return shifted

    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (_NoJITJAX(), np))
    with pytest.raises(RuntimeError, match="^JAX JIT is unavailable in the active JAX module$"):
        jax_custom_vjp_qnn_value_and_grad(_FEATURES, _LABELS, _PARAMS, jit=True)
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (_DriftedLossJAX(), np))
    with pytest.raises(RuntimeError, match="custom VJP QNN loss disagrees with parameter-shift"):
        jax_custom_vjp_qnn_value_and_grad(_FEATURES, _LABELS, _PARAMS)
