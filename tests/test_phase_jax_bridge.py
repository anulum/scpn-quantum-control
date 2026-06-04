# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase JAX Bridge
"""Tests for optional JAX phase parameter-shift interop."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_quantum_control.phase.jax_bridge as jax_bridge
from scpn_quantum_control.phase import (
    PhaseJAXGradientAgreementResult,
    PhaseJAXParameterShiftResult,
    check_jax_parameter_shift_agreement,
    is_phase_jax_available,
    jax_parameter_shift_value_and_grad,
)


class _FakeJAX:
    class ShapeDtypeStruct:
        def __init__(self, shape: tuple[int, ...], dtype: object) -> None:
            self.shape = shape
            self.dtype = dtype

    def __init__(self) -> None:
        self.jit_calls = 0
        self.callback_calls = 0

    def jit(self, fn):
        self.jit_calls += 1

        def wrapped(values):
            return fn(values)

        return wrapped

    def pure_callback(self, callback, _shape_dtypes, values):
        self.callback_calls += 1
        return callback(values)


def _objective(values: np.ndarray) -> float:
    return float(np.cos(values[0]) + 0.25 * np.sin(values[1]))


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


def test_phase_jax_bridge_reports_gradient_agreement(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_jax = _FakeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    def jax_gradient(values: np.ndarray) -> np.ndarray:
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


def test_phase_jax_bridge_reports_gradient_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_jax = _FakeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    def shifted_gradient(values: np.ndarray) -> np.ndarray:
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


def test_phase_jax_bridge_fails_closed_when_jax_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable():
        raise ImportError("blocked")

    monkeypatch.setattr(jax_bridge, "_load_jax", unavailable)

    assert not is_phase_jax_available()
    with pytest.raises(ImportError, match="blocked"):
        jax_parameter_shift_value_and_grad(_objective, np.array([0.2, -0.4], dtype=float))
    with pytest.raises(ImportError, match="blocked"):
        check_jax_parameter_shift_agreement(
            _objective,
            lambda values: values,
            np.array([0.2, -0.4], dtype=float),
        )
