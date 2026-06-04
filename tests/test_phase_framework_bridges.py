# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase Framework Bridges
"""Tests for optional PyTorch and TensorFlow phase-gradient bridges."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_quantum_control.phase.tensorflow_bridge as tensorflow_bridge
import scpn_quantum_control.phase.torch_bridge as torch_bridge
from scpn_quantum_control.phase import (
    PhaseTensorFlowParameterShiftResult,
    PhaseTorchParameterShiftResult,
    is_phase_tensorflow_available,
    is_phase_torch_available,
    multi_frequency_parameter_shift_rule,
    tensorflow_parameter_shift_value_and_grad,
    torch_parameter_shift_value_and_grad,
)


class _FakeTorchTensor:
    def __init__(self, values: object) -> None:
        self._values = np.asarray(values, dtype=float)

    def detach(self) -> _FakeTorchTensor:
        return self

    def cpu(self) -> _FakeTorchTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._values.copy()


class _FakeTorch:
    float64 = np.float64

    def __init__(self) -> None:
        self.as_tensor_calls: list[np.ndarray] = []

    def as_tensor(self, values: object, *, dtype: object | None = None) -> _FakeTorchTensor:
        del dtype
        array = np.asarray(values, dtype=float)
        self.as_tensor_calls.append(array.copy())
        return _FakeTorchTensor(array)


class _FakeTensorFlowTensor:
    def __init__(self, values: object) -> None:
        self._values = np.asarray(values, dtype=float)

    def numpy(self) -> np.ndarray:
        return self._values.copy()


class _FakeTensorFlow:
    float64 = np.float64

    def __init__(self) -> None:
        self.convert_calls: list[np.ndarray] = []

    def convert_to_tensor(
        self,
        values: object,
        *,
        dtype: object | None = None,
    ) -> _FakeTensorFlowTensor:
        del dtype
        array = np.asarray(values, dtype=float)
        self.convert_calls.append(array.copy())
        return _FakeTensorFlowTensor(array)


def _objective(values: np.ndarray) -> float:
    return float(np.cos(values[0]) + 0.25 * np.sin(values[1]))


def test_torch_bridge_returns_tensor_and_numpy_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)

    result = torch_parameter_shift_value_and_grad(
        _objective,
        _FakeTorchTensor(np.array([0.2, -0.4], dtype=float)),
    )

    assert isinstance(result, PhaseTorchParameterShiftResult)
    assert is_phase_torch_available()
    assert result.method == "parameter_shift"
    assert result.host_boundary
    assert result.evaluations == 5
    assert isinstance(result.torch_value, _FakeTorchTensor)
    assert isinstance(result.torch_gradient, _FakeTorchTensor)
    np.testing.assert_allclose(
        result.gradient,
        np.array([-np.sin(0.2), 0.25 * np.cos(-0.4)], dtype=float),
        atol=1e-12,
    )
    np.testing.assert_allclose(result.torch_gradient.numpy(), result.gradient, atol=1e-12)


def test_torch_bridge_reports_multi_frequency_parameter_shift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])

    def objective(values: np.ndarray) -> float:
        return float(np.sin(values[0]) + 0.1 * np.cos(2.0 * values[0]))

    result = torch_parameter_shift_value_and_grad(
        objective,
        _FakeTorchTensor(np.array([0.4], dtype=float)),
        rule=rule,
    )

    expected = np.array([np.cos(0.4) - 0.2 * np.sin(0.8)], dtype=float)
    assert result.method == "multi_frequency_parameter_shift"
    assert result.shift_terms == len(rule.terms)
    assert result.evaluations == 1 + 2 * len(rule.terms)
    np.testing.assert_allclose(result.gradient, expected, atol=1e-12)
    assert result.to_dict()["shift_terms"] == len(rule.terms)


def test_tensorflow_bridge_returns_tensor_and_numpy_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_tf = _FakeTensorFlow()
    monkeypatch.setattr(tensorflow_bridge, "_load_tensorflow", lambda: fake_tf)

    result = tensorflow_parameter_shift_value_and_grad(
        _objective,
        _FakeTensorFlowTensor(np.array([0.2, -0.4], dtype=float)),
    )

    assert isinstance(result, PhaseTensorFlowParameterShiftResult)
    assert is_phase_tensorflow_available()
    assert result.method == "parameter_shift"
    assert result.host_boundary
    assert result.evaluations == 5
    assert isinstance(result.tensorflow_value, _FakeTensorFlowTensor)
    assert isinstance(result.tensorflow_gradient, _FakeTensorFlowTensor)
    np.testing.assert_allclose(
        result.gradient,
        np.array([-np.sin(0.2), 0.25 * np.cos(-0.4)], dtype=float),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.tensorflow_gradient.numpy(),
        result.gradient,
        atol=1e-12,
    )


def test_tensorflow_bridge_reports_multi_frequency_parameter_shift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_tf = _FakeTensorFlow()
    monkeypatch.setattr(tensorflow_bridge, "_load_tensorflow", lambda: fake_tf)
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])

    def objective(values: np.ndarray) -> float:
        return float(np.sin(values[0]) + 0.1 * np.cos(2.0 * values[0]))

    result = tensorflow_parameter_shift_value_and_grad(
        objective,
        _FakeTensorFlowTensor(np.array([0.4], dtype=float)),
        rule=rule,
    )

    expected = np.array([np.cos(0.4) - 0.2 * np.sin(0.8)], dtype=float)
    assert result.method == "multi_frequency_parameter_shift"
    assert result.shift_terms == len(rule.terms)
    assert result.evaluations == 1 + 2 * len(rule.terms)
    np.testing.assert_allclose(result.gradient, expected, atol=1e-12)
    assert result.to_dict()["shift_terms"] == len(rule.terms)


def test_framework_bridges_fail_closed_when_optional_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_torch() -> object:
        raise ImportError("torch blocked")

    def missing_tensorflow() -> object:
        raise ImportError("tensorflow blocked")

    monkeypatch.setattr(torch_bridge, "_load_torch", missing_torch)
    monkeypatch.setattr(tensorflow_bridge, "_load_tensorflow", missing_tensorflow)

    assert not is_phase_torch_available()
    assert not is_phase_tensorflow_available()
    with pytest.raises(ImportError, match="torch blocked"):
        torch_parameter_shift_value_and_grad(_objective, np.array([0.2, -0.4], dtype=float))
    with pytest.raises(ImportError, match="tensorflow blocked"):
        tensorflow_parameter_shift_value_and_grad(
            _objective,
            np.array([0.2, -0.4], dtype=float),
        )
