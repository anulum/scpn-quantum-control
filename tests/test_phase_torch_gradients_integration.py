# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Torch Gradient Integration Tests
"""Integration tests for Torch gradient routes through the public bridge."""

from __future__ import annotations

import numpy as np
import pytest
from _phase_torch_bridge_test_helpers import (
    FloatArray,
    _FakeTorch,
    _FakeTorchTensor,
    _FakeTorchWithoutAutogradFunction,
    _objective,
)

import scpn_quantum_control.phase.torch_bridge as torch_bridge
from scpn_quantum_control.phase import (
    PhaseTorchAutogradQNNGradientResult,
    PhaseTorchParameterShiftResult,
    PhaseTorchQNNGradientResult,
    is_phase_torch_available,
    multi_frequency_parameter_shift_rule,
    parameter_shift_qnn_classifier_gradient,
    parameter_shift_qnn_classifier_loss,
    torch_autograd_qnn_value_and_grad,
    torch_bounded_qnn_value_and_grad,
    torch_parameter_shift_value_and_grad,
)


def test_torch_bridge_returns_tensor_and_numpy_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that PyTorch bridge returns tensor and NumPy gradients."""
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
    """Verify that PyTorch bridge reports multi frequency parameter shift."""
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])

    def objective(values: FloatArray) -> float:
        """Evaluate the local cosine objective used to inspect rule metadata."""
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


def test_torch_bounded_qnn_gradient_matches_parameter_shift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that PyTorch bounded QNN gradient matches parameter shift."""
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params = _FakeTorchTensor(np.array([0.45], dtype=float))

    result = torch_bounded_qnn_value_and_grad(
        features,
        labels,
        params,
        tolerance=1e-12,
    )

    expected_loss = parameter_shift_qnn_classifier_loss(
        features,
        labels,
        params.numpy(),
    )
    expected_gradient = parameter_shift_qnn_classifier_gradient(
        features,
        labels,
        params.numpy(),
    )
    assert isinstance(result, PhaseTorchQNNGradientResult)
    assert result.passed
    assert result.analytic_framework_gradient
    assert not result.native_framework_autodiff
    assert not result.host_boundary
    np.testing.assert_allclose(result.loss, expected_loss, atol=1e-12)
    np.testing.assert_allclose(result.gradient, expected_gradient, atol=1e-12)
    np.testing.assert_allclose(result.parameter_shift_gradient, expected_gradient, atol=1e-12)
    np.testing.assert_allclose(result.torch_gradient.numpy(), expected_gradient, atol=1e-12)
    assert result.to_dict()["passed"] is True


def test_torch_autograd_qnn_gradient_uses_custom_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that PyTorch autograd QNN gradient uses custom function."""
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params = _FakeTorchTensor(np.array([0.45], dtype=float))

    result = torch_autograd_qnn_value_and_grad(
        features,
        labels,
        params,
        tolerance=1e-12,
    )

    expected_gradient = parameter_shift_qnn_classifier_gradient(
        features,
        labels,
        params.numpy(),
    )
    assert isinstance(result, PhaseTorchAutogradQNNGradientResult)
    assert result.passed
    assert result.native_framework_autodiff
    assert result.custom_autograd_function
    assert not result.host_boundary
    assert result.method == "torch_bounded_phase_qnn_custom_autograd_function"
    np.testing.assert_allclose(result.gradient, expected_gradient, atol=1e-12)
    np.testing.assert_allclose(result.torch_gradient.numpy(), expected_gradient, atol=1e-12)
    assert result.to_dict()["custom_autograd_function"] is True


def test_torch_autograd_qnn_gradient_fails_closed_without_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that PyTorch autograd QNN gradient fails closed without function."""
    fake_torch = _FakeTorchWithoutAutogradFunction()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)

    with pytest.raises(RuntimeError, match="torch.autograd.Function"):
        torch_autograd_qnn_value_and_grad(
            np.array([[0.0]], dtype=float),
            np.array([0.0], dtype=float),
            np.array([0.45], dtype=float),
        )


def test_torch_bounded_qnn_gradient_fails_closed_on_shape_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that PyTorch bounded QNN gradient fails closed on shape mismatch."""
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)

    with pytest.raises(ValueError, match="params width must match feature width"):
        torch_bounded_qnn_value_and_grad(
            np.array([[0.0, 1.0]], dtype=float),
            np.array([0.0], dtype=float),
            np.array([0.45], dtype=float),
        )
