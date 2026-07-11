# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

# SCPN Quantum Control — TensorFlow Gradient Tests

"""Behavioral and structural tests for the TensorFlow gradient leaf."""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.phase.tensorflow_bridge as tensorflow_bridge
import scpn_quantum_control.phase.tensorflow_gradients as tensorflow_gradients
from scpn_quantum_control.phase import (
    PhaseTensorFlowParameterShiftResult,
    PhaseTensorFlowQNNGradientResult,
    is_phase_tensorflow_available,
    multi_frequency_parameter_shift_rule,
    parameter_shift_qnn_classifier_gradient,
    parameter_shift_qnn_classifier_loss,
    tensorflow_bounded_qnn_value_and_grad,
    tensorflow_parameter_shift_value_and_grad,
)

FloatArray = NDArray[np.float64]


class _FakeTensorFlowTensor:
    """Minimal NumPy-backed TensorFlow tensor double."""

    def __init__(self, values: object) -> None:
        """Store a deterministic copy of the tensor values."""
        if isinstance(values, _FakeTensorFlowTensor):
            values = values.numpy()
        self._values = np.asarray(values, dtype=float)

    def numpy(self) -> FloatArray:
        """Return a copy of the tensor payload."""
        return self._values.copy()


class _FakeTensorFlow:
    """Minimal TensorFlow conversion facade for gradient tests."""

    float64 = np.float64

    def __init__(self) -> None:
        """Initialize the conversion-call ledger."""
        self.convert_calls: list[FloatArray] = []

    def convert_to_tensor(
        self,
        values: object,
        *,
        dtype: object | None = None,
    ) -> _FakeTensorFlowTensor:
        """Convert values to a NumPy-backed fake TensorFlow tensor."""
        del dtype
        array = np.asarray(values, dtype=float)
        self.convert_calls.append(array.copy())
        return _FakeTensorFlowTensor(array)


def _objective(values: FloatArray) -> float:
    """Evaluate the shared two-parameter cosine objective in radians."""
    return float(np.cos(values[0]) + 0.25 * np.sin(values[1]))


def test_tensorflow_bridge_returns_tensor_and_numpy_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that TensorFlow bridge returns tensor and NumPy gradients."""
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
    """Verify that TensorFlow bridge reports multi frequency parameter shift."""
    fake_tf = _FakeTensorFlow()
    monkeypatch.setattr(tensorflow_bridge, "_load_tensorflow", lambda: fake_tf)
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])

    def objective(values: FloatArray) -> float:
        """Evaluate the local cosine objective used to inspect rule metadata."""
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


def test_tensorflow_bounded_qnn_gradient_matches_parameter_shift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that TensorFlow bounded QNN gradient matches parameter shift."""
    fake_tf = _FakeTensorFlow()
    monkeypatch.setattr(tensorflow_bridge, "_load_tensorflow", lambda: fake_tf)
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params = _FakeTensorFlowTensor(np.array([0.45], dtype=float))

    result = tensorflow_bounded_qnn_value_and_grad(
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
    assert isinstance(result, PhaseTensorFlowQNNGradientResult)
    assert result.passed
    assert result.analytic_framework_gradient
    assert not result.native_framework_autodiff
    assert not result.host_boundary
    np.testing.assert_allclose(result.loss, expected_loss, atol=1e-12)
    np.testing.assert_allclose(result.gradient, expected_gradient, atol=1e-12)
    np.testing.assert_allclose(result.parameter_shift_gradient, expected_gradient, atol=1e-12)
    np.testing.assert_allclose(result.tensorflow_gradient.numpy(), expected_gradient, atol=1e-12)
    assert result.to_dict()["passed"] is True


def test_tensorflow_bounded_qnn_gradient_fails_closed_on_shape_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that TensorFlow bounded QNN gradient fails closed on shape mismatch."""
    fake_tf = _FakeTensorFlow()
    monkeypatch.setattr(tensorflow_bridge, "_load_tensorflow", lambda: fake_tf)

    with pytest.raises(ValueError, match="params width must match feature width"):
        tensorflow_bounded_qnn_value_and_grad(
            np.array([[0.0, 1.0]], dtype=float),
            np.array([0.0], dtype=float),
            np.array([0.45], dtype=float),
        )


def test_tensorflow_gradient_leaf_has_no_facade_backedge() -> None:
    """Keep the gradient implementation independent of the compatibility facade."""
    tree = ast.parse(inspect.getsource(tensorflow_gradients))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert not any(module.endswith("tensorflow_bridge") for module in imported_modules)


def test_tensorflow_gradient_helpers_are_exact_facade_aliases() -> None:
    """Keep downstream compatibility consumers on one helper implementation."""
    helper_names = (
        "_as_parameter_vector",
        "_as_feature_matrix",
        "_as_label_vector",
        "_as_non_negative_tolerance",
        "_tensorflow_values_to_numpy",
        "_tensorflow_tensor",
    )

    for name in helper_names:
        assert getattr(tensorflow_bridge, name) is getattr(tensorflow_gradients, name)


def test_tensorflow_facade_injects_active_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify facade calls pass the currently monkeypatched loader callback."""
    sentinel = object()
    seen: dict[str, object] = {}

    def fake_parameter_shift(
        objective: Callable[[FloatArray], float],
        values: object,
        *,
        parameters: object,
        rule: object,
        _tensorflow_loader: Callable[[], Any],
    ) -> object:
        del objective, values, parameters, rule
        seen["loader"] = _tensorflow_loader
        return sentinel

    def loader() -> object:
        """Return a sentinel optional-framework module."""
        return object()

    monkeypatch.setattr(tensorflow_bridge, "_load_tensorflow", loader)
    monkeypatch.setattr(
        tensorflow_bridge,
        "_tensorflow_parameter_shift_value_and_grad",
        fake_parameter_shift,
    )

    result = tensorflow_bridge.tensorflow_parameter_shift_value_and_grad(
        _objective,
        np.array([0.0], dtype=float),
    )

    assert result is sentinel
    assert seen["loader"] is loader


def test_tensorflow_facade_does_not_redefine_gradient_helpers() -> None:
    """Keep moved gradient helpers single-owned by the execution leaf."""
    tree = ast.parse(inspect.getsource(tensorflow_bridge))
    definitions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }

    assert definitions.isdisjoint(
        {
            "_as_parameter_vector",
            "_as_feature_matrix",
            "_as_label_vector",
            "_as_non_negative_tolerance",
            "_tensorflow_values_to_numpy",
            "_tensorflow_tensor",
        }
    )
