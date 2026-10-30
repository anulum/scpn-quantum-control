# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — bounded phase-QNN framework gradient tests
"""Run the lower classifier gradient audits on real frameworks and at their boundaries."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.phase.qnn_training import (
    parameter_shift_qnn_classifier_gradient,
    parameter_shift_qnn_classifier_loss,
)
from scpn_quantum_control.phase_qnn_framework_gradients import (
    NATIVE_JAX_QNN_OPERATIONS,
    NATIVE_TORCH_QNN_OPERATIONS,
    QNNGradientOperations,
    admitted_bounded_qnn_reference,
    admitted_feature_matrix,
    admitted_label_vector,
    bounded_qnn_reference,
    jax_admitted_feature_matrix,
    jax_admitted_label_vector,
    jax_native_qnn_audit,
    native_jax_qnn_value_and_grad,
    native_tensorflow_bounded_qnn_value_and_grad,
    native_torch_autograd_qnn_value_and_grad,
    torch_autograd_qnn_audit,
)
from scpn_quantum_control.phase_qnode_audit_admission import admitted_parameter_vector

_FEATURES = np.array([[0.0], [np.pi]], dtype=np.float64)
_LABELS = np.array([0.0, 1.0], dtype=np.float64)
_PARAMS = np.array([0.8], dtype=np.float64)


def _host(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def _assert_same(left: Any, right: Any) -> None:
    for field in dataclasses.fields(left):
        np.testing.assert_array_equal(
            np.asarray(_host(getattr(left, field.name))),
            np.asarray(_host(getattr(right, field.name))),
        )


def test_native_routes_match_the_historical_bridge_routes() -> None:
    """Return exactly the historical results for every installed framework."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from scpn_quantum_control.phase import jax_bridge, torch_bridge

    _assert_same(
        native_torch_autograd_qnn_value_and_grad(torch, _FEATURES, _LABELS, _PARAMS),
        torch_bridge.torch_autograd_qnn_value_and_grad(_FEATURES, _LABELS, _PARAMS),
    )
    _assert_same(
        native_jax_qnn_value_and_grad(jax, jnp, _FEATURES, _LABELS, _PARAMS),
        jax_bridge.jax_native_qnn_value_and_grad(_FEATURES, _LABELS, _PARAMS),
    )
    jitted = native_jax_qnn_value_and_grad(jax, jnp, _FEATURES, _LABELS, _PARAMS, jit=True)
    assert jitted.passed and jitted.jitted and jitted.jit_requested


def test_native_tensorflow_route_matches_the_historical_bridge_route() -> None:
    """Return exactly the historical TensorFlow result."""
    tensorflow = pytest.importorskip("tensorflow")
    from scpn_quantum_control.phase import tensorflow_bridge

    _assert_same(
        native_tensorflow_bounded_qnn_value_and_grad(tensorflow, _FEATURES, _LABELS, _PARAMS),
        tensorflow_bridge.tensorflow_bounded_qnn_value_and_grad(_FEATURES, _LABELS, _PARAMS),
    )
    with pytest.raises(ValueError, match="params width must match feature width"):
        native_tensorflow_bounded_qnn_value_and_grad(
            tensorflow, _FEATURES, _LABELS, np.array([0.1, 0.2])
        )


def test_torch_route_refuses_mismatched_width_and_missing_audit_evidence() -> None:
    """Refuse a parameter width mismatch and a forward that records no evidence."""
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="params width must match feature width"):
        native_torch_autograd_qnn_value_and_grad(torch, _FEATURES, _LABELS, np.array([0.1, 0.2]))

    class _SilentFunction:
        """Differentiable stand-in that never records audit evidence."""

        @staticmethod
        def apply(parameters: Any) -> Any:
            """Return a differentiable scalar.

            Parameters
            ----------
            parameters
                Trainable parameter tensor.

            Returns
            -------
            Any
                Sum of squared parameters.

            """
            return (parameters * parameters).sum()

    with pytest.raises(RuntimeError, match="did not produce audit evidence"):
        torch_autograd_qnn_audit(
            torch,
            torch.autograd.grad,
            _SilentFunction,
            {},
            _PARAMS,
            1e-6,
            NATIVE_TORCH_QNN_OPERATIONS,
        )


def test_reference_and_jax_audits_refuse_disagreeing_losses() -> None:
    """Refuse an analytic or native loss that disagrees with the parameter-shift loss."""
    shifted = dataclasses.replace(
        NATIVE_TORCH_QNN_OPERATIONS,
        reference_loss=lambda: (
            lambda features, labels, params: (
                parameter_shift_qnn_classifier_loss(features, labels, params) + 1.0
            )
        ),
    )
    with pytest.raises(RuntimeError, match="PyTorch bounded phase-QNN tensor loss disagrees"):
        admitted_bounded_qnn_reference(
            _FEATURES, _LABELS, _PARAMS, 1e-6, shifted, runtime_label="PyTorch"
        )
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax_shifted = dataclasses.replace(
        NATIVE_JAX_QNN_OPERATIONS, reference_loss=shifted.reference_loss
    )
    with pytest.raises(RuntimeError, match="JAX native QNN loss disagrees"):
        jax_native_qnn_audit(
            jax,
            _FEATURES,
            _LABELS,
            _PARAMS,
            1e-6,
            lambda raw: jnp.sum(jnp.asarray(raw)) * 0.0,
            jax_shifted,
            jnp=jnp,
            jit=False,
        )


def test_jax_audit_refuses_missing_transforms_and_handles_empty_parameters() -> None:
    """Refuse a runtime without ``value_and_grad`` or ``jit``; accept an empty vector."""

    class _NoTransforms:
        """JAX stand-in without transforms."""

    with pytest.raises(RuntimeError, match="value_and_grad is required"):
        jax_native_qnn_audit(
            _NoTransforms(),
            _FEATURES,
            _LABELS,
            _PARAMS,
            1e-6,
            lambda raw: raw,
            NATIVE_JAX_QNN_OPERATIONS,
            jnp=np,
            jit=False,
        )

    class _NoJit:
        """JAX stand-in with ``value_and_grad`` but no ``jit``."""

        @staticmethod
        def value_and_grad(function: Any) -> Any:
            """Return a constant value-and-gradient function.

            Parameters
            ----------
            function
                Ignored objective.

            Returns
            -------
            Any
                Function returning zero loss and an empty gradient.

            """
            return lambda raw: (0.0, np.zeros(0))

    with pytest.raises(RuntimeError, match="JAX JIT is unavailable"):
        jax_native_qnn_audit(
            _NoJit(),
            _FEATURES,
            _LABELS,
            _PARAMS,
            1e-6,
            lambda raw: raw,
            NATIVE_JAX_QNN_OPERATIONS,
            jnp=np,
            jit=True,
        )
    empty = QNNGradientOperations(
        parameter_vector=lambda: admitted_parameter_vector,
        reference_loss=lambda: lambda features, labels, params: 0.0,
        reference_gradient=lambda: lambda features, labels, params: np.zeros(0),
        analytic_model=lambda: lambda features, labels, params: (0.0, np.zeros(0)),
    )
    result = jax_native_qnn_audit(
        _NoJit(),
        np.zeros((1, 0)),
        np.zeros(1),
        np.zeros(0),
        1e-6,
        lambda raw: raw,
        empty,
        jnp=np,
        jit=False,
    )
    assert result.passed and result.max_abs_error == 0.0
    reference = admitted_bounded_qnn_reference(
        np.zeros((1, 0)), np.zeros(1), np.zeros(0), 1e-6, empty, runtime_label="TensorFlow"
    )
    assert reference[3] == 0.0 and reference[6]


def test_admission_keeps_both_route_vocabularies() -> None:
    """Refuse malformed features and labels with each route's messages."""
    for features, message in (
        (np.zeros(2), "two-dimensional"),
        (np.zeros((0, 1)), "must not be empty"),
        (np.array([[np.nan]]), "only finite"),
    ):
        with pytest.raises(ValueError, match=message):
            admitted_feature_matrix(features)
    for labels, message in (
        (np.zeros((2, 1)), "one-dimensional"),
        (np.zeros(3), r"shape \(2,\)"),
        (np.array([np.nan, 0.0]), "only finite"),
    ):
        with pytest.raises(ValueError, match=message):
            admitted_label_vector(labels, n_samples=2)
    for features, message in (
        (np.zeros(2), "two-dimensional"),
        (np.zeros((0, 1)), "at least one sample"),
        (np.zeros((1, 0)), "at least one feature column"),
        (np.array([[np.inf]]), "only finite"),
    ):
        with pytest.raises(ValueError, match=message):
            jax_admitted_feature_matrix(features)
    for labels, message in (
        (np.zeros((2, 2)), "single-column"),
        (np.zeros(3), "same sample count"),
        (np.array([np.nan, 0.0]), "only finite"),
        (np.array([2.0, 0.0]), r"closed interval \[0, 1\]"),
    ):
        with pytest.raises(ValueError, match=message):
            jax_admitted_label_vector(labels, n_samples=2)
    np.testing.assert_array_equal(
        jax_admitted_label_vector(np.array([[0.0], [1.0]]), n_samples=2), _LABELS
    )
    reference = parameter_shift_qnn_classifier_gradient(_FEATURES, _LABELS, _PARAMS)
    assert reference.shape == (1,)
    with pytest.raises(ValueError, match="params width must match feature width"):
        bounded_qnn_reference(
            _FEATURES,
            _LABELS,
            np.array([0.1, 0.2]),
            tolerance=1e-6,
            operations=NATIVE_TORCH_QNN_OPERATIONS,
            runtime_label="PyTorch",
        )
