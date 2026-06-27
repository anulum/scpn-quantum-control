# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase JAX Bridge
"""Tests for optional JAX phase parameter-shift interop."""

from __future__ import annotations

import json

import numpy as np
import pytest

import scpn_quantum_control.phase.jax_bridge as jax_bridge
from scpn_quantum_control.phase import (
    DenseHermitianObservable,
    PauliCovarianceObservable,
    PauliTerm,
    PhaseJAXCloudValidationRunSpec,
    PhaseJAXCustomVJPQNNGradientResult,
    PhaseJAXGradientAgreementResult,
    PhaseJAXJITCompatibilityResult,
    PhaseJAXMaturityAuditResult,
    PhaseJAXNativeQNNGradientResult,
    PhaseJAXNestedTransformAlgebraResult,
    PhaseJAXParameterShiftResult,
    PhaseJAXPhaseQNodeLoweringMatrixResult,
    PhaseJAXPhaseQNodeNativeTransformResult,
    PhaseJAXPhaseQNodePyTreeTransformResult,
    PhaseJAXPhaseQNodeStatevectorResult,
    PhaseJAXPyTreeCompatibilityResult,
    PhaseJAXShardingCompatibilityResult,
    PhaseJAXVMAPCompatibilityResult,
    PhaseQNodeCircuit,
    SparsePauliHamiltonian,
    check_jax_parameter_shift_agreement,
    execute_phase_qnode_circuit,
    is_phase_jax_available,
    jax_custom_vjp_qnn_value_and_grad,
    jax_native_qnn_value_and_grad,
    jax_parameter_shift_value_and_grad,
    jax_phase_qnode_native_transform_audit,
    jax_phase_qnode_pytree_transform_audit,
    jax_phase_qnode_value_and_grad,
    multi_frequency_parameter_shift_rule,
    parameter_shift_phase_qnode_gradient,
    parameter_shift_qnn_classifier_gradient,
    plan_jax_cloud_validation_batch,
    run_jax_jit_compatibility_audit,
    run_jax_maturity_audit,
    run_jax_nested_transform_algebra_audit,
    run_jax_phase_qnode_lowering_matrix,
    run_jax_pytree_compatibility_audit,
    run_jax_sharding_compatibility_audit,
    run_jax_vmap_compatibility_audit,
)


class _FakeJAX:
    class _TreeUtil:
        @staticmethod
        def tree_flatten(tree):
            leaves = []

            def visit(node):
                if isinstance(node, dict):
                    return ("dict", tuple((key, visit(node[key])) for key in sorted(node)))
                if isinstance(node, tuple):
                    return ("tuple", tuple(visit(item) for item in node))
                if isinstance(node, list):
                    return ("list", tuple(visit(item) for item in node))
                leaves.append(node)
                return ("leaf", len(leaves) - 1)

            treedef = visit(tree)
            return leaves, treedef

        @staticmethod
        def tree_unflatten(treedef, leaves):
            kind, payload = treedef
            if kind == "leaf":
                return leaves[payload]
            if kind == "dict":
                return {
                    key: _FakeJAX._TreeUtil.tree_unflatten(child, leaves) for key, child in payload
                }
            if kind == "tuple":
                return tuple(_FakeJAX._TreeUtil.tree_unflatten(child, leaves) for child in payload)
            if kind == "list":
                return [_FakeJAX._TreeUtil.tree_unflatten(child, leaves) for child in payload]
            raise ValueError(f"unsupported fake PyTree node kind {kind}")

    class ShapeDtypeStruct:
        def __init__(self, shape: tuple[int, ...], dtype: object) -> None:
            self.shape = shape
            self.dtype = dtype

    tree_util = _TreeUtil()

    def __init__(self) -> None:
        self.jit_calls = 0
        self.callback_calls = 0
        self.callback_shape_dtypes = None
        self.custom_vjp_calls = 0
        self.custom_vjp_defvjp_calls = 0
        self.vmap_calls = 0
        self.pmap_calls = 0
        self.local_device_count_value = 2

    def jit(self, fn):
        self.jit_calls += 1

        def wrapped(values):
            return fn(values)

        return wrapped

    def pure_callback(self, callback, _shape_dtypes, values):
        self.callback_calls += 1
        self.callback_shape_dtypes = _shape_dtypes
        return callback(values)

    def custom_vjp(self, fn):
        self.custom_vjp_calls += 1
        fake_jax = self

        class _CustomVJPFunction:
            def __init__(self, primal_fn):
                self._primal_fn = primal_fn
                self._forward = None
                self._backward = None

            def defvjp(self, forward, backward):
                fake_jax.custom_vjp_defvjp_calls += 1
                self._forward = forward
                self._backward = backward

            def __call__(self, values):
                return self._primal_fn(values)

            def value_and_grad(self, values):
                if self._forward is None or self._backward is None:
                    raise RuntimeError("custom_vjp rule has not been registered")
                value, residual = self._forward(values)
                (gradient,) = self._backward(residual, np.asarray(1.0, dtype=float))
                return value, gradient

        return _CustomVJPFunction(fn)

    def value_and_grad(self, fn):
        def wrapped(values):
            if hasattr(fn, "value_and_grad"):
                return fn.value_and_grad(values)
            leaves, treedef = self.tree_util.tree_flatten(values)
            arrays = [np.asarray(leaf, dtype=float) for leaf in leaves]
            sizes = [array.size for array in arrays]
            shapes = [array.shape for array in arrays]
            array = np.concatenate([array.ravel() for array in arrays])

            def rebuild(flat_values):
                offset = 0
                rebuilt = []
                for size, shape in zip(sizes, shapes, strict=True):
                    rebuilt.append(flat_values[offset : offset + size].reshape(shape))
                    offset += size
                return self.tree_util.tree_unflatten(treedef, rebuilt)

            value = fn(rebuild(array))
            gradient = np.zeros_like(array, dtype=float)
            step = 1e-6
            for index in range(array.size):
                forward = array.copy()
                backward = array.copy()
                forward[index] += step
                backward[index] -= step
                gradient[index] = (float(fn(rebuild(forward))) - float(fn(rebuild(backward)))) / (
                    2.0 * step
                )
            return value, rebuild(gradient)

        return wrapped

    def grad(self, fn):
        def wrapped(values):
            _value, gradient = self.value_and_grad(fn)(values)
            return gradient

        return wrapped

    def jacfwd(self, fn):
        return self.grad(fn)

    def jacrev(self, fn):
        return self.grad(fn)

    def hessian(self, fn):
        gradient_fn = self.grad(fn)

        def wrapped(values):
            array = np.asarray(values, dtype=float)
            hessian = np.zeros((array.size, array.size), dtype=float)
            step = 1e-5
            for index in range(array.size):
                forward = array.copy()
                backward = array.copy()
                forward[index] += step
                backward[index] -= step
                hessian[:, index] = (gradient_fn(forward) - gradient_fn(backward)) / (2.0 * step)
            return 0.5 * (hessian + hessian.T)

        return wrapped

    def jvp(self, fn, primals, tangents):
        (values,) = primals
        (tangent,) = tangents
        value = fn(values)
        gradient = self.grad(fn)(values)
        return value, np.asarray(np.dot(gradient, tangent), dtype=float)

    def vjp(self, fn, values):
        value = fn(values)
        gradient = self.grad(fn)(values)

        def pullback(cotangent):
            return (np.asarray(cotangent, dtype=float) * gradient,)

        return value, pullback

    def vmap(self, fn):
        self.vmap_calls += 1

        def wrapped(values):
            outputs = [fn(row) for row in np.asarray(values, dtype=float)]
            if outputs and isinstance(outputs[0], tuple):
                return tuple(np.stack(items, axis=0) for items in zip(*outputs, strict=True))
            return np.asarray(outputs, dtype=float)

        return wrapped

    def pmap(self, fn):
        self.pmap_calls += 1

        def wrapped(values):
            outputs = [fn(row) for row in np.asarray(values, dtype=float)]
            if outputs and isinstance(outputs[0], tuple):
                return tuple(np.stack(items, axis=0) for items in zip(*outputs, strict=True))
            return np.asarray(outputs, dtype=float)

        return wrapped

    def local_device_count(self):
        return self.local_device_count_value

    def local_devices(self):
        return [f"fake-device-{index}" for index in range(self.local_device_count_value)]


def _objective(values: np.ndarray) -> float:
    return float(np.cos(values[0]) + 0.25 * np.sin(values[1]))


class _Float32JNP:
    @staticmethod
    def asarray(values):
        return np.asarray(values, dtype=np.float32)


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

    def objective(values: np.ndarray) -> float:
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


def test_phase_jax_bridge_reports_multi_frequency_gradient_agreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jax = _FakeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])

    def objective(values: np.ndarray) -> float:
        return float(np.sin(values[0]) + 0.1 * np.cos(2.0 * values[0]))

    def jax_gradient(values: np.ndarray) -> np.ndarray:
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
        custom_vjp = None

    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (_NoCustomVJPJAX(), np))

    with pytest.raises(RuntimeError, match="custom_vjp"):
        jax_custom_vjp_qnn_value_and_grad(
            np.array([[0.0], [np.pi]], dtype=float),
            np.array([0.0, 1.0], dtype=float),
            np.array([0.45], dtype=float),
        )


def test_phase_jax_jit_compatibility_audit_separates_native_and_callback_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jax = _FakeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    result = run_jax_jit_compatibility_audit(
        features=np.array([[0.0], [np.pi]], dtype=float),
        labels=np.array([0.0, 1.0], dtype=float),
        params=np.array([0.45], dtype=float),
        tolerance=1e-12,
    )

    assert isinstance(result, PhaseJAXJITCompatibilityResult)
    assert result.passed
    assert result.native_qnn_jitted
    assert result.custom_vjp_qnn_jitted
    assert result.custom_vjp_registered
    assert result.parameter_shift_jitted
    assert result.parameter_shift_host_callback
    assert not result.native_qnn_host_callback
    assert not result.custom_vjp_qnn_host_callback
    assert result.max_abs_error <= 1e-12
    assert result.claim_boundary == "bounded_jax_jit_compatibility"
    assert "parameter_shift_host_callback" in result.unsupported_native_routes
    assert fake_jax.jit_calls == 3
    assert result.to_dict()["passed"] is True


def test_phase_jax_jit_compatibility_audit_fails_closed_without_jit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NoJITJAX(_FakeJAX):
        jit = None

    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (_NoJITJAX(), np))

    with pytest.raises(RuntimeError, match="JAX JIT"):
        run_jax_jit_compatibility_audit(
            features=np.array([[0.0], [np.pi]], dtype=float),
            labels=np.array([0.0, 1.0], dtype=float),
            params=np.array([0.45], dtype=float),
        )


def test_phase_jax_vmap_compatibility_audit_batches_native_and_custom_vjp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jax = _FakeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params_batch = np.array([[0.25], [0.45], [0.65]], dtype=float)

    result = run_jax_vmap_compatibility_audit(
        features=features,
        labels=labels,
        params_batch=params_batch,
        tolerance=1e-10,
    )

    expected_gradients = np.vstack(
        [parameter_shift_qnn_classifier_gradient(features, labels, row) for row in params_batch]
    )
    assert isinstance(result, PhaseJAXVMAPCompatibilityResult)
    assert result.passed
    assert result.batch_size == 3
    assert result.native_qnn_vmapped
    assert result.custom_vjp_qnn_vmapped
    assert result.custom_vjp_registered
    assert not result.native_qnn_host_callback
    assert not result.custom_vjp_qnn_host_callback
    assert result.max_abs_error <= 1e-10
    assert result.claim_boundary == "bounded_jax_vmap_compatibility"
    assert "parameter_shift_host_loop_reference" in result.unsupported_native_routes
    assert fake_jax.vmap_calls == 2
    np.testing.assert_allclose(result.native_gradients, expected_gradients, atol=1e-10)
    np.testing.assert_allclose(result.custom_vjp_gradients, expected_gradients, atol=1e-10)
    assert result.to_dict()["passed"] is True


def test_phase_jax_vmap_compatibility_audit_fails_closed_without_vmap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NoVMAPJAX(_FakeJAX):
        vmap = None

    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (_NoVMAPJAX(), np))

    with pytest.raises(RuntimeError, match="JAX vmap"):
        run_jax_vmap_compatibility_audit(
            features=np.array([[0.0], [np.pi]], dtype=float),
            labels=np.array([0.0, 1.0], dtype=float),
            params_batch=np.array([[0.25], [0.45]], dtype=float),
        )


def test_phase_jax_sharding_compatibility_audit_batches_native_and_custom_vjp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jax = _FakeJAX()
    fake_jax.local_device_count_value = 2
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params_batch = np.array([[0.25], [0.45]], dtype=float)

    result = run_jax_sharding_compatibility_audit(
        features=features,
        labels=labels,
        params_batch=params_batch,
        tolerance=1e-10,
    )

    expected_gradients = np.vstack(
        [parameter_shift_qnn_classifier_gradient(features, labels, row) for row in params_batch]
    )
    assert isinstance(result, PhaseJAXShardingCompatibilityResult)
    assert result.passed
    assert result.batch_size == 2
    assert result.local_device_count == 2
    assert result.sharding_mode == "pmap_multi_device"
    assert result.native_qnn_pmapped
    assert result.custom_vjp_qnn_pmapped
    assert result.custom_vjp_registered
    assert not result.native_qnn_host_callback
    assert not result.custom_vjp_qnn_host_callback
    assert result.max_abs_error <= 1e-10
    assert result.claim_boundary == "bounded_jax_pmap_sharding_compatibility"
    assert "parameter_shift_host_loop_reference" in result.unsupported_native_routes
    assert fake_jax.pmap_calls == 2
    np.testing.assert_allclose(result.native_gradients, expected_gradients, atol=1e-10)
    np.testing.assert_allclose(result.custom_vjp_gradients, expected_gradients, atol=1e-10)
    assert result.to_dict()["passed"] is True


def test_phase_jax_sharding_compatibility_audit_fails_closed_without_pmap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NoPMAPJAX(_FakeJAX):
        pmap = None

    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (_NoPMAPJAX(), np))

    with pytest.raises(RuntimeError, match="JAX pmap"):
        run_jax_sharding_compatibility_audit(
            features=np.array([[0.0], [np.pi]], dtype=float),
            labels=np.array([0.0, 1.0], dtype=float),
            params_batch=np.array([[0.25], [0.45]], dtype=float),
        )


def test_phase_jax_pytree_compatibility_audit_round_trips_gradient_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jax = _FakeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))
    features = np.array([[0.0, 0.2, 0.4], [np.pi, np.pi + 0.2, np.pi + 0.4]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params_tree = {
        "encoder": np.array([0.25, 0.45], dtype=float),
        "readout": {"phase": np.array([0.65], dtype=float)},
    }

    result = run_jax_pytree_compatibility_audit(
        features=features,
        labels=labels,
        params_pytree=params_tree,
        tolerance=1e-10,
    )

    expected = parameter_shift_qnn_classifier_gradient(
        features,
        labels,
        np.array([0.25, 0.45, 0.65], dtype=float),
    )
    assert isinstance(result, PhaseJAXPyTreeCompatibilityResult)
    assert result.passed
    assert result.leaf_count == 2
    assert result.parameter_count == 3
    assert result.native_qnn_pytree
    assert result.custom_vjp_qnn_pytree
    assert result.custom_vjp_registered
    assert not result.native_qnn_host_callback
    assert not result.custom_vjp_qnn_host_callback
    assert result.max_abs_error <= 1e-10
    assert result.claim_boundary == "bounded_jax_pytree_compatibility"
    np.testing.assert_allclose(result.parameter_vector, np.array([0.25, 0.45, 0.65]))
    np.testing.assert_allclose(result.native_gradient_vector, expected, atol=1e-10)
    np.testing.assert_allclose(result.custom_vjp_gradient_vector, expected, atol=1e-10)
    np.testing.assert_allclose(result.native_gradient_pytree["encoder"], expected[:2], atol=1e-10)
    np.testing.assert_allclose(
        result.custom_vjp_gradient_pytree["readout"]["phase"],
        expected[2:],
        atol=1e-10,
    )
    assert result.to_dict()["passed"] is True


def test_phase_jax_pytree_compatibility_audit_fails_closed_without_tree_util(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NoTreeUtilJAX(_FakeJAX):
        tree_util = None

    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (_NoTreeUtilJAX(), np))

    with pytest.raises(RuntimeError, match="PyTree"):
        run_jax_pytree_compatibility_audit(
            features=np.array([[0.0], [np.pi]], dtype=float),
            labels=np.array([0.0, 1.0], dtype=float),
            params_pytree={"phase": np.array([0.45], dtype=float)},
        )


def test_phase_jax_maturity_audit_records_bounded_passes_and_provider_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jax = _FakeJAX()
    fake_jax.local_device_count_value = 2
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))
    features = np.array([[0.0, 0.2], [np.pi, np.pi + 0.2]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params = np.array([0.25, 0.45], dtype=float)
    params_batch = np.array([[0.25, 0.45], [0.35, 0.55]], dtype=float)
    params_tree = {"phase": params}

    result = run_jax_maturity_audit(
        features=features,
        labels=labels,
        params=params,
        params_batch=params_batch,
        params_pytree=params_tree,
        tolerance=1e-10,
    )
    payload = result.to_dict()

    assert isinstance(result, PhaseJAXMaturityAuditResult)
    assert result.bounded_model_ready
    assert not result.ready_for_provider_exceedance
    assert result.evidence["custom_vjp"].passed
    assert result.evidence["jit"].passed
    assert result.evidence["vmap"].passed
    assert result.evidence["pmap_sharding"].passed
    assert result.evidence["pytree"].passed
    assert result.evidence["cloud_validation_batch"].ready_for_cloud_dispatch
    assert "arbitrary_quantum_kernel_jax_lowering" in result.open_gaps
    assert "hardware_or_provider_callback_transform_safety" in result.open_gaps
    assert payload["required_capabilities"]["jit"] == "passed"
    assert payload["required_capabilities"]["cloud_validation_batch"] == "scheduled"
    assert payload["required_capabilities"]["arbitrary_quantum_kernel_jax_lowering"] == "blocked"
    assert payload["claim_boundary"] == "bounded_jax_provider_maturity_audit"


def test_phase_jax_cloud_validation_batch_schedules_gtx1060_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jax = _FakeJAX()
    fake_jax.local_device_count_value = 1
    fake_jax.local_devices = lambda: ("NVIDIA GeForce GTX 1060",)
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    result = plan_jax_cloud_validation_batch(runner="jarvislabs")

    assert isinstance(result, PhaseJAXCloudValidationRunSpec)
    assert result.runner == "jarvislabs"
    assert result.ready_for_cloud_dispatch
    assert result.local_execution_status == "skipped_incompatible_local_hardware"
    assert "GTX 1060" in result.local_skip_reason
    assert "jax_cuda_accelerator_device" in result.blocked_local_routes
    assert "registered_phase_qnode_pmap_multi_device_lowering" in result.blocked_local_routes
    assert "jax_cuda_device_metadata_artifact" in result.required_artifacts
    assert "registered_phase_qnode_jax_pmap_sharding_artifact" in result.required_artifacts
    assert "isolated_benchmark_artifact" in result.required_artifacts
    assert any("test_phase_jax_bridge.py" in command for command in result.commands)
    assert result.required_environment["accelerator_backend"] == "cuda"
    assert result.claim_boundary == "jax_cloud_validation_batch_plan"
    payload = result.to_dict()
    assert payload["local_execution_status"] == "skipped_incompatible_local_hardware"
    json.dumps(payload)


def test_phase_jax_nested_transform_algebra_audit_verifies_bounded_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jax = _FakeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))
    features = np.array([[0.0, 0.2], [np.pi, np.pi + 0.2]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params_batch = np.array([[0.25, 0.45], [0.35, 0.55]], dtype=float)
    params_pytree = {"encoder": np.array([0.25], dtype=float), "readout": (np.array([0.45]),)}

    result = run_jax_nested_transform_algebra_audit(
        features=features,
        labels=labels,
        params_batch=params_batch,
        params_pytree=params_pytree,
        tolerance=1e-8,
    )

    expected = np.vstack(
        [parameter_shift_qnn_classifier_gradient(features, labels, row) for row in params_batch]
    )
    assert isinstance(result, PhaseJAXNestedTransformAlgebraResult)
    assert result.passed
    assert result.bounded_transform_algebra_ready
    assert not result.ready_for_provider_exceedance
    assert result.route_status("jit_value_and_grad_under_vmap") == "passed"
    assert result.route_status("jit_vmap_value_and_grad") == "passed"
    assert result.route_status("jit_value_and_grad_pytree") == "passed"
    assert result.route_status("arbitrary_phase_qnode_jacfwd_jacrev") == "blocked"
    assert "hardware_provider_callback_transform_safety" in result.open_gaps
    assert result.claim_boundary == "bounded_jax_nested_transform_algebra"
    assert fake_jax.jit_calls >= 3
    assert fake_jax.vmap_calls >= 2
    np.testing.assert_allclose(result.jit_under_vmap_gradients, expected, atol=1e-8)
    np.testing.assert_allclose(result.jit_vmap_gradients, expected, atol=1e-8)
    np.testing.assert_allclose(
        result.pytree_gradient_vector,
        parameter_shift_qnn_classifier_gradient(features, labels, np.array([0.25, 0.45])),
        atol=1e-8,
    )
    payload = result.to_dict()
    assert payload["routes"]["arbitrary_phase_qnode_hessian"]["status"] == "blocked"


def test_phase_jax_phase_qnode_lowering_matrix_fails_closed_for_arbitrary_qnodes() -> None:
    result = run_jax_phase_qnode_lowering_matrix()

    assert isinstance(result, PhaseJAXPhaseQNodeLoweringMatrixResult)
    assert result.bounded_no_host_callback_routes_ready
    assert result.arbitrary_phase_qnode_lowering_ready
    assert not result.ready_for_provider_exceedance
    assert result.route_status("bounded_qnn_native_value_and_grad") == "passed"
    assert result.route_status("bounded_qnn_jit_value_and_grad") == "passed"
    assert result.route_status("bounded_qnn_vmap_value_and_grad") == "passed"
    assert result.route_status("registered_phase_qnode_statevector_lowering") == "passed"
    assert result.route_status("registered_phase_qnode_native_transform_lowering") == "passed"
    assert result.route_status("registered_phase_qnode_pytree_transform_lowering") == "passed"
    assert result.route_status("registered_phase_qnode_pmap_sharding_lowering") == "passed"
    assert result.route_status("registered_phase_qnode_provider_lowering") == "blocked"
    assert "registered_phase_qnode_statevector_lowering" not in result.open_gaps
    assert "registered_phase_qnode_native_transform_lowering" not in result.open_gaps
    assert "registered_phase_qnode_pytree_transform_lowering" not in result.open_gaps
    assert "registered_phase_qnode_pmap_sharding_lowering" not in result.open_gaps
    assert "isolated_benchmark_artifact" in result.open_gaps
    assert result.claim_boundary == "bounded_jax_phase_qnode_lowering_matrix"

    payload = result.to_dict()
    assert payload["routes"]["bounded_qnn_jit_value_and_grad"]["host_callback"] is False
    assert (
        payload["routes"]["registered_phase_qnode_statevector_lowering"]["host_callback"] is False
    )
    assert (
        payload["routes"]["registered_phase_qnode_native_transform_lowering"]["host_callback"]
        is False
    )
    assert (
        payload["routes"]["registered_phase_qnode_pytree_transform_lowering"]["host_callback"]
        is False
    )
    assert (
        payload["routes"]["registered_phase_qnode_pmap_sharding_lowering"]["host_callback"]
        is False
    )


def test_phase_jax_registered_qnode_native_transform_audit_uses_no_callback() -> None:
    """Registered Phase-QNode transforms should lower through native JAX APIs."""

    if not is_phase_jax_available():
        pytest.skip("JAX optional dependency is not installed")
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            ("ry", (0,), 0),
            ("rx", (1,), 1),
            ("cnot", (0, 1)),
        ),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )
    params = np.array([0.37, -0.21], dtype=float)

    result = jax_phase_qnode_native_transform_audit(
        circuit,
        params,
        tangent=np.array([0.25, -0.15], dtype=float),
        batch_offsets=np.array([[0.0, 0.0], [0.03, -0.01]], dtype=float),
        tolerance=5e-5,
    )
    reference = parameter_shift_phase_qnode_gradient(circuit, params)

    assert isinstance(result, PhaseJAXPhaseQNodeNativeTransformResult)
    assert result.passed
    assert result.native_framework_autodiff
    assert not result.host_callback
    assert result.jit_value_and_grad
    assert result.vmap_value_and_grad
    assert set(result.transform_names) == {
        "grad",
        "value_and_grad",
        "jacfwd",
        "jacrev",
        "hessian",
        "jvp",
        "vjp",
        "vmap",
        "jit",
    }
    np.testing.assert_allclose(result.value, reference.value, atol=5e-5)
    np.testing.assert_allclose(result.gradient, reference.gradient, atol=5e-5)
    np.testing.assert_allclose(result.value_and_grad_gradient, reference.gradient, atol=5e-5)
    np.testing.assert_allclose(result.jacfwd_gradient, reference.gradient, atol=5e-5)
    np.testing.assert_allclose(result.jacrev_gradient, reference.gradient, atol=5e-5)
    np.testing.assert_allclose(result.vjp_cotangent_gradient, reference.gradient, atol=5e-5)
    np.testing.assert_allclose(result.hessian, result.hessian.T, atol=5e-5)
    assert result.max_abs_hessian_symmetry_error <= 5e-5
    assert result.batch_params.shape == (2, 2)
    assert result.vmap_gradients.shape == (2, 2)
    payload = result.to_dict()
    assert payload["host_callback"] is False
    assert payload["method"] == "jax_native_registered_phase_qnode_transform_audit"
    assert payload["claim_boundary"] == "registered_phase_qnode_jax_native_transform_lowering"


def test_phase_jax_registered_qnode_pytree_transform_audit_uses_no_callback() -> None:
    """Registered Phase-QNode PyTrees should lower through native JAX transforms."""

    if not is_phase_jax_available():
        pytest.skip("JAX optional dependency is not installed")
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            ("ry", (0,), 0),
            ("rx", (1,), 1),
            ("cnot", (0, 1)),
        ),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )
    params_pytree = {
        "parameter_0": np.array([0.37], dtype=float),
        "parameter_1": (np.array([-0.21], dtype=float),),
    }
    flat_params = np.array([0.37, -0.21], dtype=float)

    result = jax_phase_qnode_pytree_transform_audit(
        circuit,
        params_pytree,
        tangent=np.array([0.25, -0.15], dtype=float),
        batch_offsets=np.array([[0.0, 0.0], [0.03, -0.01]], dtype=float),
        tolerance=5e-5,
    )
    reference = parameter_shift_phase_qnode_gradient(circuit, flat_params)

    assert isinstance(result, PhaseJAXPhaseQNodePyTreeTransformResult)
    assert result.passed
    assert result.native_framework_autodiff
    assert not result.host_callback
    assert result.jit_value_and_grad
    assert result.vmap_value_and_grad
    assert result.leaf_shapes == ((1,), (1,))
    assert set(result.transform_names) == {
        "grad",
        "value_and_grad",
        "jacfwd",
        "jacrev",
        "hessian",
        "jvp",
        "vjp",
        "vmap",
        "jit",
    }
    np.testing.assert_allclose(result.value, reference.value, atol=5e-5)
    np.testing.assert_allclose(result.gradient, reference.gradient, atol=5e-5)
    np.testing.assert_allclose(result.parameter_vector, flat_params, atol=5e-5)
    np.testing.assert_allclose(result.value_and_grad_gradient, reference.gradient, atol=5e-5)
    np.testing.assert_allclose(result.jacfwd_gradient, reference.gradient, atol=5e-5)
    np.testing.assert_allclose(result.jacrev_gradient, reference.gradient, atol=5e-5)
    np.testing.assert_allclose(result.vjp_cotangent_gradient, reference.gradient, atol=5e-5)
    assert result.hessian.shape == (2, 2)
    np.testing.assert_allclose(result.hessian, result.hessian.T, atol=5e-5)
    assert result.max_abs_hessian_symmetry_error <= 5e-5
    assert result.batch_params.shape == (2, 2)
    assert result.vmap_gradients.shape == (2, 2)
    payload = result.to_dict()
    assert payload["host_callback"] is False
    np.testing.assert_allclose(payload["hessian"], result.hessian, atol=5e-5)
    assert payload["max_abs_hessian_symmetry_error"] <= 5e-5
    assert payload["leaf_shapes"] == [[1], [1]]
    assert payload["method"] == "jax_native_registered_phase_qnode_pytree_transform_audit"
    assert payload["claim_boundary"] == "registered_phase_qnode_jax_pytree_transform_lowering"


def test_phase_jax_registered_qnode_sharding_transform_audit_uses_no_callback() -> None:
    """Registered Phase-QNode batches should lower through native JAX pmap."""

    if not is_phase_jax_available():
        pytest.skip("JAX optional dependency is not installed")
    import jax

    local_device_count = int(jax.local_device_count())
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            ("ry", (0,), 0),
            ("rx", (1,), 1),
            ("cnot", (0, 1)),
        ),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )
    base_params = np.array([0.37, -0.21], dtype=float)
    offsets = np.arange(local_device_count, dtype=float)[:, None] * np.array(
        [[0.01, -0.015]],
        dtype=float,
    )
    params_batch = base_params[None, :] + offsets

    result = jax_bridge.jax_phase_qnode_sharding_transform_audit(
        circuit,
        params_batch,
        tolerance=5e-5,
    )
    reference_gradients = np.vstack(
        [parameter_shift_phase_qnode_gradient(circuit, row).gradient for row in params_batch]
    )

    assert result.passed
    assert result.native_framework_autodiff
    assert not result.host_callback
    assert result.pmapped
    assert result.batch_size == local_device_count
    assert result.local_device_count == local_device_count
    assert result.sharding_mode in {"single_device_pmap_smoke", "multi_device_pmap"}
    assert result.values.shape == (local_device_count,)
    assert result.gradients.shape == (local_device_count, 2)
    np.testing.assert_allclose(result.gradients, reference_gradients, atol=5e-5)
    payload = result.to_dict()
    assert payload["host_callback"] is False
    assert payload["pmapped"] is True
    assert payload["local_device_count"] == local_device_count
    assert payload["claim_boundary"] == "registered_phase_qnode_jax_pmap_sharding_lowering"


def test_phase_jax_registered_qnode_native_transform_audit_fails_closed_without_transforms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial JAX modules should not pass as native-transform lowering."""

    class _MissingTransforms(_FakeJAX):
        grad = None

    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (_MissingTransforms(), np))
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )

    with pytest.raises(RuntimeError, match="grad"):
        jax_phase_qnode_native_transform_audit(circuit, np.array([0.2], dtype=float))


def test_phase_jax_registered_qnode_pytree_transform_audit_fails_closed_without_tree_util(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial JAX modules should not pass as PyTree transform lowering."""

    class _MissingTreeUtil(_FakeJAX):
        tree_util = None

    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (_MissingTreeUtil(), np))
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )

    with pytest.raises(RuntimeError, match="tree_util"):
        jax_phase_qnode_pytree_transform_audit(
            circuit,
            {"theta": np.array([0.2], dtype=float)},
        )


def test_phase_jax_registered_qnode_sharding_transform_audit_fails_closed_without_pmap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial JAX modules should not pass as PMAP sharding lowering."""

    class _MissingPMAP(_FakeJAX):
        pmap = None

    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (_MissingPMAP(), np))
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )

    with pytest.raises(RuntimeError, match="pmap"):
        jax_bridge.jax_phase_qnode_sharding_transform_audit(
            circuit,
            np.array([[0.2]], dtype=float),
        )


def test_phase_jax_registered_qnode_statevector_lowering_matches_scpn_reference() -> None:
    if not is_phase_jax_available():
        pytest.skip("JAX optional dependency is not installed")
    circuit = PhaseQNodeCircuit(
        n_qubits=3,
        operations=(
            ("h", (0,)),
            ("ry", (1,), 0),
            ("rx", (2,), 1),
            ("cnot", (1, 2)),
            ("crz", (0, 1), 2),
            ("rxx", (0, 2), 3),
            ("rzz", (1, 2), 4),
            ("ccnot", (0, 1, 2)),
        ),
        observable=SparsePauliHamiltonian(
            (
                PauliTerm(0.5, ((0, "z"),)),
                PauliTerm(-0.25, ((1, "x"), (2, "z"))),
                PauliTerm(0.75, ((0, "y"), (2, "y"))),
            )
        ),
    )
    params = np.array([0.21, -0.32, 0.43, -0.54, 0.65], dtype=float)

    result = jax_phase_qnode_value_and_grad(circuit, params, tolerance=2e-5)
    scpn_value = execute_phase_qnode_circuit(circuit, params).value
    scpn_gradient = parameter_shift_phase_qnode_gradient(circuit, params).gradient

    assert isinstance(result, PhaseJAXPhaseQNodeStatevectorResult)
    assert result.passed
    assert result.native_framework_autodiff
    assert not result.host_callback
    assert not result.jitted
    assert result.method == "jax_native_registered_phase_qnode_statevector_value_and_grad"
    np.testing.assert_allclose(result.value, scpn_value, atol=2e-5)
    np.testing.assert_allclose(result.gradient, scpn_gradient, atol=2e-5)
    np.testing.assert_allclose(result.parameter_shift_gradient, scpn_gradient, atol=1e-12)
    np.testing.assert_allclose(np.vdot(result.state, result.state).real, 1.0, atol=2e-5)
    assert result.to_dict()["host_callback"] is False


def test_phase_jax_registered_qnode_statevector_lowering_jits_without_callback() -> None:
    if not is_phase_jax_available():
        pytest.skip("JAX optional dependency is not installed")
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(("ry", (0,), 0), ("cnot", (0, 1)), ("rz", (1,), 1)),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )

    result = jax_phase_qnode_value_and_grad(
        circuit,
        np.array([0.17, -0.23], dtype=float),
        tolerance=2e-5,
        jit=True,
    )

    assert result.passed
    assert result.jit_requested
    assert result.jitted
    assert result.native_framework_autodiff
    assert not result.host_callback


def test_phase_jax_registered_qnode_lowering_covers_gate_and_observable_family() -> None:
    if not is_phase_jax_available():
        pytest.skip("JAX optional dependency is not installed")
    params = np.linspace(0.11, 0.91, 10)
    circuit = PhaseQNodeCircuit(
        n_qubits=3,
        operations=(
            ("h", (0,)),
            ("x", (1,)),
            ("y", (2,)),
            ("z", (0,)),
            ("s", (1,)),
            ("t", (2,)),
            ("sx", (0,)),
            ("rx", (0,), 0),
            ("ry", (1,), 1),
            ("rz", (2,), 2),
            ("phase", (0,), 3),
            ("cnot", (0, 1)),
            ("cz", (1, 2)),
            ("cy", (2, 0)),
            ("swap", (0, 2)),
            ("ch", (0, 1)),
            ("cs", (1, 2)),
            ("ct", (2, 0)),
            ("crx", (0, 1), 4),
            ("cry", (1, 2), 5),
            ("crz", (2, 0), 6),
            ("rxx", (0, 1), 7),
            ("ryy", (1, 2), 8),
            ("rzz", (0, 2), 9),
            ("ccnot", (0, 1, 2)),
            ("ccz", (0, 1, 2)),
            ("cswap", (0, 1, 2)),
        ),
        observable=SparsePauliHamiltonian(
            (
                PauliTerm(0.5, ((0, "x"),)),
                PauliTerm(-0.25, ((1, "y"), (2, "z"))),
                PauliTerm(0.75, ((0, "z"), (1, "z"), (2, "z"))),
            )
        ),
    )

    result = jax_phase_qnode_value_and_grad(circuit, params, tolerance=5e-6)
    reference = parameter_shift_phase_qnode_gradient(circuit, params)

    assert result.passed
    assert not result.host_callback
    np.testing.assert_allclose(result.value, reference.value, atol=5e-6)
    np.testing.assert_allclose(result.gradient, reference.gradient, atol=5e-6)


def test_phase_jax_registered_qnode_lowering_matches_dense_and_covariance_observables() -> None:
    if not is_phase_jax_available():
        pytest.skip("JAX optional dependency is not installed")
    dense_params = np.array([0.31, -0.17], dtype=float)
    covariance_params = np.array([0.23, -0.41], dtype=float)
    dense_circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0), ("rz", (0,), 1)),
        observable=DenseHermitianObservable(np.array([[1.0, 0.2], [0.2, -0.5]], dtype=float)),
    )
    covariance_circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(("ry", (0,), 0), ("cnot", (0, 1)), ("rx", (1,), 1)),
        observable=PauliCovarianceObservable(
            PauliTerm(1.0, ((0, "z"),)),
            PauliTerm(1.0, ((1, "x"),)),
        ),
    )

    dense = jax_phase_qnode_value_and_grad(
        dense_circuit,
        dense_params,
        tolerance=5e-6,
    )
    covariance = jax_phase_qnode_value_and_grad(
        covariance_circuit,
        covariance_params,
        tolerance=5e-6,
    )

    assert dense.passed
    assert covariance.passed
    np.testing.assert_allclose(
        dense.gradient,
        parameter_shift_phase_qnode_gradient(dense_circuit, dense_params).gradient,
        atol=5e-6,
    )
    np.testing.assert_allclose(
        covariance.gradient,
        parameter_shift_phase_qnode_gradient(covariance_circuit, covariance_params).gradient,
        atol=5e-6,
    )


def test_phase_jax_maturity_audit_fails_closed_on_bad_batch_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jax = _FakeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    with pytest.raises(ValueError, match="params_batch"):
        run_jax_maturity_audit(
            features=np.array([[0.0], [np.pi]], dtype=float),
            labels=np.array([0.0, 1.0], dtype=float),
            params=np.array([0.25], dtype=float),
            params_batch=np.array([0.25], dtype=float),
            params_pytree={"phase": np.array([0.25], dtype=float)},
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
    with pytest.raises(ImportError, match="blocked"):
        jax_native_qnn_value_and_grad(
            np.array([[0.0]], dtype=float),
            np.array([0.0], dtype=float),
            np.array([0.2], dtype=float),
        )
    with pytest.raises(ImportError, match="blocked"):
        jax_custom_vjp_qnn_value_and_grad(
            np.array([[0.0]], dtype=float),
            np.array([0.0], dtype=float),
            np.array([0.2], dtype=float),
        )


def test_phase_jax_bridge_import_runtime_failure_reports_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JAX import-time compatibility errors should fail closed as unavailable."""

    def incompatible_jax() -> tuple[object, object]:
        raise AttributeError("numpy dtype surface is incompatible with this JAX build")

    monkeypatch.setattr(jax_bridge, "_load_jax", incompatible_jax)

    assert not is_phase_jax_available()
