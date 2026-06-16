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
    PhaseJAXCustomVJPQNNGradientResult,
    PhaseJAXGradientAgreementResult,
    PhaseJAXJITCompatibilityResult,
    PhaseJAXMaturityAuditResult,
    PhaseJAXNativeQNNGradientResult,
    PhaseJAXNestedTransformAlgebraResult,
    PhaseJAXParameterShiftResult,
    PhaseJAXPyTreeCompatibilityResult,
    PhaseJAXShardingCompatibilityResult,
    PhaseJAXVMAPCompatibilityResult,
    check_jax_parameter_shift_agreement,
    is_phase_jax_available,
    jax_custom_vjp_qnn_value_and_grad,
    jax_native_qnn_value_and_grad,
    jax_parameter_shift_value_and_grad,
    multi_frequency_parameter_shift_rule,
    parameter_shift_qnn_classifier_gradient,
    run_jax_jit_compatibility_audit,
    run_jax_maturity_audit,
    run_jax_nested_transform_algebra_audit,
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
    assert "arbitrary_quantum_kernel_jax_lowering" in result.open_gaps
    assert "hardware_or_provider_callback_transform_safety" in result.open_gaps
    assert payload["required_capabilities"]["jit"] == "passed"
    assert payload["required_capabilities"]["arbitrary_quantum_kernel_jax_lowering"] == "blocked"
    assert payload["claim_boundary"] == "bounded_jax_provider_maturity_audit"


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
