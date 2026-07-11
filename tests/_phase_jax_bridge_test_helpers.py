# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX Bridge Test Helpers
"""Fake JAX runtime shared by bridge integration tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
FakeCallable: TypeAlias = Callable[[Any], Any]


class _FakeJAX:
    class _TreeUtil:
        @staticmethod
        def tree_flatten(tree: Any) -> tuple[list[Any], Any]:
            leaves: list[Any] = []

            def visit(node: Any) -> Any:
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
        def tree_unflatten(treedef: Any, leaves: list[Any]) -> Any:
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

    tree_util: ClassVar[Any] = _TreeUtil()

    def __init__(self) -> None:
        self.jit_calls: int = 0
        self.callback_calls: int = 0
        self.callback_shape_dtypes: Any = None
        self.custom_vjp_calls: int = 0
        self.custom_vjp_defvjp_calls: int = 0
        self.vmap_calls: int = 0
        self.pmap_calls: int = 0
        self.local_device_count_value: int = 2

    def jit(self, fn: FakeCallable) -> FakeCallable:
        self.jit_calls += 1

        def wrapped(values: Any) -> Any:
            return fn(values)

        return wrapped

    def pure_callback(self, callback: FakeCallable, _shape_dtypes: Any, values: Any) -> Any:
        self.callback_calls += 1
        self.callback_shape_dtypes = _shape_dtypes
        return callback(values)

    def custom_vjp(self, fn: FakeCallable) -> Any:
        self.custom_vjp_calls += 1
        fake_jax = self

        class _CustomVJPFunction:
            def __init__(self, primal_fn: FakeCallable) -> None:
                self._primal_fn = primal_fn
                self._forward: Any = None
                self._backward: Any = None

            def defvjp(self, forward: Any, backward: Any) -> None:
                fake_jax.custom_vjp_defvjp_calls += 1
                self._forward = forward
                self._backward = backward

            def __call__(self, values: Any) -> Any:
                return self._primal_fn(values)

            def value_and_grad(self, values: Any) -> Any:
                if self._forward is None or self._backward is None:
                    raise RuntimeError("custom_vjp rule has not been registered")
                value, residual = self._forward(values)
                (gradient,) = self._backward(residual, np.asarray(1.0, dtype=float))
                return value, gradient

        return _CustomVJPFunction(fn)

    def value_and_grad(self, fn: Any) -> FakeCallable:
        def wrapped(values: Any) -> Any:
            if hasattr(fn, "value_and_grad"):
                return fn.value_and_grad(values)
            leaves, treedef = self.tree_util.tree_flatten(values)
            arrays = [np.asarray(leaf, dtype=float) for leaf in leaves]
            sizes = [array.size for array in arrays]
            shapes = [array.shape for array in arrays]
            array = np.concatenate([array.ravel() for array in arrays])

            def rebuild(flat_values: FloatArray) -> Any:
                offset = 0
                rebuilt: list[FloatArray] = []
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

    def grad(self, fn: Any) -> FakeCallable:
        def wrapped(values: Any) -> Any:
            _value, gradient = self.value_and_grad(fn)(values)
            return gradient

        return wrapped

    def jacfwd(self, fn: Any) -> FakeCallable:
        return self.grad(fn)

    def jacrev(self, fn: Any) -> FakeCallable:
        return self.grad(fn)

    def hessian(self, fn: Any) -> FakeCallable:
        gradient_fn = self.grad(fn)

        def wrapped(values: Any) -> FloatArray:
            array = np.asarray(values, dtype=float)
            hessian = np.zeros((array.size, array.size), dtype=float)
            step = 1e-5
            for index in range(array.size):
                forward = array.copy()
                backward = array.copy()
                forward[index] += step
                backward[index] -= step
                hessian[:, index] = (gradient_fn(forward) - gradient_fn(backward)) / (2.0 * step)
            return cast(FloatArray, 0.5 * (hessian + hessian.T))

        return wrapped

    def jvp(self, fn: FakeCallable, primals: tuple[Any], tangents: tuple[Any]) -> tuple[Any, Any]:
        (values,) = primals
        (tangent,) = tangents
        value = fn(values)
        gradient = self.grad(fn)(values)
        return value, np.asarray(np.dot(gradient, tangent), dtype=float)

    def vjp(self, fn: FakeCallable, values: Any) -> tuple[Any, FakeCallable]:
        value = fn(values)
        gradient = self.grad(fn)(values)

        def pullback(cotangent: Any) -> tuple[Any]:
            return (np.asarray(cotangent, dtype=float) * gradient,)

        return value, pullback

    def vmap(self, fn: FakeCallable) -> FakeCallable:
        self.vmap_calls += 1

        def wrapped(values: Any) -> Any:
            outputs = [fn(row) for row in np.asarray(values, dtype=float)]
            if outputs and isinstance(outputs[0], tuple):
                return tuple(np.stack(items, axis=0) for items in zip(*outputs, strict=True))
            return np.asarray(outputs, dtype=float)

        return wrapped

    def pmap(self, fn: FakeCallable) -> FakeCallable:
        self.pmap_calls += 1

        def wrapped(values: Any) -> Any:
            outputs = [fn(row) for row in np.asarray(values, dtype=float)]
            if outputs and isinstance(outputs[0], tuple):
                return tuple(np.stack(items, axis=0) for items in zip(*outputs, strict=True))
            return np.asarray(outputs, dtype=float)

        return wrapped

    def local_device_count(self) -> int:
        return self.local_device_count_value

    def local_devices(self) -> list[str]:
        return [f"fake-device-{index}" for index in range(self.local_device_count_value)]


def _objective(values: FloatArray) -> float:
    return float(np.cos(values[0]) + 0.25 * np.sin(values[1]))


class _Float32JNP:
    @staticmethod
    def asarray(values: Any) -> NDArray[np.float32]:
        return np.asarray(values, dtype=np.float32)
