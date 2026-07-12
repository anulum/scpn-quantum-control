# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Torch Bridge Test Helpers
"""Strictly typed fake Torch runtime shared by bridge integration tests."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from scpn_quantum_control.phase import parameter_shift_qnn_classifier_gradient

FloatArray = NDArray[np.float64]


class _FakeTorchTensor:
    """NumPy-backed tensor implementing the PyTorch operations under test."""

    def __init__(self, values: object) -> None:
        """Initialize deterministic state for the fake framework object."""
        if isinstance(values, _FakeTorchTensor):
            values = values.numpy()
        self._values = np.asarray(values, dtype=float)
        self.grad: _FakeTorchTensor | None = None

    def detach(self) -> _FakeTorchTensor:
        """Return this tensor without changing its NumPy-backed state."""
        return self

    def clone(self) -> _FakeTorchTensor:
        """Return an independent copy of this fake tensor."""
        return _FakeTorchTensor(self._values.copy())

    def requires_grad_(self, requires_grad: bool = True) -> _FakeTorchTensor:
        """Record whether callers requested gradient tracking."""
        del requires_grad
        return self

    def cpu(self) -> _FakeTorchTensor:
        """Return this CPU-only fake tensor unchanged."""
        return self

    def numpy(self) -> FloatArray:
        """Return the tensor payload as a NumPy array."""
        return self._values.copy()

    def __mul__(self, other: object) -> _FakeTorchTensor:
        """Apply elementwise multiplication with NumPy broadcasting."""
        if isinstance(other, _FakeTorchTensor):
            return _FakeTorchTensor(self._values * other._values)
        return _FakeTorchTensor(self._values * np.asarray(other, dtype=float))

    __rmul__ = __mul__

    def __add__(self, other: object) -> _FakeTorchTensor:
        """Apply elementwise addition with NumPy broadcasting."""
        if isinstance(other, _FakeTorchTensor):
            return _FakeTorchTensor(self._values + other._values)
        return _FakeTorchTensor(self._values + np.asarray(other, dtype=float))

    __radd__ = __add__

    def __sub__(self, other: object) -> _FakeTorchTensor:
        """Apply elementwise subtraction with NumPy broadcasting."""
        if isinstance(other, _FakeTorchTensor):
            return _FakeTorchTensor(self._values - other._values)
        return _FakeTorchTensor(self._values - np.asarray(other, dtype=float))

    def __rsub__(self, other: object) -> _FakeTorchTensor:
        """Apply reflected elementwise subtraction."""
        if isinstance(other, _FakeTorchTensor):
            return _FakeTorchTensor(other._values - self._values)
        return _FakeTorchTensor(np.asarray(other, dtype=float) - self._values)

    def unsqueeze(self, axis: int) -> _FakeTorchTensor:
        """Insert one singleton dimension at the requested axis."""
        return _FakeTorchTensor(np.expand_dims(self._values, axis=axis))


class _FakeTorchAutogradFunction:
    """Minimal custom-autograd function dispatcher for bridge tests."""

    @classmethod
    def apply(cls, *args: object) -> _FakeTorchTensor:
        """Execute the supplied fake custom-autograd function."""
        ctx = type("_FakeAutogradContext", (), {})()
        result = cast(Any, cls).forward(ctx, *args)
        result._ctx = ctx
        result._function_cls = cls
        return cast(_FakeTorchTensor, result)


class _FakeTorchAutograd:
    """Finite-difference autograd facade for deterministic bridge tests."""

    Function = _FakeTorchAutogradFunction

    def grad(
        self,
        outputs: _FakeTorchTensor,
        inputs: _FakeTorchTensor,
        *,
        retain_graph: bool = False,
        create_graph: bool = False,
    ) -> tuple[_FakeTorchTensor]:
        """Build or evaluate the deterministic fake gradient operation."""
        del inputs, retain_graph, create_graph
        backward = outputs._function_cls.backward  # type: ignore[attr-defined]
        result = backward(outputs._ctx, _FakeTorchTensor(np.asarray(1.0, dtype=float)))  # type: ignore[attr-defined]
        if isinstance(result, tuple):
            return result
        return (result,)


class _FakeTorchModule:
    """Small module base exposing buffers, parameters, and calls."""

    def __init__(self) -> None:
        """Initialize deterministic state for the fake framework object."""
        self._buffers: dict[str, _FakeTorchTensor] = {}

    def register_buffer(self, name: str, tensor: _FakeTorchTensor) -> None:
        """Attach a named non-trainable buffer to the fake module."""
        self._buffers[name] = tensor
        setattr(self, name, tensor)

    def parameters(self) -> tuple[_FakeTorchTensor, ...]:
        """Return the fake module's trainable parameter tuple."""
        params = getattr(self, "params", None)
        if params is None:
            return ()
        return (params,)

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Execute the fake callable with NumPy-backed inputs."""
        return cast(Any, self).forward(*args, **kwargs)


class _FakeTorchNN:
    """Namespace containing the fake PyTorch module and parameter types."""

    Module = _FakeTorchModule

    @staticmethod
    def Parameter(values: object, *, requires_grad: bool = True) -> _FakeTorchTensor:
        """Wrap values as a trainable fake PyTorch tensor."""
        return _FakeTorchTensor(values).requires_grad_(requires_grad)


class _FakeTorchFunc:
    """Deterministic torch.func facade recording transform usage."""

    def __init__(self) -> None:
        """Initialize deterministic state for the fake framework object."""
        self.grad_calls = 0
        self.vmap_calls = 0
        self.jacrev_calls = 0

    def grad(self, loss_fn: object) -> object:
        """Build or evaluate the deterministic fake gradient operation."""
        del loss_fn
        self.grad_calls += 1

        def gradient(params: object) -> _FakeTorchTensor:
            """Evaluate the recorded gradient transform for input values."""
            return _FakeTorchTensor(self._gradient(params))

        return gradient

    def vmap(self, gradient_fn: object) -> object:
        """Build a deterministic vectorizing transform wrapper."""
        self.vmap_calls += 1

        def mapped(params_batch: object) -> _FakeTorchTensor:
            """Evaluate the wrapped function across the leading batch axis."""
            batch = np.asarray(_FakeTorchTensor(params_batch).numpy(), dtype=float)
            return _FakeTorchTensor(np.vstack([self._gradient(row) for row in batch]))

        return mapped

    def jacrev(self, loss_fn: object) -> object:
        """Build a deterministic reverse-Jacobian transform wrapper."""
        del loss_fn
        self.jacrev_calls += 1

        def jacobian(params: object) -> _FakeTorchTensor:
            """Evaluate the finite-difference Jacobian for one input vector."""
            return _FakeTorchTensor(self._gradient(params))

        return jacobian

    @staticmethod
    def _gradient(params: object) -> FloatArray:
        """Evaluate a central finite-difference gradient in float64."""
        features = np.array([[0.0], [np.pi]], dtype=float)
        labels = np.array([0.0, 1.0], dtype=float)
        return parameter_shift_qnn_classifier_gradient(
            features,
            labels,
            _FakeTorchTensor(params).numpy(),
        )


class _FakeTorch:
    """Bounded PyTorch facade backed by NumPy test doubles."""

    float64 = np.float64
    __version__ = "fake-torch"

    def __init__(self) -> None:
        """Initialize deterministic state for the fake framework object."""
        self.autograd = _FakeTorchAutograd()
        self.func = _FakeTorchFunc()
        self.nn = _FakeTorchNN()
        self.as_tensor_calls: list[FloatArray] = []
        self.compile_calls: list[dict[str, object]] = []

    def as_tensor(self, values: object, *, dtype: object | None = None) -> _FakeTorchTensor:
        """Convert values to a NumPy-backed fake PyTorch tensor."""
        del dtype
        array = np.asarray(values, dtype=float)
        self.as_tensor_calls.append(array.copy())
        return _FakeTorchTensor(array)

    def compile(
        self,
        fn: object,
        *,
        fullgraph: bool = True,
        dynamic: bool = False,
    ) -> object:
        """Record compilation and return the callable unchanged."""
        self.compile_calls.append({"fullgraph": fullgraph, "dynamic": dynamic})
        return fn

    def cos(self, values: object) -> _FakeTorchTensor:
        """Apply elementwise cosine while preserving fake tensor wrapping."""
        return _FakeTorchTensor(np.cos(_FakeTorchTensor(values).numpy()))

    def mean(self, values: object, *, dim: int | None = None) -> _FakeTorchTensor:
        """Return the scalar mean as a fake tensor."""
        return _FakeTorchTensor(np.mean(_FakeTorchTensor(values).numpy(), axis=dim))


class _FakeTorchWithoutAutogradFunction(_FakeTorch):
    """PyTorch facade lacking custom autograd support."""

    def __init__(self) -> None:
        """Initialize deterministic state for the fake framework object."""
        super().__init__()
        cast(Any, self).autograd = object()


class _FakeTorchWithoutFunc(_FakeTorch):
    """PyTorch facade lacking the torch.func transform namespace."""

    def __init__(self) -> None:
        """Initialize deterministic state for the fake framework object."""
        super().__init__()
        cast(Any, self).func = object()


class _FakeTorchWithoutCompile(_FakeTorch):
    """PyTorch facade lacking the torch.compile entry point."""

    compile: Any = None


class _FakeTorchWithoutNN(_FakeTorch):
    """PyTorch facade lacking the torch.nn module namespace."""

    def __init__(self) -> None:
        """Initialize deterministic state for the fake framework object."""
        super().__init__()
        cast(Any, self).nn = object()


def _objective(values: FloatArray) -> float:
    """Evaluate the shared two-parameter cosine objective in radians."""
    return float(np.cos(values[0]) + 0.25 * np.sin(values[1]))
