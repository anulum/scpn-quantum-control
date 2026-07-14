# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX Phase-QNode Test Helpers
"""Strictly typed fake runtimes for registered Phase-QNode JAX tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, cast

import numpy as np
from _phase_jax_bridge_test_helpers import FakeCallable, _FakeJAX
from numpy.typing import NDArray

from scpn_quantum_control.phase import PauliTerm, PhaseQNodeCircuit


class _AtUpdate:
    """NumPy implementation of JAX's immutable ``array.at[index].set`` surface."""

    def __init__(self, array: _AtArray) -> None:
        self._array = array
        self._index: Any = None

    def __getitem__(self, index: Any) -> _AtUpdate:
        """Record the index targeted by the subsequent immutable update."""
        self._index = index
        return self

    def set(self, value: object) -> _AtArray:
        """Return a copied array with the recorded index updated."""
        copied = np.asarray(self._array).copy()
        copied[self._index] = value
        return _AtArray(copied)


class _AtArray(np.ndarray[Any, Any]):
    """NumPy array subtype exposing the JAX ``at`` update property."""

    def __new__(cls, values: object) -> _AtArray:
        """View arbitrary NumPy-compatible values as an ``_AtArray``."""
        return cast(_AtArray, np.asarray(values).view(cls))

    @property
    def at(self) -> _AtUpdate:
        """Return an immutable-update indexer for this array."""
        return _AtUpdate(self)


class _NumpyJNP:
    """Small NumPy-backed ``jax.numpy`` substitute for deterministic fake tests."""

    complex128: ClassVar[type[np.complex128]] = np.complex128

    def zeros(self, shape: Any, *, dtype: Any = None) -> _AtArray:
        """Return a zero array supporting JAX-style immutable updates."""
        return _AtArray(np.zeros(shape, dtype=dtype))

    def eye(self, size: int, *, dtype: Any = None) -> _AtArray:
        """Return an identity array supporting JAX-style immutable updates."""
        return _AtArray(np.eye(size, dtype=dtype))

    def __getattr__(self, name: str) -> Any:
        """Delegate the remaining numerical surface to NumPy."""
        return getattr(np, name)


_NUMPY_JNP = _NumpyJNP()


def _single_parameter_circuit(*, gate: str = "ry") -> PhaseQNodeCircuit:
    """Return a one-qubit circuit with one indexed parameter."""
    return PhaseQNodeCircuit(
        n_qubits=1,
        operations=((gate, (0,), 0),),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )


class _FakePyTreeJAX(_FakeJAX):
    """Fake JAX transform runtime that preserves nested parameter PyTrees."""

    def _gradient(self, fn: FakeCallable, values: Any) -> Any:
        """Return the base fake runtime's finite-difference PyTree gradient."""
        _value, gradient = _FakeJAX.value_and_grad(self, fn)(values)
        return gradient

    def grad(self, fn: Any) -> FakeCallable:
        """Return a finite-difference gradient function for a PyTree input."""

        def wrapped(values: Any) -> Any:
            """Evaluate the PyTree gradient."""
            return self._gradient(fn, values)

        return wrapped

    def jacfwd(self, fn: Any) -> FakeCallable:
        """Return the scalar-output forward-Jacobian analogue."""

        def wrapped(values: Any) -> Any:
            """Evaluate the forward-Jacobian analogue."""
            return self._gradient(fn, values)

        return wrapped

    def jacrev(self, fn: Any) -> FakeCallable:
        """Return the scalar-output reverse-Jacobian analogue."""

        def wrapped(values: Any) -> Any:
            """Evaluate the reverse-Jacobian analogue."""
            return self._gradient(fn, values)

        return wrapped

    def hessian(self, _fn: Any) -> FakeCallable:
        """Return a symmetric zero Hessian with JAX-compatible PyTree blocks."""

        def wrapped(values: Any) -> tuple[tuple[NDArray[np.float64], ...], ...]:
            """Build one Hessian block for every ordered pair of leaves."""
            leaves, _treedef = self.tree_util.tree_flatten(values)
            arrays = tuple(np.asarray(leaf, dtype=float) for leaf in leaves)
            return tuple(
                tuple(np.zeros((*row.shape, *column.shape), dtype=np.float64) for column in arrays)
                for row in arrays
            )

        return wrapped

    def jvp(
        self,
        fn: FakeCallable,
        primals: tuple[Any],
        tangents: tuple[Any],
    ) -> tuple[Any, NDArray[np.float64]]:
        """Return a scalar primal and directional derivative for PyTrees."""
        (values,) = primals
        (tangent,) = tangents
        gradient = self._gradient(fn, values)
        gradient_leaves, _gradient_def = self.tree_util.tree_flatten(gradient)
        tangent_leaves, _tangent_def = self.tree_util.tree_flatten(tangent)
        gradient_vector = np.concatenate(
            [np.asarray(leaf, dtype=float).reshape(-1) for leaf in gradient_leaves]
        )
        tangent_vector = np.concatenate(
            [np.asarray(leaf, dtype=float).reshape(-1) for leaf in tangent_leaves]
        )
        return fn(values), np.asarray(np.dot(gradient_vector, tangent_vector), dtype=float)

    def vjp(self, fn: FakeCallable, values: Any) -> tuple[Any, FakeCallable]:
        """Return a scalar primal and PyTree cotangent pullback."""
        gradient = self._gradient(fn, values)
        leaves, treedef = self.tree_util.tree_flatten(gradient)

        def pullback(cotangent: Any) -> tuple[Any]:
            """Scale every gradient leaf by the scalar cotangent."""
            scalar = np.asarray(cotangent, dtype=float)
            scaled = [scalar * np.asarray(leaf, dtype=float) for leaf in leaves]
            return (self.tree_util.tree_unflatten(treedef, scaled),)

        return fn(values), pullback

    def vmap(self, fn: FakeCallable) -> FakeCallable:
        """Vectorize a value-and-gradient function over PyTree leading axes."""
        self.vmap_calls += 1

        def wrapped(values: Any) -> tuple[NDArray[np.float64], Any]:
            """Stack scalar values and gradient PyTrees over the leading axis."""
            leaves, treedef = self.tree_util.tree_flatten(values)
            arrays = tuple(np.asarray(leaf, dtype=float) for leaf in leaves)
            outputs = [
                fn(
                    self.tree_util.tree_unflatten(
                        treedef,
                        [array[index] for array in arrays],
                    )
                )
                for index in range(arrays[0].shape[0])
            ]
            scalar_values = np.asarray([output[0] for output in outputs], dtype=np.float64)
            gradient_trees = [output[1] for output in outputs]
            gradient_leaves, gradient_def = self.tree_util.tree_flatten(gradient_trees[0])
            stacked_leaves = []
            for leaf_index in range(len(gradient_leaves)):
                stacked_leaves.append(
                    np.stack(
                        [
                            self.tree_util.tree_flatten(tree)[0][leaf_index]
                            for tree in gradient_trees
                        ],
                        axis=0,
                    )
                )
            return scalar_values, self.tree_util.tree_unflatten(gradient_def, stacked_leaves)

        return wrapped


class _FakeConfig:
    """Minimal JAX config shim for x64 enablement."""

    def update(self, _key: str, _value: object) -> None:
        """Accept a configuration update without side effects."""


class _FakeShapeDtypeStruct:
    """Shape/dtype token accepted by the fake JAX export surface."""

    def __init__(self, shape: tuple[int, ...], dtype: type[np.float64] | str) -> None:
        self.shape = shape
        self.dtype = np.dtype(dtype)


class _FakeCompiled:
    """Compiled fake executable returning the reference test value."""

    def __call__(self, _values: object) -> NDArray[np.float64]:
        """Return the deterministic compiled value."""
        return np.asarray(0.0, dtype=np.float64)

    def as_text(self) -> str:
        """Return fake compiled executable text."""
        return "compiled stablehlo executable"

    def cost_analysis(self) -> dict[str, float]:
        """Return deterministic fake compiler cost metadata."""
        return {"flops": 0.0}


class _FakeLowered:
    """Lowered fake JAX stage carrying StableHLO metadata."""

    def as_text(self) -> str:
        """Return fake lowered text."""
        return "module @registered_phase_qnode { stablehlo.return }"

    def compiler_ir(self, *, dialect: str = "stablehlo") -> str:
        """Return fake compiler IR for the requested dialect."""
        return f"{dialect}.module @registered_phase_qnode"

    def compile(self) -> _FakeCompiled:
        """Return a fake compiled executable."""
        return _FakeCompiled()


class _FakeJitted:
    """Jitted fake callable exposing the AOT lower API."""

    def __call__(self, _values: object) -> NDArray[np.float64]:
        """Return the deterministic jitted value."""
        return np.asarray(0.0, dtype=np.float64)

    def lower(self, *_args: object) -> _FakeLowered:
        """Return a lowered fake stage."""
        return _FakeLowered()


class _FakeExported:
    """Fake exported JAX callable with serialization metadata."""

    platforms: ClassVar[tuple[str, ...]] = ("cpu",)
    calling_convention_version: ClassVar[int] = 10
    uses_global_constants: ClassVar[bool] = False
    disabled_safety_checks: ClassVar[tuple[str, ...]] = ()

    def mlir_module(self) -> str:
        """Return fake MLIR module text."""
        return "module @main { func.func public @main() }"

    def serialize(self, *, vjp_order: int = 0) -> bytearray:
        """Return deterministic fake serialized export bytes."""
        return bytearray(f"fake-export-vjp-{vjp_order}", encoding="ascii")

    def call(self, _values: object) -> NDArray[np.float64]:
        """Replay the deterministic exported value."""
        return np.asarray(0.0, dtype=np.float64)


class _FakeExportModule:
    """Fake ``jax.export`` module supporting export and deserialize."""

    minimum_supported_calling_convention_version: ClassVar[int] = 9
    maximum_supported_calling_convention_version: ClassVar[int] = 10

    @staticmethod
    def export(_jitted: _FakeJitted) -> Callable[[_FakeShapeDtypeStruct], _FakeExported]:
        """Return a fake export builder."""

        def build(_shape: _FakeShapeDtypeStruct) -> _FakeExported:
            """Return one fake exported program."""
            return _FakeExported()

        return build

    @staticmethod
    def deserialize(_blob: bytearray) -> _FakeExported:
        """Return a fake deserialized export."""
        return _FakeExported()


class _FakeAOTJAX:
    """Minimal JAX module shim for AOT/export diagnostics."""

    ShapeDtypeStruct: ClassVar[type[_FakeShapeDtypeStruct]] = _FakeShapeDtypeStruct
    config: ClassVar[_FakeConfig] = _FakeConfig()
    export: ClassVar[_FakeExportModule | None] = _FakeExportModule()

    def __init__(self) -> None:
        self.jit_calls = 0

    def jit(self, _fn: Callable[[object], object]) -> _FakeJitted:
        """Return a fake jitted function and count calls."""
        self.jit_calls += 1
        return _FakeJitted()


def _export_module(fake_jax: _FakeAOTJAX) -> _FakeExportModule:
    """Return the configured non-null fake export module."""
    return cast(_FakeExportModule, fake_jax.export)
