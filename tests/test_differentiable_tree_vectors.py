# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native derivative tree vector tests
"""Exercise derivative assembly and original circuit consumers with actual JAX."""

from __future__ import annotations

import builtins
import importlib
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from numpy.typing import DTypeLike, NDArray

from scpn_quantum_control.differentiable_tree_vectors import (
    batched_gradient_tree_matrix,
    gradient_tree_vector,
    hessian_tree_matrix,
    jax_batch_tree,
    jax_tree_vector,
    jax_vector_tree,
    parameter_tree_vector,
)
from scpn_quantum_control.phase.qnode_circuit import (
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeOperation,
)

pytestmark = pytest.mark.framework_imports


@pytest.fixture(autouse=True)
def _double_precision() -> Iterator[None]:
    """Use and restore actual JAX double precision independently of process defaults."""
    jax = importlib.import_module("jax")
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_mixed_shape_tree_reconstruction_retains_jit_and_derivatives() -> None:
    """Reconstruct scalar, matrix and empty leaves inside real native transforms."""
    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    tree = {"a": np.array(0.2), "b": (np.array([[0.3, -0.4]]), np.empty((0,)))}
    vector, treedef, shapes, sizes = parameter_tree_vector(jax, "params", tree, numpy=lambda: np)
    np.testing.assert_array_equal(vector, [0.2, 0.3, -0.4])
    assert shapes == ((), (1, 2), (0,)) and sizes == (1, 2, 0)

    def objective(values: Any) -> Any:
        rebuilt = jax_vector_tree(jax, jnp, values, treedef, shapes, sizes)
        packed = jax_tree_vector(jax, jnp, rebuilt)
        return jnp.sum(jnp.power(packed, 3))

    native = jnp.asarray(vector)
    np.testing.assert_allclose(jax.jit(objective)(native), np.sum(vector**3), atol=1e-14)
    np.testing.assert_allclose(jax.jit(jax.grad(objective))(native), 3 * vector**2, atol=1e-14)
    np.testing.assert_allclose(jax.jit(jax.hessian(objective))(native), np.diag(6 * vector))
    batch = np.stack([vector, vector + 0.1])
    rebuilt = jax_batch_tree(jax, jnp, jnp.asarray(batch), treedef, shapes, sizes)
    leaves, _definition = jax.tree_util.tree_flatten(rebuilt)
    assert leaves[0].shape == (2,)
    assert leaves[1].shape == (2, 1, 2)
    assert leaves[2].shape == (2, 0)
    np.testing.assert_array_equal(leaves[1][:, 0, :], batch[:, 1:])
    single = jax_tree_vector(jax, jnp, {"only": native})
    np.testing.assert_array_equal(single, vector)
    vector[0] = 9.0
    assert tree["a"] == 0.2


def test_actual_derivative_trees_match_analytic_gradient_batch_and_hessian() -> None:
    """Assemble actual SDK derivative blocks of a coupled scalar/vector objective."""
    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    tree = {"a": jnp.asarray(0.2), "b": jnp.asarray([0.3, -0.4])}
    _vector, _definition, shapes, sizes = parameter_tree_vector(
        jax, "params", tree, numpy=lambda: np
    )

    def objective(values: Any) -> Any:
        return values["a"] ** 2 + values["a"] * jnp.sum(values["b"]) + jnp.sum(values["b"] ** 3)

    gradient = gradient_tree_vector(
        jax, "gradient", jax.grad(objective)(tree), width=3, numpy=lambda: np
    )
    np.testing.assert_allclose(gradient, [0.3, 0.47, 0.68], atol=1e-14)
    blocks = jax.hessian(objective)(tree)
    hessian = hessian_tree_matrix(
        jax, "hessian", blocks, leaf_shapes=shapes, leaf_sizes=sizes, numpy=lambda: np
    )
    np.testing.assert_allclose(
        hessian, [[2.0, 1.0, 1.0], [1.0, 1.8, 0.0], [1.0, 0.0, -2.4]], atol=1e-14
    )
    batch = jax.tree_util.tree_map(lambda leaf: jnp.stack([leaf, leaf + 0.1]), tree)
    gradients = jax.jit(jax.vmap(jax.grad(objective)))(batch)
    packed = batched_gradient_tree_matrix(
        jax, "batch", gradients, batch_size=2, width=3, numpy=lambda: np
    )
    np.testing.assert_allclose(packed, [[0.3, 0.47, 0.68], [0.7, 0.78, 0.57]], atol=1e-14)
    packed[0, 0] = 99
    assert float(gradients["a"][0]) == pytest.approx(0.3)


@pytest.mark.parametrize(
    "kind,tree,width,batch_size,message",
    [
        ("parameters", {}, 0, 0, "at least one numeric leaf"),
        ("parameters", {"x": [np.nan]}, 0, 0, "leaf 0 must contain only finite"),
        ("gradient", {}, 0, 0, "at least one gradient leaf"),
        ("gradient", {"x": np.array([1.0])}, 2, 0, "must flatten to shape"),
        ("gradient", {"x": np.array([np.inf])}, 1, 0, "only finite"),
        ("batch", {}, 0, 2, "at least one gradient leaf"),
        ("batch", {"x": np.ones((3, 1))}, 1, 2, "leading batch size 2"),
        ("batch", {"x": np.ones((2, 1))}, 2, 2, "must flatten to shape"),
        ("batch", {"x": np.array([[np.nan], [1.0]])}, 1, 2, "only finite"),
    ],
)
def test_real_tree_inputs_preserve_validation_errors(
    kind: str, tree: object, width: int, batch_size: int, message: str
) -> None:
    """Reject actual malformed trees through their original ordered validations.

    Parameters
    ----------
    kind
        Numerical validation family under test.
    tree
        Actual malformed numeric tree supplied to the SDK.
    width
        Required flattened derivative width.
    batch_size
        Required leading batch dimension.
    message
        Original diagnostic fragment.

    """
    jax = importlib.import_module("jax")
    with pytest.raises(ValueError, match=message):
        if kind == "parameters":
            parameter_tree_vector(jax, "params", tree, numpy=lambda: np)
        elif kind == "gradient":
            gradient_tree_vector(jax, "gradient", tree, width=width, numpy=lambda: np)
        else:
            batched_gradient_tree_matrix(
                jax, "batch", tree, width=width, batch_size=batch_size, numpy=lambda: np
            )


@pytest.mark.parametrize(
    "blocks,shapes,sizes,message",
    [
        ([], ((),), (1,), "must contain 1 Hessian blocks"),
        ([np.ones((2,))], ((),), (1,), "block 0 must have shape"),
        ([np.array(np.nan)], ((),), (1,), "only finite"),
        ([np.array(1.0)], (), (1,), r"zip\(\) argument 2 is longer than argument 1"),
    ],
)
def test_real_hessian_blocks_preserve_shape_finiteness_and_metadata_errors(
    blocks: object, shapes: tuple[tuple[int, ...], ...], sizes: tuple[int, ...], message: str
) -> None:
    """Reject malformed blocks and inconsistent strict metadata pairings.

    Parameters
    ----------
    blocks
        Actual supplied block arrays.
    shapes
        Leaf shape metadata used by the block assembler.
    sizes
        Paired leaf sizes.
    message
        Original error fragment.

    """
    jax = importlib.import_module("jax")
    with pytest.raises(ValueError, match=message):
        hessian_tree_matrix(
            jax, "hessian", blocks, leaf_shapes=shapes, leaf_sizes=sizes, numpy=lambda: np
        )


def test_original_parameter_helper_preserves_late_numpy_namespace_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rebind the original global during actual NumPy conversion to actual JAX.

    Parameters
    ----------
    monkeypatch
        Restore the original defining module namespace after the test.

    """
    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    original = importlib.import_module("scpn_quantum_control.phase.jax_qnode_transforms")

    class RebindingLeaf:
        """Supply an actual NumPy array while rebinding the original numerical module."""

        def __array__(
            self, dtype: DTypeLike | None = None, copy: bool | None = None
        ) -> NDArray[np.float64]:
            """Return numeric values at the original conversion boundary.

            Parameters
            ----------
            dtype
                NumPy conversion dtype supplied by the original helper.
            copy
                Optional NumPy copy request.

            Returns
            -------
            NDArray
                Actual finite NumPy values, leaving subsequent reads to JAX.

            """
            monkeypatch.setattr(original, "np", jnp)
            return np.asarray([0.2, -0.3], dtype=np.float64)

    result, _tree, _shapes, _sizes = original._as_pytree_parameter_vector(
        jax, "params", RebindingLeaf()
    )
    assert isinstance(result, jax.Array)
    np.testing.assert_array_equal(result, [0.2, -0.3])


def test_empty_reconstruction_and_hessian_preserve_original_zero_width_behavior() -> None:
    """Retain empty admitted reconstruction without admitting empty parameters."""
    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    _leaves, definition = jax.tree_util.tree_flatten({})
    assert jax_vector_tree(jax, jnp, jnp.empty((0,)), definition, (), ()) == {}
    assert jax_batch_tree(jax, jnp, jnp.empty((2, 0)), definition, (), ()) == {}
    matrix = hessian_tree_matrix(
        jax, "hessian", {}, leaf_shapes=(), leaf_sizes=(), numpy=lambda: np
    )
    assert matrix.shape == (0, 0) and matrix.dtype == np.float64


def test_original_helpers_retain_scalar_error_cast_and_sequence_global_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise all seven original helpers using their actual rebound primitives.

    Parameters
    ----------
    monkeypatch
        Restore the original defining module after observing real numerical calls.

    """
    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    original = importlib.import_module("scpn_quantum_control.phase.jax_qnode_transforms")
    seen: set[str] = set()
    alias = object()

    class InputError(ValueError):
        """Identify the original caller's actual refusal constructor."""

    def conversion(type_alias: object, value: Any) -> Any:
        assert type_alias is alias
        seen.add("cast")
        return value

    for name in ("int", "tuple", "len", "enumerate", "zip", "sum"):
        actual = getattr(builtins, name)

        def observe(*args: Any, _name: str = name, _actual: Any = actual, **kwargs: Any) -> Any:
            seen.add(_name)
            return _actual(*args, **kwargs)

        monkeypatch.setattr(original, name, observe, raising=False)
    monkeypatch.setattr(original, "float", np.float32, raising=False)
    monkeypatch.setattr(original, "ValueError", InputError, raising=False)
    monkeypatch.setattr(original, "cast", conversion)
    monkeypatch.setattr(original, "FloatArray", alias)
    value = 0.1234567890123
    tree = {"x": np.array([value])}
    vector, definition, shapes, sizes = original._as_pytree_parameter_vector(jax, "p", tree)
    expected = np.asarray([value], dtype=np.float32).astype(np.float64)
    np.testing.assert_array_equal(vector, expected)
    np.testing.assert_array_equal(original._jax_flatten_pytree(jax, jnp, tree), [value])
    rebuilt = original._jax_unflatten_vector_to_pytree(jax, jnp, vector, definition, shapes, sizes)
    np.testing.assert_array_equal(rebuilt["x"], expected)
    batch_tree = original._jax_unflatten_batch_to_pytree(
        jax, jnp, vector[None, :], definition, shapes, sizes
    )
    np.testing.assert_array_equal(batch_tree["x"], expected[None, :])
    np.testing.assert_array_equal(
        original._flatten_runtime_pytree_gradient(jax, "g", tree, width=1), expected
    )
    np.testing.assert_array_equal(
        original._flatten_batched_runtime_pytree_gradient(
            jax, "g", {"x": np.array([[value]])}, batch_size=1, width=1
        ),
        expected[None, :],
    )
    np.testing.assert_array_equal(
        original._flatten_runtime_pytree_hessian(
            jax, "h", [np.array([[value]])], leaf_shapes=shapes, leaf_sizes=sizes
        ),
        expected[None, :],
    )
    assert seen == {"cast", "int", "tuple", "len", "enumerate", "zip", "sum"}
    for helper, kwargs, invalid in (
        (original._as_pytree_parameter_vector, {}, {}),
        (original._flatten_runtime_pytree_gradient, {"width": 1}, {}),
        (original._flatten_batched_runtime_pytree_gradient, {"batch_size": 1, "width": 1}, {}),
        (
            original._flatten_runtime_pytree_hessian,
            {"leaf_shapes": shapes, "leaf_sizes": sizes},
            [],
        ),
    ):
        with pytest.raises(InputError):
            helper(jax, "invalid", invalid, **kwargs)


@pytest.mark.parametrize("structured", [False, True])
def test_existing_ad_leaves_receive_the_original_numerical_objective(
    monkeypatch: pytest.MonkeyPatch, structured: bool
) -> None:
    """Observe the actual SDK transform without replacing the objective or its scope.

    Parameters
    ----------
    monkeypatch
        Restore the actual JAX transform after observing its numerical inputs.
    structured
        Exercise the original vector or tree-based public audit.

    """
    jax = importlib.import_module("jax")
    original = importlib.import_module("scpn_quantum_control.phase.jax_qnode_transforms")
    actual_grad = jax.grad
    seen: list[tuple[str, bool]] = []

    def observe(objective: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(objective, "__module__", None) == original.__name__:
            seen.append((objective.__qualname__, objective.__globals__ is original.__dict__))
        return actual_grad(objective, *args, **kwargs)

    monkeypatch.setattr(jax, "grad", observe)
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(PhaseQNodeOperation("ry", (0,), parameter_index=0),),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )
    name = "jax_phase_qnode_" + (
        "pytree_transform_audit" if structured else "native_transform_audit"
    )
    parameters = {"theta": np.array([0.2])} if structured else np.array([0.2])
    result = getattr(original, name)(circuit, parameters, tolerance=1e-10)
    assert result.passed
    assert seen and set(seen) == {(name + ".<locals>.value_function", True)}
    np.testing.assert_allclose(result.gradient, [-np.sin(0.2)], atol=1e-14)
    np.testing.assert_allclose(result.hessian, [[-np.cos(0.2)]], atol=1e-14)


def test_original_public_pytree_circuit_matches_closed_form_mixed_derivatives() -> None:
    """Reach the extracted AD owner through the real registered-circuit bridge."""
    bridge = importlib.import_module("scpn_quantum_control.phase.jax_bridge")
    a, b, c = 0.2, -0.3, 0.4
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("ry", (1,), parameter_index=1),
            PhaseQNodeOperation("rz", (0,), parameter_index=2),
        ),
        observable=PauliTerm(1.0, ((0, "x"), (1, "x"))),
    )
    result = bridge.jax_phase_qnode_pytree_transform_audit(
        circuit, {"a": np.array(a), "bc": np.array([b, c])}, tolerance=1e-10
    )
    value = np.sin(a) * np.sin(b) * np.cos(c)
    ab = np.cos(a) * np.cos(b) * np.cos(c)
    ac = -np.cos(a) * np.sin(b) * np.sin(c)
    bc = -np.sin(a) * np.cos(b) * np.sin(c)
    assert result.passed and not result.host_callback
    np.testing.assert_allclose(result.value, value, atol=1e-14)
    np.testing.assert_allclose(
        result.gradient,
        [
            np.cos(a) * np.sin(b) * np.cos(c),
            np.sin(a) * np.cos(b) * np.cos(c),
            -np.sin(a) * np.sin(b) * np.sin(c),
        ],
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result.hessian, [[-value, ab, ac], [ab, -value, bc], [ac, bc, -value]], atol=1e-14
    )
