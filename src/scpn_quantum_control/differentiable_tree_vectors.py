# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native derivative tree vector operations
"""Pack native parameter and derivative trees without circuit or model policy.

Tree traversal and native tensors remain SDK-owned. Host validation receives
the original caller numerical namespace explicitly, preserving late lookups.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sized
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


def parameter_tree_vector(
    jax_module: Any,
    name: str,
    params_pytree: object,
    *,
    numpy: Callable[[], Any],
    scalar: Callable[[], type[float]] = lambda: float,
    integer: Callable[[], type[int]] = lambda: int,
    sequence: Callable[[], type[tuple[Any, ...]]] = lambda: tuple,
    indexed: Callable[[], type[enumerate[Any]]] = lambda: enumerate,
    invalid: Callable[[], type[ValueError]] = lambda: ValueError,
    caster: Callable[[], Callable[..., Any]] = lambda: cast,
    array_type: Callable[[], object] = lambda: FloatArray,
) -> tuple[FloatArray, object, tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Validate numeric tree leaves and flatten them with reconstruction metadata.

    Parameters
    ----------
    jax_module
        Actual SDK tree flattening and reconstruction operations.
    name
        Original diagnostic label retained in validation errors.
    params_pytree
        Supplied numeric parameter tree.
    numpy
        Resolve the caller NumPy namespace at each original numerical lookup;
        this is a numerical namespace, with no model or circuit policy.

    scalar
        Resolve the caller scalar constructor at each dtype lookup.
    integer
        Resolve the original integer conversion for shape and size metadata.
    sequence
        Resolve the original tuple constructor when emitting leaf metadata.
    indexed
        Resolve the original indexed traversal of SDK leaves.
    invalid
        Resolve the original validation exception constructor at refusal.
    caster
        Resolve the original runtime type-conversion callable.
    array_type
        Resolve the original host array type alias at runtime conversion.

    Returns
    -------
    tuple
        Host vector, original tree definition, leaf shapes and leaf element counts.

    """
    leaves, treedef = jax_module.tree_util.tree_flatten(params_pytree)
    if not leaves:
        raise invalid()(f"{name} must contain at least one numeric leaf")
    arrays = []
    shapes = []
    sizes = []
    for index, leaf in indexed()(leaves):
        array = numpy().asarray(leaf, dtype=scalar())
        if not numpy().all(numpy().isfinite(array)):
            raise invalid()(f"{name} leaf {index} must contain only finite values")
        arrays.append(array.astype(numpy().float64, copy=True))
        shapes.append(sequence()(integer()(axis) for axis in array.shape))
        sizes.append(integer()(array.size))
    vector = numpy().concatenate([array.reshape(-1) for array in arrays])
    return (
        cast(FloatArray, caster()(array_type(), vector)),
        treedef,
        sequence()(shapes),
        sequence()(sizes),
    )


def jax_tree_vector(
    jax_module: Any,
    jnp: Any,
    params_pytree: object,
    *,
    length: Callable[[], Callable[[Sized], int]] = lambda: len,
) -> object:
    """Flatten actual JAX tree leaves while retaining their native transform graph.

    Parameters
    ----------
    jax_module
        Actual SDK tree flattening and reconstruction operations.
    jnp
        Actual JAX array operations within the caller precision context.
    params_pytree
        Supplied numeric parameter tree.

    length
        Resolve the original sequence-length operation.

    Returns
    -------
    object
        Native vector in the SDK tree traversal order.

    """
    leaves, _treedef = jax_module.tree_util.tree_flatten(params_pytree)
    parts = [jnp.ravel(jnp.asarray(leaf)) for leaf in leaves]
    if length()(parts) == 1:
        return parts[0]
    return jnp.concatenate(parts)


def jax_vector_tree(
    jax_module: Any,
    jnp: Any,
    vector: object,
    treedef: object,
    leaf_shapes: tuple[tuple[int, ...], ...],
    leaf_sizes: tuple[int, ...],
    *,
    paired: Callable[[], type[zip[Any]]] = lambda: zip,
) -> object:
    """Reconstruct native tree leaves from contiguous vector slices.

    Parameters
    ----------
    jax_module
        Actual SDK tree flattening and reconstruction operations.
    jnp
        Actual JAX array operations within the caller precision context.
    vector
        Native vector whose slices fill the supplied leaf shapes.
    treedef
        SDK tree definition retained by the caller.
    leaf_shapes
        Original leaf shapes in SDK traversal order.
    leaf_sizes
        Original leaf element counts, paired strictly with their shapes.

    paired
        Resolve the original strict pairing of leaf shapes and sizes.

    Returns
    -------
    object
        Native tree with the supplied definition and leaf shapes.

    """
    flat = jnp.ravel(jnp.asarray(vector))
    leaves = []
    offset = 0
    for shape, size in paired()(leaf_shapes, leaf_sizes, strict=True):
        leaves.append(jnp.reshape(flat[offset : offset + size], shape))
        offset += size
    return jax_module.tree_util.tree_unflatten(treedef, leaves)


def jax_batch_tree(
    jax_module: Any,
    jnp: Any,
    batch: object,
    treedef: object,
    leaf_shapes: tuple[tuple[int, ...], ...],
    leaf_sizes: tuple[int, ...],
    *,
    paired: Callable[[], type[zip[Any]]] = lambda: zip,
) -> object:
    """Reconstruct native batched tree leaves from contiguous matrix columns.

    Parameters
    ----------
    jax_module
        Actual SDK tree flattening and reconstruction operations.
    jnp
        Actual JAX array operations within the caller precision context.
    batch
        Native sample-by-parameter matrix.
    treedef
        SDK tree definition retained by the caller.
    leaf_shapes
        Original leaf shapes in SDK traversal order.
    leaf_sizes
        Original leaf element counts, paired strictly with their shapes.

    paired
        Resolve the original strict pairing of leaf shapes and sizes.

    Returns
    -------
    object
        Native tree whose leaves prepend the actual batch dimension.

    """
    batch_values = jnp.asarray(batch)
    leaves = []
    offset = 0
    for shape, size in paired()(leaf_shapes, leaf_sizes, strict=True):
        target_shape = (batch_values.shape[0], *shape)
        leaves.append(jnp.reshape(batch_values[:, offset : offset + size], target_shape))
        offset += size
    return jax_module.tree_util.tree_unflatten(treedef, leaves)


def gradient_tree_vector(
    jax_module: Any,
    name: str,
    gradient_pytree: object,
    *,
    width: int,
    numpy: Callable[[], Any],
    scalar: Callable[[], type[float]] = lambda: float,
    invalid: Callable[[], type[ValueError]] = lambda: ValueError,
    caster: Callable[[], Callable[..., Any]] = lambda: cast,
    array_type: Callable[[], object] = lambda: FloatArray,
) -> FloatArray:
    """Validate and flatten a supplied derivative tree into an independent host vector.

    Parameters
    ----------
    jax_module
        Actual SDK tree flattening and reconstruction operations.
    name
        Original diagnostic label retained in validation errors.
    gradient_pytree
        Supplied scalar or batched derivative tree.
    width
        Required flattened derivative width.
    numpy
        Resolve the caller NumPy namespace at each original numerical lookup;
        this is a numerical namespace, with no model or circuit policy.

    scalar
        Resolve the caller scalar constructor at each dtype lookup.
    invalid
        Resolve the original validation exception constructor at refusal.
    caster
        Resolve the original runtime type-conversion callable.
    array_type
        Resolve the original host array type alias at runtime conversion.

    Returns
    -------
    FloatArray
        Fresh finite float64 derivative vector with exactly the requested width.

    """
    leaves, _treedef = jax_module.tree_util.tree_flatten(gradient_pytree)
    if not leaves:
        raise invalid()(f"{name} must contain at least one gradient leaf")
    vector = numpy().concatenate(
        [numpy().asarray(leaf, dtype=scalar()).reshape(-1) for leaf in leaves]
    )
    if vector.shape != (width,):
        raise invalid()(f"{name} must flatten to shape ({width},), got {vector.shape}")
    if not numpy().all(numpy().isfinite(vector)):
        raise invalid()(f"{name} must contain only finite values")
    return cast(FloatArray, caster()(array_type(), vector.astype(numpy().float64, copy=True)))


def batched_gradient_tree_matrix(
    jax_module: Any,
    name: str,
    gradient_pytree: object,
    *,
    batch_size: int,
    width: int,
    numpy: Callable[[], Any],
    scalar: Callable[[], type[float]] = lambda: float,
    indexed: Callable[[], type[enumerate[Any]]] = lambda: enumerate,
    invalid: Callable[[], type[ValueError]] = lambda: ValueError,
    caster: Callable[[], Callable[..., Any]] = lambda: cast,
    array_type: Callable[[], object] = lambda: FloatArray,
) -> FloatArray:
    """Validate and assemble batched derivative leaves into a host matrix.

    Parameters
    ----------
    jax_module
        Actual SDK tree flattening and reconstruction operations.
    name
        Original diagnostic label retained in validation errors.
    gradient_pytree
        Supplied scalar or batched derivative tree.
    batch_size
        Required leading dimension of each derivative leaf.
    width
        Required flattened derivative width.
    numpy
        Resolve the caller NumPy namespace at each original numerical lookup;
        this is a numerical namespace, with no model or circuit policy.

    scalar
        Resolve the caller scalar constructor at each dtype lookup.
    indexed
        Resolve the original indexed traversal of SDK leaves.
    invalid
        Resolve the original validation exception constructor at refusal.
    caster
        Resolve the original runtime type-conversion callable.
    array_type
        Resolve the original host array type alias at runtime conversion.

    Returns
    -------
    FloatArray
        Fresh finite float64 batch matrix in the original SDK leaf order.

    """
    leaves, _treedef = jax_module.tree_util.tree_flatten(gradient_pytree)
    if not leaves:
        raise invalid()(f"{name} must contain at least one gradient leaf")
    parts = []
    for index, leaf in indexed()(leaves):
        array = numpy().asarray(leaf, dtype=scalar())
        if array.shape[0] != batch_size:
            raise invalid()(f"{name} leaf {index} must have leading batch size {batch_size}")
        parts.append(array.reshape(batch_size, -1))
    matrix = numpy().concatenate(parts, axis=1)
    if matrix.shape != (batch_size, width):
        raise invalid()(
            f"{name} must flatten to shape ({batch_size}, {width}), got {matrix.shape}"
        )
    if not numpy().all(numpy().isfinite(matrix)):
        raise invalid()(f"{name} must contain only finite values")
    return cast(FloatArray, caster()(array_type(), matrix.astype(numpy().float64, copy=True)))


def hessian_tree_matrix(
    jax_module: Any,
    name: str,
    hessian_pytree: object,
    *,
    leaf_shapes: tuple[tuple[int, ...], ...],
    leaf_sizes: tuple[int, ...],
    numpy: Callable[[], Any],
    scalar: Callable[[], type[float]] = lambda: float,
    integer: Callable[[], type[int]] = lambda: int,
    length: Callable[[], Callable[[Sized], int]] = lambda: len,
    paired: Callable[[], type[zip[Any]]] = lambda: zip,
    accumulate: Callable[[], Callable[[Iterable[int]], int]] = lambda: sum,
    invalid: Callable[[], type[ValueError]] = lambda: ValueError,
    caster: Callable[[], Callable[..., Any]] = lambda: cast,
    array_type: Callable[[], object] = lambda: FloatArray,
) -> FloatArray:
    """Assemble derivative-tree blocks into a dense host Hessian matrix.

    Parameters
    ----------
    jax_module
        Actual SDK tree flattening and reconstruction operations.
    name
        Original diagnostic label retained in validation errors.
    hessian_pytree
        Supplied tree of pairwise derivative blocks.
    leaf_shapes
        Original leaf shapes in SDK traversal order.
    leaf_sizes
        Original leaf element counts, paired strictly with their shapes.
    numpy
        Resolve the caller NumPy namespace at each original numerical lookup;
        this is a numerical namespace, with no model or circuit policy.

    scalar
        Resolve the caller scalar constructor at each dtype lookup.
    integer
        Resolve the original integer conversion for shape and size metadata.
    length
        Resolve the original sequence-length operation.
    paired
        Resolve the original strict pairing of leaf shapes and sizes.
    accumulate
        Resolve the original reduction of leaf sizes.
    invalid
        Resolve the original validation exception constructor at refusal.
    caster
        Resolve the original runtime type-conversion callable.
    array_type
        Resolve the original host array type alias at runtime conversion.

    Returns
    -------
    FloatArray
        Finite float64 block matrix retaining row and column leaf order.

    """
    blocks, _treedef = jax_module.tree_util.tree_flatten(hessian_pytree)
    leaf_count = length()(leaf_sizes)
    expected_blocks = leaf_count * leaf_count
    if length()(blocks) != expected_blocks:
        raise invalid()(f"{name} must contain {expected_blocks} Hessian blocks")
    width = integer()(accumulate()(leaf_sizes))
    matrix = numpy().zeros((width, width), dtype=numpy().float64)
    row_offset = 0
    block_index = 0
    for row_shape, row_size in paired()(leaf_shapes, leaf_sizes, strict=True):
        col_offset = 0
        for col_shape, col_size in paired()(leaf_shapes, leaf_sizes, strict=True):
            block = numpy().asarray(blocks[block_index], dtype=scalar())
            expected_shape = (*row_shape, *col_shape)
            if block.shape != expected_shape:
                raise invalid()(
                    f"{name} block {block_index} must have shape {expected_shape}, "
                    f"got {block.shape}"
                )
            matrix[
                row_offset : row_offset + row_size,
                col_offset : col_offset + col_size,
            ] = block.reshape(row_size, col_size)
            col_offset += col_size
            block_index += 1
        row_offset += row_size
    if not numpy().all(numpy().isfinite(matrix)):
        raise invalid()(f"{name} must contain only finite values")
    return cast(FloatArray, caster()(array_type(), matrix))


__all__ = [
    "parameter_tree_vector",
    "jax_tree_vector",
    "jax_vector_tree",
    "jax_batch_tree",
    "gradient_tree_vector",
    "batched_gradient_tree_matrix",
    "hessian_tree_matrix",
]
