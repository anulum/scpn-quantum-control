# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native binary tensor axis operators
"""Apply a supplied linear operator to supplied binary tensor axes.

Callers retain axis admission, ordering and inverse-permutation construction.
Native SDK arrays retain their dtype, device and differentiation graph. These
operations contain no circuit registration, gate selection or model policy.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


def jax_binary_axis_matmul(
    namespace: Any,
    state: Any,
    matrix: Any,
    *,
    dimensions: int,
    width: int,
    axes: Sequence[int],
) -> Any:
    """Multiply a native JAX tensor after moving its selected binary axes forward.

    Parameters
    ----------
    namespace
        Actual JAX array operations within the caller's precision context.
    state
        Native flattened tensor containing ``2**dimensions`` entries.
    matrix
        Native square matrix with ``2**width`` rows.
    dimensions
        Number of size-two tensor dimensions.
    width
        Number of dimensions acted on by the supplied matrix.
    axes
        Admitted permutation placing the operator's dimensions first.

    Returns
    -------
    object
        Native size-two tensor in the supplied axis order. Inverse-permutation
        construction and restoration remain with the caller. Incompatible
        shapes propagate the SDK's own error.

    """
    tensor = namespace.transpose(namespace.reshape(state, (2,) * dimensions), axes)
    front = namespace.reshape(tensor, (2**width, -1))
    return namespace.reshape(matrix @ front, (2,) * dimensions)


def torch_binary_axis_matmul(
    namespace: Any,
    state: Any,
    matrix: Any,
    *,
    dimensions: int,
    width: int,
    axes: Callable[[], tuple[int, ...]],
) -> Any:
    """Apply a native Torch matrix without changing its autograd or device graph.

    Parameters
    ----------
    namespace
        Actual PyTorch reshape and permutation operations.
    state
        Native flattened tensor containing ``2**dimensions`` entries.
    matrix
        Native square matrix with ``2**width`` rows.
    dimensions
        Number of size-two tensor dimensions.
    width
        Number of dimensions acted on by the supplied matrix.
    axes
        Resolve the admitted permutation after the initial reshape, retaining
        the original caller's tuple-construction point.

    Returns
    -------
    object
        Native size-two tensor in the supplied axis order. Shape errors and
        native resource semantics propagate unchanged.

    """
    tensor = namespace.permute(namespace.reshape(state, (2,) * dimensions), axes())
    front = namespace.reshape(tensor, (2**width, -1))
    return namespace.reshape(matrix @ front, (2,) * dimensions)


def restore_binary_axis_order(
    tensor: Any,
    axes: tuple[int, ...],
    *,
    reshape: Callable[[Any, tuple[int, ...]], Any],
    permute: Callable[[Any, tuple[int, ...]], Any],
) -> Any:
    """Restore a native tensor's dimension order using already resolved operations.

    Parameters
    ----------
    tensor
        Actual result of the binary-axis matrix multiplication.
    axes
        Inverse permutation evaluated by the caller at its original call site.
    reshape
        Exact native reshape callable resolved before the permutation callable.
    permute
        Actual native transpose or permutation callable, without wrapping it.

    Returns
    -------
    object
        Native flattened tensor in the original dimension order. The native
        operations and differentiation graph are retained without host copies.

    """
    return reshape(permute(tensor, axes), (-1,))


__all__ = ["jax_binary_axis_matmul", "torch_binary_axis_matmul", "restore_binary_axis_order"]
