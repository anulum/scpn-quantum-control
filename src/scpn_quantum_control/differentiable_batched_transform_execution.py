# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native batched derivative execution
"""Execute native batched derivatives without QNode or reporting dependencies."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def jax_batched_gradient(
    jax: Any,
    objective: Callable[[Any], Any],
    parameters: Any,
    in_axes: Any,
) -> Any:
    """Differentiate a scalar objective and map it over a native JAX parameter tree.

    Parameters
    ----------
    jax
        Actual JAX runtime. Device and precision contexts remain caller-owned.
    objective
        Native scalar numerical objective at its actual defining scope.
    parameters
        Native parameter array or PyTree. Tensors retain their caller-supplied
        units and dtype; no host conversion or report record is constructed.
    in_axes
        Native vmap axis specification. A shared scalar bias uses the tree
        ({"weights": axis, "bias": None},), so only weights are mapped.

    Returns
    -------
    object
        Native gradient tree with the mapped sample axis first, including a
        separate bias derivative for each sample rather than a batch reduction.

    """
    return jax.vmap(jax.grad(objective), in_axes=in_axes)(parameters)


def torch_batched_gradient(
    torch: Any,
    objective: Callable[[Any], Any],
    parameters: Any,
    in_dims: Any,
) -> Any:
    """Differentiate a scalar objective and map it over native torch parameters.

    Parameters
    ----------
    torch
        Actual PyTorch runtime; tensor device and precision remain caller-owned.
    objective
        Native scalar numerical objective, including a registered custom rule.
    parameters
        Native array or parameter PyTree. No conversion to host arrays is made.
    in_dims
        Native vmap dimension tree. A shared scalar bias uses the tree
        ({"weights": axis, "bias": None},), so only weights are mapped.

    Returns
    -------
    object
        Native gradient tree ordered sample first. Shared-input derivatives are
        retained per sample and are not combined into an aggregate gradient.

    """
    return torch.func.vmap(torch.func.grad(objective), in_dims=in_dims)(parameters)


def tensorflow_vectorized_derivatives(
    tf: Any,
    derivative: Callable[[Any], Any],
    parameters: Any,
) -> Any:
    """Vectorise an actual TensorFlow derivative without a serial fallback.

    Parameters
    ----------
    tf
        Actual TensorFlow runtime within the caller's device context.
    derivative
        Native numerical derivative function whose definition, closed-over
        tensors, tape construction and recording contexts remain caller-owned.
    parameters
        Native tensors with a leading sample axis. Axis-one source batches
        must be transposed by the caller before invoking this operation.

    Returns
    -------
    object
        Native derivative tensors or nested containers, ordered sample first.
        Unsupported vectorisation propagates the native failure; a Python
        loop or TensorFlow while-loop fallback is never substituted.

    """
    return tf.vectorized_map(derivative, parameters, fallback_to_while_loop=False)


__all__ = [
    "jax_batched_gradient",
    "torch_batched_gradient",
    "tensorflow_vectorized_derivatives",
]
