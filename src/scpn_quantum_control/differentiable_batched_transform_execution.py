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


def jax_vectorized_derivatives(
    jax: Any, derivative: Callable[[Any], Any], parameters: Any, in_axes: Any
) -> Any:
    """Map an existing native JAX derivative with its sample axis first.

    Parameters
    ----------
    jax
        Actual runtime within caller-owned numerical contexts.
    derivative
        Original native derivative callable, constructed and retained by the caller.
    parameters
        Native batched inputs without a host copy or substituted container.
    in_axes
        Original native mapping axis or axis tree.

    Returns
    -------
    object
        Native vectorized derivative, retaining explicit out_axes=0 semantics.

    """
    return jax.vmap(derivative, in_axes=in_axes, out_axes=0)(parameters)


def torch_vectorized_derivatives(
    torch: Any, derivative: Callable[[Any], Any], parameters: Any, in_dims: Any
) -> Any:
    """Map an existing torch.func derivative with its sample dimension first.

    Parameters
    ----------
    torch
        Actual runtime with caller-owned tensor placement and precision.
    derivative
        Original native gradient function, without a replacement closure.
    parameters
        Actual native batched inputs in their original layout.
    in_dims
        Original native mapping dimension or dimension tree.

    Returns
    -------
    object
        Native vectorized derivative with explicit out_dims=0 semantics.

    """
    return torch.func.vmap(derivative, in_dims=in_dims, out_dims=0)(parameters)


def jax_vectorized_sum(
    jax: Any, jnp: Any, objective: Callable[[Any], Any], parameters: Any, in_axes: Any
) -> Any:
    """Sum native mapped JAX objectives in the original reduction and lookup order.

    Parameters
    ----------
    jax
        Actual JAX transform provider.
    jnp
        Actual native array namespace under the caller's precision context.
    objective
        Original numerical objective whose defining scope remains unchanged.
    parameters
        Native batch in the caller-supplied layout.
    in_axes
        Native axis or axis tree used by the original vectorized map.

    Returns
    -------
    object
        Native sum of mapped outputs; the caller may differentiate this expression.

    """
    return jnp.sum(jax.vmap(objective, in_axes=in_axes)(parameters))


def torch_vectorized_sum(
    torch: Any, objective: Callable[[Any], Any], parameters: Any, in_dims: Any
) -> Any:
    """Sum native mapped torch objectives without changing transform lookup order.

    Parameters
    ----------
    torch
        Actual runtime whose sum and func.vmap attributes are read in order.
    objective
        Original scalar numerical function, including its captured native inputs.
    parameters
        Native batch, preserving its original device, dtype and layout.
    in_dims
        Native mapping dimension or dimension tree.

    Returns
    -------
    object
        Native reduction retaining its differentiation graph and original sum policy.

    """
    return torch.sum(torch.func.vmap(objective, in_dims=in_dims)(parameters))


def tensorflow_vectorized_sum(tf: Any, objective: Callable[[Any], Any], parameters: Any) -> Any:
    """Sum TensorFlow mapped values without a serial vectorization fallback.

    Parameters
    ----------
    tf
        Actual runtime under the caller's active device and recording contexts.
    objective
        Original numerical function with its existing captured tensors and scopes.
    parameters
        Native tensor batch with the sample dimension first.

    Returns
    -------
    object
        Native reduced value, retaining the original vectorized_map and reduce_sum
        expression order and the caller-owned gradient recording context.

    """
    return tf.reduce_sum(tf.vectorized_map(objective, parameters, fallback_to_while_loop=False))


__all__ = [
    "jax_batched_gradient",
    "torch_batched_gradient",
    "tensorflow_vectorized_derivatives",
    "jax_vectorized_derivatives",
    "torch_vectorized_derivatives",
    "jax_vectorized_sum",
    "torch_vectorized_sum",
    "tensorflow_vectorized_sum",
]
