# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native Jacobian and Hessian execution
"""Execute native derivative matrices without QNode or reporting dependencies."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def jax_hessian(jax: Any, objective: Callable[[Any], Any], parameters: Any) -> Any:
    """Evaluate the native JAX Hessian of a scalar numerical objective.

    Parameters
    ----------
    jax
        Actual runtime under caller-owned device and precision contexts.
    objective
        Original native scalar function of one parameter array or PyTree.
    parameters
        Native inputs, retaining their units, dtype and container structure.

    Returns
    -------
    object
        Native Hessian array or PyTree, without host conversion or report policy.

    """
    return jax.hessian(objective)(parameters)


def jax_jacfwd(jax: Any, function: Callable[[Any], Any], parameters: Any) -> Any:
    """Evaluate the native JAX forward Jacobian of a numerical function.

    Parameters
    ----------
    jax
        Actual runtime selected by the numerical caller.
    function
        Native function of one parameter argument; output may be rectangular.
    parameters
        Original native primal inputs under the caller's active contexts.

    Returns
    -------
    object
        Native forward Jacobian with output dimensions before input dimensions.

    """
    return jax.jacfwd(function)(parameters)


def jax_jacrev(jax: Any, function: Callable[[Any], Any], parameters: Any) -> Any:
    """Evaluate the native JAX reverse Jacobian of a numerical function.

    Parameters
    ----------
    jax
        Actual JAX runtime with caller-owned precision and device contexts.
    function
        Original native numerical function of one parameter argument.
    parameters
        Native inputs, with no host conversion or flattened replacement.

    Returns
    -------
    object
        Native reverse Jacobian with output dimensions before input dimensions.

    """
    return jax.jacrev(function)(parameters)


def jax_reverse_gradient_jacobian(
    jax: Any, objective: Callable[[Any], Any], parameters: Any
) -> Any:
    """Compose JAX reverse Jacobian and gradient in the original native order.

    Parameters
    ----------
    jax
        Actual JAX runtime, retaining live native transform lookups.
    objective
        Original scalar numerical objective, including registered custom rules.
    parameters
        Native primal array or parameter PyTree.

    Returns
    -------
    object
        Native reverse-over-reverse curvature, without an intermediate host copy.

    """
    return jax.jacrev(jax.grad(objective))(parameters)


def torch_hessian(torch: Any, objective: Callable[[Any], Any], parameters: Any) -> Any:
    """Evaluate the native torch.func Hessian of a scalar numerical objective.

    Parameters
    ----------
    torch
        Actual runtime; the caller supplies tensor device and precision.
    objective
        Native scalar function at its original defining scope.
    parameters
        Original native array or parameter PyTree.

    Returns
    -------
    object
        Native Hessian, retaining the runtime's output layout and dtype.

    """
    return torch.func.hessian(objective)(parameters)


def torch_jacfwd(torch: Any, function: Callable[[Any], Any], parameters: Any) -> Any:
    """Evaluate the native torch.func forward Jacobian of a numerical function.

    Parameters
    ----------
    torch
        Actual PyTorch runtime chosen by the numerical caller.
    function
        Native function of one parameter argument, with arbitrary output width.
    parameters
        Native primal inputs in their caller-supplied units and container.

    Returns
    -------
    object
        Native Jacobian with output dimensions before input dimensions.

    """
    return torch.func.jacfwd(function)(parameters)


def torch_jacrev(torch: Any, function: Callable[[Any], Any], parameters: Any) -> Any:
    """Evaluate the native torch.func reverse Jacobian of a numerical function.

    Parameters
    ----------
    torch
        Actual PyTorch runtime, with no provider selection performed here.
    function
        Native numerical function retaining its original defining namespace.
    parameters
        Original native primal input array or PyTree.

    Returns
    -------
    object
        Native reverse Jacobian, without conversion to a report or host array.

    """
    return torch.func.jacrev(function)(parameters)


def torch_forward_gradient_jacobian(
    torch: Any, objective: Callable[[Any], Any], parameters: Any
) -> Any:
    """Compose torch.func forward Jacobian and gradient in native evaluation order.

    Parameters
    ----------
    torch
        Actual runtime whose live transform attributes are read in their order.
    objective
        Original scalar objective, including its registered native custom rule.
    parameters
        Native inputs with the caller's device, dtype and parameter structure.

    Returns
    -------
    object
        Native forward-over-reverse curvature with no host conversion.

    """
    return torch.func.jacfwd(torch.func.grad(objective))(parameters)


def torch_reverse_gradient_jacobian(
    torch: Any, objective: Callable[[Any], Any], parameters: Any
) -> Any:
    """Compose torch.func reverse Jacobian and gradient in native evaluation order.

    Parameters
    ----------
    torch
        Actual PyTorch runtime under caller-owned numerical contexts.
    objective
        Original scalar numerical function and its existing custom rule.
    parameters
        Native primal array or parameter PyTree.

    Returns
    -------
    object
        Native reverse-over-reverse curvature, with no reporting wrapper.

    """
    return torch.func.jacrev(torch.func.grad(objective))(parameters)


def tensorflow_forward_jacobian(
    tf: Any, forward_product: Callable[[Any], Any], sample_matrix: Any
) -> Any:
    """Assemble a native forward Jacobian from the original float64 basis mapping.

    Parameters
    ----------
    tf
        Actual TensorFlow runtime under the caller's numerical contexts.
    forward_product
        Original vector-valued JVP function whose accumulator, recording context
        and captured primal inputs remain caller-owned.
    sample_matrix
        Caller-supplied two-dimensional sample layout. Its second dimension
        supplies the domain width; values are neither read nor converted.

    Returns
    -------
    object
        Native two-dimensional Jacobian ordered output first. Basis construction, vectorised
        mapping with no while-loop fallback, and transpose retain the original
        native lookup and expression order.

    """
    return tf.transpose(
        tf.vectorized_map(
            forward_product,
            tf.eye(sample_matrix.shape[1], dtype=tf.float64),
            fallback_to_while_loop=False,
        )
    )


def tensorflow_jacobian(tape: Any, value: Any, parameters: Any) -> Any:
    """Read a native serial Jacobian from the caller's TensorFlow recording tape.

    Parameters
    ----------
    tape
        Actual caller-owned tape that watched the primal inputs and recorded
        the value. Its construction, context, persistence and lifetime stay
        with the caller.
    value
        Recorded native output or gradient tensor.
    parameters
        Original watched native inputs or container.

    Returns
    -------
    object
        Native Jacobian from the original experimental_use_pfor=False route.
        No tape replacement, host conversion or report decision is performed.

    """
    return tape.jacobian(value, parameters, experimental_use_pfor=False)


__all__ = [
    "jax_hessian",
    "jax_jacfwd",
    "jax_jacrev",
    "jax_reverse_gradient_jacobian",
    "torch_hessian",
    "torch_jacfwd",
    "torch_jacrev",
    "torch_forward_gradient_jacobian",
    "torch_reverse_gradient_jacobian",
    "tensorflow_forward_jacobian",
    "tensorflow_jacobian",
]
