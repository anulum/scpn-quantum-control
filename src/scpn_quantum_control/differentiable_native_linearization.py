# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native Jacobian products
"""Execute native forward and reverse Jacobian products without reporting policy."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def jax_jvp(
    jax: Any, function: Callable[[Any], Any], parameters: Any, tangent: Any
) -> tuple[Any, Any]:
    """Evaluate a numerical function and its native JAX forward Jacobian product.

    Parameters
    ----------
    jax
        Actual JAX runtime under caller-owned device and precision contexts.
    function
        Native numerical function of one parameter tree, at its defining scope.
    parameters
        Native primal array or PyTree, with caller-supplied units and dtype.
    tangent
        Native tangent with the same structure and shapes as the parameters.

    Returns
    -------
    tuple[object, object]
        Actual native primal output and J times tangent. Output shape and
        container structure follow the function; no host conversion is made.

    """
    result: tuple[Any, Any] = jax.jvp(function, (parameters,), (tangent,))
    return result


def jax_vjp(
    jax: Any, function: Callable[[Any], Any], parameters: Any
) -> tuple[Any, Callable[[Any], tuple[Any]]]:
    """Evaluate a numerical function and return its actual native JAX pullback.

    Parameters
    ----------
    jax
        Actual JAX runtime under the caller's active numerical contexts.
    function
        Native function of one parameter tree; its original scope is retained.
    parameters
        Native primal inputs. Nothing is converted to host or report records.

    Returns
    -------
    tuple[object, Callable]
        Native output and the original native pullback object. The pullback
        maps an output cotangent to a one-element tuple containing the input
        cotangent, preserving the native closure and its lifetime in the caller.

    """
    result: tuple[Any, Callable[[Any], tuple[Any]]] = jax.vjp(function, parameters)
    return result


def torch_jvp(
    torch: Any, function: Callable[[Any], Any], parameters: Any, tangent: Any
) -> tuple[Any, Any]:
    """Evaluate a numerical function and its native torch forward Jacobian product.

    Parameters
    ----------
    torch
        Actual PyTorch runtime; the caller supplies device and tensor precision.
    function
        Native numerical function of one tensor or parameter PyTree.
    parameters
        Native primal inputs in their original units and container structure.
    tangent
        Native tangent with structure and shapes matching the primal inputs.

    Returns
    -------
    tuple[object, object]
        Native primal output and J times tangent, without a report wrapper or
        host conversion. The runtime's result tuple is returned unchanged.

    """
    result: tuple[Any, Any] = torch.func.jvp(function, (parameters,), (tangent,))
    return result


def torch_vjp(
    torch: Any, function: Callable[[Any], Any], parameters: Any
) -> tuple[Any, Callable[[Any], tuple[Any]]]:
    """Evaluate a numerical function and return its actual native torch pullback.

    Parameters
    ----------
    torch
        Actual PyTorch runtime with caller-owned device and precision choices.
    function
        Native numerical function, retaining its actual defining namespace.
    parameters
        Native primal inputs; they are neither copied to host nor normalised.

    Returns
    -------
    tuple[object, Callable]
        Native output and original native pullback, whose cotangent result is
        a one-element tuple for the one supplied parameter argument. The
        caller retains the original callable and its native resource lifetime.

    """
    result: tuple[Any, Callable[[Any], tuple[Any]]] = torch.func.vjp(function, parameters)
    return result


def tensorflow_jvp(accumulator: Any, value: Any) -> Any:
    """Read a native Jacobian product from a caller-owned TensorFlow accumulator.

    Parameters
    ----------
    accumulator
        Actual ForwardAccumulator that recorded the supplied value. Construction,
        active context, primal/tangent inputs and lifetime remain caller-owned.
    value
        Native output tensor or container recorded under that accumulator.

    Returns
    -------
    object
        Native forward product with the value's structure and shapes. Native
        unconnected-gradient results are retained; no synthetic zero is created.

    """
    return accumulator.jvp(value)


def tensorflow_vjp(tape: Any, value: Any, parameters: Any, cotangent: Any) -> Any:
    """Differentiate a recorded value using a caller-owned TensorFlow tape.

    Parameters
    ----------
    tape
        Actual tape that watched the primal inputs and recorded the value.
        Its construction, recording context, persistence and lifetime stay
        with the caller; this operation does not close or replace it.
    value
        Recorded native output, whose shape need not equal the parameter shape.
    parameters
        Original watched native tensor or parameter container.
    cotangent
        Native output cotangent, with structure and shapes matching the value.

    Returns
    -------
    object
        Native input cotangent, J transpose times output cotangent. A native
        unconnected-gradient result is retained rather than promoted to a zero.

    """
    return tape.gradient(value, parameters, output_gradients=cotangent)


__all__ = [
    "jax_jvp",
    "jax_vjp",
    "torch_jvp",
    "torch_vjp",
    "tensorflow_jvp",
    "tensorflow_vjp",
]
