# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native structured derivative execution
"""Execute native structured derivatives independently of QNode/report policy."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

NativeParameters: TypeAlias = dict[str, Any]
StructuredObjective: TypeAlias = Callable[[NativeParameters], Any]
StructuredHessian: TypeAlias = dict[str, dict[str, Any]]


def jax_structured_derivatives(
    jax: Any,
    objective: StructuredObjective,
    parameters: NativeParameters,
) -> tuple[NativeParameters, StructuredHessian]:
    """Execute JAX gradient and forward-over-reverse structured curvature.

    Parameters
    ----------
    jax
        Actual JAX runtime whose device and precision contexts the caller owns.
    objective
        Native scalar numerical objective over the supplied parameter mapping.
        Its actual defining namespace and closure are retained.
    parameters
        Native differentiable parameter PyTree. The structured audit supplies
        a dimensionless one-dimensional ``weights`` tensor and scalar ``bias``.

    Returns
    -------
    tuple[dict, dict]
        Native gradient and nested Hessian PyTrees in the runtime's field order.
        Values are neither copied to the host nor wrapped in report records.

    Raises
    ------
    TypeError
        If the native runtime cannot differentiate the supplied scalar objective.

    """
    gradient: NativeParameters = jax.grad(objective)(parameters)
    hessian: StructuredHessian = jax.jacfwd(jax.grad(objective))(parameters)
    return gradient, hessian


def torch_structured_derivatives(
    torch: Any,
    objective: StructuredObjective,
    parameters: NativeParameters,
) -> tuple[NativeParameters, StructuredHessian]:
    """Execute torch.func gradient and forward-over-reverse structured curvature.

    Parameters
    ----------
    torch
        Actual PyTorch runtime. The caller supplies tensors on the desired
        device and at the required precision.
    objective
        Native scalar objective accepting the original parameter mapping.
    parameters
        Native differentiable parameter PyTree. The audit's ``weights`` are
        one-dimensional and its shared ``bias`` is scalar and dimensionless.

    Returns
    -------
    tuple[dict, dict]
        Native gradient and nested Hessian PyTrees with the original native
        tensors and field identities; no report or policy decision is made.

    Raises
    ------
    RuntimeError
        If a required native transform cannot execute the supplied objective.

    """
    gradient: NativeParameters = torch.func.grad(objective)(parameters)
    hessian: StructuredHessian = torch.func.jacfwd(torch.func.grad(objective))(parameters)
    return gradient, hessian


def tensorflow_structured_gradient(
    tf: Any,
    tape: Any,
    value: Any,
    parameters: NativeParameters,
) -> tuple[NativeParameters, Any]:
    """Differentiate a recorded structured value and concatenate its native gradient.

    Parameters
    ----------
    tf
        Actual TensorFlow runtime used to concatenate native gradient tensors.
    tape
        Actual caller-owned tape that watched ``parameters`` while recording
        ``value``. Its construction, recording context and lifetime stay with
        the caller. Invoke this function inside any enclosing curvature tape.
    value
        Recorded scalar objective that depends differentiably on both fields.
    parameters
        Native ``weights`` vector and shared scalar ``bias`` mapping.

    Returns
    -------
    tuple[dict, object]
        Native gradient mapping and flat tensor, ordered weights then bias.
        The flat tensor is formed under the caller's active recording context.

    Raises
    ------
    RuntimeError
        If the supplied tape cannot produce the requested native gradient.

    """
    gradient: NativeParameters = tape.gradient(value, parameters)
    flat_gradient = tf.concat([gradient["weights"], tf.reshape(gradient["bias"], [1])], 0)
    return gradient, flat_gradient


def tensorflow_structured_hessian(
    tape: Any,
    flat_gradient: Any,
    parameters: NativeParameters,
) -> tuple[Any, Any]:
    """Differentiate the recorded flat gradient with a caller-owned TensorFlow tape.

    Parameters
    ----------
    tape
        Actual enclosing persistent tape that recorded the native flat gradient.
        The caller retains ownership and controls its lifetime.
    flat_gradient
        Native tensor ordered weights then bias, recorded by ``tape``.
    parameters
        Original native ``weights`` vector and shared scalar ``bias`` mapping.

    Returns
    -------
    tuple[object, object]
        Jacobians with respect to weights and bias. Their final output rows
        contain the bias-weight and bias-bias mixed blocks respectively.
        TensorFlow's serial Jacobian route is retained without pfor substitution.

    Raises
    ------
    RuntimeError
        If the supplied native tape cannot differentiate the recorded gradient.

    """
    weights_hessian, bias_hessian = tape.jacobian(
        flat_gradient,
        (parameters["weights"], parameters["bias"]),
        experimental_use_pfor=False,
    )
    return weights_hessian, bias_hessian


__all__ = [
    "jax_structured_derivatives",
    "torch_structured_derivatives",
    "tensorflow_structured_gradient",
    "tensorflow_structured_hessian",
]
