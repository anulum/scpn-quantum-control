# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native gate rotation and phase arithmetic
"""Construct supplied native gate operators without model or circuit selection.

Numerical constructors are explicit inputs. They retain native tensor graphs and
let compatibility callers keep their original nested factories and live reads.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def pauli_rotation(
    namespace: Any,
    theta: Any,
    imaginary_unit: Any,
    *,
    identity: Callable[[], Any],
    operator: Callable[[], Any],
) -> Any:
    """Evaluate a Pauli-generated rotation with the supplied numerical factories.

    Parameters
    ----------
    namespace
        Actual native namespace providing cosine and sine.
    theta
        Supplied native rotation parameter, retaining its differentiation graph.
    imaginary_unit
        Already constructed native imaginary unit.
    identity
        Resolve the matching identity after cosine evaluation.
    operator
        Resolve the supplied Pauli or Pauli-product matrix after sine evaluation.

    Returns
    -------
    object
        Native cosine-weighted identity minus the imaginary sine-weighted operator.
        Factories are evaluated once each, in the original arithmetic order.

    """
    return (
        namespace.cos(theta / 2.0) * identity()
        - imaginary_unit * namespace.sin(theta / 2.0) * operator()
    )


def hadamard_operator(
    namespace: Any,
    *,
    real_tensor: Callable[[object], Any],
    complex_tensor: Callable[[object], Any],
) -> Any:
    """Construct a normalized native Hadamard matrix using active constructors.

    Parameters
    ----------
    namespace
        Actual native namespace providing square root.
    real_tensor
        Active numerical constructor for the scalar normalization argument.
    complex_tensor
        Active numerical constructor for the unnormalized matrix.

    Returns
    -------
    object
        Native complex two-by-two Hadamard operator. Constructor failures retain
        their original evaluation order relative to the square root and scaling.

    """
    return (1.0 / namespace.sqrt(real_tensor(2.0))) * complex_tensor([[1.0, 1.0], [1.0, -1.0]])


def sqrt_x_operator(*, complex_tensor: Callable[[object], Any]) -> Any:
    """Construct the principal native square root of the Pauli X matrix.

    Parameters
    ----------
    complex_tensor
        Active numerical constructor retaining the caller's dtype and device.

    Returns
    -------
    object
        Native matrix whose square is Pauli X, including its original global phase.

    """
    return 0.5 * complex_tensor([[1.0 + 1.0j, 1.0 - 1.0j], [1.0 - 1.0j, 1.0 + 1.0j]])


def quarter_turn_phase(namespace: Callable[[], Any]) -> Any:
    """Evaluate the original positive quarter-turn phase with live primitive reads.

    Parameters
    ----------
    namespace
        Resolve the caller's active numerical namespace separately for exponential
        lookup and for its pi constant. No namespace snapshot is retained.

    Returns
    -------
    object
        Exponential of one quarter of pi times the imaginary unit.

    """
    return namespace().exp(1.0j * namespace().pi / 4.0)


def jax_phase_operator(
    namespace: Any, theta: Any, one: Any, zero: Any, imaginary_unit: Any, *, dtype: Any
) -> Any:
    """Construct a native JAX phase matrix without materializing traced scalars.

    Parameters
    ----------
    namespace
        Actual JAX namespace providing array construction and exponential.
    theta
        Supplied native phase parameter.
    one
        Already constructed native unit scalar.
    zero
        Already constructed native zero scalar.
    imaginary_unit
        Already constructed native imaginary unit.
    dtype
        Exact complex dtype selected by the caller before gate construction.

    Returns
    -------
    object
        Native diagonal phase operator with its parameter graph intact.

    """
    return namespace.asarray(
        [[one, zero], [zero, namespace.exp(imaginary_unit * theta)]], dtype=dtype
    )


def torch_phase_operator(
    namespace: Any, theta: Any, one: Any, zero: Any, imaginary_unit: Any
) -> Any:
    """Stack a native Torch phase matrix while preserving its autograd graph.

    Parameters
    ----------
    namespace
        Actual Torch namespace providing native stack and exponential.
    theta
        Supplied native phase parameter.
    one
        Already constructed native unit scalar.
    zero
        Already constructed native zero scalar.
    imaginary_unit
        Already constructed native imaginary unit.

    Returns
    -------
    object
        Native diagonal phase operator. Stack callables are resolved in the
        original outer-stack, first-row, second-row order.

    """
    return namespace.stack(
        (
            namespace.stack((one, zero)),
            namespace.stack((zero, namespace.exp(imaginary_unit * theta))),
        )
    )


def controlled_z_operator(
    namespace: Any, controls: int, *, complex_tensor: Callable[[object], Any]
) -> Any:
    """Construct a controlled-Z operator from an admitted number of controls.

    Parameters
    ----------
    namespace
        Actual native namespace providing diagonal matrix construction.
    controls
        Nonnegative admitted control count; callers retain resource admission.
    complex_tensor
        Active constructor for the real-valued complex diagonal entries.

    Returns
    -------
    object
        Native diagonal operator whose all-one computational basis state has sign
        minus one and whose remaining diagonal entries are one.

    """
    return namespace.diag(complex_tensor([1.0] * (2 ** (controls + 1) - 1) + [-1.0]))


__all__ = [
    "pauli_rotation",
    "hadamard_operator",
    "sqrt_x_operator",
    "quarter_turn_phase",
    "jax_phase_operator",
    "torch_phase_operator",
    "controlled_z_operator",
]
