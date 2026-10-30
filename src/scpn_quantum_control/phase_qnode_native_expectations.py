# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native observable expectation arithmetic
"""Compute quantum expectation moments from supplied native tensors.

Callers own observable selection, operator application and coefficient reads.
These kernels retain complex products until the requested real projection and
perform no normalization, host materialization or numerical admission.
"""

from __future__ import annotations

from typing import Any


def real_quadratic_form(namespace: Any, state: Any, matrix: Any) -> Any:
    """Evaluate the real Hermitian expectation without materializing native arrays.

    Parameters
    ----------
    namespace
        Actual JAX or Torch namespace providing real projection and conjugate dot.
    state
        Supplied native statevector, retaining its differentiation graph.
    matrix
        Admitted native observable matrix acting on that statevector.

    Returns
    -------
    object
        Real part of the conjugate quadratic form. Native matrix-shape errors
        propagate without conversion or replacement.

    """
    return namespace.real(namespace.vdot(state, matrix @ state))


def weighted_real_overlap(namespace: Any, state: Any, transformed: Any, coefficient: float) -> Any:
    """Evaluate a real Pauli expectation after the caller applies its operator.

    Parameters
    ----------
    namespace
        Actual native namespace providing real projection and conjugate dot.
    state
        Native original statevector.
    transformed
        Native state after application of the unweighted Pauli operator.
    coefficient
        Coefficient read by the caller after operator application.

    Returns
    -------
    object
        Supplied coefficient times the real conjugate overlap. No normalization
        or conjugation of the coefficient is introduced.

    """
    return coefficient * namespace.real(namespace.vdot(state, transformed))


def weighted_complex_overlap(
    namespace: Any,
    state: Any,
    transformed: Any,
    left_coefficient: float,
    right_coefficient: float,
) -> Any:
    """Retain a complex product moment for an ordered pair of Pauli operators.

    Parameters
    ----------
    namespace
        Actual native namespace providing the conjugate dot operation.
    state
        Native original statevector.
    transformed
        Native state after applying the right operator and then the left one.
    left_coefficient
        Left operator's coefficient, read first by the caller.
    right_coefficient
        Right operator's coefficient, read second by the caller.

    Returns
    -------
    object
        Ordered coefficient product times the full complex overlap. In particular,
        imaginary moments of anticommuting operators remain available.

    """
    return left_coefficient * right_coefficient * namespace.vdot(state, transformed)


def real_symmetrized_overlap(namespace: Any, left_right: Any, right_left: Any) -> Any:
    """Project the symmetrized sum of two already evaluated ordered moments.

    Parameters
    ----------
    namespace
        Actual native namespace providing real projection.
    left_right
        Complex moment evaluated with the right operator applied first.
    right_left
        Complex moment evaluated with the left operator applied first.

    Returns
    -------
    object
        Real part of half the sum, retaining native tensor dtype and gradients.

    """
    return namespace.real(0.5 * (left_right + right_left))


def covariance_from_moments(symmetrized: Any, left_mean: Any, right_mean: Any) -> Any:
    """Subtract the product of means from an evaluated symmetrized moment.

    Parameters
    ----------
    symmetrized
        Native real symmetrized operator-product moment.
    left_mean
        Native weighted expectation of the left operator.
    right_mean
        Native weighted expectation of the right operator.

    Returns
    -------
    object
        Native covariance with the original multiplication and subtraction order.

    """
    return symmetrized - left_mean * right_mean


__all__ = [
    "real_quadratic_form",
    "weighted_real_overlap",
    "weighted_complex_overlap",
    "real_symmetrized_overlap",
    "covariance_from_moments",
]
