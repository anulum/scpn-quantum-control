# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native controlled and permutation operators
"""Construct native QNode operators without selecting circuits or model policy.

These numerical operations preserve the native tensor graph. Gate selection,
optional runtime loading and original compatibility definitions stay with callers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def phase_operator_entries(phase: complex) -> list[list[float | complex]]:
    """Return fresh diagonal matrix entries for a supplied complex phase.

    Parameters
    ----------
    phase
        Actual admitted phase scalar, retained without coercion or normalization.

    Returns
    -------
    list of list
        Fresh unit and phase rows for the caller's active tensor constructor.

    """
    return [[1.0, 0.0], [0.0, phase]]


def two_qubit_operator_entries(gate: str) -> list[list[int | complex]]:
    """Return fresh entries for a fixed two-qubit controlled or swap operator.

    Parameters
    ----------
    gate
        Admitted fixed operator name: ``cnot``, ``cy`` or ``swap``.

    Returns
    -------
    list of list
        Fresh rows preserving the original integer and complex scalar entries.

    Raises
    ------
    ValueError
        If the supplied name does not identify one of these fixed operators.

    """
    if gate == "cnot":
        return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    if gate == "cy":
        return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1.0j], [0, 0, 1.0j, 0]]
    if gate == "swap":
        return [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    raise ValueError(f"unsupported fixed two-qubit gate: {gate}")


def pauli_operator_entries(label: str) -> list[list[float | complex]]:
    """Return fresh matrix entries for one of the three Pauli operators.

    Parameters
    ----------
    label
        Lowercase Pauli label ``x``, ``y`` or ``z``.

    Returns
    -------
    list of list
        Fresh rows retaining the original real and complex scalar values.
        Callers choose the actual tensor constructor, dtype and device.

    Raises
    ------
    ValueError
        If the label does not name a Pauli operator.

    """
    if label == "x":
        return [[0.0, 1.0], [1.0, 0.0]]
    if label == "y":
        return [[0.0, -1.0j], [1.0j, 0.0]]
    if label == "z":
        return [[1.0, 0.0], [0.0, -1.0]]
    raise ValueError(f"unsupported Pauli label: {label}")


def jax_controlled_operator(namespace: Any, target: Any) -> Any:
    """Embed a two-state target in the active-control block of a JAX operator.

    Parameters
    ----------
    namespace
        Actual JAX array namespace under the caller's precision context.
    target
        Native two-by-two operator, including any traced parameters.

    Returns
    -------
    object
        Four-by-four complex operator with an identity inactive-control block.
        Invalid target shapes propagate JAX's own assignment error.

    """
    matrix = namespace.eye(4, dtype=namespace.complex128)
    return matrix.at[2:4, 2:4].set(target)


def torch_controlled_operator_rows(
    namespace: Any, target: Any, *, complex_tensor: Callable[[object], Any]
) -> list[Any]:
    """Build the four native rows of a controlled Torch operator.

    Parameters
    ----------
    namespace
        Actual Torch namespace providing native concatenation.
    target
        Native two-by-two operator retaining its autograd and device graph.
    complex_tensor
        Resolve the caller's active complex-construction primitive at each
        invocation. No host conversion of the supplied target is performed.

    Returns
    -------
    list
        Two identity rows followed by the target rows with zero control prefixes.
        The caller retains its tuple conversion and final native stack operation.

    """
    return [
        complex_tensor([1.0, 0.0, 0.0, 0.0]),
        complex_tensor([0.0, 1.0, 0.0, 0.0]),
        namespace.cat((complex_tensor([0.0, 0.0]), target[0])),
        namespace.cat((complex_tensor([0.0, 0.0]), target[1])),
    ]


def jax_basis_transposition(namespace: Any, first: int, second: int) -> Any:
    """Exchange two computational basis states in an eight-state JAX operator.

    Parameters
    ----------
    namespace
        Actual JAX array namespace under the caller's precision context.
    first
        First admitted computational-basis row and column.
    second
        Second admitted computational-basis row and column.

    Returns
    -------
    object
        Native eight-by-eight complex identity with the two basis states exchanged.
        The input indices are used directly with JAX's native indexing semantics.

    """
    matrix = namespace.eye(8, dtype=namespace.complex128)
    matrix = matrix.at[first, first].set(0.0)
    matrix = matrix.at[second, second].set(0.0)
    matrix = matrix.at[first, second].set(1.0)
    return matrix.at[second, first].set(1.0)


def torch_basis_transposition(namespace: Any, first: int, second: int) -> Any:
    """Exchange two computational basis states in an eight-state Torch operator.

    Parameters
    ----------
    namespace
        Actual Torch namespace providing native complex identity allocation.
    first
        First admitted computational-basis row and column.
    second
        Second admitted computational-basis row and column.

    Returns
    -------
    object
        Fresh native eight-by-eight complex permutation operator. Index errors
        retain Torch's native semantics, with no mutation of caller-owned tensors.

    """
    matrix = namespace.eye(8, dtype=namespace.complex128)
    matrix[first, first] = 0.0
    matrix[second, second] = 0.0
    matrix[first, second] = 1.0
    matrix[second, first] = 1.0
    return matrix


__all__ = [
    "phase_operator_entries",
    "two_qubit_operator_entries",
    "pauli_operator_entries",
    "jax_controlled_operator",
    "torch_controlled_operator_rows",
    "jax_basis_transposition",
    "torch_basis_transposition",
]
