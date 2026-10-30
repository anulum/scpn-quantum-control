# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native local-operator state evolution
"""Evolve native statevectors using supplied numerical matrix applications.

Operator streams are consumed lazily. Circuit admission, gate construction,
observable dispatch and framework contexts remain with their existing owners.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, NamedTuple, TypeAlias

MatrixApplication: TypeAlias = Callable[[Any, Any, int, tuple[int, ...], Any], Any]


class NativeLocalOperator(NamedTuple):
    """One resolved numerical matrix application to ordered qubit dimensions.

    Attributes
    ----------
    application
        Actual numerical state/matrix application, with no circuit planner.
    namespace
        Original native SDK namespace supplied to that numerical operation.
    n_qubits
        Admitted number of binary state dimensions.
    qubits
        Ordered dimensions acted on by the matrix.
    matrix
        Actual native matrix retaining its device and derivative graph.

    """

    application: MatrixApplication
    namespace: Any
    n_qubits: int
    qubits: tuple[int, ...]
    matrix: Any


def evolve_local_operators(
    state: Any, operators: Iterable[NativeLocalOperator]
) -> tuple[Any, Any]:
    """Apply supplied local operators sequentially without materializing the stream.

    Parameters
    ----------
    state
        Actual native input vector, without a host copy or substituted tensor.
    operators
        Lazy numerical instructions. Each application completes before the next
        instruction is requested, retaining caller-owned dependency lookup order.

    Returns
    -------
    tuple[object, object]
        Final native state and the last applied matrix, or ``None`` for an empty
        stream. The caller can retain that matrix through observable evaluation,
        preserving its original native resource lifetime.

    """
    matrix = None
    for operator in operators:
        matrix = operator.matrix
        state = operator.application(
            operator.namespace, state, operator.n_qubits, operator.qubits, matrix
        )
    return state, matrix


def jax_zero_state(namespace: Any, width: Callable[[], int]) -> Any:
    """Allocate the JAX computational zero state with the original lookup order.

    Parameters
    ----------
    namespace
        Actual JAX array operations within caller-owned precision contexts.
    width
        Resolve the admitted qubit count after the native zeros callable lookup.

    Returns
    -------
    object
        Fresh native complex128 state with amplitude one at basis index zero.

    """
    state = namespace.zeros((2 ** width(),), dtype=namespace.complex128)
    return state.at[0].set(1.0 + 0.0j)


def torch_zero_state(namespace: Any, width: Callable[[], int]) -> Any:
    """Allocate the PyTorch computational zero state without a host array.

    Parameters
    ----------
    namespace
        Actual PyTorch array operations within the caller's native contexts.
    width
        Resolve the admitted qubit count after the zeros callable lookup.

    Returns
    -------
    object
        Fresh native complex128 vector whose first amplitude is one.

    """
    state = namespace.zeros((2 ** width(),), dtype=namespace.complex128)
    state[0] = 1.0 + 0.0j
    return state


__all__ = [
    "MatrixApplication",
    "NativeLocalOperator",
    "evolve_local_operators",
    "jax_zero_state",
    "torch_zero_state",
]
