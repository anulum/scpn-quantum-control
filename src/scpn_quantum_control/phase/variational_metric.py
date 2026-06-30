# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Variational Metric
"""Exact variational-dynamics linear system for a fixed parametrised ansatz.

Variational real- and imaginary-time evolution (AVQDS-style VarQRTE and VarQITE)
reduce, at each step, to a linear system ``G(θ) dθ = b(θ)`` built from the
geometry of the ansatz manifold and the Hamiltonian:

    McLachlan metric  G_ij = Re(<∂_i ψ|∂_j ψ>)               (shared by both)
    real-time force   V_i  = -Im(<∂_i ψ|H|ψ>)                (VarQRTE / AVQDS)
    imaginary force   C_i  = -Re(<∂_i ψ|(H - <H>)|ψ>)        (VarQITE)

following Yuan et al., *Quantum* 3, 191 (2019) and McArdle et al., *npj Quantum
Information* 5, 75 (2019). The only approximation in a naive implementation is the
state derivative ``∂_k|ψ>``: estimating it by central finite differences carries
an O(ε²) bias and an ε hyperparameter.

This module computes ``∂_k|ψ>`` **exactly** instead. Every parameter of the
physics-informed ansatz enters a single Pauli-generated rotation
``R_P(θ_k) = exp(-i θ_k P/2)`` (``P ∈ {X, Y, Z}``, so ``P² = I``). For such a gate
``exp(-iπP/2) = -iP``, hence ``R_P(θ_k + π) = -iP · R_P(θ_k)`` and, because the
parameter appears in exactly one gate,

    ∂_k|ψ(θ)> = ½ |ψ(θ + π e_k)> ,

an exact identity that costs one extra state evaluation per parameter — the same
cost as one arm of a finite difference, with no bias and no ε. The metric
``Re(<∂_i ψ|∂_j ψ>)`` it produces is the (real, McLachlan) quantum geometric
tensor of the ansatz, evaluated analytically.

The state is supplied through a ``state_of`` callable so the caller controls the
simulator (and tests can substitute a double); the linear-algebra assembly here is
pure NumPy and simulator-agnostic.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]
StateEvaluator: TypeAlias = Callable[[FloatArray], ComplexArray]

_PAULI_ROTATIONS = frozenset({"rx", "ry", "rz"})


def assert_single_parameter_rotations(ansatz: QuantumCircuit) -> None:
    """Validate the ansatz admits the exact π-shift state-derivative identity.

    The identity ``∂_k|ψ> = ½|ψ(θ + π e_k)>`` holds only when every free parameter
    enters exactly one Pauli-generated rotation (``rx``/``ry``/``rz``). This check
    enforces that precondition so a future ansatz change cannot silently invalidate
    the exact derivative. Non-:class:`~qiskit.QuantumCircuit` inputs (test doubles)
    are skipped — the contract applies to real circuits only.

    Raises:
        ValueError: if a parameter drives a non-Pauli-rotation gate, a gate carries
            more than one free parameter, or a parameter appears in more than one
            gate.
    """
    if not isinstance(ansatz, QuantumCircuit):
        return

    occurrences: Counter[object] = Counter()
    for instruction in ansatz.data:
        operation = instruction.operation
        free = [p for p in operation.params if getattr(p, "parameters", None)]
        if not free:
            continue
        if operation.name not in _PAULI_ROTATIONS:
            raise ValueError(
                f"parameter drives gate '{operation.name}', which is not a "
                "Pauli rotation (rx/ry/rz); the π-shift derivative identity "
                "does not apply"
            )
        symbols = {symbol for param in free for symbol in param.parameters}
        if len(symbols) != 1:
            raise ValueError(
                f"gate '{operation.name}' carries {len(symbols)} free parameters; "
                "the π-shift identity requires exactly one per gate"
            )
        occurrences[next(iter(symbols))] += 1

    repeated = sorted(str(symbol) for symbol, count in occurrences.items() if count != 1)
    if repeated:
        raise ValueError(
            "parameters reused across multiple gates break the π-shift identity: "
            + ", ".join(repeated)
        )


def analytic_state_derivatives(state_of: StateEvaluator, params: FloatArray) -> ComplexArray:
    """Exact state derivatives ``∂_k|ψ>`` via the π-shift identity.

    Args:
        state_of: returns the statevector ``|ψ(p)>`` for parameter vector ``p``.
        params: current parameter vector θ.

    Returns:
        Array of shape ``(len(params), dim)`` whose row ``k`` is ``∂_k|ψ(θ)>``.
    """
    theta = np.asarray(params, dtype=np.float64)
    reference = np.asarray(state_of(theta), dtype=np.complex128)
    derivatives = np.zeros((theta.size, reference.size), dtype=np.complex128)
    for k in range(theta.size):
        shifted = theta.copy()
        shifted[k] += np.pi
        derivatives[k] = 0.5 * np.asarray(state_of(shifted), dtype=np.complex128)
    return derivatives


def mclachlan_metric(state_derivatives: ComplexArray) -> FloatArray:
    """McLachlan metric ``G_ij = Re(<∂_i ψ|∂_j ψ>)`` (the real quantum geometric tensor).

    Args:
        state_derivatives: rows ``∂_k|ψ>``, as returned by
            :func:`analytic_state_derivatives`.

    Returns:
        Symmetric ``(n_params, n_params)`` real metric.
    """
    return np.real(state_derivatives.conj() @ state_derivatives.T).astype(np.float64)


def real_time_force(state_derivatives: ComplexArray, h_psi: ComplexArray) -> FloatArray:
    """Real-time (McLachlan) force ``V_i = -Im(<∂_i ψ|H|ψ>)``.

    Args:
        state_derivatives: rows ``∂_k|ψ>``.
        h_psi: the vector ``H|ψ>``.

    Returns:
        Length ``n_params`` force vector.
    """
    return (-np.imag(state_derivatives.conj() @ np.asarray(h_psi, dtype=np.complex128))).astype(
        np.float64
    )


def imaginary_time_force(
    state_derivatives: ComplexArray, h_shifted_psi: ComplexArray
) -> FloatArray:
    """Imaginary-time force ``C_i = -Re(<∂_i ψ|(H - <H>)|ψ>)``.

    Args:
        state_derivatives: rows ``∂_k|ψ>``.
        h_shifted_psi: the vector ``(H - <H>)|ψ>``.

    Returns:
        Length ``n_params`` force vector.
    """
    return (
        -np.real(state_derivatives.conj() @ np.asarray(h_shifted_psi, dtype=np.complex128))
    ).astype(np.float64)
