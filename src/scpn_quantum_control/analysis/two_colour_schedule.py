# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Two-edge-colour hand schedule for the XY-Trotter chain
"""Genuine width-2 hand scheduling of the 1-D XY-Trotter chain (audit AUD-7).

The campaign circuits emit the nearest-neighbour XY chain edge-by-edge, so the
transpiler lays every Trotter step out sequentially and the two-qubit depth
grows like ``O(reps · n)``. A **2-edge-colouring** of the path graph collapses
that to a constant number of layers per step:

* colour A = even-indexed edges ``{(0,1), (2,3), …}`` — disjoint qubits, one layer;
* colour B = odd-indexed edges ``{(1,2), (3,4), …}`` — disjoint qubits, one layer.

Each Trotter step becomes a single-qubit ``rz`` layer plus the two colour layers
— constant depth independent of ``n`` — so the two-qubit depth is ``O(reps)``.
This is the same first-order Trotter approximation (a reordering of the same
terms), and because every ``rz`` / ``rxx(θ)·ryy(θ)`` gate conserves total
excitation number, the scheduled circuit realises **genuine parity-preserving
width-2 dynamics** rather than a scheduling artefact of a deep serial synthesis.

This module supplies the scheduler and the classical evidence (depth reduction +
excitation-number conservation). Submitting it to hardware is an owner-gated QPU
run (AUD-5/7).
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from .dla_parity_exact_baseline import T_STEP, coupling_matrix, initial_parity


def two_colour_edges(n: int) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """2-edge-colouring of the ``n``-node path: (even-edge class, odd-edge class).

    Every edge in a class acts on disjoint qubits, so the class transpiles to a
    single parallel layer.
    """
    colour_a = tuple((i, i + 1) for i in range(0, n - 1, 2))
    colour_b = tuple((i, i + 1) for i in range(1, n - 1, 2))
    return colour_a, colour_b


def _xy_edge(circuit: QuantumCircuit, theta: float, i: int, j: int) -> None:
    """Number-conserving XX+YY rotation on edge ``(i, j)``."""
    circuit.rxx(theta, i, j)
    circuit.ryy(theta, i, j)


def build_two_colour_circuit(
    n: int, initial_bitstring: str, depth: int, t_step: float = T_STEP
) -> QuantumCircuit:
    """XY-Trotter chain scheduled as rz-layer → colour-A layer → colour-B layer."""
    colour_a, colour_b = two_colour_edges(n)
    k = coupling_matrix(n)
    omega = np.linspace(0.8, 1.2, n)
    qc = QuantumCircuit(n)
    for q, bit in enumerate(initial_bitstring):
        if bit == "1":
            qc.x(q)
    for _ in range(depth):
        for i in range(n):
            qc.rz(2.0 * omega[i] * t_step, i)
        for i, j in colour_a:
            _xy_edge(qc, 2.0 * k[i, j] * t_step, i, j)
        for i, j in colour_b:
            _xy_edge(qc, 2.0 * k[i, j] * t_step, i, j)
    return qc


def build_sequential_circuit(
    n: int, initial_bitstring: str, depth: int, t_step: float = T_STEP
) -> QuantumCircuit:
    """The same Trotter terms emitted edge-by-edge (the serial baseline)."""
    k = coupling_matrix(n)
    omega = np.linspace(0.8, 1.2, n)
    qc = QuantumCircuit(n)
    for q, bit in enumerate(initial_bitstring):
        if bit == "1":
            qc.x(q)
    for _ in range(depth):
        for i in range(n):
            qc.rz(2.0 * omega[i] * t_step, i)
        for i in range(n - 1):
            _xy_edge(qc, 2.0 * k[i, i + 1] * t_step, i, i + 1)
    return qc


def two_colour_parity_leakage(
    n: int, initial_bitstring: str, depth: int, t_step: float = T_STEP
) -> float:
    """Exact opposite-parity probability of the 2-colour circuit (zero by symmetry)."""
    probs = Statevector(
        build_two_colour_circuit(n, initial_bitstring, depth, t_step)
    ).probabilities()
    target = initial_parity(initial_bitstring)
    return float(sum(p for idx, p in enumerate(probs) if (int(idx).bit_count() % 2) != target))


def two_qubit_depth(circuit: QuantumCircuit) -> int:
    """Two-qubit gate depth (the hardware-relevant depth after basis translation)."""
    return int(circuit.depth(filter_function=lambda instr: instr.operation.num_qubits == 2))


def depth_comparison(n: int, depth: int, t_step: float = T_STEP) -> dict[str, float]:
    """Two-qubit depth of the serial vs 2-colour schedule and the reduction factor.

    Both circuits use ``initial = '0'*n`` (the schedule is state-independent) and
    the same Trotter terms; only the ordering — hence the layerisation — differs.
    """
    initial = "0" * n
    seq = two_qubit_depth(build_sequential_circuit(n, initial, depth, t_step))
    two_colour = two_qubit_depth(build_two_colour_circuit(n, initial, depth, t_step))
    return {
        "n": float(n),
        "trotter_depth": float(depth),
        "sequential_2q_depth": float(seq),
        "two_colour_2q_depth": float(two_colour),
        "reduction_factor": (float(seq) / float(two_colour)) if two_colour else float("inf"),
    }
