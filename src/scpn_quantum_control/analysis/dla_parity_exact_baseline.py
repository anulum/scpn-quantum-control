# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Exact statevector baseline for DLA-parity circuits
"""Exact, noiseless statevector reference for the DLA-parity XY-Trotter circuits.

The promoted ``ibm_kingston`` DLA-parity campaigns run the Kuramoto-XY Trotter
circuit

    H_XY = Σ K_nm (X_n X_m + Y_n Y_m) + Σ ω_n Z_n

and measure the **parity leakage**: the fraction of shots that land in the
opposite excitation-number parity to the prepared state. Every gate in the
circuit — ``rz`` and the paired ``rxx(θ)·ryy(θ)`` on nearest neighbours —
commutes with the total-excitation-number operator ``N = Σ (I − Z_i)/2`` (the
XX and YY generators on a pair commute and their sum ``XX+YY`` conserves ``N``),
so the ideal, noiseless final state stays entirely within the prepared
excitation-number sector. Its parity leakage is therefore **exactly zero** up to
floating-point round-off.

This module provides that exact reference (feasible by statevector for ``n ≤``
about 14). It closes the audit's B-9/B-10 gap: with the ideal leakage pinned at
0, all observed hardware leakage is device noise, and the even/odd leakage
asymmetry that the DLA-parity result reports is a property of the noise, not of
coherent parity-selective dynamics.

The circuit here is reconstructed to be bit-for-bit identical to the campaign
builder ``scripts/phase1_mini_bench_ibm_kingston.build_xy_trotter_circuit``; a
test asserts the two produce the same statevector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Campaign constants (verbatim from the promoted-run builder).
T_STEP: float = 0.3
_OMEGA_LO: float = 0.8
_OMEGA_HI: float = 1.2
_K_SCALE: float = 0.45
_K_DECAY: float = 0.3


def coupling_matrix(n: int) -> NDArray[np.float64]:
    """Exponential-decay coupling ``K[i, j] = 0.45·exp(-0.3·|i − j|)`` (zero diag)."""
    k = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:
                k[i, j] = _K_SCALE * np.exp(-_K_DECAY * abs(i - j))
    return k


def initial_parity(bitstring: str) -> int:
    """Excitation-number parity (popcount mod 2) of a computational basis label."""
    return bitstring.count("1") % 2


def build_statevector_circuit(
    n: int, initial_bitstring: str, depth: int, t_step: float = T_STEP
) -> QuantumCircuit:
    """The DLA-parity XY-Trotter circuit without measurement (for statevector).

    Identical gate sequence to the campaign builder: prepare the basis state
    (little-endian: ``bitstring[q]`` sets qubit ``q``), then ``depth`` Trotter
    steps of single-qubit ``rz(2·ω_i·t)`` followed by nearest-neighbour
    ``rxx(2·K_ij·t)`` and ``ryy(2·K_ij·t)``.
    """
    qc = QuantumCircuit(n)
    for q, bit in enumerate(initial_bitstring):
        if bit == "1":
            qc.x(q)
    if depth > 0:
        k = coupling_matrix(n)
        omega = np.linspace(_OMEGA_LO, _OMEGA_HI, n)
        for _ in range(depth):
            for i in range(n):
                qc.rz(2.0 * omega[i] * t_step, i)
            for i in range(n - 1):
                theta = 2.0 * k[i, i + 1] * t_step
                qc.rxx(theta, i, i + 1)
                qc.ryy(theta, i, i + 1)
    return qc


def exact_parity_leakage(
    n: int, initial_bitstring: str, depth: int, t_step: float = T_STEP
) -> float:
    """Exact probability that lands in the opposite parity to ``initial_bitstring``.

    Zero (to round-off) for every depth, because the XY-Trotter dynamics conserve
    total excitation number.
    """
    state = Statevector(build_statevector_circuit(n, initial_bitstring, depth, t_step))
    probs = state.probabilities()
    target_parity = initial_parity(initial_bitstring)
    leakage = 0.0
    for index, prob in enumerate(probs):
        # Qiskit orders amplitudes by integer basis index; popcount parity is
        # bit-order independent, so the integer's bit count suffices.
        if (int(index).bit_count() % 2) != target_parity:
            leakage += float(prob)
    return leakage


@dataclass(frozen=True)
class ExactBaselineRow:
    """Exact vs hardware leakage for one (depth, initial-state) circuit."""

    depth: int
    initial: str
    n_qubits: int
    initial_parity: int
    exact_leakage: float


def exact_baseline_grid(
    n: int, initials: tuple[str, ...], depths: tuple[int, ...], t_step: float = T_STEP
) -> tuple[ExactBaselineRow, ...]:
    """Exact leakage over the (initial-state × depth) grid the campaign measured."""
    rows: list[ExactBaselineRow] = []
    for initial in initials:
        for depth in depths:
            rows.append(
                ExactBaselineRow(
                    depth=depth,
                    initial=initial,
                    n_qubits=n,
                    initial_parity=initial_parity(initial),
                    exact_leakage=exact_parity_leakage(n, initial, depth, t_step),
                )
            )
    return tuple(rows)
