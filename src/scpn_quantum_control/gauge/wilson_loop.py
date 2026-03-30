# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Wilson Loop
"""U(1) Wilson loop measurement on the Kuramoto-XY coupling graph.

The XY model is equivalent to a U(1) lattice gauge theory. The
Wilson loop W(C) = exp(i Σ_{(ij)∈C} θ_ij) measures the phase
circulation around a closed path C on the coupling graph.

For the quantum XY Hamiltonian:
    W(C) = Π_{(i,j)∈C} (X_i X_j + Y_i Y_j) / 2

In the ordered phase (K > K_c):
    <W(C)> → constant (perimeter law) → confined
In the disordered phase (K < K_c):
    <W(C)> ~ exp(-Area(C)) → area law → deconfined

At the BKT transition:
    <W(C)> ~ exp(-σ × Perimeter(C))  with universal σ

The Wilson loop is the order parameter for confinement in the
U(1) gauge theory picture. It connects the synchronization
transition to the vortex (de)confinement transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from ..hardware.classical import classical_exact_diag


@dataclass
class WilsonLoopResult:
    """Wilson loop measurement result."""

    loop: list[int]  # qubit indices forming the loop
    loop_length: int
    expectation_value: complex
    magnitude: float
    phase_angle: float  # arg(W) in radians


def _build_wilson_operator(loop: list[int], n_qubits: int) -> SparsePauliOp:
    """Build Wilson loop operator for closed path on qubit indices.

    W(C) = Π_{(i,j)∈C} (X_i X_j + Y_i Y_j) / 2

    For a two-site loop (i,j): W = (X_i X_j + Y_i Y_j) / 2.
    For longer loops, we compose the link operators.
    """
    if len(loop) < 2:
        raise ValueError(f"Loop must have at least 2 sites, got {len(loop)}")

    # Build link operators for each edge (i, i+1) in the loop
    # Each link: (X_i X_j + Y_i Y_j) / 2
    result = None
    for k in range(len(loop)):
        i = loop[k]
        j = loop[(k + 1) % len(loop)]

        # X_i X_j term
        xx_label = ["I"] * n_qubits
        xx_label[i] = "X"
        xx_label[j] = "X"
        xx_op = SparsePauliOp("".join(reversed(xx_label)), coeffs=[0.5])

        # Y_i Y_j term
        yy_label = ["I"] * n_qubits
        yy_label[i] = "Y"
        yy_label[j] = "Y"
        yy_op = SparsePauliOp("".join(reversed(yy_label)), coeffs=[0.5])

        link = (xx_op + yy_op).simplify()

        if result is None:
            result = link
        else:
            result = (result @ link).simplify()

    assert result is not None
    return result


def wilson_loop_expectation(
    psi: np.ndarray,
    loop: list[int],
    n_qubits: int,
) -> complex:
    """Compute <ψ|W(C)|ψ> for a given state and loop."""
    W = _build_wilson_operator(loop, n_qubits)
    W_mat = W.to_matrix()
    if hasattr(W_mat, "toarray"):
        W_mat = W_mat.toarray()
    return complex(psi.conj() @ W_mat @ psi)


def _find_loops(K: np.ndarray, max_length: int = 4) -> list[list[int]]:
    """Find closed loops of length 3 and 4 on the coupling graph."""
    n = K.shape[0]
    loops: list[list[int]] = []

    # Triangles (length 3)
    for i, j, k in combinations(range(n), 3):
        if K[i, j] > 0 and K[j, k] > 0 and K[k, i] > 0:
            loops.append([i, j, k])

    # Squares (length 4)
    if max_length >= 4:
        for i, j, k, m in combinations(range(n), 4):
            if K[i, j] > 0 and K[j, k] > 0 and K[k, m] > 0 and K[m, i] > 0:
                loops.append([i, j, k, m])

    return loops


def compute_wilson_loops(
    K: np.ndarray,
    omega: np.ndarray,
    max_length: int = 4,
    max_loops: int = 20,
) -> list[WilsonLoopResult]:
    """Compute Wilson loop expectation values for the ground state.

    Finds all loops up to max_length on the coupling graph and
    measures <W(C)> for each.
    """
    n = K.shape[0]
    exact = classical_exact_diag(n, K=K, omega=omega)
    psi = exact["ground_state"]

    loops = _find_loops(K, max_length=max_length)[:max_loops]
    results: list[WilsonLoopResult] = []

    for loop in loops:
        w = wilson_loop_expectation(psi, loop, n)
        results.append(
            WilsonLoopResult(
                loop=loop,
                loop_length=len(loop),
                expectation_value=w,
                magnitude=float(abs(w)),
                phase_angle=float(np.angle(w)),
            )
        )

    return results
