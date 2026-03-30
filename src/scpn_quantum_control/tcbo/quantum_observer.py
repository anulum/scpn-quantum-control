# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Quantum Observer
"""Quantum TCBO: topological coherence via persistent Dirac operator.

The classical TCBO computes persistent homology (Betti numbers β_0, β_1)
from the oscillator phase configuration. The quantum TCBO measures
topological content directly from the quantum state.

Quantum topological observables:
    1. p_h1: fraction of H1 cycles detected (vortex count / plaquettes)
       — The consciousness gate threshold at p_h1 = 0.72

    2. Topological entanglement entropy (TEE):
       S_topo = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC
       For topologically ordered states, S_topo = -ln(D) where D
       is the total quantum dimension (Kitaev-Preskill, Levin-Wen 2006).

    3. String order parameter:
       O_string = <Z_i × Π_{k=i+1}^{j-1} X_k × Z_j>
       Non-zero in symmetry-protected topological phases.

For the XY model at the BKT transition, the TEE distinguishes
trivial (TEE=0) from topological (TEE≠0) phases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

from ..analysis.quantum_phi import partial_trace, von_neumann_entropy
from ..gauge.vortex_detector import measure_vortex_density
from ..hardware.classical import classical_exact_diag


@dataclass
class TCBOResult:
    """Quantum TCBO measurement result."""

    p_h1: float  # vortex density (persistent H1 proxy)
    tee: float  # topological entanglement entropy
    string_order: float  # string order parameter
    n_qubits: int
    betti_0_proxy: float  # connected components proxy
    betti_1_proxy: float  # loop proxy (= p_h1)


def _string_order_parameter(psi: np.ndarray, n: int, i: int = 0, j: int | None = None) -> float:
    """String order parameter: <Z_i × Π X_k × Z_j> for k in (i,j)."""
    if j is None:
        j = n - 1
    if j <= i + 1:
        return 0.0

    sv = Statevector(np.ascontiguousarray(psi))
    label = ["I"] * n
    label[i] = "Z"
    label[j] = "Z"
    for k in range(i + 1, j):
        label[k] = "X"
    op = SparsePauliOp("".join(reversed(label)))
    return float(sv.expectation_value(op).real)


def _topological_entanglement_entropy(psi: np.ndarray, n: int) -> float:
    """Topological entanglement entropy via Kitaev-Preskill construction.

    For 3 contiguous regions A, B, C:
        S_topo = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC

    Uses n//3 partition into 3 equal-ish regions.
    """
    if n < 4:
        return 0.0

    rho = np.outer(psi, psi.conj())
    size_a = n // 3
    size_b = n // 3

    A = list(range(size_a))
    B = list(range(size_a, size_a + size_b))
    C = list(range(size_a + size_b, n))
    AB = A + B
    BC = B + C
    AC = A + C

    s_a = von_neumann_entropy(partial_trace(rho, A, n))
    s_b = von_neumann_entropy(partial_trace(rho, B, n))
    s_c = von_neumann_entropy(partial_trace(rho, C, n))
    s_ab = von_neumann_entropy(partial_trace(rho, AB, n))
    s_bc = von_neumann_entropy(partial_trace(rho, BC, n))
    s_ac = von_neumann_entropy(partial_trace(rho, AC, n))
    s_abc = von_neumann_entropy(rho)  # full system (should be 0 for pure state)

    tee = s_a + s_b + s_c - s_ab - s_bc - s_ac + s_abc
    return float(tee)


def _betti_0_proxy(psi: np.ndarray, n: int) -> float:
    """β_0 proxy: fraction of qubits with |<Z>| > 0.5 (non-trivially polarised).

    High β_0 = many connected components = desynchronised.
    """
    sv = Statevector(np.ascontiguousarray(psi))
    count = 0
    for i in range(n):
        label = ["I"] * n
        label[i] = "Z"
        z_exp = float(sv.expectation_value(SparsePauliOp("".join(reversed(label)))).real)
        if abs(z_exp) > 0.5:
            count += 1
    return count / max(n, 1)


def compute_tcbo_observables(
    K: np.ndarray,
    omega: np.ndarray,
) -> TCBOResult:
    """Compute all quantum TCBO observables from ground state."""
    n = K.shape[0]
    exact = classical_exact_diag(n, K=K, omega=omega)
    psi = exact["ground_state"]

    # p_h1: vortex density from gauge module
    vortex = measure_vortex_density(K, omega)
    p_h1 = vortex.vortex_density

    # TEE
    tee = _topological_entanglement_entropy(psi, n)

    # String order
    string_ord = _string_order_parameter(psi, n)

    # β_0 proxy
    b0 = _betti_0_proxy(psi, n)

    return TCBOResult(
        p_h1=p_h1,
        tee=tee,
        string_order=string_ord,
        n_qubits=n,
        betti_0_proxy=b0,
        betti_1_proxy=p_h1,
    )
