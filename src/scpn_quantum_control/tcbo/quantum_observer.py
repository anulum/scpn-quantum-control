# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum Observer
"""Compute small-system TCBO proxy diagnostics from exact ground states.

The observer diagonalises the XY Kuramoto Hamiltonian and aggregates four
diagnostics: gauge-module vortex density as ``p_h1``; a seven-term,
Kitaev-Preskill-style entropy inclusion-exclusion value in bits; a Pauli-string
expectation; and the fraction of qubits whose absolute Z expectation exceeds
0.5. The two Betti-labelled fields are independent bounded proxies.

This module does not construct a persistent Dirac operator, compute persistent
homology, certify topological order, or execute on hardware. The related
coupling-weighted persistent-homology reconstruction lives in
``analysis.tcbo_weighted_complex``. The target ``p_h1 = 0.72`` remains an open
empirical/theoretical question rather than a result of this observer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from qiskit.quantum_info import SparsePauliOp, Statevector

from ..analysis.quantum_phi import partial_trace, von_neumann_entropy
from ..gauge.vortex_detector import measure_vortex_density
from ..hardware.classical import classical_exact_diag


@dataclass
class TCBOResult:
    """Collect small-system TCBO proxy diagnostics.

    Attributes
    ----------
    p_h1 : float
        Gauge vortex density, reused as the Betti-1-labelled proxy.
    tee : float
        Seven-term entropy inclusion-exclusion proxy in bits.
    string_order : float
        Real expectation of the endpoint-Z/interior-X Pauli string.
    n_qubits : int
        Number of oscillators represented by the exact ground state.
    betti_0_proxy : float
        Fraction of qubits with absolute Z expectation above 0.5.
    betti_1_proxy : float
        Alias of ``p_h1``; not a computed persistent-homology Betti number.

    """

    p_h1: float  # vortex density (persistent H1 proxy)
    tee: float  # seven-term entropy inclusion-exclusion proxy [bits]
    string_order: float  # endpoint-Z/interior-X Pauli-string expectation
    n_qubits: int
    betti_0_proxy: float  # thresholded polarization fraction
    betti_1_proxy: float  # vortex-density alias (= p_h1)


def _string_order_parameter(
    psi: NDArray[np.complex128], n: int, i: int = 0, j: int | None = None
) -> float:
    """Return the endpoint-Z/interior-X string expectation for ``(i, j)``."""
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


def _topological_entanglement_entropy(psi: NDArray[np.complex128], n: int) -> float:
    """Return the seven-term entropy inclusion-exclusion proxy in bits.

    For three contiguous qubit-index regions A, B, and C, the value is
    ``S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC``.

    This small-system partition is inspired by the Kitaev-Preskill
    inclusion-exclusion form. It does not establish the spatial geometry,
    scaling limit, or state family needed to certify topological entanglement
    entropy or topological order.
    """
    if n < 4:
        return 0.0

    rho = np.outer(psi, psi.conj()).astype(np.complex128)
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


def _betti_0_proxy(psi: NDArray[np.complex128], n: int) -> float:
    """Return the fraction of qubits whose absolute Z expectation exceeds 0.5."""
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
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
) -> TCBOResult:
    """Compute TCBO proxy diagnostics from a small-system exact ground state.

    Parameters
    ----------
    K : NDArray[np.float64]
        Square oscillator-coupling matrix passed to the exact-diagonalisation
        and gauge-vortex owners.
    omega : NDArray[np.float64]
        Oscillator-frequency vector with one entry per row of ``K``.

    Returns
    -------
    TCBOResult
        Vortex-density, entropy inclusion-exclusion, Pauli-string, and
        polarization-fraction diagnostics.

    Notes
    -----
    The state-based fields use ``classical_exact_diag``. The ``p_h1`` field
    delegates independently to ``measure_vortex_density``, whose gauge-module
    contract owns its own ground-state solve. Consequently this aggregator is a
    small-system library utility, not a large-system or hardware pipeline.

    """
    n = K.shape[0]
    exact = classical_exact_diag(n, K=K, omega=omega)
    psi = exact["ground_state"]

    # p_h1: vortex density from gauge module
    vortex = measure_vortex_density(K, omega)
    p_h1 = vortex.vortex_density

    # Seven-term entropy inclusion-exclusion proxy
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
