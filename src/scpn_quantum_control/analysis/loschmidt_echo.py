# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Loschmidt echo and dynamical quantum phase transitions.

Quantum quench: prepare ground state of H(K_i), suddenly switch to H(K_f).
The Loschmidt amplitude G(t) = ⟨ψ_i|e^{-iH_f t}|ψ_i⟩ and rate function
g(t) = -log|G(t)|²/N probe the quench dynamics.

DQPTs manifest as cusps (non-analyticities) in g(t) at critical times t*.
For quenches across the BKT point K_c:
- Zunkovic et al. (2016) showed DQPTs are NOT guaranteed at BKT for the
  infinite-range XY model — they depend on the quench protocol.
- The specific behavior for the Kuramoto Hamiltonian with heterogeneous
  frequencies is unstudied.

Ref: Heyl et al. PRL 110, 135704 (2013) — foundational DQPT paper.
     Zunkovic et al. Phil. Trans. R. Soc. A 374, 20150160 (2016).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bridge.knm_hamiltonian import knm_to_hamiltonian


@dataclass
class LoschmidtResult:
    """Loschmidt echo / DQPT result."""

    times: np.ndarray
    loschmidt_amplitude: np.ndarray  # |G(t)| = |⟨ψ_i|e^{-iH_f t}|ψ_i⟩|
    rate_function: np.ndarray  # g(t) = -log|G(t)|²/N
    n_cusps: int  # number of detected cusps in g(t)
    cusp_times: list[float]  # times of cusps
    K_initial: float
    K_final: float


def loschmidt_quench(
    omega: np.ndarray,
    K_topology: np.ndarray,
    K_initial: float,
    K_final: float,
    t_max: float = 10.0,
    n_times: int = 200,
) -> LoschmidtResult:
    """Compute Loschmidt echo after quench K_initial → K_final.

    G(t) = Σ_n |⟨n_f|ψ_i⟩|² exp(-iE_n^f t)
    where |n_f⟩, E_n^f are eigenstates of H(K_final).
    """
    n = len(omega)

    # Initial ground state
    K_i = K_initial * K_topology
    H_i = knm_to_hamiltonian(K_i, omega).to_matrix()
    eigvals_i, eigvecs_i = np.linalg.eigh(H_i)
    psi_i = eigvecs_i[:, 0]

    # Final Hamiltonian eigenbasis
    K_f = K_final * K_topology
    H_f = knm_to_hamiltonian(K_f, omega).to_matrix()
    eigvals_f, eigvecs_f = np.linalg.eigh(H_f)

    # Overlaps |⟨n_f|ψ_i⟩|²
    overlaps = np.abs(eigvecs_f.conj().T @ psi_i) ** 2

    # Loschmidt amplitude
    times = np.linspace(0, t_max, n_times)
    G: np.ndarray = np.zeros(n_times, dtype=complex)
    for idx, t in enumerate(times):
        G[idx] = np.sum(overlaps * np.exp(-1j * eigvals_f * t))

    amplitude: np.ndarray = np.abs(G)

    # Rate function g(t) = -log|G(t)|²/N
    rate: np.ndarray = np.zeros(n_times)
    for idx in range(n_times):
        if amplitude[idx] > 1e-15:
            rate[idx] = -2.0 * np.log(amplitude[idx]) / n
        else:
            rate[idx] = 30.0 / n  # cap for log(0)

    # Detect cusps: local maxima of |dg/dt|
    cusp_times: list[float] = []
    if n_times > 4:
        dg = np.gradient(rate, times)
        d2g = np.gradient(dg, times)
        # Cusps: where d²g/dt² changes sign AND |dg/dt| is large
        dg_threshold = np.std(dg) * 1.5
        for i in range(1, len(d2g) - 1):
            if d2g[i - 1] * d2g[i + 1] < 0 and abs(dg[i]) > dg_threshold:
                cusp_times.append(float(times[i]))

    return LoschmidtResult(
        times=times,
        loschmidt_amplitude=amplitude,
        rate_function=rate,
        n_cusps=len(cusp_times),
        cusp_times=cusp_times,
        K_initial=K_initial,
        K_final=K_final,
    )


def quench_scan(
    omega: np.ndarray,
    K_topology: np.ndarray,
    K_initial: float,
    K_final_range: np.ndarray | None = None,
    t_max: float = 10.0,
    n_times: int = 100,
) -> dict[str, list]:
    """Scan quenches from fixed K_i to varying K_f.

    Tests whether crossing K_c produces more/fewer DQPTs.
    """
    if K_final_range is None:
        K_final_range = np.linspace(0.5, 5.0, 10)

    results: dict[str, list] = {
        "K_final": [],
        "n_cusps": [],
        "max_rate": [],
        "mean_amplitude": [],
    }

    for kf in K_final_range:
        lr = loschmidt_quench(omega, K_topology, K_initial, float(kf), t_max, n_times)
        results["K_final"].append(float(kf))
        results["n_cusps"].append(lr.n_cusps)
        results["max_rate"].append(float(np.max(lr.rate_function)))
        results["mean_amplitude"].append(float(np.mean(lr.loschmidt_amplitude)))

    return results
