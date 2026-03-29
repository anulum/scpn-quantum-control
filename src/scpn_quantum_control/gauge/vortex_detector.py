# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Vortex density measurement for the Kuramoto-XY quantum model.

In the XY/U(1) lattice gauge theory, vortices are topological defects
where the phase winds by ±2π around a plaquette. The BKT transition
is characterised by vortex-antivortex pair binding/unbinding:

    T < T_BKT: bound pairs only (ordered, confined)
    T > T_BKT: free vortices proliferate (disordered, deconfined)

For a plaquette P = (i, j, k) on the coupling graph, the vorticity is:

    q_P = round(Σ_{(a,b)∈P} Δθ_{ab} / (2π))

where Δθ_{ab} = θ_b - θ_a (mod [-π, π]).

In the quantum model, we measure the phases θ_i = atan2(<Y_i>, <X_i>)
from the ground state and compute vorticity from phase differences.

The vortex density ρ_v = N_vortices / N_plaquettes is the BKT order
parameter: ρ_v → 0 in the ordered phase, ρ_v > 0 in the disordered.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

from ..hardware.classical import classical_exact_diag


@dataclass
class VortexResult:
    """Vortex density measurement result."""

    n_vortices: int
    n_antivortices: int
    n_plaquettes: int
    vortex_density: float  # (n_v + n_av) / n_plaquettes
    net_charge: int  # n_v - n_av (should be 0 on closed manifold)
    plaquette_vorticities: list[int]
    phases: np.ndarray  # extracted oscillator phases


def _extract_phases(psi: np.ndarray, n_qubits: int) -> np.ndarray:
    """Extract oscillator phases from statevector via Pauli expectations."""
    sv = Statevector(np.ascontiguousarray(psi))
    phases = np.zeros(n_qubits)
    for i in range(n_qubits):
        label_x = ["I"] * n_qubits
        label_x[i] = "X"
        label_y = ["I"] * n_qubits
        label_y[i] = "Y"
        exp_x = float(sv.expectation_value(SparsePauliOp("".join(reversed(label_x)))).real)
        exp_y = float(sv.expectation_value(SparsePauliOp("".join(reversed(label_y)))).real)
        phases[i] = np.arctan2(exp_y, exp_x)
    result: np.ndarray = phases
    return result


def _find_plaquettes(K: np.ndarray) -> list[list[int]]:
    """Find triangular plaquettes on the coupling graph.

    A plaquette is a minimal closed loop. For a general graph,
    triangles (3-cycles) are the smallest plaquettes.
    """
    n = K.shape[0]
    plaquettes: list[list[int]] = []
    for i, j, k in combinations(range(n), 3):
        if K[i, j] > 0 and K[j, k] > 0 and K[k, i] > 0:
            plaquettes.append([i, j, k])
    return plaquettes


def _angle_diff(a: float, b: float) -> float:
    """Signed angle difference in [-π, π]."""
    d = b - a
    return float(d - 2 * np.pi * round(d / (2 * np.pi)))


def plaquette_vorticity(phases: np.ndarray, plaquette: list[int]) -> int:
    """Compute vorticity (winding number) around a plaquette.

    q = round(Σ Δθ / (2π)) where Δθ is the signed phase difference.
    """
    total = 0.0
    for idx in range(len(plaquette)):
        i = plaquette[idx]
        j = plaquette[(idx + 1) % len(plaquette)]
        total += _angle_diff(float(phases[i]), float(phases[j]))
    return int(round(total / (2 * np.pi)))


def measure_vortex_density(
    K: np.ndarray,
    omega: np.ndarray,
) -> VortexResult:
    """Measure vortex density from the ground state of H(K, omega)."""
    n = K.shape[0]
    exact = classical_exact_diag(n, K=K, omega=omega)
    psi = exact["ground_state"]
    phases = _extract_phases(psi, n)

    plaquettes = _find_plaquettes(K)
    vorticities: list[int] = []

    for plaq in plaquettes:
        v = plaquette_vorticity(phases, plaq)
        vorticities.append(v)

    n_v = sum(1 for v in vorticities if v > 0)
    n_av = sum(1 for v in vorticities if v < 0)
    n_plaq = len(plaquettes)
    density = (n_v + n_av) / max(n_plaq, 1)

    return VortexResult(
        n_vortices=n_v,
        n_antivortices=n_av,
        n_plaquettes=n_plaq,
        vortex_density=density,
        net_charge=n_v - n_av,
        plaquette_vorticities=vorticities,
        phases=phases,
    )


def vortex_density_vs_coupling(
    omega: np.ndarray,
    k_base_values: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Scan vortex density vs coupling strength.

    At the BKT transition, vortex density should jump from 0 to finite.
    """
    from ..bridge.knm_hamiltonian import build_knm_paper27

    if k_base_values is None:
        k_base_values = np.linspace(0.01, 3.0, 20)

    n = len(omega)
    results: dict[str, list[float]] = {
        "k_base": [],
        "vortex_density": [],
        "n_vortices": [],
        "net_charge": [],
    }

    for kb in k_base_values:
        K = build_knm_paper27(L=n, K_base=float(kb))
        vr = measure_vortex_density(K, omega)
        results["k_base"].append(float(kb))
        results["vortex_density"].append(vr.vortex_density)
        results["n_vortices"].append(float(vr.n_vortices + vr.n_antivortices))
        results["net_charge"].append(float(vr.net_charge))

    return results
