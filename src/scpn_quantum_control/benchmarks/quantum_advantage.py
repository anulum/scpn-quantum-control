# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Quantum Advantage
"""Quantum vs classical scaling benchmark for Kuramoto Hamiltonian simulation.

Measures wall-clock time for classical (exact diag + matrix exp) vs quantum
(Trotter on statevector), then extrapolates scaling crossover.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit

from ..bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27, knm_to_hamiltonian
from ..hardware.classical import classical_exact_diag, classical_exact_evolution

log = logging.getLogger(__name__)

MAX_CLASSICAL_QUBITS = 14  # 2^14 = 16384 state dim; beyond this expm is infeasible


@dataclass
class AdvantageResult:
    """Scaling benchmark result for one system size."""

    n_qubits: int
    t_classical_ms: float
    t_quantum_ms: float
    errors: dict = field(default_factory=dict)
    crossover_predicted: int | None = None


def classical_benchmark(n: int, t_max: float = 1.0, dt: float = 0.1) -> dict:
    """Time classical exact evolution of XY Hamiltonian.

    For n > MAX_CLASSICAL_QUBITS, returns inf (matrix expm infeasible).
    """
    if n > MAX_CLASSICAL_QUBITS:
        return {"t_total_ms": float("inf"), "ground_energy": None, "R_final": None}

    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n].copy()

    t0 = time.perf_counter()
    evo = classical_exact_evolution(n, t_max, dt, K=K, omega=omega)
    t_evo = time.perf_counter() - t0

    t0 = time.perf_counter()
    diag = classical_exact_diag(n, K=K, omega=omega)
    t_diag = time.perf_counter() - t0

    return {
        "t_total_ms": (t_evo + t_diag) * 1000,
        "ground_energy": diag["ground_energy"],
        "R_final": float(evo["R"][-1]),
    }


def quantum_benchmark(
    n: int,
    t_max: float = 1.0,
    dt: float = 0.1,
    trotter_reps: int = 5,
) -> dict:
    """Time Trotter evolution on statevector simulator."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import Statevector
    from qiskit.synthesis import LieTrotter

    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n].copy()
    H = knm_to_hamiltonian(K, omega)
    n_steps = max(1, int(t_max / dt))

    t0 = time.perf_counter()

    qc_init = QuantumCircuit(n)
    for i in range(n):
        qc_init.ry(float(omega[i]) % (2 * np.pi), i)
    sv = Statevector.from_instruction(qc_init)

    synth = LieTrotter(reps=trotter_reps)
    evo_gate = PauliEvolutionGate(H, time=dt, synthesis=synth)
    step_qc = QuantumCircuit(n)
    step_qc.append(evo_gate, range(n))

    for _ in range(n_steps):
        sv = sv.evolve(step_qc)

    t_total = time.perf_counter() - t0

    return {
        "t_total_ms": t_total * 1000,
        "n_trotter_steps": n_steps * trotter_reps,
    }


def estimate_crossover(results: list[AdvantageResult]) -> int | None:
    """Fit exponential scaling, extrapolate where quantum becomes faster.

    Returns predicted qubit count at crossover, or None.
    """
    feasible = [r for r in results if np.isfinite(r.t_classical_ms)]
    if len(feasible) < 3:
        return None

    ns = np.array([r.n_qubits for r in feasible], dtype=float)
    t_c = np.array([r.t_classical_ms for r in feasible])
    t_q = np.array([r.t_quantum_ms for r in feasible])

    def exp_fit(x: np.ndarray, a: float, b: float) -> np.ndarray:
        result: np.ndarray = a * np.exp(b * x)
        return result

    try:
        popt_c, _ = curve_fit(exp_fit, ns, t_c, p0=[0.01, 0.5], maxfev=2000)
        popt_q, _ = curve_fit(exp_fit, ns, t_q, p0=[0.01, 0.1], maxfev=2000)
    except RuntimeError:
        return None

    if popt_c[1] <= popt_q[1]:
        return None

    ratio = popt_q[0] / popt_c[0]
    if ratio <= 0:
        return None
    n_cross = np.log(ratio) / (popt_c[1] - popt_q[1])
    return int(np.ceil(n_cross))


def run_scaling_benchmark(
    sizes: list[int] | None = None,
    t_max: float = 1.0,
    dt: float = 0.1,
) -> list[AdvantageResult]:
    """Full scaling benchmark across system sizes.

    Default sizes=[4, 8, 12, 16, 20]. N=20 is ~8 MB statevector (quantum only).
    Classical exact evolution infeasible beyond n=14.
    """
    if sizes is None:
        sizes = [4, 8, 12, 16, 20]

    results: list[AdvantageResult] = []
    for n in sizes:
        if n > 23:
            warnings.warn(
                f"n={n} requires {2**n * 16 / 1e9:.1f} GB statevector memory",
                stacklevel=2,
            )

        log.info("Benchmarking n=%d", n)
        c = classical_benchmark(n, t_max, dt)
        q = quantum_benchmark(n, t_max, dt)

        results.append(
            AdvantageResult(
                n_qubits=n,
                t_classical_ms=c["t_total_ms"],
                t_quantum_ms=q["t_total_ms"],
            )
        )

    if len(results) >= 3:
        crossover = estimate_crossover(results)
        for r in results:
            r.crossover_predicted = crossover

    return results
