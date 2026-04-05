# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Experiment Dynamics
"""Kuramoto evolution experiments on quantum hardware."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from ..bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from ._experiment_helpers import (
    _build_evo_base,
    _build_xyz_circuits,
    _R_from_xyz,
)
from .classical import classical_exact_evolution


def kuramoto_4osc_experiment(
    runner, shots: int = 10000, n_time_steps: int = 8, dt: float = 0.1
) -> dict:
    """4-oscillator Kuramoto XY dynamics on hardware.

    Measures order parameter R(t) via X, Y, Z basis shots at each time step.
    Compares against exact matrix-exponential evolution.

    Returns:
        dict with keys:
            experiment (str): Experiment name identifier.
            n_oscillators (int): Number of oscillators.
            dt (float): Time step size.
            hw_times (list[float]): Hardware measurement times.
            hw_R (list[float]): Hardware order parameter per step.
            hw_R_std (list[float]): Shot-noise std of R per step.
            classical_times (list[float]): Exact evolution times.
            classical_R (list[float]): Exact order parameter per step.
            classical_R_std (float): Always 0.0 (exact).
            hw_expectations (list[dict]): Per-step exp_x, exp_y, exp_z.
    """
    n = 4
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]

    print(f"\n=== Kuramoto 4-oscillator, {n_time_steps} steps, dt={dt} ===")

    all_circuits: list[QuantumCircuit] = []
    step_indices: list[int] = []
    for step in range(1, n_time_steps + 1):
        t = step * dt
        base = _build_evo_base(n, K, omega, t, trotter_reps=step * 2)
        qc_z, qc_x, qc_y = _build_xyz_circuits(base, n)
        idx = len(all_circuits)
        all_circuits.extend([qc_z, qc_x, qc_y])
        step_indices.append(idx)

    hw_results = runner.run_sampler(all_circuits, shots=shots, name="kuramoto_4osc")

    hw_R = []
    hw_R_std = []
    hw_exp = []
    for idx in step_indices:
        R, R_std, ex, ey, ez, sx, sy, sz = _R_from_xyz(
            hw_results[idx].counts,
            hw_results[idx + 1].counts,
            hw_results[idx + 2].counts,
            n,
        )
        hw_R.append(R)
        hw_R_std.append(R_std)
        hw_exp.append({"exp_x": ex.tolist(), "exp_y": ey.tolist(), "exp_z": ez.tolist()})

    hw_times = [i * dt for i in range(1, n_time_steps + 1)]
    classical = classical_exact_evolution(n, n_time_steps * dt, dt, K, omega)

    result = {
        "experiment": "kuramoto_4osc",
        "n_oscillators": n,
        "dt": dt,
        "hw_times": hw_times,
        "hw_R": hw_R,
        "hw_R_std": hw_R_std,
        "classical_times": classical["times"].tolist(),
        "classical_R": classical["R"].tolist(),
        "classical_R_std": 0.0,
        "hw_expectations": hw_exp,
    }
    runner.save_result(hw_results[0], "kuramoto_4osc.json")
    return result


def kuramoto_8osc_experiment(
    runner, shots: int = 10000, n_time_steps: int = 6, dt: float = 0.1
) -> dict:
    """8-oscillator Kuramoto XY dynamics.

    Returns:
        dict with keys:
            experiment (str): Experiment name identifier.
            n_oscillators (int): Number of oscillators.
            dt (float): Time step size.
            hw_times (list[float]): Hardware measurement times.
            hw_R (list[float]): Hardware order parameter per step.
            hw_R_std (list[float]): Shot-noise std of R per step.
            classical_times (list[float]): Exact evolution times.
            classical_R (list[float]): Exact order parameter per step.
            classical_R_std (float): Always 0.0 (exact).
    """
    n = 8
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]

    print(f"\n=== Kuramoto 8-oscillator, {n_time_steps} steps, dt={dt} ===")

    all_circuits: list[QuantumCircuit] = []
    step_indices: list[int] = []
    for step in range(1, n_time_steps + 1):
        t = step * dt
        base = _build_evo_base(n, K, omega, t, trotter_reps=step)
        qc_z, qc_x, qc_y = _build_xyz_circuits(base, n)
        idx = len(all_circuits)
        all_circuits.extend([qc_z, qc_x, qc_y])
        step_indices.append(idx)

    hw_results = runner.run_sampler(all_circuits, shots=shots, name="kuramoto_8osc")

    hw_R = []
    hw_R_std = []
    for idx in step_indices:
        R, R_std, *_ = _R_from_xyz(
            hw_results[idx].counts,
            hw_results[idx + 1].counts,
            hw_results[idx + 2].counts,
            n,
        )
        hw_R.append(R)
        hw_R_std.append(R_std)

    hw_times = [i * dt for i in range(1, n_time_steps + 1)]
    classical = classical_exact_evolution(n, n_time_steps * dt, dt, K, omega)

    result = {
        "experiment": "kuramoto_8osc",
        "n_oscillators": n,
        "dt": dt,
        "hw_times": hw_times,
        "hw_R": hw_R,
        "hw_R_std": hw_R_std,
        "classical_times": classical["times"].tolist(),
        "classical_R": classical["R"].tolist(),
        "classical_R_std": 0.0,
    }
    runner.save_result(hw_results[0], "kuramoto_8osc.json")
    return result


def kuramoto_4osc_trotter2_experiment(
    runner, shots: int = 10000, n_time_steps: int = 8, dt: float = 0.1
) -> dict:
    """4-oscillator Kuramoto with second-order Suzuki-Trotter.

    Same structure as kuramoto_4osc_experiment but uses SuzukiTrotter(order=2).
    Produces order-1 vs order-2 comparison data.

    Returns:
        dict with keys:
            experiment (str): Experiment name identifier.
            n_oscillators (int): Number of oscillators.
            trotter_order (int): Suzuki-Trotter order (2).
            dt (float): Time step size.
            hw_times (list[float]): Hardware measurement times.
            hw_R (list[float]): Hardware order parameter per step.
            hw_R_std (list[float]): Shot-noise std of R per step.
            classical_times (list[float]): Exact evolution times.
            classical_R (list[float]): Exact order parameter per step.
            classical_R_std (float): Always 0.0 (exact).
            hw_expectations (list[dict]): Per-step exp_x, exp_y, exp_z.
    """
    n = 4
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]

    print(f"\n=== Kuramoto 4-osc Trotter-2, {n_time_steps} steps, dt={dt} ===")

    all_circuits: list[QuantumCircuit] = []
    step_indices: list[int] = []
    for step in range(1, n_time_steps + 1):
        t = step * dt
        base = _build_evo_base(n, K, omega, t, trotter_reps=step * 2, trotter_order=2)
        qc_z, qc_x, qc_y = _build_xyz_circuits(base, n)
        idx = len(all_circuits)
        all_circuits.extend([qc_z, qc_x, qc_y])
        step_indices.append(idx)

    hw_results = runner.run_sampler(all_circuits, shots=shots, name="kuramoto_4osc_trotter2")

    hw_R = []
    hw_R_std = []
    hw_exp = []
    for idx in step_indices:
        R, R_std, ex, ey, ez, *_ = _R_from_xyz(
            hw_results[idx].counts,
            hw_results[idx + 1].counts,
            hw_results[idx + 2].counts,
            n,
        )
        hw_R.append(R)
        hw_R_std.append(R_std)
        hw_exp.append({"exp_x": ex.tolist(), "exp_y": ey.tolist(), "exp_z": ez.tolist()})

    hw_times = [i * dt for i in range(1, n_time_steps + 1)]
    classical = classical_exact_evolution(n, n_time_steps * dt, dt, K, omega)

    return {
        "experiment": "kuramoto_4osc_trotter2",
        "n_oscillators": n,
        "trotter_order": 2,
        "dt": dt,
        "hw_times": hw_times,
        "hw_R": hw_R,
        "hw_R_std": hw_R_std,
        "classical_times": classical["times"].tolist(),
        "classical_R": classical["R"].tolist(),
        "classical_R_std": 0.0,
        "hw_expectations": hw_exp,
    }


def sync_threshold_experiment(
    runner,
    shots: int = 10000,
    k_values: list[float] | None = None,
) -> dict:
    """Kuramoto synchronization phase transition on quantum hardware.

    Sweeps coupling strength K_base and measures R at fixed t=0.1.
    Maps the bifurcation: below K_c, R~0 (incoherent); above K_c,
    R grows (synchronised). K_c depends on frequency spread.

    Science: first measurement of Kuramoto phase transition on
    superconducting qubits. Validates quantum XY <-> classical Kuramoto
    correspondence at the critical point.

    Returns:
        dict with keys:
            experiment (str): Experiment name identifier.
            n_oscillators (int): Number of oscillators.
            dt (float): Time step size.
            k_values (list[float]): Coupling strengths swept.
            results (list[dict]): Per-K dicts with K_base, hw_R, classical_R.
    """
    if k_values is None:
        k_values = [0.05, 0.15, 0.30, 0.45, 0.60, 0.80]

    n = 4
    omega = OMEGA_N_16[:n]
    dt = 0.1

    print(f"\n=== Sync threshold sweep, {len(k_values)} K values ===")

    alpha = 0.3  # Paper 27, Eq. 3

    results_per_k = []
    for k_base in k_values:
        # Pure exponential-decay K (no anchors) for clean bifurcation sweep
        idx = np.arange(n)
        K = k_base * np.exp(-alpha * np.abs(idx[:, None] - idx[None, :]))
        base = _build_evo_base(n, K, omega, dt, trotter_reps=2)
        qc_z, qc_x, qc_y = _build_xyz_circuits(base, n)
        hw = runner.run_sampler([qc_z, qc_x, qc_y], shots=shots, name=f"sync_K{k_base:.2f}")
        R, R_std, ex, ey, ez, *_ = _R_from_xyz(hw[0].counts, hw[1].counts, hw[2].counts, n)

        classical = classical_exact_evolution(n, dt, dt, K, omega)
        cl_R = float(classical["R"][-1])

        results_per_k.append(
            {
                "K_base": k_base,
                "hw_R": R,
                "classical_R": cl_R,
            }
        )
        print(f"  K={k_base:.2f}: hw_R={R:.4f}, exact_R={cl_R:.4f}")

    return {
        "experiment": "sync_threshold",
        "n_oscillators": n,
        "dt": dt,
        "k_values": k_values,
        "results": results_per_k,
    }
