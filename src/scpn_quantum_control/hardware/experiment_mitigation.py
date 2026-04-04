# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Experiment Mitigation
"""ZNE, dynamical decoupling, and noise characterisation experiments."""

from __future__ import annotations

import numpy as np

from ..bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from ._experiment_helpers import (
    _build_evo_base,
    _build_xyz_circuits,
    _R_from_xyz,
)
from .classical import classical_exact_evolution


def kuramoto_4osc_zne_experiment(
    runner, shots: int = 10000, dt: float = 0.1, scales: list[int] | None = None
) -> dict:
    """4-oscillator Kuramoto with ZNE error mitigation.

    Runs the evolution at multiple noise scales via unitary folding,
    then extrapolates to zero noise.

    Returns:
        dict with keys:
            experiment (str): Experiment name identifier.
            scales (list[int]): Noise scale factors used.
            R_per_scale (list[float]): Measured R at each scale.
            zne_R (float): Zero-noise extrapolated R.
            classical_R (float): Exact order parameter.
            fit_residual (float): Polynomial fit residual.
    """
    from ..mitigation.zne import gate_fold_circuit, zne_extrapolate

    if scales is None:
        scales = [1, 3, 5]

    n = 4
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    t = dt

    print(f"\n=== Kuramoto 4-osc ZNE, dt={dt}, scales={scales} ===")

    base = _build_evo_base(n, K, omega, t, trotter_reps=2)

    R_per_scale = []
    R_std_per_scale = []
    for s in scales:
        folded = gate_fold_circuit(base, s)
        qc_z, qc_x, qc_y = _build_xyz_circuits(folded, n)
        hw = runner.run_sampler([qc_z, qc_x, qc_y], shots=shots, name=f"zne_s{s}")
        R, R_std, *_ = _R_from_xyz(hw[0].counts, hw[1].counts, hw[2].counts, n)
        R_per_scale.append(R)
        R_std_per_scale.append(R_std)
        print(f"  scale={s}: R={R:.4f} ± {R_std:.4f}")

    zne = zne_extrapolate(scales, R_per_scale, order=1)
    classical = classical_exact_evolution(n, dt, dt, K, omega)

    print(f"  ZNE R(0) = {zne.zero_noise_estimate:.4f}")
    print(f"  Exact R  = {classical['R'][-1]:.4f}")

    return {
        "experiment": "kuramoto_4osc_zne",
        "scales": scales,
        "R_per_scale": R_per_scale,
        "R_std_per_scale": R_std_per_scale,
        "zne_R": zne.zero_noise_estimate,
        "classical_R": float(classical["R"][-1]),
        "fit_residual": zne.fit_residual,
    }


def noise_baseline_experiment(runner, shots: int = 10000) -> dict:
    """4-qubit near-identity circuit for calibration drift detection.

    Single Trotter step at dt=0.01 (near-identity). Measures R + per-qubit
    expectations. Compare Feb->Mar to detect backend drift.

    Returns:
        dict with keys:
            experiment (str): Experiment name identifier.
            n_qubits (int): Number of qubits.
            dt (float): Time step size (0.01, near-identity).
            hw_R (float): Hardware order parameter.
            hw_R_std (float): Shot-noise std of R.
            classical_R (float): Exact order parameter.
            classical_R_std (float): Always 0.0 (exact).
            hw_exp_x (list[float]): Per-qubit X expectations.
            hw_exp_y (list[float]): Per-qubit Y expectations.
            hw_exp_z (list[float]): Per-qubit Z expectations.
    """
    n = 4
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    dt = 0.01

    print(f"\n=== Noise baseline, {n}q, dt={dt} ===")

    base = _build_evo_base(n, K, omega, dt, trotter_reps=1)
    qc_z, qc_x, qc_y = _build_xyz_circuits(base, n)

    hw_results = runner.run_sampler([qc_z, qc_x, qc_y], shots=shots, name="noise_baseline")
    R, R_std, exp_x, exp_y, exp_z, *_ = _R_from_xyz(
        hw_results[0].counts, hw_results[1].counts, hw_results[2].counts, n
    )

    classical = classical_exact_evolution(n, dt, dt, K, omega)

    result = {
        "experiment": "noise_baseline",
        "n_qubits": n,
        "dt": dt,
        "hw_R": R,
        "hw_R_std": R_std,
        "classical_R": float(classical["R"][-1]),
        "classical_R_std": 0.0,
        "hw_exp_x": exp_x.tolist(),
        "hw_exp_y": exp_y.tolist(),
        "hw_exp_z": exp_z.tolist(),
    }
    runner.save_result(hw_results[0], "noise_baseline.json")
    return result


def kuramoto_8osc_zne_experiment(
    runner, shots: int = 10000, dt: float = 0.1, scales: list[int] | None = None
) -> dict:
    """8-oscillator Kuramoto with ZNE error mitigation.

    Gate-fold at each noise scale, Richardson extrapolation to zero noise.
    Extends the 4-osc ZNE result to depth-233 territory.

    Returns:
        dict with keys:
            experiment (str): Experiment name identifier.
            n_oscillators (int): Number of oscillators.
            scales (list[int]): Noise scale factors used.
            R_per_scale (list[float]): Measured R at each scale.
            zne_R (float): Zero-noise extrapolated R.
            classical_R (float): Exact order parameter.
            fit_residual (float): Polynomial fit residual.
    """
    from ..mitigation.zne import gate_fold_circuit, zne_extrapolate

    if scales is None:
        scales = [1, 3, 5]

    n = 8
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    t = dt

    print(f"\n=== Kuramoto 8-osc ZNE, dt={dt}, scales={scales} ===")

    base = _build_evo_base(n, K, omega, t, trotter_reps=2)

    R_per_scale = []
    R_std_per_scale = []
    for s in scales:
        folded = gate_fold_circuit(base, s)
        qc_z, qc_x, qc_y = _build_xyz_circuits(folded, n)
        hw = runner.run_sampler([qc_z, qc_x, qc_y], shots=shots, name=f"zne8_s{s}")
        R, R_std, *_ = _R_from_xyz(hw[0].counts, hw[1].counts, hw[2].counts, n)
        R_per_scale.append(R)
        R_std_per_scale.append(R_std)
        print(f"  scale={s}: R={R:.4f} ± {R_std:.4f}")

    zne = zne_extrapolate(scales, R_per_scale, order=1)
    classical = classical_exact_evolution(n, dt, dt, K, omega)

    print(f"  ZNE R(0) = {zne.zero_noise_estimate:.4f}")
    print(f"  Exact R  = {classical['R'][-1]:.4f}")

    return {
        "experiment": "kuramoto_8osc_zne",
        "n_oscillators": n,
        "scales": scales,
        "R_per_scale": R_per_scale,
        "R_std_per_scale": R_std_per_scale,
        "zne_R": zne.zero_noise_estimate,
        "classical_R": float(classical["R"][-1]),
        "fit_residual": zne.fit_residual,
    }


def upde_16_dd_experiment(runner, shots: int = 20000, trotter_steps: int = 1) -> dict:
    """16-layer UPDE with dynamical decoupling.

    Same structure as upde_16_snapshot but applies DD (XY4) to each
    basis circuit before submission. Compares R(DD) vs R(no-DD) vs classical.

    Returns:
        dict with keys:
            experiment (str): Experiment name identifier.
            n_layers (int): Number of UPDE layers (16).
            dt (float): Time step size.
            trotter_steps (int): Number of Trotter repetitions.
            hw_R_raw (float): Hardware R without DD.
            hw_R_dd (float): Hardware R with dynamical decoupling.
            classical_R (float): Exact order parameter.
            hw_exp_x_dd (list[float]): Per-qubit X expectations (DD).
            hw_exp_y_dd (list[float]): Per-qubit Y expectations (DD).
            hw_exp_z_dd (list[float]): Per-qubit Z expectations (DD).
    """
    n = 16
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16.copy()
    dt = 0.05

    print(f"\n=== UPDE 16-layer DD, dt={dt}, {trotter_steps} Trotter steps ===")

    base = _build_evo_base(n, K, omega, dt, trotter_reps=trotter_steps)
    qc_z, qc_x, qc_y = _build_xyz_circuits(base, n)

    # No-DD run
    hw_raw = runner.run_sampler([qc_z, qc_x, qc_y], shots=shots, name="upde16_raw")
    R_raw, R_raw_std, exp_x_raw, exp_y_raw, exp_z_raw, *_ = _R_from_xyz(
        hw_raw[0].counts, hw_raw[1].counts, hw_raw[2].counts, n
    )

    # DD run
    dd_z = runner.transpile_with_dd(qc_z)
    dd_x = runner.transpile_with_dd(qc_x)
    dd_y = runner.transpile_with_dd(qc_y)
    hw_dd = runner.run_sampler([dd_z, dd_x, dd_y], shots=shots, name="upde16_dd")
    R_dd, R_dd_std, exp_x_dd, exp_y_dd, exp_z_dd, *_ = _R_from_xyz(
        hw_dd[0].counts, hw_dd[1].counts, hw_dd[2].counts, n
    )

    classical = classical_exact_evolution(n, dt, dt, K, omega)

    print(f"  R(raw)={R_raw:.4f}, R(DD)={R_dd:.4f}, R(exact)={classical['R'][-1]:.4f}")

    result = {
        "experiment": "upde_16_dd",
        "n_layers": n,
        "dt": dt,
        "trotter_steps": trotter_steps,
        "hw_R_raw": R_raw,
        "hw_R_dd": R_dd,
        "classical_R": float(classical["R"][-1]),
        "hw_exp_x_dd": exp_x_dd.tolist(),
        "hw_exp_y_dd": exp_y_dd.tolist(),
        "hw_exp_z_dd": exp_z_dd.tolist(),
    }
    runner.save_result(hw_raw[0], "upde_16_dd.json")
    return result


def zne_higher_order_experiment(
    runner,
    shots: int = 10000,
    dt: float = 0.1,
    scales: list[int] | None = None,
    poly_order: int = 2,
) -> dict:
    """ZNE with extended noise scales and higher-order polynomial extrapolation.

    Default: scales=[1,3,5,7,9], quadratic fit. Tests whether 5-point
    polynomial extrapolation recovers more signal than the 3-point linear
    version (kuramoto_4osc_zne).

    Science: systematic ZNE study -- linear vs quadratic vs cubic on the
    same data. Determines optimal extrapolation order for XY evolution on
    Heron r2.

    Returns:
        dict with keys:
            experiment (str): Experiment name identifier.
            scales (list[int]): Noise scale factors used.
            R_per_scale (list[float]): Measured R at each scale.
            extrapolations (dict): Per-order dicts with zne_R, fit_residual.
            classical_R (float): Exact order parameter.
    """
    from ..mitigation.zne import gate_fold_circuit, zne_extrapolate

    if scales is None:
        scales = [1, 3, 5, 7, 9]

    n = 4
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    t = dt

    print(f"\n=== ZNE higher-order, scales={scales}, poly_order={poly_order} ===")

    base = _build_evo_base(n, K, omega, t, trotter_reps=2)

    R_per_scale = []
    R_std_per_scale = []
    for s in scales:
        folded = gate_fold_circuit(base, s)
        qc_z, qc_x, qc_y = _build_xyz_circuits(folded, n)
        hw = runner.run_sampler([qc_z, qc_x, qc_y], shots=shots, name=f"zne_ho_s{s}")
        R, R_std, *_ = _R_from_xyz(hw[0].counts, hw[1].counts, hw[2].counts, n)
        R_per_scale.append(R)
        R_std_per_scale.append(R_std)
        print(f"  scale={s}: R={R:.4f} ± {R_std:.4f}")

    zne_results = {}
    for order in range(1, poly_order + 1):
        zne = zne_extrapolate(scales, R_per_scale, order=order)
        zne_results[f"order_{order}"] = {
            "zne_R": zne.zero_noise_estimate,
            "fit_residual": zne.fit_residual,
        }
        print(
            f"  order-{order} ZNE R(0) = {zne.zero_noise_estimate:.4f} (residual={zne.fit_residual:.4f})"
        )

    classical = classical_exact_evolution(n, dt, dt, K, omega)

    return {
        "experiment": "zne_higher_order",
        "scales": scales,
        "R_per_scale": R_per_scale,
        "R_std_per_scale": R_std_per_scale,
        "extrapolations": zne_results,
        "classical_R": float(classical["R"][-1]),
    }


def decoherence_scaling_experiment(
    runner,
    shots: int = 10000,
    qubit_counts: list[int] | None = None,
) -> dict:
    """Systematic decoherence scaling: R vs circuit depth across qubit counts.

    Runs 1-Trotter-step evolution at fixed dt=0.1 for each qubit count,
    records depth, R, and exact R. Provides data for fitting
    R_hw = R_exact * exp(-gamma * depth).

    Science: extracts per-gate depolarization rate gamma from a single
    calibration run. Enables predictive modeling of experiment fidelity.

    Returns:
        dict with keys:
            experiment (str): Experiment name identifier.
            dt (float): Time step size.
            data_points (list[dict]): Per-qubit-count dicts with n_qubits,
                depth, hw_R, classical_R.
            fit_gamma (float): Fitted per-gate depolarization rate.
            fit_r_squared (float): R-squared of exponential fit.
    """
    if qubit_counts is None:
        qubit_counts = [2, 4, 6, 8, 10, 12]

    dt = 0.1

    print(f"\n=== Decoherence scaling, qubits={qubit_counts} ===")

    data_points = []
    for n in qubit_counts:
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        base = _build_evo_base(n, K, omega, dt, trotter_reps=1)
        qc_z, qc_x, qc_y = _build_xyz_circuits(base, n)
        hw = runner.run_sampler([qc_z, qc_x, qc_y], shots=shots, name=f"decoherence_{n}q")
        R, R_std, *_ = _R_from_xyz(hw[0].counts, hw[1].counts, hw[2].counts, n)

        isa = runner.transpile(base)
        depth = isa.depth()

        classical = classical_exact_evolution(n, dt, dt, K, omega)
        cl_R = float(classical["R"][-1])

        data_points.append(
            {
                "n_qubits": n,
                "depth": depth,
                "hw_R": R,
                "classical_R": cl_R,
            }
        )
        print(f"  {n}q: depth={depth}, hw_R={R:.4f}, exact_R={cl_R:.4f}")

    # Fit exponential decay: R_hw = R_exact * exp(-gamma * depth)
    depths = np.array([d["depth"] for d in data_points], dtype=float)
    ratios = np.array([d["hw_R"] / max(d["classical_R"], 1e-10) for d in data_points])
    valid = ratios > 0
    if np.sum(valid) >= 2:
        log_ratios = np.log(np.clip(ratios[valid], 1e-10, None))
        coeffs = np.polyfit(depths[valid], log_ratios, 1)
        gamma = -coeffs[0]
        r_squared = 1.0 - np.var(log_ratios - np.polyval(coeffs, depths[valid])) / max(
            float(np.var(log_ratios)), 1e-10
        )
    else:
        gamma = np.float64("nan")
        r_squared = np.float64("nan")

    print(f"  Fit: gamma={gamma:.6f} per gate, R²={r_squared:.4f}")

    return {
        "experiment": "decoherence_scaling",
        "dt": dt,
        "data_points": data_points,
        "fit_gamma": gamma,
        "fit_r_squared": r_squared,
    }
