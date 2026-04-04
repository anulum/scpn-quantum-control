# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Experiment Control
"""QAOA-MPC, UPDE snapshot, Bell test, correlator, and QKD experiments."""

from __future__ import annotations

import numpy as np

from ..bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from ..crypto.entanglement_qkd import (
    bell_inequality_test,
    correlator_matrix,
    scpn_qkd_protocol,
)
from ..crypto.knm_key import estimate_qber, extract_raw_key, prepare_key_state
from ..crypto.noise_analysis import devetak_winter_rate
from ._experiment_helpers import (
    _build_evo_base,
    _build_xyz_circuits,
    _correlator_from_counts,
    _expectation_per_qubit,
    _qaoa_cost_from_counts,
    _R_from_xyz,
)
from .classical import classical_brute_mpc, classical_exact_evolution


def qaoa_mpc_4_experiment(runner, shots: int = 10000) -> dict:
    """QAOA-MPC binary control, horizon=4, p=1 and p=2.

    Cost Hamiltonian is diagonal in Z, so Z-basis measurement is exact.
    Compares QAOA solution quality vs brute-force optimal.

    Returns:
        dict with keys:
            experiment (str): Experiment name identifier.
            horizon (int): MPC planning horizon.
            brute_force_cost (float): Optimal cost from exhaustive search.
            brute_force_actions (list[int]): Optimal binary actions.
            qaoa_p1 (dict): p=1 results (qaoa_cost, qaoa_actions, n_iterations).
            qaoa_p2 (dict): p=2 results (qaoa_cost, qaoa_actions, n_iterations).
    """
    from scipy.optimize import minimize

    B = np.eye(2)
    target = np.array([0.8, 0.6])
    horizon = 4

    print(f"\n=== QAOA-MPC h={horizon}, target={target} ===")

    brute = classical_brute_mpc(B, target, horizon)
    print(f"  Brute-force optimal cost: {brute['optimal_cost']:.6f}")
    print(f"  Brute-force optimal actions: {brute['optimal_actions']}")

    from ..control.qaoa_mpc import QAOA_MPC

    results_by_p = {}
    for p in [1, 2]:
        mpc = QAOA_MPC(B, target, horizon=horizon, p_layers=p)
        mpc.build_cost_hamiltonian()

        def cost_fn(params, _p=p, _mpc=mpc):
            gamma = params[:_p]
            beta = params[_p:]
            qc = _mpc._build_qaoa_circuit(gamma, beta)
            qc.measure_all()
            hw = runner.run_sampler(qc, shots=shots, name=f"qaoa_p{_p}")
            counts = hw[0].counts
            return _qaoa_cost_from_counts(counts, _mpc._cost_ham, _mpc.n_qubits)

        x0 = np.random.default_rng(42).uniform(0, np.pi, 2 * p)
        opt = minimize(cost_fn, x0, method="COBYLA", options={"maxiter": 100})

        gamma_opt = opt.x[:p]
        beta_opt = opt.x[p:]
        final_qc = mpc._build_qaoa_circuit(gamma_opt, beta_opt)
        final_qc.measure_all()
        final_hw = runner.run_sampler(final_qc, shots=shots, name=f"qaoa_p{p}_final")
        counts = final_hw[0].counts
        best_bitstring = max(counts, key=counts.get)
        actions = np.array([int(b) for b in reversed(best_bitstring)])

        print(f"  QAOA p={p}: cost={opt.fun:.6f}, actions={actions}, iters={opt.nfev}")

        results_by_p[p] = {
            "qaoa_cost": float(opt.fun),
            "qaoa_actions": actions.tolist(),
            "n_iterations": opt.nfev,
        }

    result = {
        "experiment": "qaoa_mpc_4",
        "horizon": horizon,
        "brute_force_cost": brute["optimal_cost"],
        "brute_force_actions": brute["optimal_actions"].tolist(),
        "qaoa_p1": results_by_p[1],
        "qaoa_p2": results_by_p[2],
    }
    return result


def upde_16_snapshot_experiment(runner, shots: int = 20000, trotter_steps: int = 1) -> dict:
    """Full 16-layer UPDE single Trotter snapshot.

    Measures in X, Y, Z bases. Compares R against exact evolution.
    ~240 ECR gates. On real hardware, needs error mitigation.

    Returns:
        dict with keys:
            experiment (str): Experiment name identifier.
            n_layers (int): Number of UPDE layers (16).
            dt (float): Time step size.
            trotter_steps (int): Number of Trotter repetitions.
            hw_R (float): Hardware order parameter.
            hw_R_std (float): Shot-noise std of R.
            classical_R (float): Exact order parameter.
            classical_R_std (float): Always 0.0 (exact).
            hw_exp_x (list[float]): Per-qubit X expectations.
            hw_exp_y (list[float]): Per-qubit Y expectations.
            hw_exp_z (list[float]): Per-qubit Z expectations.
    """
    n = 16
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16.copy()
    dt = 0.05

    print(f"\n=== UPDE 16-layer snapshot, dt={dt}, {trotter_steps} Trotter steps ===")

    base = _build_evo_base(n, K, omega, dt, trotter_reps=trotter_steps)
    qc_z, qc_x, qc_y = _build_xyz_circuits(base, n)

    hw_results = runner.run_sampler([qc_z, qc_x, qc_y], shots=shots, name="upde_16")

    R, R_std, exp_x, exp_y, exp_z, *_ = _R_from_xyz(
        hw_results[0].counts,
        hw_results[1].counts,
        hw_results[2].counts,
        n,
    )

    classical = classical_exact_evolution(n, dt, dt, K, omega)

    result = {
        "experiment": "upde_16_snapshot",
        "n_layers": n,
        "dt": dt,
        "trotter_steps": trotter_steps,
        "hw_R": R,
        "hw_R_std": R_std,
        "classical_R": float(classical["R"][-1]),
        "classical_R_std": 0.0,
        "hw_exp_x": exp_x.tolist(),
        "hw_exp_y": exp_y.tolist(),
        "hw_exp_z": exp_z.tolist(),
    }
    runner.save_result(hw_results[0], "upde_16_snapshot.json")
    return result


def bell_test_4q_experiment(runner, shots: int = 10000, maxiter: int = 100) -> dict:
    """CHSH Bell test on 4-qubit K_nm ground state.

    Certifies entanglement between qubits 0 and 1 via CHSH inequality
    violation. S > 2 proves non-classical correlations on hardware.
    ~20s QPU budget (4 circuits x ~5s each).

    Returns:
        dict with keys:
            experiment (str): Experiment name identifier.
            S_hw (float): CHSH S value from hardware.
            S_sim (float): CHSH S value from simulation.
            violates_classical_hw (bool): S_hw > 2.0.
            violates_classical_sim (bool): S_sim > 2.0.
            correlators_hw (dict): ZZ, ZX, XZ, XX correlators (hardware).
            correlators_sim (dict): Correlators from simulation.
    """
    n = 4
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]

    print(f"\n=== Bell test 4q, maxiter={maxiter} ===")

    key_state = prepare_key_state(K, omega, ansatz_reps=2, maxiter=maxiter)
    base_circuit = key_state["circuit"]
    sv = key_state["statevector"]

    # 4 measurement circuits: ZZ, ZX, XZ, XX on qubits 0,1
    qc_zz = base_circuit.copy()
    qc_zz.measure_all()

    qc_zx = base_circuit.copy()
    qc_zx.h(1)
    qc_zx.measure_all()

    qc_xz = base_circuit.copy()
    qc_xz.h(0)
    qc_xz.measure_all()

    qc_xx = base_circuit.copy()
    qc_xx.h(0)
    qc_xx.h(1)
    qc_xx.measure_all()

    hw_results = runner.run_sampler([qc_zz, qc_zx, qc_xz, qc_xx], shots=shots, name="bell_test_4q")

    e_zz = _correlator_from_counts(hw_results[0].counts, 0, 1)
    e_zx = _correlator_from_counts(hw_results[1].counts, 0, 1)
    e_xz = _correlator_from_counts(hw_results[2].counts, 0, 1)
    e_xx = _correlator_from_counts(hw_results[3].counts, 0, 1)

    s_hw = abs(e_zz - e_zx + e_xz + e_xx)

    sim_bell = bell_inequality_test(sv, 0, 1, n)
    s_sim = sim_bell["S"]

    print(f"  S_hw={s_hw:.4f}, S_sim={s_sim:.4f}")

    return {
        "experiment": "bell_test_4q",
        "S_hw": s_hw,
        "S_sim": s_sim,
        "violates_classical_hw": s_hw > 2.0,
        "violates_classical_sim": sim_bell["violates_classical"],
        "correlators_hw": {"ZZ": e_zz, "ZX": e_zx, "XZ": e_xz, "XX": e_xx},
        "correlators_sim": sim_bell["correlators"],
    }


def correlator_4q_experiment(runner, shots: int = 10000, maxiter: int = 100) -> dict:
    """ZZ cross-correlation of 4-qubit K_nm ground state on hardware.

    Validates that the K_ij coupling topology maps to measurable quantum
    correlations. Connected correlation C[i,j] = <Z_i Z_j> - <Z_i><Z_j>.
    ~25s QPU budget (1 circuit).

    Returns:
        dict with keys:
            experiment (str): Experiment name identifier.
            corr_hw (list[list[float]]): 4x4 connected correlation matrix (hw).
            corr_sim (list[list[float]]): 4x4 connected correlation matrix (sim).
            frobenius_error (float): ||corr_hw - corr_sim||_F.
            max_correlation_hw (float): Max absolute correlation on hardware.
    """
    n = 4
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]

    print(f"\n=== Correlator 4q, maxiter={maxiter} ===")

    key_state = prepare_key_state(K, omega, ansatz_reps=2, maxiter=maxiter)
    base_circuit = key_state["circuit"]
    sv = key_state["statevector"]

    qc_z = base_circuit.copy()
    qc_z.measure_all()
    hw_results = runner.run_sampler(qc_z, shots=shots, name="correlator_4q")
    counts = hw_results[0].counts

    # Per-qubit <Z_i> from counts
    exp_z, _ = _expectation_per_qubit(counts, n)

    # <Z_i Z_j> for all pairs
    corr_hw = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                corr_hw[i, j] = 1.0 - exp_z[i] ** 2
            else:
                zz = _correlator_from_counts(counts, i, j)
                corr_hw[i, j] = zz - exp_z[i] * exp_z[j]

    # Simulator reference: correlator_matrix gives alice×bob block
    corr_sim_block = correlator_matrix(sv, [0, 1], [2, 3])
    corr_sim = np.zeros((n, n))
    corr_sim[:2, 2:] = corr_sim_block
    corr_sim[2:, :2] = corr_sim_block.T

    frob_error = float(np.linalg.norm(corr_hw - corr_sim))
    max_corr = float(np.max(np.abs(corr_hw)))

    print(f"  frobenius_error={frob_error:.4f}, max_correlation={max_corr:.4f}")

    return {
        "experiment": "correlator_4q",
        "corr_hw": corr_hw.tolist(),
        "corr_sim": corr_sim.tolist(),
        "frobenius_error": frob_error,
        "max_correlation_hw": max_corr,
    }


def qkd_qber_4q_experiment(runner, shots: int = 10000, maxiter: int = 100) -> dict:
    """QBER measurement from hardware for BB84-family security validation.

    Measures in Z and X bases, extracts Alice (qubits 0,1) and Bob (qubits 2,3)
    raw keys, computes QBER. Secure if QBER < 0.11 (BB84 threshold).
    ~15s QPU budget (2 circuits).

    Returns:
        dict with keys:
            experiment (str): Experiment name identifier.
            qber_z_hw (float): Z-basis QBER from hardware.
            qber_x_hw (float): X-basis QBER from hardware.
            qber_sim (float): QBER from simulator.
            secure_hw (bool): Both QBERs < 0.11.
            secure_sim (bool): Simulator QBER < 0.11.
            key_rate_hw (float): Devetak-Winter secret key rate.
    """
    n = 4
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]

    print(f"\n=== QKD QBER 4q, maxiter={maxiter} ===")

    key_state = prepare_key_state(K, omega, ansatz_reps=2, maxiter=maxiter)
    base_circuit = key_state["circuit"]

    # Z-basis and X-basis measurement circuits
    qc_z = base_circuit.copy()
    qc_z.measure_all()

    qc_x = base_circuit.copy()
    for q in range(n):
        qc_x.h(q)
    qc_x.measure_all()

    hw_results = runner.run_sampler([qc_z, qc_x], shots=shots, name="qkd_qber_4q")

    alice_z = extract_raw_key(hw_results[0].counts, "Z", [0, 1])
    bob_z = extract_raw_key(hw_results[0].counts, "Z", [2, 3])
    alice_x = extract_raw_key(hw_results[1].counts, "X", [0, 1])
    bob_x = extract_raw_key(hw_results[1].counts, "X", [2, 3])

    qber_z_hw = estimate_qber(alice_z, bob_z)
    qber_x_hw = estimate_qber(alice_x, bob_x)

    # Simulator reference
    sim_result = scpn_qkd_protocol(K, omega, [0, 1], [2, 3], shots=shots)
    qber_sim = sim_result["qber"]

    rate_z = devetak_winter_rate(qber_z_hw)
    rate_x = devetak_winter_rate(qber_x_hw)

    print(f"  QBER_Z={qber_z_hw:.4f}, QBER_X={qber_x_hw:.4f}, QBER_sim={qber_sim:.4f}")
    print(f"  key_rate_z={rate_z:.4f}, key_rate_x={rate_x:.4f}")

    return {
        "experiment": "qkd_qber_4q",
        "qber_z_hw": qber_z_hw,
        "qber_x_hw": qber_x_hw,
        "qber_sim": qber_sim,
        "secure_hw": qber_z_hw < 0.11 and qber_x_hw < 0.11,
        "secure_sim": qber_sim < 0.11,
        "key_rate_hw": max(rate_z, rate_x),
    }
