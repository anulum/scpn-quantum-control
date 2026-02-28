"""Concrete hardware experiments for IBM Quantum.

Each experiment function takes a HardwareRunner and returns results + classical
reference for comparison. Designed to fit within the 10-min/month free tier.

17 experiments total. See docs/EXPERIMENT_ROADMAP.md for budget allocation.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from scipy.optimize import minimize

from ..bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    knm_to_ansatz,
    knm_to_hamiltonian,
)
from .classical import (
    classical_brute_mpc,
    classical_exact_diag,
    classical_exact_evolution,
)


def _build_evo_base(n, K, omega, t, trotter_reps, trotter_order=1):
    """Build evolution circuit without measurement gates."""
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(float(omega[i]) % (2 * np.pi), i)
    H = knm_to_hamiltonian(K, omega)
    if trotter_order >= 2:
        synthesis = SuzukiTrotter(order=trotter_order, reps=trotter_reps)
    else:
        synthesis = LieTrotter(reps=trotter_reps)
    evo = PauliEvolutionGate(H, time=t, synthesis=synthesis)
    qc.append(evo, range(n))
    return qc


def _build_xyz_circuits(base_circuit, n):
    """Build 3 copies of base_circuit measuring in Z, X, Y bases.

    X-basis: H before measurement.
    Y-basis: Sdg then H before measurement.
    Returns (z_circuit, x_circuit, y_circuit).
    """
    qc_z = base_circuit.copy()
    qc_z.measure_all()

    qc_x = base_circuit.copy()
    for q in range(n):
        qc_x.h(q)
    qc_x.measure_all()

    qc_y = base_circuit.copy()
    for q in range(n):
        qc_y.sdg(q)
        qc_y.h(q)
    qc_y.measure_all()

    return qc_z, qc_x, qc_y


def _expectation_per_qubit(counts, n_qubits):
    """Compute per-qubit <Z> (or <X>/<Y> if measured in rotated basis)."""
    total = sum(counts.values())
    exp_vals = np.zeros(n_qubits)
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        for q in range(min(n_qubits, len(bits))):
            bit = int(bits[-(q + 1)])
            exp_vals[q] += (1 - 2 * bit) * count
    exp_vals /= total
    return exp_vals


def _R_from_xyz(z_counts, x_counts, y_counts, n_qubits):
    """Compute Kuramoto order parameter R from X, Y, Z basis measurements.

    R = |1/N sum_q (exp_X_q + i*exp_Y_q)|
    The Z measurement is recorded but R uses XY-plane expectations.
    """
    exp_x = _expectation_per_qubit(x_counts, n_qubits)
    exp_y = _expectation_per_qubit(y_counts, n_qubits)
    exp_z = _expectation_per_qubit(z_counts, n_qubits)
    z_complex = np.mean(exp_x + 1j * exp_y)
    return float(abs(z_complex)), exp_x, exp_y, exp_z


def kuramoto_4osc_experiment(
    runner, shots: int = 10000, n_time_steps: int = 8, dt: float = 0.1
) -> dict:
    """4-oscillator Kuramoto XY dynamics on hardware.

    Measures order parameter R(t) via X, Y, Z basis shots at each time step.
    Compares against exact matrix-exponential evolution.
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
    hw_exp = []
    for idx in step_indices:
        R, ex, ey, ez = _R_from_xyz(
            hw_results[idx].counts,
            hw_results[idx + 1].counts,
            hw_results[idx + 2].counts,
            n,
        )
        hw_R.append(R)
        hw_exp.append({"exp_x": ex.tolist(), "exp_y": ey.tolist(), "exp_z": ez.tolist()})

    hw_times = [i * dt for i in range(1, n_time_steps + 1)]
    classical = classical_exact_evolution(n, n_time_steps * dt, dt, K, omega)

    result = {
        "experiment": "kuramoto_4osc",
        "n_oscillators": n,
        "dt": dt,
        "hw_times": hw_times,
        "hw_R": hw_R,
        "classical_times": classical["times"].tolist(),
        "classical_R": classical["R"].tolist(),
        "hw_expectations": hw_exp,
    }
    runner.save_result(hw_results[0], "kuramoto_4osc.json")
    return result


def kuramoto_8osc_experiment(
    runner, shots: int = 10000, n_time_steps: int = 6, dt: float = 0.1
) -> dict:
    """8-oscillator Kuramoto XY dynamics."""
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
    for idx in step_indices:
        R, _, _, _ = _R_from_xyz(
            hw_results[idx].counts,
            hw_results[idx + 1].counts,
            hw_results[idx + 2].counts,
            n,
        )
        hw_R.append(R)

    hw_times = [i * dt for i in range(1, n_time_steps + 1)]
    classical = classical_exact_evolution(n, n_time_steps * dt, dt, K, omega)

    result = {
        "experiment": "kuramoto_8osc",
        "n_oscillators": n,
        "dt": dt,
        "hw_times": hw_times,
        "hw_R": hw_R,
        "classical_times": classical["times"].tolist(),
        "classical_R": classical["R"].tolist(),
    }
    runner.save_result(hw_results[0], "kuramoto_8osc.json")
    return result


def _run_vqe(n: int, maxiter: int = 200) -> dict:
    """Statevector VQE on the n-oscillator XY Hamiltonian."""
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    H = knm_to_hamiltonian(K, omega)
    ansatz = knm_to_ansatz(K, reps=2)
    n_params = ansatz.num_parameters

    print(f"\n=== VQE {n}-qubit, {n_params} params, maxiter={maxiter} ===")

    exact = classical_exact_diag(n, K, omega)
    print(f"  Exact ground energy: {exact['ground_energy']:.6f}")

    energy_history: list[float] = []

    def cost_fn(params):
        bound = ansatz.assign_parameters(params)
        sv = Statevector.from_instruction(bound)
        energy = float(sv.expectation_value(H).real)
        energy_history.append(energy)
        return energy

    x0 = np.random.default_rng(42).uniform(-np.pi, np.pi, n_params)
    opt_result = minimize(cost_fn, x0, method="COBYLA", options={"maxiter": maxiter})

    print(f"  VQE converged: {opt_result.success}, iterations: {opt_result.nfev}")
    print(f"  VQE energy: {opt_result.fun:.6f}")

    return {
        "experiment": f"vqe_{n}q",
        "n_qubits": n,
        "vqe_energy": float(opt_result.fun),
        "exact_ground_energy": exact["ground_energy"],
        "energy_gap": float(opt_result.fun - exact["ground_energy"]),
        "n_iterations": opt_result.nfev,
        "converged": bool(opt_result.success),
        "energy_history": energy_history,
    }


def vqe_4q_experiment(runner, shots: int = 10000, maxiter: int = 200) -> dict:
    """VQE ground state of 4-oscillator XY Hamiltonian."""
    return _run_vqe(4, maxiter)


def vqe_8q_experiment(runner, shots: int = 10000, maxiter: int = 150) -> dict:
    """VQE ground state of 8-oscillator XY Hamiltonian."""
    return _run_vqe(8, maxiter)


def qaoa_mpc_4_experiment(runner, shots: int = 10000) -> dict:
    """QAOA-MPC binary control, horizon=4, p=1 and p=2.

    Cost Hamiltonian is diagonal in Z, so Z-basis measurement is exact.
    Compares QAOA solution quality vs brute-force optimal.
    """
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
    """
    n = 16
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16.copy()
    dt = 0.05

    print(f"\n=== UPDE 16-layer snapshot, dt={dt}, {trotter_steps} Trotter steps ===")

    base = _build_evo_base(n, K, omega, dt, trotter_reps=trotter_steps)
    qc_z, qc_x, qc_y = _build_xyz_circuits(base, n)

    hw_results = runner.run_sampler([qc_z, qc_x, qc_y], shots=shots, name="upde_16")

    R, exp_x, exp_y, exp_z = _R_from_xyz(
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
        "classical_R": float(classical["R"][-1]),
        "hw_exp_x": exp_x.tolist(),
        "hw_exp_y": exp_y.tolist(),
        "hw_exp_z": exp_z.tolist(),
    }
    runner.save_result(hw_results[0], "upde_16_snapshot.json")
    return result


def _qaoa_cost_from_counts(counts: dict, cost_ham: SparsePauliOp, n_qubits: int) -> float:
    """Evaluate QAOA cost Hamiltonian (diagonal in Z) from counts."""
    total = sum(counts.values())
    energy = 0.0
    for pauli, coeff in zip(cost_ham.paulis, cost_ham.coeffs):
        label = str(pauli)
        exp_val = 0.0
        for bitstring, count in counts.items():
            bits = bitstring.replace(" ", "")
            sign = 1
            for q in range(n_qubits):
                p = label[-(q + 1)]
                if p == "Z":
                    bit = int(bits[-(q + 1)])
                    sign *= (-1) ** bit
                elif p in ("X", "Y"):
                    sign = 0
                    break
            exp_val += sign * count
        exp_val /= total
        energy += float(coeff.real) * exp_val
    return energy


def kuramoto_4osc_zne_experiment(
    runner, shots: int = 10000, dt: float = 0.1, scales: list[int] | None = None
) -> dict:
    """4-oscillator Kuramoto with ZNE error mitigation.

    Runs the evolution at multiple noise scales via unitary folding,
    then extrapolates to zero noise.
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
    for s in scales:
        folded = gate_fold_circuit(base, s)
        qc_z, qc_x, qc_y = _build_xyz_circuits(folded, n)
        hw = runner.run_sampler([qc_z, qc_x, qc_y], shots=shots, name=f"zne_s{s}")
        R, _, _, _ = _R_from_xyz(hw[0].counts, hw[1].counts, hw[2].counts, n)
        R_per_scale.append(R)
        print(f"  scale={s}: R={R:.4f}")

    zne = zne_extrapolate(scales, R_per_scale, order=1)
    classical = classical_exact_evolution(n, dt, dt, K, omega)

    print(f"  ZNE R(0) = {zne.zero_noise_estimate:.4f}")
    print(f"  Exact R  = {classical['R'][-1]:.4f}")

    return {
        "experiment": "kuramoto_4osc_zne",
        "scales": scales,
        "R_per_scale": R_per_scale,
        "zne_R": zne.zero_noise_estimate,
        "classical_R": float(classical["R"][-1]),
        "fit_residual": zne.fit_residual,
    }


def noise_baseline_experiment(runner, shots: int = 10000) -> dict:
    """4-qubit near-identity circuit for calibration drift detection.

    Single Trotter step at dt=0.01 (near-identity). Measures R + per-qubit
    expectations. Compare Feb→Mar to detect backend drift.
    """
    n = 4
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    dt = 0.01

    print(f"\n=== Noise baseline, {n}q, dt={dt} ===")

    base = _build_evo_base(n, K, omega, dt, trotter_reps=1)
    qc_z, qc_x, qc_y = _build_xyz_circuits(base, n)

    hw_results = runner.run_sampler([qc_z, qc_x, qc_y], shots=shots, name="noise_baseline")
    R, exp_x, exp_y, exp_z = _R_from_xyz(
        hw_results[0].counts, hw_results[1].counts, hw_results[2].counts, n
    )

    classical = classical_exact_evolution(n, dt, dt, K, omega)

    result = {
        "experiment": "noise_baseline",
        "n_qubits": n,
        "dt": dt,
        "hw_R": R,
        "classical_R": float(classical["R"][-1]),
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
    for s in scales:
        folded = gate_fold_circuit(base, s)
        qc_z, qc_x, qc_y = _build_xyz_circuits(folded, n)
        hw = runner.run_sampler([qc_z, qc_x, qc_y], shots=shots, name=f"zne8_s{s}")
        R, _, _, _ = _R_from_xyz(hw[0].counts, hw[1].counts, hw[2].counts, n)
        R_per_scale.append(R)
        print(f"  scale={s}: R={R:.4f}")

    zne = zne_extrapolate(scales, R_per_scale, order=1)
    classical = classical_exact_evolution(n, dt, dt, K, omega)

    print(f"  ZNE R(0) = {zne.zero_noise_estimate:.4f}")
    print(f"  Exact R  = {classical['R'][-1]:.4f}")

    return {
        "experiment": "kuramoto_8osc_zne",
        "n_oscillators": n,
        "scales": scales,
        "R_per_scale": R_per_scale,
        "zne_R": zne.zero_noise_estimate,
        "classical_R": float(classical["R"][-1]),
        "fit_residual": zne.fit_residual,
    }


def vqe_8q_hardware_experiment(runner, shots: int = 10000, maxiter: int = 150) -> dict:
    """VQE 8-qubit: Statevector optimization → hardware energy evaluation.

    1. COBYLA on Statevector → optimal params
    2. Bind optimal params into Knm-informed ansatz
    3. Send bound circuit to hardware via run_estimator with H observable
    4. Return sim energy, hw energy, exact ground energy
    """
    n = 8
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    H = knm_to_hamiltonian(K, omega)
    ansatz = knm_to_ansatz(K, reps=2)

    print(f"\n=== VQE 8q hardware, {ansatz.num_parameters} params ===")

    exact = classical_exact_diag(n, K, omega)
    print(f"  Exact ground energy: {exact['ground_energy']:.6f}")

    # Step 1: classical VQE on Statevector
    def cost_fn(params):
        bound = ansatz.assign_parameters(params)
        sv = Statevector.from_instruction(bound)
        return float(sv.expectation_value(H).real)

    x0 = np.random.default_rng(42).uniform(-np.pi, np.pi, ansatz.num_parameters)
    opt = minimize(cost_fn, x0, method="COBYLA", options={"maxiter": maxiter})
    sim_energy = float(opt.fun)
    print(f"  Sim VQE energy: {sim_energy:.6f}")

    # Step 2: bind optimal params and evaluate on hardware
    bound_circuit = ansatz.assign_parameters(opt.x)
    hw_result = runner.run_estimator(bound_circuit, [H], name="vqe_8q_hw")
    hw_energy = float(hw_result.expectation_values[0])
    print(f"  HW energy: {hw_energy:.6f}")

    return {
        "experiment": "vqe_8q_hardware",
        "n_qubits": n,
        "sim_energy": sim_energy,
        "hw_energy": hw_energy,
        "exact_energy": exact["ground_energy"],
        "sim_gap": sim_energy - exact["ground_energy"],
        "hw_gap": hw_energy - exact["ground_energy"],
        "n_iterations": opt.nfev,
    }


def upde_16_dd_experiment(runner, shots: int = 20000, trotter_steps: int = 1) -> dict:
    """16-layer UPDE with dynamical decoupling.

    Same structure as upde_16_snapshot but applies DD (XY4) to each
    basis circuit before submission. Compares R(DD) vs R(no-DD) vs classical.
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
    R_raw, exp_x_raw, exp_y_raw, exp_z_raw = _R_from_xyz(
        hw_raw[0].counts, hw_raw[1].counts, hw_raw[2].counts, n
    )

    # DD run
    dd_z = runner.transpile_with_dd(qc_z)
    dd_x = runner.transpile_with_dd(qc_x)
    dd_y = runner.transpile_with_dd(qc_y)
    hw_dd = runner.run_sampler([dd_z, dd_x, dd_y], shots=shots, name="upde16_dd")
    R_dd, exp_x_dd, exp_y_dd, exp_z_dd = _R_from_xyz(
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


def kuramoto_4osc_trotter2_experiment(
    runner, shots: int = 10000, n_time_steps: int = 8, dt: float = 0.1
) -> dict:
    """4-oscillator Kuramoto with second-order Suzuki-Trotter.

    Same structure as kuramoto_4osc_experiment but uses SuzukiTrotter(order=2).
    Produces order-1 vs order-2 comparison data.
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
    hw_exp = []
    for idx in step_indices:
        R, ex, ey, ez = _R_from_xyz(
            hw_results[idx].counts,
            hw_results[idx + 1].counts,
            hw_results[idx + 2].counts,
            n,
        )
        hw_R.append(R)
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
        "classical_times": classical["times"].tolist(),
        "classical_R": classical["R"].tolist(),
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
    R grows (synchronized). K_c depends on frequency spread.

    Science: first measurement of Kuramoto phase transition on
    superconducting qubits. Validates quantum XY ↔ classical Kuramoto
    correspondence at the critical point.
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
        R, ex, ey, ez = _R_from_xyz(hw[0].counts, hw[1].counts, hw[2].counts, n)

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


def ansatz_comparison_hw_experiment(runner, shots: int = 10000, maxiter: int = 100) -> dict:
    """VQE ansatz comparison on hardware: Knm-informed vs TwoLocal vs EfficientSU2.

    For each ansatz:
    1. COBYLA on Statevector → optimal params
    2. Bind optimal params, evaluate energy on hardware via Estimator
    3. Compare: sim energy, hw energy, exact ground energy, param count

    Science: proves physics-informed (Knm) ansatz advantage is real
    on noisy hardware, not just an artifact of noiseless simulation.
    """
    from qiskit.circuit.library import efficient_su2, n_local

    n = 4
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    H = knm_to_hamiltonian(K, omega)
    exact = classical_exact_diag(n, K, omega)

    print(f"\n=== Ansatz comparison (hw), exact E={exact['ground_energy']:.6f} ===")

    ansatz_builders = {
        "knm_informed": lambda: knm_to_ansatz(K, reps=2),
        "two_local": lambda: n_local(
            n, rotation_blocks=["ry", "rz"], entanglement_blocks="cz", reps=2
        ),
        "efficient_su2": lambda: efficient_su2(n, reps=2),
    }

    comparison = []
    for name, build_fn in ansatz_builders.items():
        ansatz = build_fn()

        def cost_fn(params, _a=ansatz, _H=H):
            bound = _a.assign_parameters(params)
            sv = Statevector.from_instruction(bound)
            return float(sv.expectation_value(_H).real)

        x0 = np.random.default_rng(42).uniform(-np.pi, np.pi, ansatz.num_parameters)
        opt = minimize(cost_fn, x0, method="COBYLA", options={"maxiter": maxiter})
        sim_energy = float(opt.fun)

        bound = ansatz.assign_parameters(opt.x)
        hw_result = runner.run_estimator(bound, [H], name=f"ansatz_{name}")
        hw_energy = float(hw_result.expectation_values[0])

        entry = {
            "ansatz": name,
            "n_params": ansatz.num_parameters,
            "sim_energy": sim_energy,
            "hw_energy": hw_energy,
            "exact_energy": exact["ground_energy"],
            "sim_gap": sim_energy - exact["ground_energy"],
            "hw_gap": hw_energy - exact["ground_energy"],
            "n_iterations": opt.nfev,
        }
        comparison.append(entry)
        print(f"  {name}: sim={sim_energy:.4f}, hw={hw_energy:.4f}, gap={entry['hw_gap']:.4f}")

    return {
        "experiment": "ansatz_comparison_hw",
        "n_qubits": n,
        "exact_energy": exact["ground_energy"],
        "comparison": comparison,
    }


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

    Science: systematic ZNE study — linear vs quadratic vs cubic on the
    same data. Determines optimal extrapolation order for XY evolution on
    Heron r2.
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
    for s in scales:
        folded = gate_fold_circuit(base, s)
        qc_z, qc_x, qc_y = _build_xyz_circuits(folded, n)
        hw = runner.run_sampler([qc_z, qc_x, qc_y], shots=shots, name=f"zne_ho_s{s}")
        R, _, _, _ = _R_from_xyz(hw[0].counts, hw[1].counts, hw[2].counts, n)
        R_per_scale.append(R)
        print(f"  scale={s}: R={R:.4f}")

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
        R, _, _, _ = _R_from_xyz(hw[0].counts, hw[1].counts, hw[2].counts, n)

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
            np.var(log_ratios), 1e-10
        )
    else:
        gamma = float("nan")
        r_squared = float("nan")

    print(f"  Fit: gamma={gamma:.6f} per gate, R²={r_squared:.4f}")

    return {
        "experiment": "decoherence_scaling",
        "dt": dt,
        "data_points": data_points,
        "fit_gamma": gamma,
        "fit_r_squared": r_squared,
    }


def vqe_landscape_experiment(runner, shots: int = 10000, n_samples: int = 50) -> dict:
    """Sample VQE cost landscape to detect barren plateaus.

    Evaluates the 4-qubit Hamiltonian at random parameter vectors and
    computes the variance of the energy. Low variance = barren plateau
    (exponentially flat landscape). Compares Knm-informed vs TwoLocal.

    Science: barren plateaus are the #1 obstacle to scaling VQE.
    Showing Knm-informed ansatz avoids them (higher variance) is a
    publishable result on its own.

    Reference: McClean et al., "Barren plateaus in quantum neural network
    training landscapes", Nature Comm. 9, 4812 (2018).
    """
    from qiskit.circuit.library import n_local

    n = 4
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    H = knm_to_hamiltonian(K, omega)
    rng = np.random.default_rng(42)

    print(f"\n=== VQE landscape, {n_samples} samples ===")

    results = {}
    for name, ansatz in [
        ("knm_informed", knm_to_ansatz(K, reps=2)),
        ("two_local", n_local(n, rotation_blocks=["ry", "rz"], entanglement_blocks="cz", reps=2)),
    ]:
        energies = []
        for _ in range(n_samples):
            params = rng.uniform(-np.pi, np.pi, ansatz.num_parameters)
            bound = ansatz.assign_parameters(params)
            sv = Statevector.from_instruction(bound)
            e = float(sv.expectation_value(H).real)
            energies.append(e)

        energies_arr = np.array(energies)
        results[name] = {
            "n_params": ansatz.num_parameters,
            "mean_energy": float(energies_arr.mean()),
            "std_energy": float(energies_arr.std()),
            "min_energy": float(energies_arr.min()),
            "max_energy": float(energies_arr.max()),
        }
        print(f"  {name}: mean={energies_arr.mean():.4f}, std={energies_arr.std():.4f}")

    exact = classical_exact_diag(n, K, omega)

    return {
        "experiment": "vqe_landscape",
        "n_qubits": n,
        "n_samples": n_samples,
        "exact_ground_energy": exact["ground_energy"],
        "landscapes": results,
    }


ALL_EXPERIMENTS = {
    "kuramoto_4osc": kuramoto_4osc_experiment,
    "kuramoto_8osc": kuramoto_8osc_experiment,
    "vqe_4q": vqe_4q_experiment,
    "vqe_8q": vqe_8q_experiment,
    "qaoa_mpc_4": qaoa_mpc_4_experiment,
    "upde_16_snapshot": upde_16_snapshot_experiment,
    "kuramoto_4osc_zne": kuramoto_4osc_zne_experiment,
    "noise_baseline": noise_baseline_experiment,
    "kuramoto_8osc_zne": kuramoto_8osc_zne_experiment,
    "vqe_8q_hardware": vqe_8q_hardware_experiment,
    "upde_16_dd": upde_16_dd_experiment,
    "kuramoto_4osc_trotter2": kuramoto_4osc_trotter2_experiment,
    "sync_threshold": sync_threshold_experiment,
    "ansatz_comparison_hw": ansatz_comparison_hw_experiment,
    "zne_higher_order": zne_higher_order_experiment,
    "decoherence_scaling": decoherence_scaling_experiment,
    "vqe_landscape": vqe_landscape_experiment,
}
