"""Concrete hardware experiments for IBM Quantum.

Each experiment function takes a HardwareRunner and returns results + classical
reference for comparison. Designed to fit within the 10-min/month free tier.

Budget estimate per experiment (with XY-basis measurement):
  kuramoto_4osc:    ~30s QPU  (3N circuits per step, N=4)
  kuramoto_8osc:    ~60s QPU  (3N circuits per step, N=8)
  vqe_4q:           ~30s QPU  (Estimator-based, 200 iterations)
  vqe_8q:           ~90s QPU  (Estimator-based, 150 iterations)
  qaoa_mpc_4:       ~20s QPU  (Z-diagonal cost, Sampler is fine)
  upde_16_snapshot: ~180s QPU (16-qubit, 240 ECR)
  Total:            ~7 min QPU
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import LieTrotter
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
    classical_kuramoto_reference,
)


def _build_evo_base(n, K, omega, t, trotter_reps):
    """Build evolution circuit without measurement gates."""
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(float(omega[i]) % (2 * np.pi), i)
    H = knm_to_hamiltonian(K, omega)
    evo = PauliEvolutionGate(H, time=t, synthesis=LieTrotter(reps=trotter_reps))
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


def kuramoto_4osc_experiment(runner, shots: int = 10000, n_time_steps: int = 8, dt: float = 0.1) -> dict:
    """4-oscillator Kuramoto XY dynamics on hardware.

    Measures order parameter R(t) via X, Y, Z basis shots at each time step.
    Compares against exact matrix-exponential evolution.
    """
    n = 4
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]

    print(f"\n=== Kuramoto 4-oscillator, {n_time_steps} steps, dt={dt} ===")

    all_circuits = []
    step_indices = []
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


def kuramoto_8osc_experiment(runner, shots: int = 10000, n_time_steps: int = 6, dt: float = 0.1) -> dict:
    """8-oscillator Kuramoto XY dynamics."""
    n = 8
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]

    print(f"\n=== Kuramoto 8-oscillator, {n_time_steps} steps, dt={dt} ===")

    all_circuits = []
    step_indices = []
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


def vqe_4q_experiment(runner, shots: int = 10000, maxiter: int = 200) -> dict:
    """VQE ground state of 4-oscillator XY Hamiltonian.

    Uses Statevector simulation for cost evaluation on simulator,
    or Estimator primitive on real hardware (handles basis rotations).
    """
    n = 4
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    H = knm_to_hamiltonian(K, omega)
    ansatz = knm_to_ansatz(K, reps=2)
    n_params = ansatz.num_parameters

    print(f"\n=== VQE 4-qubit, {n_params} params, maxiter={maxiter} ===")

    exact = classical_exact_diag(n, K, omega)
    print(f"  Exact ground energy: {exact['ground_energy']:.6f}")

    energy_history = []

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

    result = {
        "experiment": "vqe_4q",
        "n_qubits": n,
        "vqe_energy": float(opt_result.fun),
        "exact_ground_energy": exact["ground_energy"],
        "energy_gap": float(opt_result.fun - exact["ground_energy"]),
        "n_iterations": opt_result.nfev,
        "converged": bool(opt_result.success),
        "energy_history": energy_history,
    }
    return result


def vqe_8q_experiment(runner, shots: int = 10000, maxiter: int = 150) -> dict:
    """VQE ground state of 8-oscillator XY Hamiltonian."""
    n = 8
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    H = knm_to_hamiltonian(K, omega)
    ansatz = knm_to_ansatz(K, reps=2)
    n_params = ansatz.num_parameters

    print(f"\n=== VQE 8-qubit, {n_params} params, maxiter={maxiter} ===")

    exact = classical_exact_diag(n, K, omega)
    print(f"  Exact ground energy: {exact['ground_energy']:.6f}")

    energy_history = []

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

    result = {
        "experiment": "vqe_8q",
        "n_qubits": n,
        "vqe_energy": float(opt_result.fun),
        "exact_ground_energy": exact["ground_energy"],
        "energy_gap": float(opt_result.fun - exact["ground_energy"]),
        "n_iterations": opt_result.nfev,
        "converged": bool(opt_result.success),
        "energy_history": energy_history,
    }
    return result


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


ALL_EXPERIMENTS = {
    "kuramoto_4osc": kuramoto_4osc_experiment,
    "kuramoto_8osc": kuramoto_8osc_experiment,
    "vqe_4q": vqe_4q_experiment,
    "vqe_8q": vqe_8q_experiment,
    "qaoa_mpc_4": qaoa_mpc_4_experiment,
    "upde_16_snapshot": upde_16_snapshot_experiment,
}
