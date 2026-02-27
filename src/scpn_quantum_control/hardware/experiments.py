"""Concrete hardware experiments for IBM Quantum.

Each experiment function takes a HardwareRunner and returns results + classical
reference for comparison. Designed to fit within the 10-min/month free tier.

Budget estimate per experiment:
  kuramoto_4osc:    ~30s QPU (8 circuits x 10k shots, 12 ECR each)
  kuramoto_8osc:    ~60s QPU (8 circuits x 10k shots, 56 ECR each)
  vqe_4q:           ~30s QPU (200 optimizer iterations x 10k shots)
  vqe_8q:           ~90s QPU (200 iterations, 56 ECR per eval)
  qaoa_mpc_4:       ~20s QPU (200 optimizer iterations, 12 ECR)
  upde_16_snapshot:  ~180s QPU (1 Trotter step, 240 ECR, needs mitigation)
  ─────────────────────────────────
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


def kuramoto_4osc_experiment(runner, shots: int = 10000, n_time_steps: int = 8, dt: float = 0.1) -> dict:
    """Experiment 1: 4-oscillator Kuramoto XY dynamics on hardware.

    Measures order parameter R(t) via shot-based sampling at each time step.
    Compares against exact matrix-exponential evolution.
    """
    n = 4
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    H = knm_to_hamiltonian(K, omega)

    print(f"\n=== Kuramoto 4-oscillator, {n_time_steps} steps, dt={dt} ===")

    circuits = []
    for step in range(1, n_time_steps + 1):
        t = step * dt
        qc = QuantumCircuit(n)
        for i in range(n):
            qc.ry(float(omega[i]) % (2 * np.pi), i)
        evo = PauliEvolutionGate(H, time=t, synthesis=LieTrotter(reps=step * 2))
        qc.append(evo, range(n))
        qc.measure_all()
        circuits.append(qc)

    hw_results = runner.run_sampler(circuits, shots=shots, name="kuramoto_4osc")

    hw_R = [_R_from_counts(r.counts, n) for r in hw_results]
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
        "hw_results": [r.to_dict() for r in hw_results],
    }
    runner.save_result(hw_results[0], "kuramoto_4osc.json")
    return result


def kuramoto_8osc_experiment(runner, shots: int = 10000, n_time_steps: int = 6, dt: float = 0.1) -> dict:
    """Experiment 2: 8-oscillator Kuramoto XY dynamics."""
    n = 8
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    H = knm_to_hamiltonian(K, omega)

    print(f"\n=== Kuramoto 8-oscillator, {n_time_steps} steps, dt={dt} ===")

    circuits = []
    for step in range(1, n_time_steps + 1):
        t = step * dt
        qc = QuantumCircuit(n)
        for i in range(n):
            qc.ry(float(omega[i]) % (2 * np.pi), i)
        evo = PauliEvolutionGate(H, time=t, synthesis=LieTrotter(reps=step))
        qc.append(evo, range(n))
        qc.measure_all()
        circuits.append(qc)

    hw_results = runner.run_sampler(circuits, shots=shots, name="kuramoto_8osc")
    hw_R = [_R_from_counts(r.counts, n) for r in hw_results]
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
    """Experiment 3: VQE ground state of 4-oscillator XY Hamiltonian.

    Compares Knm-informed ansatz energy against exact diagonalization.
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
        bound.measure_all()
        results = runner.run_sampler(bound, shots=shots, name="vqe_4q_eval")
        counts = results[0].counts
        energy = _energy_from_counts(counts, H, n)
        energy_history.append(energy)
        return energy

    x0 = np.random.default_rng(42).uniform(-np.pi, np.pi, n_params)
    opt_result = minimize(cost_fn, x0, method="COBYLA", options={"maxiter": maxiter})

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
    """Experiment 4: VQE ground state of 8-oscillator XY Hamiltonian."""
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
        bound.measure_all()
        results = runner.run_sampler(bound, shots=shots, name="vqe_8q_eval")
        counts = results[0].counts
        energy = _energy_from_counts(counts, H, n)
        energy_history.append(energy)
        return energy

    x0 = np.random.default_rng(42).uniform(-np.pi, np.pi, n_params)
    opt_result = minimize(cost_fn, x0, method="COBYLA", options={"maxiter": maxiter})

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
    """Experiment 5: QAOA-MPC binary control, horizon=4, p=1 and p=2.

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

        def cost_fn(params):
            gamma = params[:p]
            beta = params[p:]
            qc = mpc._build_qaoa_circuit(gamma, beta)
            qc.measure_all()
            hw = runner.run_sampler(qc, shots=shots, name=f"qaoa_p{p}")
            counts = hw[0].counts
            return _qaoa_cost_from_counts(counts, mpc._cost_ham, mpc.n_qubits)

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
    """Experiment 6: Full 16-layer UPDE single Trotter snapshot.

    Marginal circuit depth (~240 ECR). Uses higher shots for noise averaging.
    Compare per-qubit Z expectations against exact evolution.
    """
    n = 16
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16.copy()
    H = knm_to_hamiltonian(K, omega)
    dt = 0.05  # short time to limit Trotter error

    print(f"\n=== UPDE 16-layer snapshot, dt={dt}, {trotter_steps} Trotter steps ===")

    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(float(omega[i]) % (2 * np.pi), i)
    evo = PauliEvolutionGate(H, time=dt, synthesis=LieTrotter(reps=trotter_steps))
    qc.append(evo, range(n))
    qc.measure_all()

    hw_results = runner.run_sampler(qc, shots=shots, name="upde_16")
    counts = hw_results[0].counts

    classical = classical_exact_evolution(n, dt, dt, K, omega)

    result = {
        "experiment": "upde_16_snapshot",
        "n_layers": n,
        "dt": dt,
        "trotter_steps": trotter_steps,
        "hw_R": _R_from_counts(counts, n),
        "classical_R": float(classical["R"][-1]),
        "hw_result": hw_results[0].to_dict(),
    }
    runner.save_result(hw_results[0], "upde_16_snapshot.json")
    return result


# ── Helpers ──────────────────────────────────────────────────────────

def _R_from_counts(counts: dict, n_qubits: int) -> float:
    """Estimate order parameter R from measurement counts.

    Each bitstring encodes qubit states. Map |0> -> phase 0, |1> -> phase pi.
    R = |mean(exp(i*phase))|.
    """
    total = sum(counts.values())
    z_complex = 0.0 + 0.0j

    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        for q in range(min(n_qubits, len(bits))):
            bit = int(bits[-(q + 1)])  # Qiskit little-endian
            phase = np.pi * bit
            z_complex += (np.cos(phase) + 1j * np.sin(phase)) * count

    z_complex /= (total * n_qubits)
    return float(abs(z_complex))


def _energy_from_counts(counts: dict, H: SparsePauliOp, n_qubits: int) -> float:
    """Estimate <H> from shot counts by evaluating each Pauli term."""
    total = sum(counts.values())
    energy = 0.0

    for pauli, coeff in zip(H.paulis, H.coeffs):
        label = str(pauli)
        exp_val = 0.0
        for bitstring, count in counts.items():
            bits = bitstring.replace(" ", "")
            sign = 1
            for q in range(n_qubits):
                p = label[-(q + 1)]
                if p in ("Z",):
                    bit = int(bits[-(q + 1)])
                    sign *= (-1) ** bit
                elif p in ("X", "Y"):
                    # X,Y expectations average to 0 in Z-basis measurement
                    sign = 0
                    break
            exp_val += sign * count
        exp_val /= total
        energy += float(coeff.real) * exp_val

    return energy


def _qaoa_cost_from_counts(counts: dict, cost_ham: SparsePauliOp, n_qubits: int) -> float:
    """Evaluate QAOA cost Hamiltonian (diagonal in Z) from counts."""
    return _energy_from_counts(counts, cost_ham, n_qubits)


ALL_EXPERIMENTS = {
    "kuramoto_4osc": kuramoto_4osc_experiment,
    "kuramoto_8osc": kuramoto_8osc_experiment,
    "vqe_4q": vqe_4q_experiment,
    "vqe_8q": vqe_8q_experiment,
    "qaoa_mpc_4": qaoa_mpc_4_experiment,
    "upde_16_snapshot": upde_16_snapshot_experiment,
}
