# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Experiment VQE
"""VQE and ansatz experiments on quantum hardware."""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from ..bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    knm_to_ansatz,
    knm_to_hamiltonian,
)
from .classical import classical_exact_diag


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

    tail = energy_history[-10:] if len(energy_history) >= 10 else energy_history
    energy_std = float(np.std(tail))

    return {
        "experiment": f"vqe_{n}q",
        "n_qubits": n,
        "vqe_energy": float(opt_result.fun),
        "exact_ground_energy": exact["ground_energy"],
        "energy_gap": float(opt_result.fun - exact["ground_energy"]),
        "energy_std": energy_std,
        "n_iterations": opt_result.nfev,
        "converged": bool(opt_result.success),
        "energy_history": energy_history,
    }


def vqe_4q_experiment(runner, shots: int = 10000, maxiter: int = 200) -> dict:
    """VQE ground state of 4-oscillator XY Hamiltonian.

    Returns:
        dict with keys:
            experiment (str): Experiment name identifier.
            n_qubits (int): Number of qubits.
            vqe_energy (float): Optimized VQE energy.
            exact_ground_energy (float): Exact diagonalization ground energy.
            energy_gap (float): vqe_energy - exact_ground_energy.
            n_iterations (int): COBYLA function evaluations.
            converged (bool): Whether optimizer converged.
            energy_history (list[float]): Energy at each iteration.
    """
    return _run_vqe(4, maxiter)


def vqe_8q_experiment(runner, shots: int = 10000, maxiter: int = 150) -> dict:
    """VQE ground state of 8-oscillator XY Hamiltonian.

    Returns:
        dict with keys:
            experiment (str): Experiment name identifier.
            n_qubits (int): Number of qubits.
            vqe_energy (float): Optimized VQE energy.
            exact_ground_energy (float): Exact diagonalization ground energy.
            energy_gap (float): vqe_energy - exact_ground_energy.
            n_iterations (int): COBYLA function evaluations.
            converged (bool): Whether optimizer converged.
            energy_history (list[float]): Energy at each iteration.
    """
    return _run_vqe(8, maxiter)


def vqe_8q_hardware_experiment(runner, shots: int = 10000, maxiter: int = 150) -> dict:
    """VQE 8-qubit: Statevector optimization -> hardware energy evaluation.

    1. COBYLA on Statevector -> optimal params
    2. Bind optimal params into Knm-informed ansatz
    3. Send bound circuit to hardware via run_estimator with H observable
    4. Return sim energy, hw energy, exact ground energy

    Returns:
        dict with keys:
            experiment (str): Experiment name identifier.
            n_qubits (int): Number of qubits.
            sim_energy (float): Statevector-optimized VQE energy.
            hw_energy (float): Hardware-evaluated energy.
            exact_energy (float): Exact diagonalization ground energy.
            sim_gap (float): sim_energy - exact_energy.
            hw_gap (float): hw_energy - exact_energy.
            n_iterations (int): COBYLA function evaluations.
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


def ansatz_comparison_hw_experiment(runner, shots: int = 10000, maxiter: int = 100) -> dict:
    """VQE ansatz comparison on hardware: Knm-informed vs TwoLocal vs EfficientSU2.

    For each ansatz:
    1. COBYLA on Statevector -> optimal params
    2. Bind optimal params, evaluate energy on hardware via Estimator
    3. Compare: sim energy, hw energy, exact ground energy, param count

    Science: proves physics-informed (Knm) ansatz advantage is real
    on noisy hardware, not just an artifact of noiseless simulation.

    Returns:
        dict with keys:
            experiment (str): Experiment name identifier.
            n_qubits (int): Number of qubits.
            exact_energy (float): Exact diagonalization ground energy.
            comparison (list[dict]): Per-ansatz dicts with ansatz, n_params,
                sim_energy, hw_energy, exact_energy, sim_gap, hw_gap,
                n_iterations.
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

    Returns:
        dict with keys:
            experiment (str): Experiment name identifier.
            n_qubits (int): Number of qubits.
            n_samples (int): Random parameter vectors sampled.
            exact_ground_energy (float): Exact diagonalization ground energy.
            landscapes (dict): Per-ansatz dicts with n_params, mean_energy,
                std_energy, min_energy, max_energy.
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
