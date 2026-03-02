"""End-to-end integration tests exercising the full analysis pipeline."""

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.hardware.classical import classical_exact_evolution
from scpn_quantum_control.mitigation.zne import gate_fold_circuit, zne_extrapolate
from scpn_quantum_control.phase.xy_kuramoto import QuantumKuramotoSolver


def test_quantum_vs_classical_kuramoto_4osc():
    """Quantum Trotter evolution should approximate classical exact evolution.

    At high Trotter reps on 4 oscillators, the quantum R(t) should track the
    classical R(t) within reasonable tolerance (Trotter error + finite dt).
    """
    n = 4
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    dt = 0.1
    n_steps = 5

    quantum = QuantumKuramotoSolver(n, K, omega)
    q_result = quantum.run(t_max=n_steps * dt, dt=dt, trotter_per_step=10)

    classical = classical_exact_evolution(n, n_steps * dt, dt, K, omega)

    # R values should be within 0.15 of each other (Trotter error at reps=10)
    for q_r, c_r in zip(q_result["R"], classical["R"]):
        assert abs(q_r - c_r) < 0.15, f"quantum R={q_r:.4f} vs classical R={c_r:.4f}"


def test_zne_on_kuramoto_noiseless():
    """ZNE on a noiseless simulator: extrapolated R should match scale=1 R.

    On a noiseless backend, folding should not change the result, so all
    noise-scale R values should be identical and extrapolation trivial.
    """
    n = 4
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]

    solver = QuantumKuramotoSolver(n, K, omega)
    solver.build_hamiltonian()
    qc = solver.evolve(0.1, trotter_steps=2)

    from qiskit.quantum_info import Statevector

    def measure_R(circuit):
        sv = Statevector.from_instruction(circuit)
        R, _ = solver.measure_order_parameter(sv)
        return R

    R_values = []
    for scale in [1, 3, 5]:
        folded = gate_fold_circuit(qc, scale)
        R_values.append(measure_R(folded))

    # Noiseless: all scales should give the same R (unitary folding is identity)
    np.testing.assert_allclose(R_values, R_values[0], atol=1e-10)

    zne = zne_extrapolate([1, 3, 5], R_values, order=1)
    np.testing.assert_allclose(zne.zero_noise_estimate, R_values[0], atol=1e-10)


def test_energy_conservation_trotter():
    """Energy should be approximately conserved under Trotter evolution.

    For a time-independent Hamiltonian, <H(t)> = <H(0)> exactly.
    With Trotter decomposition, small drift is expected but should be bounded.
    """
    n = 3
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]

    solver = QuantumKuramotoSolver(n, K, omega)
    solver.build_hamiltonian()

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    init = QuantumCircuit(n)
    for i in range(n):
        init.ry(float(omega[i]) % (2 * np.pi), i)
    sv = Statevector.from_instruction(init)

    E0 = solver.energy_expectation(sv)
    energies = [E0]

    for _ in range(5):
        evo = solver.evolve(0.1, trotter_steps=5)
        sv = sv.evolve(evo)
        energies.append(solver.energy_expectation(sv))

    # Energy drift should be small over 5 steps
    drift = max(abs(e - E0) for e in energies)
    assert drift < 0.25, f"energy drift {drift:.4f} exceeds tolerance"


@pytest.mark.parametrize("n_osc", [2, 3, 4, 6])
def test_quantum_vs_classical_parametrized(n_osc):
    """Quantum R(t) tracks classical R(t) at multiple system sizes."""
    K = build_knm_paper27(L=n_osc)
    omega = OMEGA_N_16[:n_osc]
    dt = 0.1
    n_steps = 3

    quantum = QuantumKuramotoSolver(n_osc, K, omega)
    q_result = quantum.run(t_max=n_steps * dt, dt=dt, trotter_per_step=10)
    classical = classical_exact_evolution(n_osc, n_steps * dt, dt, K, omega)

    for q_r, c_r in zip(q_result["R"], classical["R"]):
        assert abs(q_r - c_r) < 0.2, f"n={n_osc}: quantum R={q_r:.4f} vs classical R={c_r:.4f}"


def test_exact_diag_ground_energy_matches_hamiltonian():
    """Ground energy from exact diag should match min eigenvalue of H matrix."""
    from scpn_quantum_control.hardware.classical import classical_exact_diag

    n = 3
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]

    result = classical_exact_diag(n, K=K, omega=omega)

    from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_hamiltonian

    H = knm_to_hamiltonian(K, omega)
    H_mat = np.array(H.to_matrix())
    evals_direct = np.linalg.eigvalsh(H_mat)

    np.testing.assert_allclose(result["ground_energy"], evals_direct[0], atol=1e-10)
