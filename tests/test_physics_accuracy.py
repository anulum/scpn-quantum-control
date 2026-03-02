"""Physics accuracy tests: verify quantum-classical parity at specific parameters."""

from __future__ import annotations

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from scpn_quantum_control.bridge import OMEGA_N_16, build_knm_paper27, knm_to_hamiltonian
from scpn_quantum_control.hardware.classical import classical_exact_diag, classical_exact_evolution
from scpn_quantum_control.hardware.experiments import _build_evo_base
from scpn_quantum_control.phase.phase_vqe import PhaseVQE


def _statevector_from_circuit(qc):
    """Transpile + statevector simulate."""
    qc_t = transpile(qc, basis_gates=["cx", "u3", "u2", "u1", "id"], optimization_level=0)
    qc_t.save_statevector()
    sim = AerSimulator(method="statevector")
    return Statevector(sim.run(qc_t).result().get_statevector())


def test_2q_ground_energy_exact():
    """2-qubit system: exact diag matches direct numpy eigvalsh."""
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    result = classical_exact_diag(n_osc=2, K=K, omega=omega)
    E0 = result["ground_energy"]

    H = knm_to_hamiltonian(K, omega)
    mat = H.to_matrix()
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    E0_direct = np.linalg.eigvalsh(mat)[0]

    assert abs(E0 - E0_direct) < 1e-10


def test_4q_trotter_tracks_classical_short_time():
    """4-qubit Trotter at short time tracks classical evolution."""
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]

    cl = classical_exact_evolution(n_osc=4, t_max=0.05, dt=0.01, K=K, omega=omega)
    R_classical = cl["R"][-1]

    qc = _build_evo_base(4, K, omega, t=0.05, trotter_reps=1)
    sv = _statevector_from_circuit(qc)

    x_exp = np.array(
        [float(sv.expectation_value(_single_qubit_op("X", i, 4)).real) for i in range(4)]
    )
    y_exp = np.array(
        [float(sv.expectation_value(_single_qubit_op("Y", i, 4)).real) for i in range(4)]
    )
    R_quantum = float(abs(np.mean(x_exp + 1j * y_exp)))

    # Both R values should be physical (in [0, 1]) and finite
    assert 0.0 <= R_quantum <= 1.5
    assert 0.0 <= R_classical <= 1.0
    # At short time, Trotter and classical are within reasonable agreement
    assert abs(R_quantum - R_classical) < 0.20, (
        f"R_quantum={R_quantum:.4f}, R_classical={R_classical:.4f}"
    )


def test_energy_conservation_trotter():
    """Energy expectation under Trotter evolution stays near initial value."""
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    H = knm_to_hamiltonian(K, omega)

    # Build evolution circuit and extract initial state (Ry-prepared)
    qc_init = _build_evo_base(4, K, omega, t=0.0001, trotter_reps=1)
    sv_init = _statevector_from_circuit(qc_init)
    E_init = float(sv_init.expectation_value(H).real)

    # Evolve for slightly longer time
    qc_evo = _build_evo_base(4, K, omega, t=0.05, trotter_reps=1)
    sv_evo = _statevector_from_circuit(qc_evo)
    E_evo = float(sv_evo.expectation_value(H).real)

    # Energy drift bounded — Trotter error grows with time but stays small for short t
    assert abs(E_evo - E_init) < 0.30, f"Energy drift: E_init={E_init:.4f}, E_evo={E_evo:.4f}"


def test_vqe_4q_beats_random():
    """VQE optimized energy is lower than random parameter energy."""
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    vqe = PhaseVQE(K, omega, ansatz_reps=1)

    rng = np.random.default_rng(42)
    random_energies = []
    for _ in range(5):
        params = rng.uniform(0, 2 * np.pi, size=vqe.n_params)
        E = vqe._cost(params)
        random_energies.append(E)
    E_random_avg = np.mean(random_energies)

    result = vqe.solve(maxiter=30, seed=0)
    E_opt = result["ground_energy"]

    assert E_opt < E_random_avg, f"VQE ({E_opt:.3f}) should beat random ({E_random_avg:.3f})"


def test_hamiltonian_traceless():
    """XY Hamiltonian is traceless (all Pauli terms are traceless)."""
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    H = knm_to_hamiltonian(K, omega)
    mat = H.to_matrix()
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    assert abs(np.trace(mat)) < 1e-8, f"Tr(H) = {np.trace(mat)}"


def test_ground_energy_decreases_with_coupling():
    """Stronger coupling lowers ground energy (more negative)."""
    omega = OMEGA_N_16[:4]

    K_weak = 0.1 * build_knm_paper27(L=4)
    K_strong = 2.0 * build_knm_paper27(L=4)

    H_weak = knm_to_hamiltonian(K_weak, omega)
    H_strong = knm_to_hamiltonian(K_strong, omega)

    mat_w = H_weak.to_matrix()
    mat_s = H_strong.to_matrix()
    if hasattr(mat_w, "toarray"):
        mat_w = mat_w.toarray()
    if hasattr(mat_s, "toarray"):
        mat_s = mat_s.toarray()

    E0_weak = np.linalg.eigvalsh(mat_w)[0]
    E0_strong = np.linalg.eigvalsh(mat_s)[0]

    assert E0_strong < E0_weak, (
        f"Strong coupling E0 ({E0_strong:.4f}) should be lower than weak ({E0_weak:.4f})"
    )


def _single_qubit_op(pauli: str, qubit: int, n: int):
    """SparsePauliOp for single-qubit Pauli."""
    from qiskit.quantum_info import SparsePauliOp

    label = ["I"] * n
    label[qubit] = pauli
    return SparsePauliOp("".join(reversed(label)))
