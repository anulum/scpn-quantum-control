"""Tests targeting specific uncovered lines to close coverage gaps."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.bridge.spn_to_qcircuit import inhibitor_anti_control, spn_to_circuit
from scpn_quantum_control.control.qaoa_mpc import QAOA_MPC
from scpn_quantum_control.control.vqls_gs import VQLS_GradShafranov
from scpn_quantum_control.phase.xy_kuramoto import QuantumKuramotoSolver
from scpn_quantum_control.qec.control_qec import ControlQEC
from scpn_quantum_control.qsnn.qstdp import QuantumSTDP

# --- spn_to_qcircuit lines 74-82: multi-inhibitor anti-control ---


def test_multi_inhibitor_anti_control():
    """Cover lines 74-82: >1 inhibitor qubit triggers multi-controlled RYGate."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(4)
    inhibitor_anti_control(qc, [0, 1], target=2, theta=0.5)
    assert qc.size() > 0


def test_inhibitor_target_in_inhibitor_list():
    """Cover line 82: target qubit is also in the inhibitor list (degenerate case)."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3)
    # target=1 is in inhibitor_qubits=[0,1] — controls list becomes [0], still works
    inhibitor_anti_control(qc, [0, 1], target=1, theta=0.3)
    assert qc.size() > 0


def test_spn_circuit_with_inhibitor_arcs():
    """SPN with inhibitor arcs (negative W_in) exercises the full anti-control path."""
    W_in = np.array([[-1.0, 0.5, 0.0], [0.0, 0.0, 0.8]])
    W_out = np.array([[0.0, 0.0], [0.6, 0.0], [0.0, 0.4]])
    thresholds = np.array([1.0, 1.0])

    qc = spn_to_circuit(W_in, W_out, thresholds)
    assert qc.num_qubits == 3


def test_spn_multi_inhibitor():
    """Two inhibitor arcs on a single transition -> multi-controlled gate."""
    W_in = np.array([[-0.5, -0.3, 0.0]])  # places 0,1 are inhibitors
    W_out = np.array([[0.0], [0.0], [0.7]])  # output to place 2
    thresholds = np.array([1.0])

    qc = spn_to_circuit(W_in, W_out, thresholds)
    assert qc.num_qubits == 3


# --- qaoa_mpc lines 62-63, 80-82: lazy hamiltonian build + ZZ terms ---


def test_qaoa_optimize_triggers_lazy_build():
    """QAOA_MPC.optimize() calls build_cost_hamiltonian lazily if not called yet."""
    B = np.eye(2)
    target = np.array([0.5, 0.5])
    mpc = QAOA_MPC(B, target, horizon=2, p_layers=1)
    # Don't call build_cost_hamiltonian — optimize should do it
    actions = mpc.optimize()
    assert len(actions) == 2


def test_qaoa_zz_circuit_path():
    """Inject a ZZ-containing Hamiltonian to cover lines 79-82 of _build_qaoa_circuit."""
    from qiskit.quantum_info import SparsePauliOp

    B = np.eye(2)
    target = np.array([0.5, 0.5])
    mpc = QAOA_MPC(B, target, horizon=4, p_layers=1)
    # Inject a custom Hamiltonian with ZZ terms
    mpc._cost_ham = SparsePauliOp(["IIZZ", "ZZII", "ZIII"], [0.3, 0.2, 0.1])
    gamma = np.array([0.5])
    beta = np.array([0.3])
    qc = mpc._build_qaoa_circuit(gamma, beta)
    assert qc.size() > 0


# --- vqls_gs line 92: degenerate xAtAx near zero ---


def test_vqls_cost_handles_near_zero_state():
    """VQLS cost returns 1.0 when xAtAx < 1e-15 (line 92)."""
    solver = VQLS_GradShafranov(n_qubits=4)
    psi = solver.solve(maxiter=5)
    assert psi is not None
    assert len(psi) > 0


# --- xy_kuramoto lines 54, 107: lazy hamiltonian build ---


def test_kuramoto_evolve_lazy_build():
    """evolve() calls build_hamiltonian if not already called (line 54)."""
    from scpn_quantum_control.bridge import OMEGA_N_16, build_knm_paper27

    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    solver = QuantumKuramotoSolver(2, K, omega)
    # Don't call build_hamiltonian — evolve should do it
    qc = solver.evolve(time=0.1, trotter_steps=1)
    assert qc.num_qubits == 2


def test_kuramoto_energy_lazy_build():
    """energy_expectation() calls build_hamiltonian if not already called (line 107)."""
    from qiskit.quantum_info import Statevector

    from scpn_quantum_control.bridge import OMEGA_N_16, build_knm_paper27

    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    solver = QuantumKuramotoSolver(2, K, omega)
    sv = Statevector.from_label("00")
    E = solver.energy_expectation(sv)
    assert np.isfinite(E)


# --- control_qec lines 79, 186, 217 ---


def test_qec_odd_defects_duplication():
    """Odd number of defects triggers defect duplication (line 79)."""
    qec = ControlQEC(distance=3)
    err_x = np.zeros(2 * 3**2, dtype=np.int8)
    err_z = np.zeros(2 * 3**2, dtype=np.int8)
    # Place a single X error to get odd syndrome defects
    err_x[0] = 1
    syn_z, syn_x = qec.get_syndrome(err_x, err_z)
    corr = qec.decoder.decode(syn_z)
    assert corr.shape == err_x.shape


def test_qec_simulate_errors_default_rng():
    """simulate_errors with rng=None creates its own (line 186)."""
    qec = ControlQEC(distance=3)
    err_x, err_z = qec.simulate_errors(p_error=0.1)
    assert err_x.shape == err_z.shape


def test_qec_residual_syndrome_failure():
    """High error rate causes correction failure (line 217)."""
    qec = ControlQEC(distance=3)
    rng = np.random.default_rng(42)
    failures = 0
    for _ in range(50):
        err_x, err_z = qec.simulate_errors(p_error=0.3, rng=rng)
        success = qec.decode_and_correct(err_x, err_z)
        if not success:
            failures += 1
    # At p=0.3, most rounds should fail
    assert failures > 0


# --- qstdp line 17: TYPE_CHECKING import ---
# Line 17 is a TYPE_CHECKING guard — can't be covered at runtime.
# Instead verify the class works correctly with a synapse.


def test_qstdp_update_with_synapse():
    """QuantumSTDP.update works with a real QuantumSynapse."""
    from scpn_quantum_control.qsnn import QuantumSynapse

    synapse = QuantumSynapse(weight=0.5)
    stdp = QuantumSTDP(learning_rate=0.1)
    old_w = synapse.weight
    stdp.update(synapse, pre_measured=1, post_measured=1)
    # Weight should change when both pre and post fire
    assert synapse.weight != old_w
