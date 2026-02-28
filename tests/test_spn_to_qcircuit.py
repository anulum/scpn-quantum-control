"""Tests for bridge/spn_to_qcircuit.py."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from scpn_quantum_control.bridge.spn_to_qcircuit import inhibitor_anti_control, spn_to_circuit


def test_spn_to_circuit_qubit_count():
    W_in = np.array([[0.5, 0.3], [0.0, 0.7]])
    W_out = np.array([[0.4, 0.0], [0.0, 0.6]])
    thresholds = np.array([0.5, 0.5])
    qc = spn_to_circuit(W_in, W_out, thresholds)
    assert qc.num_qubits == 2


def test_inhibitor_anti_control_produces_x_gates():
    qc = QuantumCircuit(2)
    inhibitor_anti_control(qc, [0], 1, np.pi / 4)
    ops = [inst.operation.name for inst in qc.data]
    assert ops.count("x") == 2
    assert "cry" in ops


def test_zero_weights_no_gates():
    W_in = np.zeros((1, 2))
    W_out = np.zeros((2, 1))
    thresholds = np.array([0.5])
    qc = spn_to_circuit(W_in, W_out, thresholds)
    assert qc.num_qubits == 2
    assert len(qc.data) == 0


def test_negative_weight_inhibitor():
    W_in = np.array([[-0.5, 0.3]])
    W_out = np.array([[0.4], [0.0]])
    thresholds = np.array([0.5])
    qc = spn_to_circuit(W_in, W_out, thresholds)
    ops = [inst.operation.name for inst in qc.data]
    assert "x" in ops


def test_inhibitor_blocks_when_place_occupied():
    """Output should fire only when inhibitor place is empty (|0>)."""
    # 2 places, 1 transition. Place 0 is inhibitor (-0.5), output goes to place 1.
    W_in = np.array([[-0.5, 0.0]])
    W_out = np.array([[0.0], [0.9]])
    thresholds = np.array([1.0])

    # Case 1: inhibitor place empty (|0>) → output should fire
    qc_empty = QuantumCircuit(2)
    qc_full = spn_to_circuit(W_in, W_out, thresholds)
    qc_empty.compose(qc_full, inplace=True)
    sv_empty = Statevector.from_instruction(qc_empty)
    p1_prob_empty = sv_empty.probabilities([1])[1]

    # Case 2: inhibitor place occupied (|1>) → output should be suppressed
    qc_occ = QuantumCircuit(2)
    qc_occ.x(0)
    qc_occ.compose(qc_full, inplace=True)
    sv_occ = Statevector.from_instruction(qc_occ)
    p1_prob_occ = sv_occ.probabilities([1])[1]

    assert p1_prob_empty > p1_prob_occ
