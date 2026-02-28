"""Tests for bridge/spn_to_qcircuit.py."""

import numpy as np
from qiskit import QuantumCircuit

from scpn_quantum_control.bridge.spn_to_qcircuit import inhibitor_to_anti_control, spn_to_circuit


def test_spn_to_circuit_qubit_count():
    W_in = np.array([[0.5, 0.3], [0.0, 0.7]])
    W_out = np.array([[0.4, 0.0], [0.0, 0.6]])
    thresholds = np.array([0.5, 0.5])
    qc = spn_to_circuit(W_in, W_out, thresholds)
    assert qc.num_qubits == 2


def test_inhibitor_pattern():
    qc = QuantumCircuit(1)
    inhibitor_to_anti_control(qc, 0, np.pi / 4)
    ops = [inst.operation.name for inst in qc.data]
    assert ops.count("x") == 2
    assert "ry" in ops


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
    assert "x" in ops  # inhibitor pattern present
