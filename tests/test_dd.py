"""Tests for dynamical decoupling."""

import pytest
from qiskit import QuantumCircuit

from scpn_quantum_control.mitigation.dd import DDSequence, insert_dd_sequence


def test_xy4_adds_gates():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)

    result = insert_dd_sequence(qc, idle_qubits=[2], sequence=DDSequence.XY4)
    ops = result.count_ops()
    assert ops.get("x", 0) >= 2
    assert ops.get("y", 0) >= 2


def test_x2_adds_gates():
    qc = QuantumCircuit(2)
    qc.h(0)

    result = insert_dd_sequence(qc, idle_qubits=[1], sequence=DDSequence.X2)
    ops = result.count_ops()
    assert ops.get("x", 0) >= 2


def test_qubit_count_preserved():
    qc = QuantumCircuit(4)
    qc.h(0)
    result = insert_dd_sequence(qc, idle_qubits=[1, 2, 3])
    assert result.num_qubits == 4


def test_invalid_qubit_raises():
    qc = QuantumCircuit(2)
    with pytest.raises(ValueError, match="out of range"):
        insert_dd_sequence(qc, idle_qubits=[5])


def test_empty_idle_qubits():
    qc = QuantumCircuit(2)
    qc.h(0)
    result = insert_dd_sequence(qc, idle_qubits=[])
    original_ops = sum(qc.count_ops().values())
    result_ops = sum(result.count_ops().values())
    assert result_ops == original_ops


def test_multiple_idle_qubits():
    qc = QuantumCircuit(4)
    qc.h(0)
    result = insert_dd_sequence(qc, idle_qubits=[1, 2, 3], sequence=DDSequence.XY4)
    ops = result.count_ops()
    # 3 qubits * 2 x gates each = 6 x gates, + 3*2=6 y gates
    assert ops.get("x", 0) >= 6
    assert ops.get("y", 0) >= 6


def test_cpmg_adds_gates():
    """CPMG sequence (YXYX) should add both x and y gates."""
    qc = QuantumCircuit(2)
    qc.h(0)
    result = insert_dd_sequence(qc, idle_qubits=[1], sequence=DDSequence.CPMG)
    ops = result.count_ops()
    assert ops.get("y", 0) >= 2
    assert ops.get("x", 0) >= 2
