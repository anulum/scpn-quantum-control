# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Spn To Qcircuit
"""Tests for bridge/spn_to_qcircuit.py — elite multi-angle coverage."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from scpn_quantum_control.bridge.spn_to_qcircuit import inhibitor_anti_control, spn_to_circuit

# ---------------------------------------------------------------------------
# spn_to_circuit — basic structure
# ---------------------------------------------------------------------------


class TestSPNToCircuit:
    def test_qubit_count_matches_places(self):
        W_in = np.array([[0.5, 0.3], [0.0, 0.7]])
        W_out = np.array([[0.4, 0.0], [0.0, 0.6]])
        thresholds = np.array([0.5, 0.5])
        qc = spn_to_circuit(W_in, W_out, thresholds)
        assert qc.num_qubits == 2

    def test_zero_weights_no_gates(self):
        W_in = np.zeros((1, 2))
        W_out = np.zeros((2, 1))
        thresholds = np.array([0.5])
        qc = spn_to_circuit(W_in, W_out, thresholds)
        assert qc.num_qubits == 2
        assert len(qc.data) == 0

    def test_larger_network(self):
        W_in = np.array([[0.5, 0.3, 0.0], [0.0, 0.0, 0.8]])
        W_out = np.array([[0.0, 0.4], [0.6, 0.0], [0.0, 0.5]])
        thresholds = np.array([0.5, 0.5])
        qc = spn_to_circuit(W_in, W_out, thresholds)
        assert qc.num_qubits == 3

    def test_single_transition_single_place(self):
        W_in = np.array([[0.5]])
        W_out = np.array([[0.5]])
        thresholds = np.array([1.0])
        qc = spn_to_circuit(W_in, W_out, thresholds)
        assert qc.num_qubits == 1


# ---------------------------------------------------------------------------
# inhibitor_anti_control
# ---------------------------------------------------------------------------


class TestInhibitorAntiControl:
    def test_produces_x_gates(self):
        qc = QuantumCircuit(2)
        inhibitor_anti_control(qc, [0], 1, np.pi / 4)
        ops = [inst.operation.name for inst in qc.data]
        assert ops.count("x") == 2
        assert "cry" in ops

    def test_self_inhibitor_bare_ry(self):
        """When inhibitor == target, use bare Ry (no control)."""
        qc = QuantumCircuit(3)
        inhibitor_anti_control(qc, [1], target=1, theta=0.5)
        from qiskit.circuit.library import RYGate

        assert any(isinstance(inst.operation, RYGate) for inst in qc.data)

    def test_all_inhibitors_equal_target(self):
        qc = QuantumCircuit(3)
        inhibitor_anti_control(qc, [1, 1], target=1, theta=0.5)
        from qiskit.circuit.library import RYGate

        assert any(isinstance(inst.operation, RYGate) for inst in qc.data)

    def test_multiple_inhibitors(self):
        """Multi-controlled anti-pattern should produce controlled RY."""
        qc = QuantumCircuit(3)
        inhibitor_anti_control(qc, [0, 1], target=2, theta=np.pi / 4)
        ops = [inst.operation.name for inst in qc.data]
        assert ops.count("x") == 4  # 2 inhibitors × 2 (flip+restore)


# ---------------------------------------------------------------------------
# Negative weight → inhibitor behaviour
# ---------------------------------------------------------------------------


class TestNegativeWeightInhibitor:
    def test_negative_input_produces_x_gates(self):
        W_in = np.array([[-0.5, 0.3]])
        W_out = np.array([[0.4], [0.0]])
        thresholds = np.array([0.5])
        qc = spn_to_circuit(W_in, W_out, thresholds)
        ops = [inst.operation.name for inst in qc.data]
        assert "x" in ops

    def test_inhibitor_blocks_when_place_occupied(self):
        """Output fires when inhibitor place is empty, suppressed when occupied."""
        W_in = np.array([[-0.5, 0.0]])
        W_out = np.array([[0.0], [0.9]])
        thresholds = np.array([1.0])

        qc_full = spn_to_circuit(W_in, W_out, thresholds)

        # Case 1: inhibitor empty (|0>) → output fires
        qc_empty = QuantumCircuit(2)
        qc_empty.compose(qc_full, inplace=True)
        sv_empty = Statevector.from_instruction(qc_empty)
        p1_empty = sv_empty.probabilities([1])[1]

        # Case 2: inhibitor occupied (|1>) → output suppressed
        qc_occ = QuantumCircuit(2)
        qc_occ.x(0)
        qc_occ.compose(qc_full, inplace=True)
        sv_occ = Statevector.from_instruction(qc_occ)
        p1_occ = sv_occ.probabilities([1])[1]

        assert p1_empty > p1_occ


# ---------------------------------------------------------------------------
# Circuit unitarity
# ---------------------------------------------------------------------------


class TestCircuitUnitarity:
    def test_circuit_preserves_norm(self):
        """Any SPN circuit should preserve statevector normalisation."""
        W_in = np.array([[0.5, -0.3], [0.0, 0.7]])
        W_out = np.array([[0.4, 0.0], [0.0, 0.6]])
        thresholds = np.array([0.5, 0.5])
        qc = spn_to_circuit(W_in, W_out, thresholds)
        sv = Statevector.from_instruction(qc)
        np.testing.assert_allclose(float(np.sum(np.abs(sv) ** 2)), 1.0, atol=1e-12)
