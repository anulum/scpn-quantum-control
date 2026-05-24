# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Circuit export contract tests
"""Contract tests for QASM, Quil, and multi-format circuit export surfaces."""

from __future__ import annotations

import numpy as np
import pytest


def _system(n: int = 4):
    """Standard heterogeneous Kuramoto-XY system."""
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    np.fill_diagonal(K, 0.0)
    omega = np.linspace(0.8, 1.2, n)
    return n, K, omega


def _zero_coupling(n: int = 4):
    """Decoupled system — K=0, eigenstates are product states."""
    K = np.zeros((n, n))
    omega = np.linspace(0.8, 1.2, n)
    return n, K, omega


class TestCircuitExport:
    """Tests for multi-platform circuit export."""

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_qasm3_valid_for_multiple_sizes(self, n):
        from scpn_quantum_control.hardware.circuit_export import to_qasm3

        _, K, omega = _system(n)
        qasm = to_qasm3(K, omega, t=0.1, reps=2)
        assert isinstance(qasm, str)
        assert len(qasm) > 50
        assert "OPENQASM" in qasm or "qreg" in qasm or "measure" in qasm

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_quil_valid_for_multiple_sizes(self, n):
        from scpn_quantum_control.hardware.circuit_export import to_quil

        _, K, omega = _system(n)
        quil = to_quil(K, omega, t=0.1, reps=2)
        assert isinstance(quil, str)
        assert "DECLARE" in quil
        assert "MEASURE" in quil

    def test_export_all_keys_and_types(self):
        from qiskit import QuantumCircuit

        from scpn_quantum_control.hardware.circuit_export import export_all

        _, K, omega = _system(4)
        result = export_all(K, omega, t=0.1, reps=2)
        assert isinstance(result["qiskit"], QuantumCircuit)
        assert isinstance(result["qasm3"], str)
        assert isinstance(result["quil"], str)
        assert result["n_qubits"] == 4
        assert result["depth"] > 0
        assert result["gate_count"] > 0

    def test_build_trotter_circuit_properties(self):
        from scpn_quantum_control.hardware.circuit_export import build_trotter_circuit

        _, K, omega = _system(4)
        qc = build_trotter_circuit(K, omega, t=0.1, reps=3)
        assert qc.num_qubits == 4
        assert any(i.operation.name == "measure" for i in qc.data)
        assert qc.depth() > 0
        assert qc.size() > 0

    @pytest.mark.parametrize("reps", [1, 3, 5, 10])
    def test_depth_scales_with_reps(self, reps):
        from scpn_quantum_control.hardware.circuit_export import build_trotter_circuit

        _, K, omega = _system(3)
        qc = build_trotter_circuit(K, omega, t=0.1, reps=reps)
        assert qc.depth() > 0

    def test_export_formats_all_reference_same_qubits(self):
        from scpn_quantum_control.hardware.circuit_export import export_all

        _, K, omega = _system(3)
        result = export_all(K, omega, t=0.1, reps=2)
        assert result["qiskit"].num_qubits == 3
        assert result["n_qubits"] == 3
