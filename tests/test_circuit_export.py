# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Multi-Platform Circuit Export
"""Tests for circuit_export: Qiskit → QASM / Quil / Cirq / export_all.

Covers:
    - build_trotter_circuit: correct qubit count, depth, measure_all
    - to_qasm3: valid QASM string with OPENQASM header
    - to_quil: Quil string with DECLARE, gate lines, MEASURE
    - to_cirq: Cirq conversion (mocked via mitiq fallback)
    - export_all: all keys present, metadata correct
    - Edge cases: n=2, zero coupling, zero omega
    - Physics: initial RY rotations from omega, Trotter evolution
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scpn_quantum_control.hardware.circuit_export import (
    build_trotter_circuit,
    export_all,
    to_cirq,
    to_qasm3,
    to_quil,
)


def _system(n: int = 4):
    """Build coupling matrix and frequencies for n oscillators."""
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    np.fill_diagonal(K, 0.0)
    omega = np.linspace(0.8, 1.2, n)
    return K, omega


class TestBuildTrotterCircuit:
    """Tests for Trotterised evolution circuit construction."""

    def test_returns_quantum_circuit(self):
        from qiskit import QuantumCircuit

        K, omega = _system(3)
        qc = build_trotter_circuit(K, omega)
        assert isinstance(qc, QuantumCircuit)

    def test_correct_qubit_count(self):
        for n in (2, 3, 4):
            K, omega = _system(n)
            qc = build_trotter_circuit(K, omega)
            assert qc.num_qubits == n

    def test_has_measurements(self):
        """measure_all() should add classical bits."""
        K, omega = _system(3)
        qc = build_trotter_circuit(K, omega)
        assert qc.num_clbits == 3

    def test_depth_positive(self):
        K, omega = _system(3)
        qc = build_trotter_circuit(K, omega, reps=3)
        assert qc.depth() > 0

    def test_different_reps_same_structure(self):
        """PauliEvolutionGate wraps reps internally; circuit structure stable."""
        K, omega = _system(3)
        qc1 = build_trotter_circuit(K, omega, reps=1)
        qc5 = build_trotter_circuit(K, omega, reps=5)
        assert qc1.num_qubits == qc5.num_qubits == 3

    def test_zero_time(self):
        """t=0 should still produce valid circuit (identity evolution)."""
        K, omega = _system(2)
        qc = build_trotter_circuit(K, omega, t=0.0, reps=1)
        assert qc.num_qubits == 2

    def test_ry_gates_from_omega(self):
        """Initial RY rotations encode natural frequencies."""
        K, omega = _system(3)
        qc = build_trotter_circuit(K, omega)
        ry_ops = [instr for instr in qc.data if instr.operation.name == "ry"]
        assert len(ry_ops) == 3


class TestToQasm3:
    """Tests for QASM export."""

    def test_contains_openqasm_header(self):
        K, omega = _system(3)
        qasm = to_qasm3(K, omega)
        assert "OPENQASM" in qasm

    def test_is_nonempty_string(self):
        K, omega = _system(2)
        qasm = to_qasm3(K, omega)
        assert isinstance(qasm, str)
        assert len(qasm) > 50

    def test_contains_qubit_declaration(self):
        K, omega = _system(3)
        qasm = to_qasm3(K, omega)
        assert "qreg" in qasm.lower() or "qubit" in qasm.lower()

    def test_various_sizes(self):
        for n in (2, 3, 4):
            K, omega = _system(n)
            qasm = to_qasm3(K, omega)
            assert "OPENQASM" in qasm


class TestToQuil:
    """Tests for Quil (Rigetti) export."""

    def test_contains_declare(self):
        K, omega = _system(3)
        quil = to_quil(K, omega)
        assert "DECLARE" in quil

    def test_contains_measure(self):
        K, omega = _system(3)
        quil = to_quil(K, omega)
        assert "MEASURE" in quil

    def test_contains_gate_ops(self):
        """Should have rotation gates (RX, RY, RZ) or CNOT."""
        K, omega = _system(3)
        quil = to_quil(K, omega)
        has_gates = any(g in quil for g in ("RX", "RY", "RZ", "CNOT"))
        assert has_gates

    def test_is_multiline(self):
        K, omega = _system(3)
        quil = to_quil(K, omega)
        lines = quil.strip().split("\n")
        assert len(lines) > 3


class TestToCirq:
    """Tests for Cirq conversion — uses mocked fallback paths."""

    def test_mitiq_path(self):
        """Cover to_cirq via mitiq.interface.mitiq_qiskit.conversions."""
        K, omega = _system(2)
        fake_circuit = MagicMock()
        mock_from_qiskit = MagicMock(return_value=fake_circuit)

        with patch.dict(
            "sys.modules",
            {
                "mitiq": MagicMock(),
                "mitiq.interface": MagicMock(),
                "mitiq.interface.mitiq_qiskit": MagicMock(),
                "mitiq.interface.mitiq_qiskit.conversions": MagicMock(
                    from_qiskit=mock_from_qiskit
                ),
            },
        ):
            result = to_cirq(K, omega)
        assert result is not None

    def test_mitiq_tuple_result(self):
        """Cover tuple branch: from_qiskit returns (circuit, qubits)."""
        K, omega = _system(2)
        fake_circuit = MagicMock()
        mock_from_qiskit = MagicMock(return_value=(fake_circuit, [0, 1]))

        with patch.dict(
            "sys.modules",
            {
                "mitiq": MagicMock(),
                "mitiq.interface": MagicMock(),
                "mitiq.interface.mitiq_qiskit": MagicMock(),
                "mitiq.interface.mitiq_qiskit.conversions": MagicMock(
                    from_qiskit=mock_from_qiskit
                ),
            },
        ):
            result = to_cirq(K, omega)
        assert result is fake_circuit

    def test_cirq_fallback_path(self):
        """Cover cirq.contrib.qasm_import fallback when mitiq absent."""
        K, omega = _system(2)
        fake_circuit = MagicMock()

        mock_qasm_import = MagicMock()
        mock_qasm_import.circuit_from_qasm = MagicMock(return_value=fake_circuit)

        with patch.dict(
            "sys.modules",
            {
                "mitiq": None,
                "mitiq.interface": None,
                "mitiq.interface.mitiq_qiskit": None,
                "mitiq.interface.mitiq_qiskit.conversions": None,
                "cirq": MagicMock(),
                "cirq.contrib": MagicMock(),
                "cirq.contrib.qasm_import": mock_qasm_import,
            },
        ):
            result = to_cirq(K, omega)
        assert result is fake_circuit

    def test_no_cirq_no_mitiq_raises(self):
        """Cover ImportError when neither mitiq nor cirq installed."""
        K, omega = _system(2)

        with (
            patch.dict(
                "sys.modules",
                {
                    "mitiq": None,
                    "mitiq.interface": None,
                    "mitiq.interface.mitiq_qiskit": None,
                    "mitiq.interface.mitiq_qiskit.conversions": None,
                    "cirq": None,
                    "cirq.contrib": None,
                    "cirq.contrib.qasm_import": None,
                },
            ),
            pytest.raises(ImportError, match="cirq not installed"),
        ):
            to_cirq(K, omega)


class TestExportAll:
    """Tests for export_all: combined multi-format export."""

    def test_returns_dict(self):
        K, omega = _system(3)
        result = export_all(K, omega)
        assert isinstance(result, dict)

    def test_contains_required_keys(self):
        K, omega = _system(3)
        result = export_all(K, omega)
        for key in ("qiskit", "qasm3", "quil", "n_qubits", "depth", "gate_count"):
            assert key in result, f"Missing key: {key}"

    def test_n_qubits_correct(self):
        K, omega = _system(4)
        result = export_all(K, omega)
        assert result["n_qubits"] == 4

    def test_depth_positive(self):
        K, omega = _system(3)
        result = export_all(K, omega)
        assert result["depth"] > 0

    def test_gate_count_positive(self):
        K, omega = _system(3)
        result = export_all(K, omega)
        assert result["gate_count"] > 0

    def test_qasm3_is_string(self):
        K, omega = _system(3)
        result = export_all(K, omega)
        assert isinstance(result["qasm3"], str)
        assert "OPENQASM" in result["qasm3"]

    def test_quil_is_string(self):
        K, omega = _system(3)
        result = export_all(K, omega)
        assert isinstance(result["quil"], str)
        assert "DECLARE" in result["quil"]

    def test_cirq_included_when_available(self):
        """export_all tries to include cirq via suppress(ImportError)."""
        K, omega = _system(2)
        fake_circuit = MagicMock()
        mock_from_qiskit = MagicMock(return_value=fake_circuit)

        with patch.dict(
            "sys.modules",
            {
                "mitiq": MagicMock(),
                "mitiq.interface": MagicMock(),
                "mitiq.interface.mitiq_qiskit": MagicMock(),
                "mitiq.interface.mitiq_qiskit.conversions": MagicMock(
                    from_qiskit=mock_from_qiskit
                ),
            },
        ):
            result = export_all(K, omega)
        assert "cirq" in result

    def test_cirq_absent_graceful(self):
        """export_all should not fail when cirq/mitiq unavailable."""
        K, omega = _system(2)
        with patch(
            "scpn_quantum_control.hardware.circuit_export.to_cirq",
            side_effect=ImportError("no cirq"),
        ):
            result = export_all(K, omega)
        assert "cirq" not in result


class TestEdgeCases:
    """Edge cases: zero coupling, zero omega, smallest system."""

    def test_n2_minimum_system(self):
        K, omega = _system(2)
        qc = build_trotter_circuit(K, omega)
        assert qc.num_qubits == 2
        qasm = to_qasm3(K, omega)
        assert "OPENQASM" in qasm

    def test_zero_coupling(self):
        """Zero K → only Z field rotations, no XX+YY terms."""
        K = np.zeros((3, 3))
        omega = np.array([1.0, 1.5, 2.0])
        qc = build_trotter_circuit(K, omega)
        assert qc.num_qubits == 3
        qasm = to_qasm3(K, omega)
        assert isinstance(qasm, str)

    def test_zero_omega(self):
        """Zero omega → coupling-only evolution."""
        K = np.array([[0, 0.5, 0.1], [0.5, 0, 0.3], [0.1, 0.3, 0]])
        omega = np.zeros(3)
        qc = build_trotter_circuit(K, omega)
        assert qc.num_qubits == 3
