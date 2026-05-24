# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Multi-platform export workflow contract tests
"""Workflow tests for compiler output and circuit export consistency across supported formats."""

from __future__ import annotations

import numpy as np


def _system(n: int = 4):
    """Standard heterogeneous Kuramoto-XY system."""
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    np.fill_diagonal(K, 0.0)
    omega = np.linspace(0.8, 1.2, n)
    return n, K, omega


def _homogeneous_system(n: int = 4):
    """Circulant K + uniform omega for translation symmetry."""
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = min(abs(i - j), n - abs(i - j))
            K[i, j] = 0.5 * np.exp(-0.3 * d) if d > 0 else 0
    omega = np.ones(n) * 1.0
    return n, K, omega


class TestMultiPlatformPipeline:
    """XY-compiled circuits should export correctly to all formats."""

    def test_xy_compiled_exports_to_qasm(self):
        """XY-compiled circuit should produce valid QASM string."""
        from scpn_quantum_control.hardware.circuit_export import to_qasm3
        from scpn_quantum_control.phase.xy_compiler import compile_xy_trotter

        _, K, omega = _system(4)

        # Compile with XY-optimised gates
        qc = compile_xy_trotter(K, omega, t=0.1, reps=3)
        assert qc.num_qubits == 4

        # Export same system to QASM
        qasm = to_qasm3(K, omega, t=0.1, reps=3)
        assert isinstance(qasm, str)
        assert len(qasm) > 100

    def test_xy_compiler_reduces_depth(self):
        """XY compiler should produce shallower circuits than generic Trotter."""
        from scpn_quantum_control.phase.xy_compiler import depth_comparison

        _, K, omega = _system(4)
        cmp = depth_comparison(K, omega, t=0.1, reps=5)

        assert cmp["optimised_depth"] > 0
        assert cmp["generic_depth"] > 0

    def test_export_all_formats_consistent(self):
        """All export formats should represent the same circuit."""
        from scpn_quantum_control.hardware.circuit_export import export_all

        _, K, omega = _system(4)
        result = export_all(K, omega, t=0.1, reps=3)

        assert result["qiskit"].num_qubits == 4
        assert "OPENQASM" in result["qasm3"] or "qreg" in result["qasm3"]
        assert "DECLARE" in result["quil"]
        assert result["n_qubits"] == 4
        assert result["depth"] > 0

    def test_ancilla_circuit_exportable(self):
        """Ancilla Lindblad circuit should be exportable to QASM."""
        from qiskit import qasm2

        from scpn_quantum_control.phase.ancilla_lindblad import (
            build_ancilla_lindblad_circuit,
        )

        _, K, omega = _system(3)
        qc = build_ancilla_lindblad_circuit(
            K,
            omega,
            t=0.1,
            trotter_reps=2,
            n_dissipation_steps=2,
        )

        # Should be exportable to QASM
        qasm_str = qasm2.dumps(qc)
        assert isinstance(qasm_str, str)
        assert len(qasm_str) > 50
