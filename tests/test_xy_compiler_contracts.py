# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — XY compiler contract tests
"""Contract tests for XY compiler qubit counts, depth behaviour, and gate-unitarity boundaries."""

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


class TestXYCompiler:
    """Tests for XY-optimised gate decomposition."""

    @pytest.mark.parametrize("n", [2, 3, 4, 6])
    def test_compile_returns_correct_qubit_count(self, n):
        from scpn_quantum_control.phase.xy_compiler import compile_xy_trotter

        _, K, omega = _system(n)
        qc = compile_xy_trotter(K, omega, t=0.1, reps=3)
        assert qc.num_qubits == n

    def test_order2_deeper_than_order1(self):
        from scpn_quantum_control.phase.xy_compiler import compile_xy_trotter

        _, K, omega = _system(4)
        qc1 = compile_xy_trotter(K, omega, t=0.1, reps=3, order=1)
        qc2 = compile_xy_trotter(K, omega, t=0.1, reps=3, order=2)
        assert qc2.depth() >= qc1.depth()
        assert qc2.size() >= qc1.size()

    def test_depth_comparison_values(self):
        from scpn_quantum_control.phase.xy_compiler import depth_comparison

        _, K, omega = _system(4)
        result = depth_comparison(K, omega, t=0.1, reps=5)
        assert result["generic_depth"] > 0
        assert result["optimised_depth"] > 0
        assert isinstance(result["reduction_pct"], (int, float))

    @pytest.mark.parametrize("angle", [0.0, 0.1, 0.5, np.pi / 4, np.pi / 2])
    def test_xy_gate_is_unitary(self, angle):
        """XY gate must be unitary for all rotation angles."""
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator

        from scpn_quantum_control.phase.xy_compiler import xy_gate

        qc = QuantumCircuit(2)
        xy_gate(qc, 0, 1, angle)
        op = Operator(qc)
        product = op.data @ op.data.conj().T
        np.testing.assert_allclose(product, np.eye(4), atol=1e-10)

    def test_xy_gate_zero_angle_is_identity(self):
        """XY gate at angle=0 should be identity."""
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator

        from scpn_quantum_control.phase.xy_compiler import xy_gate

        qc = QuantumCircuit(2)
        xy_gate(qc, 0, 1, 0.0)
        op = Operator(qc)
        np.testing.assert_allclose(np.abs(op.data), np.eye(4), atol=1e-10)

    def test_reps_increases_depth(self):
        from scpn_quantum_control.phase.xy_compiler import compile_xy_trotter

        _, K, omega = _system(3)
        d1 = compile_xy_trotter(K, omega, t=0.1, reps=1).depth()
        d3 = compile_xy_trotter(K, omega, t=0.1, reps=3).depth()
        assert d3 >= d1
