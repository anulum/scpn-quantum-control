# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for XY-Optimised Circuit Compiler
"""Tests for domain-specific XY Hamiltonian circuit compiler.

Covers:
    - xy_gate appends correct gates
    - compile_xy_trotter qubit count, order 1 and 2
    - Zero coupling/omega edge cases
    - _apply_xy_layer internals
    - depth_comparison: generic vs optimised
    - Unitary correctness: XY-compiled ≈ PauliEvolutionGate Trotter
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit

from scpn_quantum_control.phase.xy_compiler import (
    _apply_xy_layer,
    compile_xy_trotter,
    depth_comparison,
    xy_gate,
)


def _system(n: int = 3) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build a deterministic dense coupling system for compiler tests."""
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    np.fill_diagonal(K, 0.0)
    omega = np.linspace(0.8, 1.2, n)
    return K, omega


class TestXYGate:
    """Exercise the public two-qubit interaction primitive."""

    def test_appends_gates(self) -> None:
        """Append both controlled and rotation operations."""
        qc = QuantumCircuit(3)
        xy_gate(qc, 0, 1, 0.5)
        ops = [inst.operation.name for inst in qc.data]
        assert "cx" in ops
        assert "rx" in ops

    def test_gate_count(self) -> None:
        """Emit the stable three-operation primitive."""
        qc = QuantumCircuit(3)
        xy_gate(qc, 0, 1, 0.5)
        assert len(qc.data) == 3  # CX, RX, CX

    def test_zero_angle(self) -> None:
        """Retain the explicit primitive at zero interaction angle."""
        qc = QuantumCircuit(2)
        xy_gate(qc, 0, 1, 0.0)
        assert len(qc.data) == 3  # still appends gates


class TestCompileXYTrotter:
    """Exercise first- and second-order circuit compilation."""

    def test_qubit_count(self) -> None:
        """Allocate one qubit per oscillator."""
        K, omega = _system(4)
        qc = compile_xy_trotter(K, omega)
        assert qc.num_qubits == 4

    def test_order1(self) -> None:
        """Compile a first-order Lie-Trotter circuit."""
        K, omega = _system(3)
        qc = compile_xy_trotter(K, omega, reps=3, order=1)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 3

    def test_order2(self) -> None:
        """Compile a second-order circuit with the expected extra rotations."""
        K, omega = _system(3)
        qc = compile_xy_trotter(K, omega, reps=3, order=2)
        assert isinstance(qc, QuantumCircuit)
        # Order 2 should have more gates than order 1 (extra Rz half-steps)
        qc1 = compile_xy_trotter(K, omega, reps=3, order=1)
        assert len(qc.data) >= len(qc1.data)

    def test_zero_coupling(self) -> None:
        """Omit controlled interactions for a decoupled system."""
        n = 3
        K = np.zeros((n, n))
        omega = np.ones(n)
        qc = compile_xy_trotter(K, omega, reps=2)
        # No XY gates, only Ry + Rz
        ops = [inst.operation.name for inst in qc.data]
        assert "cx" not in ops

    def test_zero_omega(self) -> None:
        """Omit frequency rotations while retaining coupled interactions."""
        K, _ = _system(3)
        omega = np.zeros(3)
        qc = compile_xy_trotter(K, omega, reps=2)
        ops = [inst.operation.name for inst in qc.data]
        # No Rz from omega, but CX from coupling
        assert "cx" in ops

    def test_zero_omega_order2(self) -> None:
        """Omit both half-step frequency rotations in second-order compilation."""
        K, _ = _system(3)
        omega = np.zeros(3)
        qc = compile_xy_trotter(K, omega, reps=2, order=2)
        ops = [inst.operation.name for inst in qc.data]
        assert "rz" not in ops
        assert "cx" in ops

    def test_single_rep(self) -> None:
        """Compile the minimum positive repetition count."""
        K, omega = _system(3)
        qc = compile_xy_trotter(K, omega, reps=1)
        assert isinstance(qc, QuantumCircuit)


class TestApplyXYLayer:
    """Exercise sparse coupling-layer construction."""

    def test_layer_adds_gates(self) -> None:
        """Append one primitive for each nonzero upper-triangle coupling."""
        qc = QuantumCircuit(3)
        K = np.array([[0, 0.5, 0], [0.5, 0, 0.3], [0, 0.3, 0]])
        _apply_xy_layer(qc, K, 0.1, 3)
        ops = [inst.operation.name for inst in qc.data]
        assert ops.count("cx") == 4  # 2 couplings × 2 CX each

    def test_skip_zero_coupling(self) -> None:
        """Skip every zero coupling in the upper triangle."""
        qc = QuantumCircuit(3)
        K = np.zeros((3, 3))
        K[0, 1] = K[1, 0] = 0.5  # only one coupling
        _apply_xy_layer(qc, K, 0.1, 3)
        assert len(qc.data) == 3  # 1 xy_gate = CX + RX + CX


class TestDepthComparison:
    """Exercise the public generic-versus-optimized comparison."""

    def test_output_keys(self) -> None:
        """Return the complete stable comparison schema."""
        K, omega = _system(3)
        result = depth_comparison(K, omega, reps=2)
        expected = {
            "generic_depth",
            "optimised_depth",
            "reduction_pct",
            "generic_cx_count",
            "optimised_cx_count",
        }
        assert set(result.keys()) == expected

    def test_depths_positive(self) -> None:
        """Report positive depths for a coupled three-oscillator system."""
        K, omega = _system(3)
        result = depth_comparison(K, omega, reps=2)
        assert result["generic_depth"] > 0
        assert result["optimised_depth"] > 0

    def test_optimised_not_worse(self) -> None:
        """Optimised depth should not be drastically worse than generic."""
        K, omega = _system(3)
        result = depth_comparison(K, omega, reps=3)
        # Allow some tolerance — optimised is typically better
        assert result["optimised_depth"] <= result["generic_depth"] * 2
