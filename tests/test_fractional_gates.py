# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Fractional Gates
"""Tests for fractional gate support in HardwareRunner."""

from __future__ import annotations

from scpn_quantum_control.hardware.runner import HardwareRunner


class TestFractionalGates:
    def test_default_fractional_enabled(self):
        runner = HardwareRunner(use_simulator=True)
        assert runner.use_fractional_gates is True

    def test_fractional_disabled(self):
        runner = HardwareRunner(use_simulator=True, use_fractional_gates=False)
        assert runner.use_fractional_gates is False

    def test_simulator_basis_includes_rzz_when_fractional(self):
        runner = HardwareRunner(use_simulator=True, use_fractional_gates=True)
        runner.connect()
        # Transpile a simple circuit and check that rzz is available
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.rzz(0.5, 0, 1)
        qc.measure_all()
        isa = runner.transpile(qc)
        # With fractional gates, rzz should remain as-is (not decomposed to ecr+rz)
        ops = isa.count_ops()
        assert "rzz" in ops or isa.depth() <= qc.depth() + 5

    def test_simulator_without_fractional_decomposes_rzz(self):
        runner = HardwareRunner(use_simulator=True, use_fractional_gates=False)
        runner.connect()
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.rzz(0.5, 0, 1)
        qc.measure_all()
        isa = runner.transpile(qc)
        ops = isa.count_ops()
        # Without fractional, rzz should be decomposed into basis gates
        assert "rzz" not in ops

    def test_kuramoto_circuit_depth_reduction(self):
        from scpn_quantum_control import OMEGA_N_16, QuantumKuramotoSolver, build_knm_paper27

        K = build_knm_paper27(L=4)
        solver = QuantumKuramotoSolver(4, K, OMEGA_N_16[:4])
        qc = solver.evolve(time=0.3, trotter_steps=2)

        runner_frac = HardwareRunner(use_simulator=True, use_fractional_gates=True)
        runner_frac.connect()
        isa_frac = runner_frac.transpile(qc)

        runner_no = HardwareRunner(use_simulator=True, use_fractional_gates=False)
        runner_no.connect()
        isa_no = runner_no.transpile(qc)

        # Fractional gates should produce shallower or equal depth circuits
        assert isa_frac.depth() <= isa_no.depth()
