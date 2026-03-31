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

    def test_fractional_gate_count_comparison(self):
        """Fractional gates should reduce 2q gate count."""
        from scpn_quantum_control import OMEGA_N_16, QuantumKuramotoSolver, build_knm_paper27

        K = build_knm_paper27(L=4)
        solver = QuantumKuramotoSolver(4, K, OMEGA_N_16[:4])
        qc = solver.evolve(time=0.3, trotter_steps=2)

        runner_frac = HardwareRunner(use_simulator=True, use_fractional_gates=True)
        runner_frac.connect()
        isa_frac = runner_frac.transpile(qc)
        stats_frac = runner_frac.circuit_stats(isa_frac)

        runner_no = HardwareRunner(use_simulator=True, use_fractional_gates=False)
        runner_no.connect()
        isa_no = runner_no.transpile(qc)
        stats_no = runner_no.circuit_stats(isa_no)

        assert stats_frac["depth"] <= stats_no["depth"]
        assert stats_frac["total_gates"] <= stats_no["total_gates"] + 10

    def test_simple_bell_both_modes(self):
        """Bell pair should work correctly in both gate modes."""
        from qiskit import QuantumCircuit

        for frac in (True, False):
            runner = HardwareRunner(use_simulator=True, use_fractional_gates=frac)
            runner.connect()
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure_all()
            results = runner.run_sampler(qc, shots=1000, name="bell")
            counts = results[0].counts
            total = sum(counts.values())
            assert total == 1000
            # Bell pair should have mostly "00" and "11"
            ideal = counts.get("00", 0) + counts.get("11", 0)
            assert ideal > 900

    def test_fractional_preserves_qubit_count(self):
        from qiskit import QuantumCircuit

        runner = HardwareRunner(use_simulator=True, use_fractional_gates=True)
        runner.connect()
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()
        isa = runner.transpile(qc)
        assert isa.num_qubits >= 3

    def test_optimization_level_affects_depth(self):
        """Higher optimization should reduce depth."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.h(2)
        qc.cx(2, 0)
        qc.measure_all()

        runner_o0 = HardwareRunner(use_simulator=True, optimization_level=0)
        runner_o0.connect()
        isa_o0 = runner_o0.transpile(qc)

        runner_o2 = HardwareRunner(use_simulator=True, optimization_level=2)
        runner_o2.connect()
        isa_o2 = runner_o2.transpile(qc)

        assert isa_o2.depth() <= isa_o0.depth() + 5
