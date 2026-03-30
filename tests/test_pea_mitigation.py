# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Pea Mitigation
"""Tests for PEA error mitigation default in HardwareRunner."""

from __future__ import annotations

from scpn_quantum_control.hardware.runner import HardwareRunner


class TestPEAMitigation:
    def test_default_resilience_level_is_2(self):
        runner = HardwareRunner(use_simulator=True)
        assert runner.resilience_level == 2

    def test_resilience_level_override(self):
        runner = HardwareRunner(use_simulator=True, resilience_level=0)
        assert runner.resilience_level == 0

    def test_resilience_level_1_still_works(self):
        runner = HardwareRunner(use_simulator=True, resilience_level=1)
        assert runner.resilience_level == 1

    def test_simulator_ignores_resilience(self):
        """Simulator path doesn't use resilience_level (no IBM Runtime)."""
        runner = HardwareRunner(use_simulator=True, resilience_level=2)
        runner.connect()
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        # Should work without error — simulator has no resilience options
        isa = runner.transpile(qc)
        assert isa.num_qubits >= 2
