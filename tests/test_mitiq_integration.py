# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
"""Tests for Mitiq error mitigation integration."""

from __future__ import annotations

import pytest

mitiq = pytest.importorskip("mitiq")

from scpn_quantum_control.mitigation.mitiq_integration import (
    is_mitiq_available,
    zne_mitigated_expectation,
)


class TestMitiqAvailable:
    def test_mitiq_installed(self):
        assert is_mitiq_available()


class TestZNE:
    def test_zne_returns_float(self):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        result = zne_mitigated_expectation(qc, scale_factors=[1.0, 2.0, 3.0])
        assert isinstance(result, float)

    def test_zne_bounded(self):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.x(0)
        qc.measure_all()

        result = zne_mitigated_expectation(qc, scale_factors=[1.0, 2.0, 3.0])
        assert -1.1 <= result <= 1.1

    def test_zne_identity_circuit(self):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.measure_all()

        result = zne_mitigated_expectation(qc, scale_factors=[1.0, 2.0, 3.0])
        assert result > 0.5, "Identity circuit should give positive Z expectation"

    def test_zne_custom_executor(self):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure_all()

        def mock_executor(circuit):
            return 0.5

        result = zne_mitigated_expectation(
            qc, executor=mock_executor, scale_factors=[1.0, 2.0, 3.0]
        )
        assert isinstance(result, float)
