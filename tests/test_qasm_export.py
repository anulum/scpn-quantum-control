# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for OpenQASM 3 circuit export."""

from __future__ import annotations

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.hardware.qasm_export import (
    QASMExportResult,
    export_ansatz_qasm,
    export_measurement_qasm,
    export_trotter_qasm,
)


class TestExportTrotterQASM:
    def test_returns_result(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = export_trotter_qasm(K, omega, t=0.5, reps=2)
        assert isinstance(result, QASMExportResult)

    def test_contains_openqasm(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = export_trotter_qasm(K, omega, t=0.5, reps=2)
        assert "OPENQASM" in result.qasm_string

    def test_n_qubits(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = export_trotter_qasm(K, omega)
        assert result.n_qubits == 4

    def test_gate_count_positive(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = export_trotter_qasm(K, omega)
        assert result.gate_count > 0

    def test_format_version(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = export_trotter_qasm(K, omega)
        assert result.format_version == "OpenQASM 3.0"


class TestExportAnsatzQASM:
    def test_returns_result(self):
        K = build_knm_paper27(L=4)
        result = export_ansatz_qasm(K)
        assert isinstance(result, QASMExportResult)

    def test_contains_openqasm(self):
        K = build_knm_paper27(L=4)
        result = export_ansatz_qasm(K)
        assert "OPENQASM" in result.qasm_string


class TestExportMeasurementQASM:
    def test_returns_result(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = export_measurement_qasm(K, omega, t=0.5, reps=2)
        assert isinstance(result, QASMExportResult)

    def test_contains_measure(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = export_measurement_qasm(K, omega, t=0.5, reps=2)
        assert "measure" in result.qasm_string.lower() or "bit" in result.qasm_string.lower()
