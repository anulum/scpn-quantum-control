# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Qasm Export
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
    """Exporting a Trotter circuit as QASM."""

    def test_returns_result(self) -> None:
        """The export returns a result record."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = export_trotter_qasm(K, omega, t=0.5, reps=2)
        assert isinstance(result, QASMExportResult)

    def test_contains_openqasm(self) -> None:
        """The text declares OpenQASM."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = export_trotter_qasm(K, omega, t=0.5, reps=2)
        assert "OPENQASM" in result.qasm_string

    def test_n_qubits(self) -> None:
        """The result records four qubits."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = export_trotter_qasm(K, omega)
        assert result.n_qubits == 4

    def test_gate_count_positive(self) -> None:
        """The gate count is positive."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = export_trotter_qasm(K, omega)
        assert result.gate_count > 0

    def test_format_version(self) -> None:
        """The format is OpenQASM 3.0."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = export_trotter_qasm(K, omega)
        assert result.format_version == "OpenQASM 3.0"


class TestExportAnsatzQASM:
    """Exporting an ansatz as QASM."""

    def test_returns_result(self) -> None:
        """The export returns a result record."""
        K = build_knm_paper27(L=4)
        result = export_ansatz_qasm(K)
        assert isinstance(result, QASMExportResult)

    def test_contains_openqasm(self) -> None:
        """The text declares OpenQASM."""
        K = build_knm_paper27(L=4)
        result = export_ansatz_qasm(K)
        assert "OPENQASM" in result.qasm_string


class TestExportMeasurementQASM:
    """Exporting a measurement circuit as QASM."""

    def test_returns_result(self) -> None:
        """The export returns a result record."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = export_measurement_qasm(K, omega, t=0.5, reps=2)
        assert isinstance(result, QASMExportResult)

    def test_contains_measure(self) -> None:
        """The text contains measurements or bits."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = export_measurement_qasm(K, omega, t=0.5, reps=2)
        assert "measure" in result.qasm_string.lower() or "bit" in result.qasm_string.lower()


# ---------------------------------------------------------------------------
# QASM validity: parseable, contains qubit declarations
# ---------------------------------------------------------------------------


class TestQASMValidity:
    """Validity of the exported text."""

    def test_qasm_contains_qubit_declaration(self) -> None:
        """The text declares its qubits."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = export_trotter_qasm(K, omega)
        assert "qubit" in result.qasm_string.lower() or "qreg" in result.qasm_string.lower()

    def test_qasm_nonempty(self) -> None:
        """The text is longer than 50 characters."""
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = export_trotter_qasm(K, omega)
        assert len(result.qasm_string) > 50

    def test_various_sizes(self) -> None:
        """The qubit count follows the layer count."""
        for L in [2, 3, 4]:
            K = build_knm_paper27(L=L)
            omega = OMEGA_N_16[:L]
            result = export_trotter_qasm(K, omega)
            assert result.n_qubits == L


# ---------------------------------------------------------------------------
# Pipeline: Knm → QASM export → valid string → wired
# ---------------------------------------------------------------------------


class TestQASMPipeline:
    """The coupling-to-QASM pipeline."""

    def test_pipeline_knm_to_qasm(self) -> None:
        """Full pipeline: build_knm → Trotter circuit → QASM3 export.

        Verifies QASM export is wired and produces valid output.
        """
        import time

        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]

        t0 = time.perf_counter()
        result = export_trotter_qasm(K, omega, t=0.5, reps=3)
        dt = (time.perf_counter() - t0) * 1000

        assert "OPENQASM" in result.qasm_string
        assert result.gate_count > 0

        print(f"\n  PIPELINE Knm→QASM3 (4q, 3 reps): {dt:.1f} ms")
        print(f"  Gates = {result.gate_count}, QASM length = {len(result.qasm_string)} chars")
