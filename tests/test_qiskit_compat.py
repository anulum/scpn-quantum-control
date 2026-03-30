# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Qiskit Compat
"""Tests for Qiskit compatibility layer."""

from __future__ import annotations

from scpn_quantum_control.hardware.qiskit_compat import (
    check_qiskit_compatibility,
    get_lie_trotter,
    get_pauli_evolution_gate,
    get_sparse_pauli_op,
    get_statevector,
    qiskit_major,
    qiskit_version,
)


class TestQiskitCompat:
    def test_version_string(self):
        v = qiskit_version()
        assert isinstance(v, str)
        assert "." in v

    def test_major_is_int(self):
        m = qiskit_major()
        assert isinstance(m, int)
        assert m >= 1

    def test_pauli_evolution_gate(self):
        PEG = get_pauli_evolution_gate()
        assert PEG is not None

    def test_lie_trotter(self):
        LT = get_lie_trotter()
        assert LT is not None

    def test_statevector(self):
        SV = get_statevector()
        assert SV is not None

    def test_sparse_pauli_op(self):
        SPO = get_sparse_pauli_op()
        assert SPO is not None

    def test_check_compatibility(self):
        result = check_qiskit_compatibility()
        assert "version" in result
        assert "compatible" in result
        assert isinstance(result["compatible"], bool)
