# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for qec/fault_tolerant.py."""

import pytest

from scpn_quantum_control.qec.fault_tolerant import FaultTolerantUPDE, LogicalQubit


def test_logical_qubit_data_count():
    lq = LogicalQubit(code_distance=3, phase_angle=1.0)
    assert lq.data_qubits == 3


def test_init_basic():
    ft = FaultTolerantUPDE(n_osc=2, code_distance=3)
    assert ft.n_osc == 2
    assert ft.d == 3
    assert ft.total_qubits == 2 * (3 + 2)


def test_init_rejects_even_distance():
    with pytest.raises(ValueError, match="odd positive"):
        FaultTolerantUPDE(n_osc=2, code_distance=4)


def test_init_rejects_single_osc():
    with pytest.raises(ValueError, match="2"):
        FaultTolerantUPDE(n_osc=1)


def test_physical_qubit_count():
    ft = FaultTolerantUPDE(n_osc=3, code_distance=5)
    assert ft.physical_qubit_count() == 3 * (5 + 4)


def test_build_step_circuit():
    ft = FaultTolerantUPDE(n_osc=2, code_distance=3)
    qc = ft.build_step_circuit(dt=0.1)
    assert qc.num_qubits == ft.total_qubits
    assert qc.size() > 0


def test_step_with_qec():
    ft = FaultTolerantUPDE(n_osc=2, code_distance=3)
    result = ft.step_with_qec(dt=0.1)
    assert "syndromes" in result
    assert len(result["syndromes"]) == 2
    assert result["code_distance"] == 3


def test_step_no_errors_clean_syndromes():
    ft = FaultTolerantUPDE(n_osc=2, code_distance=3)
    result = ft.step_with_qec(dt=0.1)
    assert result["errors_detected"] == 0
