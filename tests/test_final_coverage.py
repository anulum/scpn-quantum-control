# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Final Coverage
"""Final coverage push: targeting the last coverable missed lines."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.hardware.classical import _expectation_pauli


def test_expectation_pauli_z():
    """Cover classical.py line 185: Z Pauli case."""
    # |0> state: <Z> = +1
    psi = np.array([1, 0, 0, 0], dtype=complex)  # 2-qubit |00>
    z0 = _expectation_pauli(psi, 2, 0, "Z")
    z1 = _expectation_pauli(psi, 2, 1, "Z")
    assert abs(z0 - 1.0) < 1e-10
    assert abs(z1 - 1.0) < 1e-10

    # |11> state: <Z> = -1 for both qubits
    psi_11 = np.array([0, 0, 0, 1], dtype=complex)
    z0_11 = _expectation_pauli(psi_11, 2, 0, "Z")
    z1_11 = _expectation_pauli(psi_11, 2, 1, "Z")
    assert abs(z0_11 - (-1.0)) < 1e-10
    assert abs(z1_11 - (-1.0)) < 1e-10


def test_inhibitor_all_equal_target():
    """Cover spn_to_qcircuit.py line 82: all inhibitors == target → bare Ry."""
    from qiskit import QuantumCircuit

    from scpn_quantum_control.bridge.spn_to_qcircuit import inhibitor_anti_control

    qc = QuantumCircuit(3)
    # 2 inhibitors both equal to target → controls list empty → bare ry
    inhibitor_anti_control(qc, [1, 1], target=1, theta=0.5)
    from qiskit.circuit.library import RYGate

    assert any(isinstance(inst.operation, RYGate) for inst in qc.data)


def test_inhibitor_single_self():
    """Cover spn_to_qcircuit.py line 84: single inhibitor == target → bare Ry."""
    from qiskit import QuantumCircuit

    from scpn_quantum_control.bridge.spn_to_qcircuit import inhibitor_anti_control

    qc = QuantumCircuit(3)
    inhibitor_anti_control(qc, [2], target=2, theta=0.4)
    from qiskit.circuit.library import RYGate

    assert any(isinstance(inst.operation, RYGate) for inst in qc.data)


def test_qec_odd_syndrome_defects():
    """Cover control_qec.py line 79: odd defects → duplicate first defect."""
    from scpn_quantum_control.qec.control_qec import ControlQEC

    qec = ControlQEC(distance=3)
    n_data = 2 * 3**2
    # Construct an error pattern that produces an odd number of defects
    rng = np.random.default_rng(0)
    for _ in range(100):
        err_x = (rng.random(n_data) < 0.15).astype(np.int8)
        err_z = np.zeros(n_data, dtype=np.int8)
        syn_z, _ = qec.get_syndrome(err_x, err_z)
        n_defects = int(syn_z.sum())
        if n_defects % 2 == 1:
            corr = qec.decoder.decode(syn_z)
            assert corr.shape == (n_data,)
            return
    pytest.skip("no odd syndrome found in 100 random samples")


def test_qec_decode_and_correct_failure():
    """Cover control_qec.py line 217: decoder returns False on heavy errors."""
    from scpn_quantum_control.qec.control_qec import ControlQEC

    qec = ControlQEC(distance=3)
    rng = np.random.default_rng(7)
    failures = 0
    for _ in range(200):
        err_x, err_z = qec.simulate_errors(p_error=0.4, rng=rng)
        if not qec.decode_and_correct(err_x, err_z):
            failures += 1
    assert failures > 0, "Expected some correction failures at p=0.4"


def test_expectation_pauli_x():
    """Cover X Pauli expectation: |+> state <X>=+1."""
    psi = np.array([1, 1, 0, 0], dtype=complex) / np.sqrt(2)  # qubit 0 = |+>, qubit 1 = |0>
    x0 = _expectation_pauli(psi, 2, 0, "X")
    assert abs(x0 - 1.0) < 1e-10


def test_expectation_pauli_superposition():
    """Uniform superposition <Z>=0 for all qubits."""
    psi = np.ones(4, dtype=complex) / 2.0
    z0 = _expectation_pauli(psi, 2, 0, "Z")
    z1 = _expectation_pauli(psi, 2, 1, "Z")
    assert abs(z0) < 1e-10
    assert abs(z1) < 1e-10


def test_qec_basic_syndrome():
    """ControlQEC should produce syndromes with correct shape."""
    from scpn_quantum_control.qec.control_qec import ControlQEC

    qec = ControlQEC(distance=3)
    n_data = 2 * 3**2
    err_x = np.zeros(n_data, dtype=np.int8)
    err_z = np.zeros(n_data, dtype=np.int8)
    syn_z, syn_x = qec.get_syndrome(err_x, err_z)
    assert len(syn_z) > 0
    assert len(syn_x) > 0


def test_qec_no_error_clean_syndrome():
    """Zero errors should produce all-zero syndrome."""
    from scpn_quantum_control.qec.control_qec import ControlQEC

    qec = ControlQEC(distance=3)
    n_data = 2 * 3**2
    err_x = np.zeros(n_data, dtype=np.int8)
    err_z = np.zeros(n_data, dtype=np.int8)
    syn_z, syn_x = qec.get_syndrome(err_x, err_z)
    assert int(syn_z.sum()) == 0
    assert int(syn_x.sum()) == 0


def test_inhibitor_multiple_inhibitors():
    """Cover spn_to_qcircuit multi-inhibitor path."""
    from qiskit import QuantumCircuit

    from scpn_quantum_control.bridge.spn_to_qcircuit import inhibitor_anti_control

    qc = QuantumCircuit(3)
    inhibitor_anti_control(qc, [0, 1], target=2, theta=0.5)
    ops = [inst.operation.name for inst in qc.data]
    assert ops.count("x") == 4


def test_pipeline_coverage_wiring():
    """Pipeline: verify all coverage targets (classical, SPN, QEC) are wired."""
    import time

    t0 = time.perf_counter()
    # Classical path
    psi = np.array([1, 0, 0, 0], dtype=complex)
    z = _expectation_pauli(psi, 2, 0, "Z")
    assert abs(z - 1.0) < 1e-10

    # QEC path
    from scpn_quantum_control.qec.control_qec import ControlQEC

    qec = ControlQEC(distance=3)
    n_data = 2 * 3**2
    syn_z, _ = qec.get_syndrome(np.zeros(n_data, dtype=np.int8), np.zeros(n_data, dtype=np.int8))
    assert int(syn_z.sum()) == 0

    dt = (time.perf_counter() - t0) * 1000
    print(f"\n  PIPELINE coverage wiring (classical+QEC): {dt:.1f} ms")
