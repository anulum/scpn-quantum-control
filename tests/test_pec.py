# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for mitigation/pec.py."""

import numpy as np
import pytest
from qiskit import QuantumCircuit

from scpn_quantum_control.mitigation.pec import PECResult, pauli_twirl_decompose, pec_sample


def test_decompose_zero_error():
    coeffs = pauli_twirl_decompose(0.0)
    assert coeffs[0] == pytest.approx(1.0)
    assert coeffs[1] == pytest.approx(0.0)
    assert coeffs[2] == pytest.approx(0.0)
    assert coeffs[3] == pytest.approx(0.0)


def test_decompose_small_error():
    coeffs = pauli_twirl_decompose(0.01)
    assert coeffs[0] > 1.0
    assert coeffs[1] < 0.0
    assert np.isclose(coeffs[1], coeffs[2])
    assert np.isclose(coeffs[2], coeffs[3])


def test_decompose_normalization():
    coeffs = pauli_twirl_decompose(0.1)
    assert np.sum(coeffs) == pytest.approx(1.0)


def test_decompose_rejects_invalid_rate():
    with pytest.raises(ValueError, match="gate_error_rate"):
        pauli_twirl_decompose(1.0)
    with pytest.raises(ValueError, match="gate_error_rate"):
        pauli_twirl_decompose(-0.1)


def test_decompose_multi_qubit_not_implemented():
    with pytest.raises(NotImplementedError):
        pauli_twirl_decompose(0.01, n_qubits=2)


def test_pec_sample_returns_result():
    qc = QuantumCircuit(1)
    qc.x(0)
    result = pec_sample(qc, 0.01, n_samples=50, rng=np.random.default_rng(42))
    assert isinstance(result, PECResult)
    assert result.n_samples == 50
    assert len(result.sign_distribution) == 50


def test_pec_zero_noise_recovers_ideal():
    qc = QuantumCircuit(1)
    qc.x(0)  # |0> -> |1>, <Z> = -1
    result = pec_sample(qc, 0.0, n_samples=100, rng=np.random.default_rng(42))
    assert result.mitigated_value == pytest.approx(-1.0, abs=0.01)
    assert result.overhead == pytest.approx(1.0)


def test_pec_overhead_increases_with_error():
    qc = QuantumCircuit(1)
    qc.h(0)
    r1 = pec_sample(qc, 0.01, n_samples=10, rng=np.random.default_rng(0))
    r2 = pec_sample(qc, 0.1, n_samples=10, rng=np.random.default_rng(0))
    assert r2.overhead > r1.overhead


def test_pec_sample_deterministic_with_seed():
    qc = QuantumCircuit(1)
    qc.ry(1.0, 0)
    r1 = pec_sample(qc, 0.05, 20, rng=np.random.default_rng(42))
    r2 = pec_sample(qc, 0.05, 20, rng=np.random.default_rng(42))
    assert r1.mitigated_value == pytest.approx(r2.mitigated_value)
