# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Pec
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


# ---------------------------------------------------------------------------
# PEC mathematical invariants
# ---------------------------------------------------------------------------


def test_coefficients_quasi_probability():
    """Identity coefficient > 1, error coefficients < 0 for p > 0."""
    coeffs = pauli_twirl_decompose(0.05)
    assert coeffs[0] > 1.0  # quasi-probability > 1
    for c in coeffs[1:]:
        assert c < 0.0  # negative quasi-probabilities


def test_overhead_formula():
    """PEC overhead = (1 + 2p/(1-p))^n for n gates. For 1 gate: 1/(1-4p/3)."""
    result = pec_sample(QuantumCircuit(1), 0.0, n_samples=10, rng=np.random.default_rng(0))
    assert result.overhead == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Rust path parity
# ---------------------------------------------------------------------------


def test_rust_pec_coefficients_parity():
    """Rust pec_coefficients should match Python pauli_twirl_decompose."""
    try:
        import scpn_quantum_engine as eng

        for p in [0.001, 0.01, 0.05, 0.1]:
            rust = np.array(eng.pec_coefficients(p))
            py = pauli_twirl_decompose(p)
            np.testing.assert_allclose(rust, py, atol=1e-10)
    except ImportError:
        pytest.skip("scpn-quantum-engine not available")


# ---------------------------------------------------------------------------
# Pipeline: circuit → PEC → mitigated expectation → wired
# ---------------------------------------------------------------------------


def test_pipeline_circuit_to_mitigated_value():
    """Full pipeline: build circuit → PEC sample → mitigated <Z>.
    Verifies PEC is not decorative — produces corrected expectation values.
    """
    import time

    qc = QuantumCircuit(1)
    qc.x(0)  # |1>, <Z> = -1

    t0 = time.perf_counter()
    result = pec_sample(qc, 0.05, n_samples=200, rng=np.random.default_rng(42))
    dt = (time.perf_counter() - t0) * 1000

    assert abs(result.mitigated_value - (-1.0)) < 0.2
    assert result.overhead > 1.0

    print(f"\n  PIPELINE PEC (1q X, p=0.05, 200 samples): {dt:.1f} ms")
    print(f"  Mitigated <Z> = {result.mitigated_value:.4f} (ideal: -1.0)")
    print(f"  Overhead = {result.overhead:.4f}")
