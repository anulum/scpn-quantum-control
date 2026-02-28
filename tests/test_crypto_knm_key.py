"""Tests for knm_key: coupling matrix to key material pipeline."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.bridge import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.crypto.knm_key import (
    estimate_qber,
    extract_raw_key,
    prepare_key_state,
    privacy_amplification,
)


def test_prepare_key_state_returns_valid():
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    result = prepare_key_state(K, omega, ansatz_reps=1, maxiter=20)
    assert result["n_qubits"] == 4
    assert result["energy"] < 0
    assert result["statevector"].num_qubits == 4
    assert abs(result["statevector"].trace() - 1.0) < 1e-10


def test_extract_raw_key_basic():
    counts = {"00": 100, "01": 50, "10": 30, "11": 20}
    bits = extract_raw_key(counts, "Z")
    assert len(bits) == 2
    assert bits.dtype == np.uint8


def test_extract_raw_key_subset():
    counts = {"000": 100, "001": 50, "110": 30}
    bits = extract_raw_key(counts, "Z", keep_qubits=[0, 2])
    assert len(bits) == 2


def test_qber_identical_keys():
    a = np.array([0, 1, 0, 1, 1], dtype=np.uint8)
    b = np.array([0, 1, 0, 1, 1], dtype=np.uint8)
    assert estimate_qber(a, b) == 0.0


def test_qber_half_errors():
    a = np.array([0, 0, 0, 0], dtype=np.uint8)
    b = np.array([0, 0, 1, 1], dtype=np.uint8)
    assert estimate_qber(a, b) == 0.5


def test_qber_empty():
    assert estimate_qber(np.array([]), np.array([])) == 1.0


def test_privacy_amplification_low_qber():
    raw = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1], dtype=np.uint8)
    key = privacy_amplification(raw, qber=0.02)
    assert len(key) == 32  # SHA-256


def test_privacy_amplification_high_qber():
    raw = np.array([0, 1, 0, 1], dtype=np.uint8)
    key = privacy_amplification(raw, qber=0.15)
    assert key == b""  # Above threshold
