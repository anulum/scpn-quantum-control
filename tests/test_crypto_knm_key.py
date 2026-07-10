# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Crypto Knm Key
"""Tests for knm_key: coupling matrix to key material pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.crypto.knm_key import (
    estimate_qber,
    extract_raw_key,
    prepare_key_state,
    privacy_amplification,
)


def test_prepare_key_state_returns_valid() -> None:
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    result = prepare_key_state(K, omega, ansatz_reps=1, maxiter=20)
    assert result["n_qubits"] == 4
    assert result["energy"] < 0
    assert result["statevector"].num_qubits == 4
    assert abs(result["statevector"].trace() - 1.0) < 1e-10


def test_extract_raw_key_basic() -> None:
    counts = {"00": 100, "01": 50, "10": 30, "11": 20}
    bits = extract_raw_key(counts, "Z")
    assert len(bits) == 2
    assert bits.dtype == np.uint8


def test_extract_raw_key_subset() -> None:
    counts = {"000": 100, "001": 50, "110": 30}
    bits = extract_raw_key(counts, "Z", keep_qubits=[0, 2])
    assert len(bits) == 2


def test_qber_identical_keys() -> None:
    a = np.array([0, 1, 0, 1, 1], dtype=np.uint8)
    b = np.array([0, 1, 0, 1, 1], dtype=np.uint8)
    assert estimate_qber(a, b) == 0.0


def test_qber_half_errors() -> None:
    a = np.array([0, 0, 0, 0], dtype=np.uint8)
    b = np.array([0, 0, 1, 1], dtype=np.uint8)
    assert estimate_qber(a, b) == 0.5


def test_qber_empty() -> None:
    assert estimate_qber(np.array([]), np.array([])) == 1.0


def _expected_secure_bits(n_bits: int, qber: float) -> int:
    h2 = -qber * np.log2(qber + 1e-15) - (1 - qber) * np.log2(1 - qber + 1e-15)
    return int(n_bits * max(0.0, 1.0 - 2.0 * h2))


def test_privacy_amplification_output_length_is_n_secure_bits() -> None:
    """The extracted key is exactly the leftover-hash-lemma length, in bits."""
    rng = np.random.default_rng(11)
    raw = rng.integers(0, 2, size=256, dtype=np.uint8)
    for qber in (0.0, 0.02, 0.05, 0.09):
        key = privacy_amplification(raw, qber=qber, seed=7)
        assert key.dtype == np.uint8
        assert key.size == _expected_secure_bits(raw.size, qber)
        assert set(np.unique(key)).issubset({0, 1})


def test_privacy_amplification_deterministic_per_seed() -> None:
    """Same raw key + same Toeplitz seed reproduces the same output bits."""
    rng = np.random.default_rng(12)
    raw = rng.integers(0, 2, size=128, dtype=np.uint8)
    k1 = privacy_amplification(raw, qber=0.02, seed=1234)
    k2 = privacy_amplification(raw, qber=0.02, seed=1234)
    assert np.array_equal(k1, k2)


def test_privacy_amplification_seed_sensitivity() -> None:
    """A different Toeplitz seed selects a different family member (different output)."""
    rng = np.random.default_rng(13)
    raw = rng.integers(0, 2, size=128, dtype=np.uint8)
    k1 = privacy_amplification(raw, qber=0.02, seed=1)
    k2 = privacy_amplification(raw, qber=0.02, seed=2)
    assert k1.size == k2.size > 0
    assert not np.array_equal(k1, k2)


def test_privacy_amplification_input_sensitivity() -> None:
    """Flipping one raw-key bit changes the extracted key (linearity over GF(2))."""
    rng = np.random.default_rng(14)
    raw = rng.integers(0, 2, size=128, dtype=np.uint8)
    flipped = raw.copy()
    flipped[0] ^= 1
    k1 = privacy_amplification(raw, qber=0.02, seed=5)
    k2 = privacy_amplification(flipped, qber=0.02, seed=5)
    assert not np.array_equal(k1, k2)


def test_privacy_amplification_matches_explicit_toeplitz_product() -> None:
    """The output equals T @ raw_key mod 2 for the seed-derived Toeplitz matrix."""
    rng = np.random.default_rng(15)
    raw = rng.integers(0, 2, size=64, dtype=np.uint8)
    qber = 0.02
    key = privacy_amplification(raw, qber=qber, seed=99)
    m = _expected_secure_bits(raw.size, qber)
    diagonals = np.random.default_rng(99).integers(0, 2, size=m + raw.size - 1, dtype=np.uint8)
    expected = np.zeros(m, dtype=np.uint8)
    for i in range(m):
        acc = 0
        for j in range(raw.size):
            acc ^= int(diagonals[i - j + raw.size - 1]) & int(raw[j])
        expected[i] = acc
    assert np.array_equal(key, expected)


def test_privacy_amplification_high_qber_yields_empty_key() -> None:
    raw = np.array([0, 1, 0, 1], dtype=np.uint8)
    key = privacy_amplification(raw, qber=0.15, seed=3)
    assert key.size == 0  # At/above threshold: no extractable bits


def test_privacy_amplification_zero_fraction_yields_empty_key() -> None:
    """Just below the threshold the secret fraction rounds to zero bits — honest empty."""
    raw = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.uint8)
    key = privacy_amplification(raw, qber=0.109, seed=3)
    assert key.size == 0


def test_privacy_amplification_seed_is_required() -> None:
    raw = np.array([0, 1, 0, 1], dtype=np.uint8)
    with pytest.raises(TypeError):
        privacy_amplification(raw, 0.02)  # type: ignore[call-arg]


def test_prepare_key_state_returns_dict() -> None:
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    state = prepare_key_state(K, omega, ansatz_reps=1, maxiter=10)
    assert "statevector" in state
    assert "energy" in state


def test_prepare_key_state_energy_finite() -> None:
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    state = prepare_key_state(K, omega, ansatz_reps=1, maxiter=10)
    assert np.isfinite(state["energy"])


def test_estimate_qber_identical_keys() -> None:
    alice = np.array([0, 1, 0, 1], dtype=np.uint8)
    bob = np.array([0, 1, 0, 1], dtype=np.uint8)
    qber = estimate_qber(alice, bob)
    assert qber == 0.0
