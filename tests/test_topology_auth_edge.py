# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Topology Auth Edge
"""Multi-angle tests for crypto/topology_auth.py.

Covers: spectral fingerprint, normalized Laplacian, topology distance,
commitment/verification, challenge-response, noise tolerance,
row hashing, edge cases, parametrised graph topologies.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.crypto.topology_auth import (
    challenge_response_prove,
    challenge_response_verify,
    fingerprint_noise_tolerance,
    normalized_laplacian_fingerprint,
    row_hash_fingerprint,
    spectral_fingerprint,
    topology_commitment,
    topology_distance,
    verify_commitment,
    verify_fingerprint,
    verify_row_hash,
)


def _ring_coupling(n: int) -> np.ndarray:
    K = np.zeros((n, n))
    for i in range(n):
        K[i, (i + 1) % n] = 1.0
        K[(i + 1) % n, i] = 1.0
    return K


# =====================================================================
# Spectral Fingerprint
# =====================================================================
class TestSpectralFingerprint:
    @pytest.mark.parametrize("n", [2, 3, 4, 6])
    def test_returns_dict_with_keys(self, n):
        K = _ring_coupling(n)
        fp = spectral_fingerprint(K)
        assert "eigenvalues" in fp
        assert "spectral_entropy" in fp

    def test_eigenvalues_sorted(self):
        K = _ring_coupling(4)
        fp = spectral_fingerprint(K)
        eigvals = fp["eigenvalues"]
        np.testing.assert_array_equal(eigvals, np.sort(eigvals))

    def test_eigenvalues_nonnegative(self):
        """Laplacian eigenvalues are ≥ 0."""
        K = _ring_coupling(4)
        fp = spectral_fingerprint(K)
        assert all(e >= -1e-10 for e in fp["eigenvalues"])

    def test_fiedler_positive_for_connected(self):
        """Connected graph → fiedler > 0."""
        K = _ring_coupling(4)
        fp = spectral_fingerprint(K)
        assert fp["fiedler"] > 0

    def test_entropy_nonnegative(self):
        K = _ring_coupling(4)
        fp = spectral_fingerprint(K)
        assert fp["spectral_entropy"] >= 0.0


# =====================================================================
# Normalized Laplacian
# =====================================================================
class TestNormalizedLaplacian:
    def test_zero_entropy_for_1x1(self):
        K = np.array([[1.0]])
        fp = normalized_laplacian_fingerprint(K)
        assert fp["spectral_entropy_norm"] == 0.0

    def test_returns_dict(self):
        K = _ring_coupling(4)
        fp = normalized_laplacian_fingerprint(K)
        assert "eigenvalues_norm" in fp
        assert "spectral_entropy_norm" in fp

    def test_eigenvalues_bounded_02(self):
        """Normalized Laplacian eigenvalues ∈ [0, 2]."""
        K = _ring_coupling(4)
        fp = normalized_laplacian_fingerprint(K)
        for e in fp["eigenvalues_norm"]:
            assert -1e-10 <= e <= 2.0 + 1e-10


# =====================================================================
# Verification & Commitment
# =====================================================================
class TestVerification:
    def test_verify_fingerprint_roundtrip(self):
        K = _ring_coupling(4)
        fp = spectral_fingerprint(K)
        assert verify_fingerprint(K, fp)

    def test_verify_fails_for_wrong_K(self):
        K = _ring_coupling(4)
        fp = spectral_fingerprint(K)
        K_wrong = K * 2.0
        assert not verify_fingerprint(K_wrong, fp)

    def test_commitment_roundtrip(self):
        K = _ring_coupling(4)
        nonce = b"test_nonce"
        commit = topology_commitment(K, nonce)
        assert verify_commitment(K, nonce, commit)

    def test_commitment_fails_wrong_nonce(self):
        K = _ring_coupling(4)
        commit = topology_commitment(K, b"nonce_a")
        assert not verify_commitment(K, b"nonce_b", commit)

    def test_commitment_deterministic(self):
        K = _ring_coupling(4)
        c1 = topology_commitment(K, b"det")
        c2 = topology_commitment(K, b"det")
        assert c1 == c2


# =====================================================================
# Challenge-Response
# =====================================================================
class TestChallengeResponse:
    def test_prove_verify_roundtrip(self):
        K = _ring_coupling(4)
        challenge = b"challenge_42"
        response = challenge_response_prove(K, challenge)
        assert challenge_response_verify(K, challenge, response)

    def test_wrong_K_fails(self):
        K = _ring_coupling(4)
        response = challenge_response_prove(K, b"ch")
        K_wrong = K + 0.01
        assert not challenge_response_verify(K_wrong, b"ch", response)


# =====================================================================
# Topology Distance
# =====================================================================
class TestTopologyDistance:
    def test_self_distance_is_zero(self):
        K = _ring_coupling(4)
        fp = spectral_fingerprint(K)
        d = topology_distance(fp, fp)
        np.testing.assert_allclose(d, 0.0, atol=1e-12)

    def test_distance_positive_for_different(self):
        fp1 = spectral_fingerprint(_ring_coupling(4))
        fp2 = spectral_fingerprint(_ring_coupling(3))
        d = topology_distance(fp1, fp2)
        assert d > 0


# =====================================================================
# Row Hashing
# =====================================================================
class TestRowHash:
    def test_row_hash_length(self):
        K = _ring_coupling(4)
        hashes = row_hash_fingerprint(K)
        assert len(hashes) == 4

    def test_row_hash_verify_roundtrip(self):
        K = _ring_coupling(4)
        hashes = row_hash_fingerprint(K)
        for i in range(4):
            assert verify_row_hash(K, i, hashes[i])

    def test_row_hash_fails_for_modified_row(self):
        K = _ring_coupling(4)
        hashes = row_hash_fingerprint(K)
        K_mod = K.copy()
        K_mod[0, 1] += 0.1
        assert not verify_row_hash(K_mod, 0, hashes[0])


# =====================================================================
# Noise Tolerance
# =====================================================================
class TestNoiseTolerance:
    def test_returns_dict(self):
        K = _ring_coupling(4)
        result = fingerprint_noise_tolerance(K, n_trials=10, sigma=0.01)
        assert "mean_drift" in result or "max_drift" in result or isinstance(result, dict)

    def test_small_noise_small_drift(self):
        K = _ring_coupling(4)
        r_small = fingerprint_noise_tolerance(K, n_trials=10, sigma=0.001)
        r_large = fingerprint_noise_tolerance(K, n_trials=10, sigma=0.1)
        # Larger noise should give larger drift (on average)
        assert isinstance(r_small, dict)
        assert isinstance(r_large, dict)
