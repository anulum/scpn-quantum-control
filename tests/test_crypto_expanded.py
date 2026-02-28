"""Tests for expanded crypto functions: topology_auth, hierarchical_keys, percolation."""

from __future__ import annotations

import os

import numpy as np

from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27
from scpn_quantum_control.crypto.hierarchical_keys import (
    evolve_key_phases,
    group_key,
    hmac_sign,
    hmac_verify_key,
    rotating_key_schedule,
)
from scpn_quantum_control.crypto.percolation import (
    best_entanglement_path,
    robustness_random_removal,
    robustness_targeted_removal,
)
from scpn_quantum_control.crypto.topology_auth import (
    challenge_response_prove,
    challenge_response_verify,
    fingerprint_noise_tolerance,
    normalized_laplacian_fingerprint,
    row_hash_fingerprint,
    topology_commitment,
    verify_commitment,
    verify_row_hash,
)

K4 = build_knm_paper27(L=4)
OMEGA4 = np.array([1.329, 1.247, 1.183, 1.131])


# --- topology_auth expansions ---


class TestNormalizedLaplacianFingerprint:
    def test_returns_expected_keys(self):
        fp = normalized_laplacian_fingerprint(K4)
        assert "fiedler_norm" in fp
        assert "spectral_entropy_norm" in fp
        assert "eigenvalues_norm" in fp
        assert "spectral_radius" in fp

    def test_eigenvalues_in_0_2_range(self):
        fp = normalized_laplacian_fingerprint(K4)
        for e in fp["eigenvalues_norm"]:
            assert -1e-8 <= e <= 2.0 + 1e-8


class TestTopologyCommitment:
    def test_deterministic(self):
        c1 = topology_commitment(K4, b"nonce1")
        c2 = topology_commitment(K4, b"nonce1")
        assert c1 == c2

    def test_different_nonce_different_commitment(self):
        c1 = topology_commitment(K4, b"a")
        c2 = topology_commitment(K4, b"b")
        assert c1 != c2

    def test_verify_roundtrip(self):
        nonce = os.urandom(16)
        c = topology_commitment(K4, nonce)
        assert verify_commitment(K4, nonce, c)

    def test_verify_rejects_wrong_matrix(self):
        nonce = b"test"
        c = topology_commitment(K4, nonce)
        K_fake = K4 * 1.01
        assert not verify_commitment(K_fake, nonce, c)


class TestChallengeResponse:
    def test_prove_verify_roundtrip(self):
        challenge = os.urandom(32)
        response = challenge_response_prove(K4, challenge)
        assert challenge_response_verify(K4, challenge, response)

    def test_wrong_challenge_fails(self):
        challenge = b"real_challenge"
        response = challenge_response_prove(K4, challenge)
        assert not challenge_response_verify(K4, b"fake_challenge", response)

    def test_wrong_matrix_fails(self):
        challenge = os.urandom(32)
        response = challenge_response_prove(K4, challenge)
        K_fake = K4 * 0.99
        assert not challenge_response_verify(K_fake, challenge, response)


class TestFingerprintNoiseTolerance:
    def test_returns_expected_keys(self):
        result = fingerprint_noise_tolerance(K4, n_trials=10, sigma=0.01)
        assert "sigma" in result
        assert "mean_drift" in result
        assert "max_drift" in result
        assert "safe_tol" in result

    def test_larger_sigma_more_drift(self):
        r_small = fingerprint_noise_tolerance(K4, n_trials=20, sigma=0.001)
        r_large = fingerprint_noise_tolerance(K4, n_trials=20, sigma=0.1)
        assert r_large["mean_drift"] > r_small["mean_drift"]


class TestRowHashFingerprint:
    def test_length_matches_rows(self):
        hashes = row_hash_fingerprint(K4)
        assert len(hashes) == K4.shape[0]

    def test_verify_row_hash_correct(self):
        hashes = row_hash_fingerprint(K4)
        for i in range(K4.shape[0]):
            assert verify_row_hash(K4, i, hashes[i])

    def test_verify_row_hash_wrong_row(self):
        hashes = row_hash_fingerprint(K4)
        assert not verify_row_hash(K4, 0, hashes[1])


# --- hierarchical_keys expansions ---


class TestEvolveKeyPhases:
    def test_output_shape(self):
        theta_0 = np.zeros(4)
        trajectory = evolve_key_phases(K4, OMEGA4, theta_0, t_window=1.0, n_samples=16)
        assert trajectory.shape == (4, 16)

    def test_phases_evolve(self):
        theta_0 = np.zeros(4)
        trajectory = evolve_key_phases(K4, OMEGA4, theta_0, t_window=2.0)
        assert not np.allclose(trajectory[:, 0], trajectory[:, -1])


class TestRotatingKeySchedule:
    def test_returns_correct_number_of_windows(self):
        theta_0 = np.zeros(4)
        schedule = rotating_key_schedule(K4, OMEGA4, theta_0, n_windows=3)
        assert len(schedule) == 3

    def test_each_window_has_master_and_layers(self):
        theta_0 = np.zeros(4)
        schedule = rotating_key_schedule(K4, OMEGA4, theta_0, n_windows=2)
        for entry in schedule:
            assert "master" in entry
            assert "layers" in entry
            assert "R_global" in entry
            assert len(entry["master"]) == 32

    def test_different_windows_different_keys(self):
        theta_0 = np.zeros(4)
        schedule = rotating_key_schedule(K4, OMEGA4, theta_0, n_windows=3)
        masters = [e["master"] for e in schedule]
        assert len(set(masters)) == 3


class TestGroupKey:
    def test_deterministic(self):
        phases = np.array([0.1, 0.2, 0.3, 0.4])
        k1 = group_key(K4, [0, 1], phases, b"nonce")
        k2 = group_key(K4, [0, 1], phases, b"nonce")
        assert k1 == k2

    def test_different_members_different_key(self):
        phases = np.array([0.1, 0.2, 0.3, 0.4])
        k_01 = group_key(K4, [0, 1], phases, b"n")
        k_23 = group_key(K4, [2, 3], phases, b"n")
        assert k_01 != k_23

    def test_key_length(self):
        phases = np.array([0.1, 0.2, 0.3, 0.4])
        k = group_key(K4, [0, 1, 2], phases)
        assert len(k) == 32


class TestHMAC:
    def test_sign_verify_roundtrip(self):
        key = os.urandom(32)
        msg = b"test message"
        mac = hmac_sign(key, msg)
        assert hmac_verify_key(key, msg, mac)

    def test_wrong_key_fails(self):
        key = os.urandom(32)
        msg = b"test message"
        mac = hmac_sign(key, msg)
        assert not hmac_verify_key(os.urandom(32), msg, mac)

    def test_tampered_message_fails(self):
        key = os.urandom(32)
        mac = hmac_sign(key, b"original")
        assert not hmac_verify_key(key, b"tampered", mac)


# --- percolation expansions ---


class TestRobustnessRandomRemoval:
    def test_returns_expected_keys(self):
        result = robustness_random_removal(K4, n_trials=5)
        assert "mean_resilience" in result
        assert "min_resilience" in result
        assert "n_edges" in result

    def test_resilience_bounded(self):
        result = robustness_random_removal(K4, n_trials=10)
        assert 0 <= result["mean_resilience"] <= 1.0
        assert 0 <= result["min_resilience"] <= 1.0


class TestRobustnessTargetedRemoval:
    def test_returns_expected_keys(self):
        result = robustness_targeted_removal(K4)
        assert "edges_to_disconnect" in result
        assert "fraction" in result

    def test_fraction_bounded(self):
        result = robustness_targeted_removal(K4)
        assert 0 < result["fraction"] <= 1.0


class TestBestEntanglementPath:
    def test_path_exists_connected_graph(self):
        result = best_entanglement_path(K4, 0, 3)
        assert len(result["path"]) >= 2
        assert result["path"][0] == 0
        assert result["path"][-1] == 3
        assert result["bottleneck"] > 0

    def test_direct_neighbors(self):
        result = best_entanglement_path(K4, 0, 1)
        assert 0 in result["path"]
        assert 1 in result["path"]
        assert result["bottleneck"] > 0

    def test_disconnected_returns_empty(self):
        K_disconnected = np.zeros((4, 4))
        result = best_entanglement_path(K_disconnected, 0, 3)
        assert result["path"] == []
        assert result["bottleneck"] == 0.0

    def test_bottleneck_at_most_max_weight(self):
        result = best_entanglement_path(K4, 0, 3)
        assert result["bottleneck"] <= K4.max()
