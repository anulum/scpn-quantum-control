# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Crypto Properties
"""Property-based tests for cryptographic key derivation — elite multi-angle coverage."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scpn_quantum_control.crypto.hierarchical_keys import (
    derive_layer_key,
    derive_master_key,
    evolve_key_phases,
    group_key,
    hmac_sign,
    hmac_verify_key,
    key_hierarchy,
    rotating_key_schedule,
    verify_key_chain,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_symmetric_K(rng, n):
    K = rng.uniform(0, 1, (n, n))
    return (K + K.T) / 2


def _random_phases(rng, n):
    return rng.uniform(0, 2 * np.pi, n)


def _random_R(rng, phases):
    return float(abs(np.mean(np.exp(1j * phases))))


# ---------------------------------------------------------------------------
# derive_master_key
# ---------------------------------------------------------------------------


class TestDeriveMasterKey:
    def test_returns_32_bytes(self):
        K = np.eye(4)
        key = derive_master_key(K, R_global=0.5)
        assert isinstance(key, bytes)
        assert len(key) == 32

    def test_deterministic(self):
        K = np.eye(4)
        k1 = derive_master_key(K, R_global=0.5, nonce=b"test")
        k2 = derive_master_key(K, R_global=0.5, nonce=b"test")
        assert k1 == k2

    def test_different_R_gives_different_key(self):
        K = np.eye(4)
        k1 = derive_master_key(K, R_global=0.5)
        k2 = derive_master_key(K, R_global=0.9)
        assert k1 != k2

    def test_different_nonce_gives_different_key(self):
        K = np.eye(4)
        k1 = derive_master_key(K, R_global=0.5, nonce=b"a")
        k2 = derive_master_key(K, R_global=0.5, nonce=b"b")
        assert k1 != k2

    @pytest.mark.parametrize("n", [2, 4, 8])
    def test_various_sizes(self, n):
        K = np.eye(n)
        key = derive_master_key(K, R_global=0.5)
        assert len(key) == 32


# ---------------------------------------------------------------------------
# derive_layer_key
# ---------------------------------------------------------------------------


class TestDeriveLayerKey:
    def test_returns_32_bytes(self):
        K = np.eye(4)
        phases = np.zeros(1)
        key = derive_layer_key(K, 0, phases)
        assert len(key) == 32

    def test_different_layers_different_keys(self):
        K = np.eye(4)
        phases = np.zeros(1)
        keys = [derive_layer_key(K, i, phases) for i in range(4)]
        assert len(set(keys)) == 4

    def test_deterministic(self):
        K = np.eye(4) * 0.5
        phases = np.array([1.0])
        k1 = derive_layer_key(K, 2, phases, nonce=b"x")
        k2 = derive_layer_key(K, 2, phases, nonce=b"x")
        assert k1 == k2


# ---------------------------------------------------------------------------
# key_hierarchy (property-based)
# ---------------------------------------------------------------------------


@given(seed=st.integers(min_value=0, max_value=10000))
@settings(max_examples=20, deadline=5000)
def test_key_hierarchy_deterministic(seed: int) -> None:
    """Same inputs must always produce the same key hierarchy."""
    rng = np.random.default_rng(seed)
    n = 4
    K = _random_symmetric_K(rng, n)
    phases = _random_phases(rng, n)
    R = _random_R(rng, phases)

    h1 = key_hierarchy(K, phases, R, nonce=b"test")
    h2 = key_hierarchy(K, phases, R, nonce=b"test")

    assert h1["master"] == h2["master"]
    for i in range(n):
        assert h1["layers"][i] == h2["layers"][i]


@given(seed=st.integers(min_value=0, max_value=10000))
@settings(max_examples=20, deadline=5000)
def test_verify_key_chain_roundtrip(seed: int) -> None:
    """verify_key_chain must return True for freshly generated keys."""
    rng = np.random.default_rng(seed)
    n = 4
    K = _random_symmetric_K(rng, n)
    phases = _random_phases(rng, n)
    R = _random_R(rng, phases)

    h = key_hierarchy(K, phases, R, nonce=b"verify")
    assert verify_key_chain(h["master"], h["layers"], K, phases, R, nonce=b"verify")


class TestKeyHierarchyStructure:
    def test_has_master_and_layers(self):
        K = np.eye(4) * 0.3
        phases = np.zeros(4)
        h = key_hierarchy(K, phases, R_global=0.5)
        assert "master" in h
        assert "layers" in h
        assert len(h["layers"]) == 4

    def test_all_keys_are_32_bytes(self):
        K = np.eye(4)
        phases = np.ones(4)
        h = key_hierarchy(K, phases, R_global=0.8)
        assert len(h["master"]) == 32
        for key in h["layers"].values():
            assert len(key) == 32

    @pytest.mark.parametrize("n", [2, 3, 6, 8])
    def test_layers_count_matches_n(self, n):
        K = np.eye(n)
        phases = np.zeros(n)
        h = key_hierarchy(K, phases, R_global=0.5)
        assert len(h["layers"]) == n


# ---------------------------------------------------------------------------
# verify_key_chain
# ---------------------------------------------------------------------------


class TestVerifyKeyChain:
    def test_tampered_master_fails(self):
        K = np.eye(4) * 0.3
        phases = np.zeros(4)
        h = key_hierarchy(K, phases, R_global=0.5, nonce=b"n")
        bad_master = b"\x00" * 32
        assert not verify_key_chain(bad_master, h["layers"], K, phases, 0.5, nonce=b"n")

    def test_tampered_layer_key_fails(self):
        K = np.eye(4) * 0.3
        phases = np.zeros(4)
        h = key_hierarchy(K, phases, R_global=0.5, nonce=b"n")
        bad_layers = dict(h["layers"])
        bad_layers[0] = b"\xff" * 32
        assert not verify_key_chain(h["master"], bad_layers, K, phases, 0.5, nonce=b"n")

    def test_wrong_nonce_fails(self):
        K = np.eye(4) * 0.3
        phases = np.zeros(4)
        h = key_hierarchy(K, phases, R_global=0.5, nonce=b"a")
        assert not verify_key_chain(h["master"], h["layers"], K, phases, 0.5, nonce=b"b")


# ---------------------------------------------------------------------------
# HMAC sign / verify
# ---------------------------------------------------------------------------


@given(seed=st.integers(min_value=0, max_value=10000))
@settings(max_examples=20, deadline=5000)
def test_hmac_sign_verify_roundtrip(seed: int) -> None:
    """hmac_sign then hmac_verify_key must always agree."""
    rng = np.random.default_rng(seed)
    key = rng.bytes(32)
    msg = rng.bytes(64)
    tag = hmac_sign(key, msg)
    assert hmac_verify_key(key, msg, tag)


class TestHMAC:
    def test_tag_is_32_bytes(self):
        tag = hmac_sign(b"key123", b"message")
        assert len(tag) == 32

    def test_wrong_key_fails(self):
        tag = hmac_sign(b"correct_key", b"message")
        assert not hmac_verify_key(b"wrong_key", b"message", tag)

    def test_wrong_message_fails(self):
        tag = hmac_sign(b"key", b"correct_message")
        assert not hmac_verify_key(b"key", b"wrong_message", tag)

    def test_tampered_tag_fails(self):
        tag = hmac_sign(b"key", b"message")
        bad_tag = bytes([b ^ 0xFF for b in tag])
        assert not hmac_verify_key(b"key", b"message", bad_tag)

    def test_deterministic(self):
        t1 = hmac_sign(b"k", b"m")
        t2 = hmac_sign(b"k", b"m")
        assert t1 == t2

    def test_empty_message(self):
        tag = hmac_sign(b"key", b"")
        assert hmac_verify_key(b"key", b"", tag)


# ---------------------------------------------------------------------------
# evolve_key_phases
# ---------------------------------------------------------------------------


class TestEvolveKeyPhases:
    def test_output_shape(self):
        K = np.array([[0, 0.3], [0.3, 0]])
        omega = np.array([1.0, -1.0])
        theta_0 = np.array([0.0, np.pi / 4])
        traj = evolve_key_phases(K, omega, theta_0, t_window=1.0, n_samples=16)
        assert traj.shape == (2, 16)

    def test_initial_condition_preserved(self):
        K = np.eye(3) * 0.1
        omega = np.zeros(3)
        theta_0 = np.array([0.1, 0.2, 0.3])
        traj = evolve_key_phases(K, omega, theta_0, t_window=0.5)
        np.testing.assert_allclose(traj[:, 0], theta_0, atol=1e-6)

    @pytest.mark.parametrize("n", [2, 4, 6])
    def test_various_sizes(self, n):
        rng = np.random.default_rng(42)
        K = _random_symmetric_K(rng, n)
        omega = rng.uniform(-1, 1, n)
        theta_0 = rng.uniform(0, 2 * np.pi, n)
        traj = evolve_key_phases(K, omega, theta_0, t_window=0.5, n_samples=8)
        assert traj.shape == (n, 8)

    def test_all_values_finite(self):
        K = np.array([[0, 0.5], [0.5, 0]])
        omega = np.array([1.0, 2.0])
        theta_0 = np.zeros(2)
        traj = evolve_key_phases(K, omega, theta_0, t_window=2.0)
        assert np.all(np.isfinite(traj))


# ---------------------------------------------------------------------------
# rotating_key_schedule
# ---------------------------------------------------------------------------


class TestRotatingKeySchedule:
    def test_returns_correct_number_of_windows(self):
        K = np.array([[0, 0.3], [0.3, 0]])
        omega = np.array([1.0, -1.0])
        theta_0 = np.zeros(2)
        schedule = rotating_key_schedule(K, omega, theta_0, n_windows=3)
        assert len(schedule) == 3

    def test_each_window_has_expected_keys(self):
        K = np.array([[0, 0.5], [0.5, 0]])
        omega = np.array([1.0, -1.0])
        theta_0 = np.zeros(2)
        schedule = rotating_key_schedule(K, omega, theta_0, n_windows=2)
        for entry in schedule:
            assert "window" in entry
            assert "master" in entry
            assert "layers" in entry
            assert "R_global" in entry
            assert "final_phases" in entry
            assert 0.0 <= entry["R_global"] <= 1.0

    def test_different_windows_produce_different_master_keys(self):
        K = np.array([[0, 0.5], [0.5, 0]])
        omega = np.array([1.0, -1.0])
        theta_0 = np.array([0.0, np.pi / 3])
        schedule = rotating_key_schedule(K, omega, theta_0, n_windows=4)
        masters = [s["master"] for s in schedule]
        assert len(set(masters)) == 4

    def test_window_indices_sequential(self):
        K = np.array([[0, 0.3], [0.3, 0]])
        omega = np.ones(2)
        theta_0 = np.zeros(2)
        schedule = rotating_key_schedule(K, omega, theta_0, n_windows=5)
        assert [s["window"] for s in schedule] == list(range(5))


# ---------------------------------------------------------------------------
# group_key
# ---------------------------------------------------------------------------


class TestGroupKey:
    def test_returns_32_bytes(self):
        K = np.eye(4)
        phases = np.zeros(4)
        gk = group_key(K, [0, 1], phases)
        assert len(gk) == 32

    def test_different_subsets_different_keys(self):
        K = np.eye(4) * 0.5
        phases = np.linspace(0, np.pi, 4)
        gk1 = group_key(K, [0, 1], phases)
        gk2 = group_key(K, [2, 3], phases)
        assert gk1 != gk2

    def test_deterministic(self):
        K = np.eye(4)
        phases = np.ones(4)
        g1 = group_key(K, [0, 2], phases, nonce=b"x")
        g2 = group_key(K, [0, 2], phases, nonce=b"x")
        assert g1 == g2

    def test_single_member_group(self):
        K = np.eye(4)
        phases = np.zeros(4)
        gk = group_key(K, [0], phases)
        assert len(gk) == 32
