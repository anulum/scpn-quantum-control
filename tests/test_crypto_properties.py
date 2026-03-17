# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Property-based tests for cryptographic key derivation."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from scpn_quantum_control.crypto.hierarchical_keys import (
    hmac_sign,
    hmac_verify_key,
    key_hierarchy,
    verify_key_chain,
)


@given(seed=st.integers(min_value=0, max_value=10000))
@settings(max_examples=20, deadline=5000)
def test_key_hierarchy_deterministic(seed: int) -> None:
    """Same inputs must always produce the same key hierarchy."""
    rng = np.random.default_rng(seed)
    n = 4
    K = rng.uniform(0, 1, (n, n))
    K = (K + K.T) / 2
    phases = rng.uniform(0, 2 * np.pi, n)
    R = float(abs(np.mean(np.exp(1j * phases))))

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
    K = rng.uniform(0, 1, (n, n))
    K = (K + K.T) / 2
    phases = rng.uniform(0, 2 * np.pi, n)
    R = float(abs(np.mean(np.exp(1j * phases))))

    h = key_hierarchy(K, phases, R, nonce=b"verify")
    assert verify_key_chain(h["master"], h["layers"], K, phases, R, nonce=b"verify")


@given(seed=st.integers(min_value=0, max_value=10000))
@settings(max_examples=20, deadline=5000)
def test_hmac_sign_verify_roundtrip(seed: int) -> None:
    """hmac_sign then hmac_verify_key must always agree."""
    rng = np.random.default_rng(seed)
    key = rng.bytes(32)
    msg = rng.bytes(64)
    tag = hmac_sign(key, msg)
    assert hmac_verify_key(key, msg, tag)
