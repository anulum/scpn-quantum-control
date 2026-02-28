"""SCPN layer hierarchy to key derivation tree.

The 16-layer SCPN hierarchy maps to a key tree where:
- Master key = hash(K_nm_full || R_global || nonce)
- Layer-n subkey = hash(K_nm[n,:] || phase_sequence_n || nonce)

Time-varying keys: Kuramoto phase sequences add temporal entropy.
Different time windows produce different keys from the same K_nm.

Ref: Improved group QKD with multi-party collaboration, Sci. Reports 2025
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np


def derive_master_key(
    K: np.ndarray,
    R_global: float,
    nonce: bytes = b"",
) -> bytes:
    """Master key from full coupling matrix + order parameter.

    32-byte SHA-256 digest of K_nm flattened || R_global || nonce.
    """
    h = hashlib.sha256()
    h.update(K.tobytes())
    h.update(struct.pack("!d", R_global))
    h.update(nonce)
    return h.digest()


def derive_layer_key(
    K: np.ndarray,
    layer_idx: int,
    phase_sequence: np.ndarray,
    nonce: bytes = b"",
) -> bytes:
    """Layer-specific subkey from coupling row + phase trajectory.

    Args:
        K: Full coupling matrix (uses row layer_idx).
        layer_idx: 0-indexed layer number.
        phase_sequence: Array of phase values theta_n(t) over time window.
        nonce: Additional entropy.
    """
    h = hashlib.sha256()
    h.update(K[layer_idx, :].tobytes())
    h.update(struct.pack("!i", layer_idx))
    h.update(phase_sequence.tobytes())
    h.update(nonce)
    return h.digest()


def key_hierarchy(
    K: np.ndarray,
    phases: np.ndarray,
    R_global: float,
    nonce: bytes = b"",
) -> dict:
    """Full hierarchy: master key + all layer subkeys.

    Args:
        K: n×n coupling matrix.
        phases: n-element array of current phase values.
        R_global: Global order parameter.
        nonce: Session nonce.

    Returns dict with 'master' (bytes) and 'layers' (dict[int, bytes]).
    """
    master = derive_master_key(K, R_global, nonce)
    n = K.shape[0]
    layers = {}
    for i in range(n):
        layers[i] = derive_layer_key(K, i, phases[i : i + 1], nonce)
    return {"master": master, "layers": layers}


def verify_key_chain(
    master: bytes,
    layer_keys: dict[int, bytes],
    K: np.ndarray,
    phases: np.ndarray,
    R_global: float,
    nonce: bytes = b"",
) -> bool:
    """Verify layer keys are consistent with master and K_nm.

    Recomputes all keys from K and checks equality.
    """
    expected = key_hierarchy(K, phases, R_global, nonce)
    if master != expected["master"]:
        return False
    return all(key == expected["layers"].get(idx) for idx, key in layer_keys.items())
