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
import hmac
import struct

import numpy as np
from scipy.integrate import solve_ivp


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


# --- Time-Evolving Key Rotation ---


def _kuramoto_rhs(t: float, theta: np.ndarray, K: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Kuramoto ODE right-hand side: dθ/dt = ω + Σ K_nm sin(θ_m - θ_n)."""
    n = len(theta)
    dtheta = omega.copy()
    for i in range(n):
        for j in range(n):
            dtheta[i] += K[i, j] * np.sin(theta[j] - theta[i])
    return dtheta


def evolve_key_phases(
    K: np.ndarray,
    omega: np.ndarray,
    theta_0: np.ndarray,
    t_window: float,
    n_samples: int = 32,
) -> np.ndarray:
    """Evolve Kuramoto dynamics and sample phase trajectory.

    Returns (n_layers, n_samples) array of phase values over the time window.
    Each column is a snapshot at a different time.
    """
    t_eval = np.linspace(0, t_window, n_samples)
    sol = solve_ivp(
        _kuramoto_rhs,
        (0, t_window),
        theta_0,
        args=(K, omega),
        t_eval=t_eval,
        method="RK45",
    )
    return sol.y  # shape (n_layers, n_samples)


def rotating_key_schedule(
    K: np.ndarray,
    omega: np.ndarray,
    theta_0: np.ndarray,
    n_windows: int = 4,
    window_duration: float = 1.0,
) -> list[dict]:
    """Generate a sequence of key hierarchies from evolving Kuramoto dynamics.

    Each window produces a different key hierarchy because the phase
    trajectory changes. Natural key rotation without re-keying.

    Returns list of dicts, each with 'window', 'master', 'layers', 'R_global'.
    """
    theta = theta_0.copy()
    schedule = []

    for w in range(n_windows):
        trajectory = evolve_key_phases(K, omega, theta, window_duration)
        # R_global from final phases
        final_phases = trajectory[:, -1]
        R = float(abs(np.mean(np.exp(1j * final_phases))))
        nonce = struct.pack("!i", w)
        h = key_hierarchy(K, final_phases, R, nonce)
        schedule.append(
            {
                "window": w,
                "master": h["master"],
                "layers": h["layers"],
                "R_global": R,
                "final_phases": final_phases,
            }
        )
        theta = final_phases  # chain windows

    return schedule


# --- Group Key Agreement ---


def group_key(
    K: np.ndarray,
    member_layers: list[int],
    phases: np.ndarray,
    nonce: bytes = b"",
) -> bytes:
    """Derive a shared key for a subset of SCPN layers.

    Uses the sub-matrix K[members, members] and their phase values.
    Any subset of layers can form a group with a shared key.
    """
    sub_K = K[np.ix_(member_layers, member_layers)]
    sub_phases = phases[member_layers]
    h = hashlib.sha256()
    h.update(sub_K.tobytes())
    for idx in sorted(member_layers):
        h.update(struct.pack("!i", idx))
    h.update(sub_phases.tobytes())
    h.update(nonce)
    return h.digest()


def hmac_verify_key(key: bytes, message: bytes, expected_mac: bytes) -> bool:
    """HMAC-SHA256 verification for key-authenticated messages."""
    computed = hmac.new(key, message, hashlib.sha256).digest()
    return hmac.compare_digest(computed, expected_mac)


def hmac_sign(key: bytes, message: bytes) -> bytes:
    """HMAC-SHA256 signature for a message using a derived key."""
    return hmac.new(key, message, hashlib.sha256).digest()
