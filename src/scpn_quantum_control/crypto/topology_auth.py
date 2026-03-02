"""Spectral fingerprint authentication for K_nm topology.

The Laplacian spectrum of K_nm provides a public authentication token.
Co-spectral graphs (different K_nm with same spectrum) exist, so
publishing the spectrum doesn't reveal K_nm — but any party with the
true K_nm can verify consistency.

Refs:
- Van Dam & Haemers (2003), "Which graphs are determined by their spectrum?"
- AAAI Symposium 2025, "Quantum Network Science: Graph Structure to Entanglement Performance"
"""

from __future__ import annotations

import hashlib
import hmac

import numpy as np
from scipy.stats import entropy as scipy_entropy

EIGENVALUE_ZERO_ATOL = 1e-12  # absolute tolerance for Laplacian zero eigenvalues
EIGENVALUE_ZERO_RTOL = 1e-8   # relative tolerance for Laplacian eigenvalue ratios


def spectral_fingerprint(K: np.ndarray) -> dict:
    """Compute public spectral fingerprint of coupling matrix.

    Returns dict with:
        fiedler: Second-smallest eigenvalue of graph Laplacian (algebraic connectivity).
        gap_ratio: lambda_1 / lambda_2 (spectral gap quality).
        spectral_entropy: Shannon entropy of normalized eigenvalue distribution.
        n_components: Number of connected components (eigenvalues ≈ 0).
    """
    n = K.shape[0]
    D = np.diag(K.sum(axis=1))
    L = D - K

    eigvals = np.sort(np.linalg.eigvalsh(L))

    fiedler = float(eigvals[1]) if n > 1 else 0.0
    gap_ratio = float(eigvals[1] / eigvals[2]) if n > 2 and eigvals[2] > EIGENVALUE_ZERO_ATOL else 0.0

    pos_eigvals = eigvals[eigvals > EIGENVALUE_ZERO_ATOL]
    if len(pos_eigvals) > 0:
        p = pos_eigvals / pos_eigvals.sum()
        s_entropy = float(scipy_entropy(p, base=2))
    else:
        s_entropy = 0.0

    n_components = int(np.sum(eigvals < EIGENVALUE_ZERO_RTOL))

    return {
        "fiedler": fiedler,
        "gap_ratio": gap_ratio,
        "spectral_entropy": s_entropy,
        "n_components": n_components,
        "eigenvalues": eigvals.tolist(),
    }


def normalized_laplacian_fingerprint(K: np.ndarray) -> dict:
    """Fingerprint from the normalized Laplacian L_sym = I - D^{-1/2} K D^{-1/2}.

    More robust to degree heterogeneity than the combinatorial Laplacian.
    Eigenvalues lie in [0, 2] for connected graphs.
    """
    n = K.shape[0]
    d = K.sum(axis=1)
    d_inv_sqrt = np.where(d > EIGENVALUE_ZERO_ATOL, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_sym = np.eye(n) - D_inv_sqrt @ K @ D_inv_sqrt

    eigvals = np.sort(np.linalg.eigvalsh(L_sym))
    fiedler = float(eigvals[1]) if n > 1 else 0.0

    pos_eigvals = eigvals[eigvals > EIGENVALUE_ZERO_ATOL]
    if len(pos_eigvals) > 0:
        p = pos_eigvals / pos_eigvals.sum()
        s_entropy = float(scipy_entropy(p, base=2))
    else:
        s_entropy = 0.0

    return {
        "fiedler_norm": fiedler,
        "spectral_entropy_norm": s_entropy,
        "eigenvalues_norm": eigvals.tolist(),
        "spectral_radius": float(eigvals[-1]),
    }


def verify_fingerprint(K: np.ndarray, fingerprint: dict, tol: float = 1e-6) -> bool:
    """Check K against a claimed spectral fingerprint."""
    computed = spectral_fingerprint(K)
    return (
        abs(computed["fiedler"] - fingerprint["fiedler"]) < tol
        and abs(computed["spectral_entropy"] - fingerprint["spectral_entropy"]) < tol
        and computed["n_components"] == fingerprint["n_components"]
    )


def topology_distance(fp1: dict, fp2: dict) -> float:
    """L2 distance between two spectral fingerprints.

    Useful for detecting calibration drift or K_nm tampering.
    """
    e1 = np.array(fp1["eigenvalues"])
    e2 = np.array(fp2["eigenvalues"])
    if len(e1) != len(e2):
        return float("inf")
    return float(np.linalg.norm(e1 - e2))


# --- Challenge-Response Authentication ---


def topology_commitment(K: np.ndarray, nonce: bytes = b"") -> bytes:
    """Commit to K_nm without revealing it.

    Returns SHA-256(K_nm_bytes || nonce). The commitment binds the prover
    to a specific K_nm. Later, the prover opens by revealing K_nm + nonce,
    and the verifier recomputes the hash.
    """
    h = hashlib.sha256()
    h.update(K.tobytes())
    h.update(nonce)
    return h.digest()


def verify_commitment(K: np.ndarray, nonce: bytes, commitment: bytes) -> bool:
    """Verify that K_nm matches a previously issued commitment."""
    return topology_commitment(K, nonce) == commitment


def challenge_response_prove(K: np.ndarray, challenge: bytes) -> bytes:
    """Prover: compute HMAC(K_nm, challenge) as proof of K_nm knowledge.

    The challenge is a random nonce from the verifier. The response
    proves the prover knows K_nm without transmitting it.
    """
    return hmac.new(K.tobytes(), challenge, hashlib.sha256).digest()


def challenge_response_verify(
    K: np.ndarray,
    challenge: bytes,
    response: bytes,
) -> bool:
    """Verifier: check that the response matches HMAC(K_nm, challenge)."""
    expected = hmac.new(K.tobytes(), challenge, hashlib.sha256).digest()
    return hmac.compare_digest(response, expected)


# --- Noise Tolerance ---


def fingerprint_noise_tolerance(K: np.ndarray, n_trials: int = 100, sigma: float = 0.01) -> dict:
    """Estimate fingerprint stability under small perturbations to K.

    Adds Gaussian noise N(0, sigma²) to K, recomputes fingerprint,
    measures drift. Returns mean and max drift across trials.
    """
    fp_ref = spectral_fingerprint(K)
    rng = np.random.default_rng(42)
    drifts = []

    for _ in range(n_trials):
        noise = rng.normal(0, sigma, K.shape)
        noise = (noise + noise.T) / 2  # keep symmetric
        K_noisy = np.maximum(K + noise, 0)  # keep non-negative
        np.fill_diagonal(K_noisy, 0)  # keep zero diagonal
        fp_noisy = spectral_fingerprint(K_noisy)
        drifts.append(topology_distance(fp_ref, fp_noisy))

    return {
        "sigma": sigma,
        "mean_drift": float(np.mean(drifts)),
        "max_drift": float(np.max(drifts)),
        "std_drift": float(np.std(drifts)),
        "safe_tol": float(np.percentile(drifts, 99)),
    }


def row_hash_fingerprint(K: np.ndarray) -> list[bytes]:
    """Per-row SHA-256 hashes of K_nm.

    Enables selective verification: prove knowledge of specific coupling
    rows without revealing the full matrix. Useful for hierarchical
    authentication where different parties control different SCPN layers.
    """
    return [hashlib.sha256(K[i, :].tobytes()).digest() for i in range(K.shape[0])]


def verify_row_hash(K: np.ndarray, row_idx: int, expected_hash: bytes) -> bool:
    """Verify a single row of K_nm against its hash."""
    return hashlib.sha256(K[row_idx, :].tobytes()).digest() == expected_hash
